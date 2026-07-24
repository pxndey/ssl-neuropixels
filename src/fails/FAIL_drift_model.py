"""Differentiable joint drift + localization model.

Wraps the existing ``SetLocalizer`` encoder (centroid-relative x/y/z/alpha) and
adds a global, time-indexed drift parameter ``D(t)`` plus a population-level
soft-rasterization / cross-correlation ("Diff-DREDge") loss so that the encoder
is pushed to predict drift-invariant brain coordinates while ``D(t)`` absorbs
global probe motion.

The encoder predicts per-spike ``(x, y, z, alpha)`` relative to the
neighborhood centroid. The local monopole geometry is therefore independent of
global drift:

    z_probe = centroid_z + z_local
    dz_monopole = z_local - z_channel_local
    z_brain = z_probe - D(t)

When D(t) correctly tracks global probe motion, the same neuron appears at the
same Z_brain at all times, so R_t is temporally stable and the Diff-DREDge
cross-correlation shifts collapse to ~0. The loss pushes D(t) to absorb drift
and the encoder to predict consistent z_local.

The Diff-DREDge loss uses **normalized** cross-correlation (each R_t row is
L2-normalized before conv1d, like DREDge's normxcorr1d), so bins with different
spike counts are comparable. See /scratch/ap7151/dredge-source/dredge.py.
"""

import torch
import torch.nn as nn

from model import (
    SetLocalizer,
    fourier_positional_embedding,
    build_knn_attention_mask,
    compute_feature,
    masked_recon_loss,
)
from drift_registration import (
    RegistrationConfig,
    sample_drift,
    soft_rasterize,
    time_bin_counts,
    zero_shift_nll,
)


class UnifiedDriftLocalizer(nn.Module):
    """SetLocalizer encoder + differentiable drift field + Diff-DREDge loss.

    Drift is stored as a per-time-bin scalar vector ``global_drift`` of length
    ``num_bins = ceil(total_duration / bin_width) + 1`` and sampled at arbitrary
    continuous timestamps by linear interpolation. ``D(t_0)`` is pinned to zero
    by subtracting ``global_drift[0]`` from every sampled value.
    """

    def __init__(self, n_channels, n_samples, total_recording_duration_sec,
                 pos_dim=8, feat_dim=32, hidden=128, num_heads=4, max_freq=0.1,
                 use_knn=False, knn_k=16, bin_width_sec=1.0, max_z=3840.0,
                 spatial_grid_step=1.0, sigma=2.0, temporal_window_bins=20,
                 max_shift_bins=30, beta=3.0, b=1.0, loss_type="mse",
                 recon_feature="ptp", init_drift_scale=0.0):
        super().__init__()
        self.encoder = SetLocalizer(
            n_channels=n_channels, n_samples=n_samples, pos_dim=pos_dim,
            feat_dim=feat_dim, hidden=hidden, num_heads=num_heads)
        self.pos_dim = pos_dim
        self.use_knn = use_knn
        self.knn_k = knn_k
        self.max_freq = max_freq
        self.b = b
        self.loss_type = loss_type
        self.recon_feature = recon_feature

        self.bin_width_sec = bin_width_sec
        self.num_bins = int(total_recording_duration_sec / bin_width_sec) + 1
        self.global_drift = nn.Parameter(
            torch.zeros(self.num_bins) + init_drift_scale)
        self.channel_insulation_constant = nn.Parameter(torch.tensor([5.0]))

        self.max_z = max_z
        self.spatial_grid_step = spatial_grid_step
        self.sigma = sigma
        self.window_bins = temporal_window_bins
        self.max_shift_bins = max_shift_bins
        self.beta = beta
        self.registration_config = RegistrationConfig(
            num_time_bins=self.num_bins, bin_width_sec=bin_width_sec,
            spatial_min=0.0, spatial_max=max_z,
            spatial_grid_step=spatial_grid_step, sigma=sigma,
            temporal_window_bins=temporal_window_bins,
            max_shift_bins=max_shift_bins, beta=beta)
        self.last_registration_diagnostics = None

    def freeze_encoder(self):
        """Freeze encoder parameters (for M-step)."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.channel_insulation_constant.requires_grad = False

    def freeze_drift(self):
        """Freeze drift parameters (for E-step)."""
        self.global_drift.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.channel_insulation_constant.requires_grad = True
        self.global_drift.requires_grad = True

    def init_drift_smooth(self, mode="smooth_random", amplitude=30.0, seed=0):
        """Initialize ``global_drift`` with a smooth non-zero probe-motion trace.

        Modes:
            smooth_random: interpolate and smooth random coarse points
            gaussian_bump: single Gaussian bump centered at midpoint
            linear: linear ramp from 0 to amplitude
            zero: keep as zeros (default init)
        """
        if mode == "zero":
            return
        with torch.no_grad():
            import numpy as np
            if mode == "smooth_random":
                rng = np.random.default_rng(seed)
                num_knots = min(20, self.num_bins)
                coarse = rng.uniform(-amplitude, amplitude, num_knots)
                x_coarse = np.linspace(0, self.num_bins - 1, num_knots)
                x_fine = np.arange(self.num_bins)
                init = np.interp(x_fine, x_coarse, coarse)
                kernel_size = max(self.num_bins // (4 * num_knots), 3)
                sigma = kernel_size / 2.0
                kernel_points = torch.arange(
                    -kernel_size // 2, kernel_size // 2 + 1).float()
                kernel = torch.exp(-kernel_points.square() / (2 * sigma**2))
                kernel = kernel / kernel.sum()
                init_t = torch.tensor(init, dtype=torch.float32).reshape(1, 1, -1)
                init_smooth = torch.nn.functional.conv1d(
                    init_t, kernel.reshape(1, 1, -1), padding=kernel_size // 2)
                init = init_smooth.squeeze().numpy()
            elif mode == "gaussian_bump":
                t = np.arange(self.num_bins)
                center = self.num_bins / 2
                width = self.num_bins / 6
                init = amplitude * np.exp(-(t - center)**2 / (2 * width**2))
            elif mode == "linear":
                init = amplitude * np.linspace(-1, 1, self.num_bins)
            else:
                raise ValueError(f"unknown init_drift mode: {mode}")

            init = torch.tensor(
                init, dtype=self.global_drift.dtype, device=self.global_drift.device)
            self.global_drift.copy_(init - init[0])

    def get_drift(self, times_sec):
        """Sample the gauge-fixed global drift field at continuous timestamps."""
        return sample_drift(self.global_drift, times_sec, self.bin_width_sec)

    def _encoder_forward(self, wf, coords, mask):
        xc, zc = coords[..., 0], coords[..., 1]
        pos_emb = fourier_positional_embedding(
            xc, zc, self.pos_dim, self.max_freq)
        knn = (build_knn_attention_mask(xc, zc, mask, k=self.knn_k)
               if self.use_knn else None)
        x, y, z, alpha = self.encoder(wf, pos_emb, mask, knn_allowed=knn)
        return x, y, z, alpha, xc, zc

    def monopole_decoder(self, z_local, zc_local):
        """Return local source-to-channel depth offsets for the PTP decoder."""
        dz = z_local.unsqueeze(-1) - zc_local
        return dz

    def soft_rasterize(self, z_brain, weights, times_sec):
        return soft_rasterize(z_brain, weights, times_sec, self.registration_config)

    def compute_diff_dredge_loss(self, raster):
        """Compatibility entry point when event-count diagnostics are unavailable."""
        loss, diagnostics = zero_shift_nll(raster, self.registration_config)
        self.last_registration_diagnostics = diagnostics
        return loss

    def compute_registration_loss(self, raster, times_sec):
        """Evaluate registration with exact per-bin event-count diagnostics."""
        spike_counts = time_bin_counts(times_sec, self.registration_config)
        loss, diagnostics = zero_shift_nll(
            raster, self.registration_config, spike_counts=spike_counts)
        self.last_registration_diagnostics = diagnostics
        return loss

    def encode_probe_frame(self, wf, coords, mask, centroid, times_sec=None):
        """Encode localizations and optionally sample drift without rasterization.

        This is the prediction/export path: it never constructs a population
        raster or registration objective.
        """
        x, y, z_local, alpha, xc, zc = self._encoder_forward(wf, coords, mask)
        z_probe = centroid[:, 1] + z_local
        drift = self.get_drift(times_sec) if times_sec is not None else None
        z_brain = z_probe - drift if drift is not None else None
        return {
            "x_probe": centroid[:, 0] + x,
            "z_probe": z_probe,
            "z_brain": z_brain,
            "y": y,
            "alpha": alpha,
            "drift": drift,
            "_x_local": x,
            "_z_local": z_local,
            "_xc_local": xc,
            "_zc_local": zc,
        }

    def forward(self, wf, coords, mask, centroid, times_sec, gt_ptp=None,
                phase="all"):
        """Returns (loss_monopole, loss_dredge, loss_smooth, extras dict).

        phase: "monopole" | "dredge" | "inference" | "all"
            monopole: compute only monopole loss (skip raster/dredge)
            dredge: compute only dredge+smooth (skip monopole PTP)
            inference: encode and sample drift only (skip both losses)
            all: compute everything

        wf:        (S, N, T_samples)
        coords:    (S, N, 2)   local channel coords (x, z) relative to centroid
        mask:      (S, N) bool
        centroid:  (S, 2)       absolute probe-frame centroid (x, z)
        times_sec: (S,)
        gt_ptp:    (S, N) optional; if None, computed from wf via recon_feature.
        """
        if phase not in ("monopole", "dredge", "inference", "all"):
            raise ValueError(f"unknown phase: {phase}")
        drift_times = times_sec if phase != "monopole" else None
        extras = self.encode_probe_frame(wf, coords, mask, centroid, drift_times)
        x = extras.pop("_x_local")
        z_local = extras.pop("_z_local")
        xc = extras.pop("_xc_local")
        zc = extras.pop("_zc_local")
        z_probe = extras["z_probe"]
        y = extras["y"]
        alpha = extras["alpha"]

        loss_monopole = z_probe.new_zeros(())
        ptp_pred = None
        if phase in ("monopole", "all"):
            dz = self.monopole_decoder(z_local, zc)
            dx = x.unsqueeze(-1) - xc
            r2 = (dx.square() + dz.square() + y.unsqueeze(-1).square()
                  + self.channel_insulation_constant.square())
            ptp_pred = alpha.unsqueeze(-1) / torch.sqrt(r2)
            if gt_ptp is None:
                gt_ptp = compute_feature(wf, self.recon_feature)
            loss_monopole = masked_recon_loss(gt_ptp, ptp_pred, mask, self.loss_type)

        loss_dredge = z_probe.new_zeros(())
        loss_smooth = z_probe.new_zeros(())
        registration = None
        if phase in ("dredge", "all"):
            raster = self.soft_rasterize(extras["z_brain"], alpha, times_sec)
            loss_dredge = self.compute_registration_loss(raster, times_sec)
            curvature = torch.diff(self.global_drift, n=2)
            loss_smooth = (curvature.square().mean() if curvature.numel()
                           else z_probe.new_zeros(()))
            registration = self.last_registration_diagnostics

        extras["ptp_pred"] = ptp_pred
        extras["registration"] = registration
        return loss_monopole, loss_dredge, loss_smooth, extras
