"""Differentiable joint drift + localization model.

Wraps the existing ``SetLocalizer`` encoder (centroid-relative x/y/z/alpha) and
adds a global, time-indexed drift parameter ``D(t)`` plus a population-level
soft-rasterization / cross-correlation ("Diff-DREDge") loss so that the encoder
is pushed to predict drift-invariant brain coordinates while ``D(t)`` absorbs
global probe motion.

Adaptation to the existing local-coords convention: the encoder predicts
per-spike ``(x, y, z, alpha)`` relative to the neighborhood centroid. Because
both the source and the channels are offset by the same ``centroid`` constant,
drift enters the monopole distance purely additively in ``dz``:

    dz = (z_local + D(t)) - zc_local

The rasterizer builds R_t(z) in **drift-corrected brain frame**:

    Z_brain = (centroid_z + z_local) - D(t)

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
import torch.nn.functional as F

from model import (
    SetLocalizer,
    fourier_positional_embedding,
    build_knn_attention_mask,
    compute_feature,
    masked_recon_loss,
)


class UnifiedDriftLocalizer(nn.Module):
    """SetLocalizer encoder + differentiable drift field + Diff-DREDge loss.

    Drift is stored as a per-time-bin scalar vector ``global_drift`` of length
    ``num_bins = ceil(total_duration / bin_width) + 1`` and sampled at arbitrary
    continuous timestamps via bilinear ``grid_sample``. ``D(t_0)`` is pinned to 0
    (gauge anchor) by subtracting ``global_drift[0]`` from every sampled value.
    """

    def __init__(self, n_channels, n_samples, total_recording_duration_sec,
                 pos_dim=8, feat_dim=32, hidden=128, num_heads=4, max_freq=0.1,
                 use_knn=False, knn_k=16, bin_width_sec=1.0, max_z=3840.0,
                 spatial_grid_step=1.0, sigma=2.0, temporal_window_bins=20,
                 max_shift_bins=30, beta=15.0, b=1.0, loss_type="mse",
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

    def get_drift(self, times_sec):
        """Bilinearly sample ``D(t)`` at continuous timestamps, pinned to 0 at t0."""
        bin_coords = times_sec / self.bin_width_sec
        normalized = (bin_coords / max(self.num_bins - 1, 1)) * 2.0 - 1.0
        normalized = normalized.clamp(-1.0, 1.0)

        grid_input = torch.stack([normalized, torch.zeros_like(normalized)], dim=-1)
        grid_input = grid_input.unsqueeze(0).unsqueeze(0)

        drift_grid = self.global_drift.view(1, 1, 1, -1)
        sampled = F.grid_sample(drift_grid, grid_input, align_corners=True,
                                mode='bilinear', padding_mode='border')
        return sampled.squeeze(0).squeeze(0).squeeze(0) - self.global_drift[0]

    def _encoder_forward(self, wf, coords, mask):
        xc, zc = coords[..., 0], coords[..., 1]
        pos_emb = fourier_positional_embedding(xc, zc, self.pos_dim, self.max_freq)
        knn = build_knn_attention_mask(xc, zc, mask, k=self.knn_k) if self.use_knn else None
        x, y, z, alpha = self.encoder(wf, pos_emb, mask, knn_allowed=knn)
        return x, y, z, alpha, xc, zc

    def monopole_decoder(self, z_local, drift, zc_local):
        """Per-channel PTP with drift folded into dz."""
        dz = (z_local.unsqueeze(-1) + drift.unsqueeze(-1)) - zc_local
        return dz

    def soft_rasterize(self, z_brain_driftcorr, alpha, times_sec):
        """Continuous spatial density R_t(z) over the full recording's time bins.

        R: (num_bins, grid_len). Each row t sums Gaussian bumps from all spikes
        in time bin t, placed at their drift-corrected brain-frame depth. When
        D(t) is correct, the same neuron contributes to the same spatial location
        across all time bins, so R is temporally stable.

        z_brain_driftcorr: (S,)  drift-corrected brain-frame depth
        alpha:             (S,)
        times_sec:         (S,)
        """
        device = z_brain_driftcorr.device
        grid = torch.arange(0.0, self.max_z, self.spatial_grid_step, device=device)
        grid_len = grid.numel()

        bin_ids = torch.clamp((times_sec / self.bin_width_sec).long(),
                              0, self.num_bins - 1)
        dist_sq = (grid.unsqueeze(0) - z_brain_driftcorr.unsqueeze(1)) ** 2
        weights = alpha.unsqueeze(1) * torch.exp(-dist_sq / (2 * self.sigma ** 2))

        R = torch.zeros(self.num_bins, grid_len, device=device)
        R.index_add_(0, bin_ids, weights)
        return R

    def compute_diff_dredge_loss(self, R):
        """Normalized cross-correlation Diff-DREDge loss via grouped conv1d + soft-argmax.

        Each R_t row is L2-normalized before cross-correlation (like DREDge's
        normxcorr1d), so bins with different spike counts produce comparable
        correlation values. For each lag w in [1, window_bins], we correlate
        R[t] with R[t+w] across a [-max_shift, max_shift] spatial shift search,
        pass through soft-argmax to get the estimated shift, and penalize
        shift^2 with exponential temporal decay.

        R: (num_bins, grid_len)
        """
        T, grid_len = R.size()
        if T < 2:
            return torch.tensor(0.0, device=R.device)

        row_norm = R.norm(dim=1, keepdim=True).clamp(min=1e-8)
        R_norm = R / row_norm

        max_shift = self.max_shift_bins
        padded = F.pad(R_norm, (max_shift, max_shift), mode='constant', value=0)

        shifts = torch.arange(-max_shift, max_shift + 1, dtype=torch.float32, device=R.device)
        loss = torch.tensor(0.0, device=R.device)
        denom = 0.0
        for w in range(1, self.window_bins + 1):
            if T <= w:
                break
            n = T - w
            signals = padded[0:n].unsqueeze(0)
            kernels = R_norm[w:T].unsqueeze(1)
            corr = F.conv1d(signals, kernels, groups=n).squeeze(0)
            probs = F.softmax(corr * self.beta, dim=-1)
            est = torch.sum(shifts.unsqueeze(0) * probs, dim=-1)
            decay = torch.exp(torch.tensor(-float(w) / self.window_bins, device=R.device))
            loss = loss + decay * torch.mean(est ** 2)
            denom += 1.0
        return loss / max(denom, 1.0)

    def forward(self, wf, coords, mask, centroid, times_sec, gt_ptp=None):
        """Returns (loss_monopole, loss_dredge, loss_smooth, extras dict).

        wf:        (S, N, T_samples)
        coords:    (S, N, 2)   local channel coords (x, z) relative to centroid
        mask:      (S, N) bool
        centroid:  (S, 2)       absolute probe-frame centroid (x, z)
        times_sec: (S,)
        gt_ptp:    (S, N) optional; if None, computed from wf via recon_feature.
        """
        x, y, z_local, alpha, xc, zc = self._encoder_forward(wf, coords, mask)

        drift = self.get_drift(times_sec)
        dz = self.monopole_decoder(z_local, drift, zc)
        dx = x.unsqueeze(-1) - xc
        r2 = dx ** 2 + dz ** 2 + y.unsqueeze(-1) ** 2 + self.channel_insulation_constant ** 2
        ptp_pred = alpha.unsqueeze(-1) / torch.sqrt(r2)

        if gt_ptp is None:
            gt_ptp = compute_feature(wf, self.recon_feature)
        loss_monopole = masked_recon_loss(gt_ptp, ptp_pred, mask, self.loss_type)

        z_probe = centroid[:, 1] + z_local
        z_brain_driftcorr = z_probe - drift
        R = self.soft_rasterize(z_brain_driftcorr, alpha, times_sec)
        loss_dredge = self.compute_diff_dredge_loss(R)

        loss_smooth = torch.mean(torch.diff(self.global_drift) ** 2)

        extras = {
            "x_brain_abs": centroid[:, 0] + x,
            "z_brain_abs": z_probe,
            "z_brain_driftcorr": z_brain_driftcorr,
            "y": y, "alpha": alpha,
            "drift": drift, "ptp_pred": ptp_pred,
        }
        return loss_monopole, loss_dredge, loss_smooth, extras
