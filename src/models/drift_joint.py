"""Alternating dense-population joint localization and rigid-drift model."""

import torch
import torch.nn as nn

from drift_population import RigidSplineDrift, resample_trace
from drift_registration import sample_drift
from model import (
    SetLocalizer,
    build_knn_attention_mask,
    compute_feature,
    fourier_positional_embedding,
    masked_recon_loss,
    physics_forward,
)


class JointDriftLocalizer(nn.Module):
    """Probe-frame waveform localizer plus a session-specific rigid spline."""

    def __init__(self, localizer_config, num_time_bins, num_drift_knots,
                 time_bin_sec, initial_drift=None):
        super().__init__()
        self.localizer_config = dict(localizer_config)
        self.time_bin_sec = float(time_bin_sec)
        self.num_time_bins = int(num_time_bins)
        if self.time_bin_sec <= 0 or self.num_time_bins < 2:
            raise ValueError("invalid drift time grid")

        cfg = self.localizer_config
        self.encoder = SetLocalizer(
            n_channels=cfg["n_channels"],
            n_samples=cfg["n_samples"],
            pos_dim=cfg["pos_dim"],
            feat_dim=cfg["feat_dim"],
            hidden=cfg["hidden"],
            num_heads=cfg["num_heads"],
        )
        if initial_drift is None:
            initial_drift = torch.zeros(self.num_time_bins)
        if initial_drift.shape != (self.num_time_bins,):
            raise ValueError("initial_drift must contain one value per time bin")
        self.drift = RigidSplineDrift(initial_drift, num_drift_knots)
        self.register_buffer(
            "channel_insulation_constant",
            torch.tensor(float(cfg["b"]), dtype=torch.float32),
        )

    def encode(self, waveforms, local_coords, mask, centroids):
        xc = local_coords[..., 0]
        zc = local_coords[..., 1]
        cfg = self.localizer_config
        position = fourier_positional_embedding(
            xc, zc, cfg["pos_dim"], cfg["max_freq"])
        knn = (
            build_knn_attention_mask(
                xc, zc, mask, k=cfg.get("knn_k", 16))
            if cfg["use_knn"] else None
        )
        x_local, y, z_local, alpha, embedding = self.encoder(
            waveforms, position, mask, knn_allowed=knn,
            return_embedding=True)
        return {
            "x_local": x_local,
            "y": y,
            "z_local": z_local,
            "alpha": alpha,
            "embedding": embedding,
            "x_probe": centroids[:, 0] + x_local,
            "z_probe": centroids[:, 1] + z_local,
            "xc_local": xc,
            "zc_local": zc,
        }

    def local_reconstruction_loss(self, waveforms, local_coords, mask,
                                  centroids):
        outputs = self.encode(waveforms, local_coords, mask, centroids)
        predicted = physics_forward(
            outputs["x_local"], outputs["y"], outputs["z_local"],
            outputs["alpha"], outputs["xc_local"], outputs["zc_local"],
            self.channel_insulation_constant)
        target = compute_feature(
            waveforms, self.localizer_config.get("recon_feature", "ptp"))
        loss = masked_recon_loss(
            target, predicted, mask,
            self.localizer_config.get("loss_type", "mse"))
        outputs["ptp_pred"] = predicted
        return loss, outputs

    def drift_trace(self):
        return self.drift(self.num_time_bins)

    def sample_drift(self, times_sec):
        return sample_drift(
            self.drift_trace(), times_sec, self.time_bin_sec)

    def correct_depth(self, z_probe, times_sec):
        return z_probe - self.sample_drift(times_sec)

    def set_drift_trace_(self, trace):
        if trace.shape != (self.num_time_bins,):
            raise ValueError("trace must contain one value per time bin")
        knots = resample_trace(trace, self.drift.knots.numel())
        with torch.no_grad():
            self.drift.knots.copy_(knots)

    def clamp_drift_(self, max_abs_um):
        self.drift.clamp_(max_abs_um)

    def set_encoder_trainable(self, trainable):
        for parameter in self.encoder.parameters():
            parameter.requires_grad = bool(trainable)

    def set_drift_trainable(self, trainable):
        self.drift.knots.requires_grad = bool(trainable)

