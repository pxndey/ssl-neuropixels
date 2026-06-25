"""End-to-end pipeline: waveform encoder (B) -> differentiable DREDge (A).

Wires Part B's transformer output into Part A's soft-binning seam exactly as the
Integration section specifies:

  1. The encoder produces per-channel tokens for a batch of spike neighborhoods;
     the peak channel's token is projected to a **cleaned per-spike feature**
     (amplitude) and an optional **position refinement** (dx, dy in microns).
  2. Each spike's continuous ``(x, y, t)`` -- absolute peak position (+ refinement)
     and time bin -- together with its cleaned feature are handed to differentiable
     soft-binning to build the space-time activity raster.
  3. The raster flows through differentiable DREDge (A.2->A.3->A.1) to the motion
     trace ``P``.

Because every stage is differentiable, a downstream loss on ``P`` backpropagates
through: linear solve -> sigmoid gate -> soft-argmax -> soft-binning ->
decoder/transformer/encoder, training the feature extractor jointly with the
motion-correction objective.  Part B's own masked-reconstruction ``dredge_loss``
remains available as an auxiliary/pretraining term, and
:meth:`compute_losses` can combine the two.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dredge_diff import DiffDredge, DredgeConfig
from waveform_encoder.model import EncoderConfig, WaveformMAE
from waveform_encoder.loss import dredge_loss, target_mask

_AXIS_COL = {"x": 0, "y": 1}


class SpikeLocalizationMotionPipeline(nn.Module):
    """Couples :class:`WaveformMAE` (upstream) with :class:`DiffDredge` (downstream)."""

    def __init__(
        self,
        encoder_cfg: Optional[EncoderConfig] = None,
        dredge_cfg: Optional[DredgeConfig] = None,
        *,
        refine_positions: bool = True,
        feature_activation: str = "abs",   # "abs" | "softplus" | "none"
    ):
        super().__init__()
        self.encoder = WaveformMAE(encoder_cfg or EncoderConfig())
        self.dredge = DiffDredge(dredge_cfg or DredgeConfig())
        self.refine_positions = refine_positions
        self.feature_activation = feature_activation

    # ------------------------------------------------------------------ #
    def _feature(self, amplitude: torch.Tensor) -> torch.Tensor:
        """Non-negative activity weight from the raw per-spike amplitude head."""
        if self.feature_activation == "abs":
            return amplitude.abs()
        if self.feature_activation == "softplus":
            return F.softplus(amplitude)
        return amplitude

    def encode_spikes(
        self,
        waveforms: torch.Tensor,
        coords: torch.Tensor,
        content_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        peak_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Encoder forward -> cleaned per-spike feature, position offset, recon."""
        out = self.encoder(waveforms, coords, content_mask, padding_mask, peak_idx=peak_idx)
        return {
            "recon": out["recon"],
            "tokens": out["tokens"],
            "feature": self._feature(out["amplitude"]),    # (B,)
            "pos_offset": out["pos_offset"],               # (B, 2)
        }

    # ------------------------------------------------------------------ #
    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_extras: bool = False,
    ):
        """Run the full pipeline for one batch of spikes.

        ``batch`` must provide:
            waveforms      (B, N, 90)
            coords         (B, N, 2)    microns, relative to peak channel
            content_mask   (B, N) bool
            padding_mask   (B, N) bool
            peak_idx       (B,)   long
            peak_xy        (B, 2) microns, absolute peak-channel position on the probe
            time_idx       (B,)   long, time-bin index of each spike
            n_time         int
        Optional: ``spatial_centers`` (dict), ``contact_depths`` (tensor).

        Returns ``P`` (motion trace), and ``extras`` if requested.
        """
        enc = self.encode_spikes(
            batch["waveforms"], batch["coords"],
            batch["content_mask"], batch["padding_mask"], batch["peak_idx"],
        )

        peak_xy = batch["peak_xy"].to(enc["feature"].dtype)
        pos = peak_xy + enc["pos_offset"] if self.refine_positions else peak_xy   # (B, 2)

        spike_coords = {d: pos[:, _AXIS_COL[d]] for d in self.dredge.spatial_axes}

        n_time = int(batch["n_time"])
        P, dredge_extras = self.dredge(
            spike_coords=spike_coords,
            spike_features=enc["feature"],
            spike_time_idx=batch["time_idx"].to(torch.long),
            n_time=n_time,
            spatial_centers=batch.get("spatial_centers"),
            contact_depths=batch.get("contact_depths"),
            return_extras=True,
        )

        if not return_extras:
            return P

        extras = {
            "recon": enc["recon"],
            "feature": enc["feature"],
            "position": pos,
            "pos_offset": enc["pos_offset"],
            **dredge_extras,
        }
        return P, extras

    # ------------------------------------------------------------------ #
    def compute_losses(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        motion_target: Optional[torch.Tensor] = None,
        w_ssl: float = 1.0,
        w_motion: float = 1.0,
        ssl_kind: str = "mse",
    ) -> Dict[str, torch.Tensor]:
        """Combine the auxiliary SSL reconstruction loss with a motion loss.

        The motion loss is MSE to ``motion_target`` when provided, otherwise a
        temporal-smoothness regularizer on ``P`` (a stand-in downstream objective
        useful for end-to-end gradient checks / fine-tuning experiments).
        """
        P, extras = self.forward(batch, return_extras=True)

        tmask = target_mask(batch["content_mask"], batch["padding_mask"])
        ssl = dredge_loss(extras["recon"], batch["waveforms"], mask=tmask, kind=ssl_kind)

        if motion_target is not None:
            motion = F.mse_loss(P, motion_target.to(P.dtype))
        else:
            # temporal smoothness of the motion trace over time bins
            if P.shape[0] > 1:
                motion = (P[1:] - P[:-1]).pow(2).mean()
            else:
                motion = P.pow(2).mean()

        total = w_ssl * ssl + w_motion * motion
        return {"total": total, "ssl": ssl, "motion": motion, "P": P}
