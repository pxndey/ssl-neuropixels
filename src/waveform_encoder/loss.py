"""``dredge_loss`` -- the masked waveform reconstruction loss.

Named for the pipeline it feeds (the cleaned waveforms it shapes are what flow
into differentiable DREDge), this is the self-supervised objective for Part B:
reconstruct the hidden (masked) channels' waveforms from the visible ones.

It is computed **only on real, masked positions** -- ``content_mask & ~padding_mask``
-- so padding slots and visible channels never contribute.

Default is mean-squared error; a Huber (smooth-L1) variant is available for
robustness to the heavy-tailed outliers common in extracellular snippets.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def dredge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    kind: str = "mse",
    delta: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Masked reconstruction loss.

    Parameters
    ----------
    pred, target : tensors of the same shape
        If ``mask`` is given they are ``(B, N, L)`` and the loss is taken over the
        masked rows; otherwise they are assumed pre-indexed (``(n_masked, L)``).
    mask : (B, N) bool, optional
        ``True`` at positions to include (typically ``content_mask & ~padding_mask``).
    kind : {"mse", "huber"}
    delta : float
        Huber transition point.
    reduction : {"mean", "sum", "none"}
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    if pred.numel() == 0:
        return pred.sum() * 0.0   # keeps the graph / device, yields 0

    if kind == "mse":
        loss = F.mse_loss(pred, target, reduction="none")
    elif kind == "huber":
        loss = F.huber_loss(pred, target, reduction="none", delta=delta)
    else:
        raise ValueError(f"unknown loss kind {kind!r}")

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def target_mask(content_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """``content_mask & ~padding_mask`` -- real, hidden positions only."""
    return content_mask & (~padding_mask)
