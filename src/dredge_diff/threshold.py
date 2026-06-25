"""A.3 -- Smooth reliability gate replacing the hard correlation threshold.

DREDge zeroes out unreliable time-bin pairs with a hard switch
(``threshold_correlation_matrix``: ``Ss = (Cs >= mincorr) * Cs``, then squared),
and separately sets a bin's variance to infinity (weight 0) when its local
activity is zero.  Both are 0/1 indicators with no gradient.

Here the indicator ``[C >= theta_C]`` is replaced by a sharp sigmoid of
``slope * (C - theta_C)``, smoothly scaling each pair's weight in ``[0, 1]``.  As
``slope -> infinity`` the sigmoid approaches the step function and we recover the
exact reference behaviour (asserted by ``tests/test_sigmoid_gate.py``).

The output ``S`` plays the role of DREDge's ``Ss`` and is used directly as the
weight matrix ``U`` in the Hessian (the spec's "scale the Hessian weight between
0 and 1").  An optional per-time *activity* gate reproduces the
"``V_bt == 0`` -> weight 0" behaviour, also smoothly.
"""

from __future__ import annotations

from typing import Optional

import torch

from .config import ThresholdConfig


def reliability_gate(Cs: torch.Tensor, cfg: ThresholdConfig) -> torch.Tensor:
    """Smooth (or hard) gate in ``[0, 1]`` from correlation vs threshold."""
    if cfg.mode == "hard":
        return (Cs >= cfg.mincorr).to(Cs.dtype)
    if cfg.mode == "sigmoid":
        return torch.sigmoid(cfg.slope * (Cs - cfg.mincorr))
    raise ValueError(f"unknown threshold mode {cfg.mode!r}")


def threshold_correlation_matrix(
    Cs: torch.Tensor,
    cfg: ThresholdConfig,
    *,
    activity: Optional[torch.Tensor] = None,
    activity_slope: float = 10.0,
    bin_s: float = 1.0,
    time_horizon_s: Optional[float] = None,
) -> torch.Tensor:
    """Turn correlation matrices ``Cs`` (B,T,T) into solver weights ``S`` (B,T,T).

    Parameters
    ----------
    Cs : (B, T, T) tensor
        Per-window pairwise correlations.
    cfg : ThresholdConfig
    activity : (B, T) tensor, optional
        Per-window per-time-bin local activity ``V_bt`` (e.g. window heat).  When
        given, a smooth gate ``sigmoid(activity_slope * V)`` multiplies both the
        row and column of ``S`` so that empty bins get ~0 weight.
    bin_s, time_horizon_s :
        If set, pairs of bins farther apart than ``time_horizon_s`` are zeroed
        (hard in time -- time is not a learnable coordinate).

    Returns
    -------
    S : (B, T, T) tensor, non-negative weights.
    """
    gate = reliability_gate(Cs, cfg)
    S = gate * Cs
    if cfg.square:
        S = S * S

    if activity is not None:
        # smooth "is this bin alive?" gate, applied symmetrically (row & col)
        a_gate = torch.sigmoid(activity_slope * activity)           # (B, T)
        S = S * a_gate.unsqueeze(2) * a_gate.unsqueeze(1)

    if time_horizon_s is not None and time_horizon_s > 0:
        T = Cs.shape[-1]
        tt = bin_s * torch.arange(T, device=Cs.device, dtype=Cs.dtype)
        dt = (tt.unsqueeze(1) - tt.unsqueeze(0)).abs()
        mask = (dt <= time_horizon_s).to(Cs.dtype)
        S = S * mask.unsqueeze(0)

    # guard against tiny negatives from the squaring of near-zero gated corrs
    return S.clamp_min(0.0)
