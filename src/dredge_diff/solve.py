"""A.1 -- The core quadratic solve ``H P = g`` (differentiable).

DREDge recovers the motion trace by solving a Bayesian inverse problem whose
objective is fully quadratic, so its Hessian ``H`` is a structured
block-tridiagonal matrix and ``P = H^{-1} g``.  Stock DREDge uses
``scipy.linalg.solve`` inside a custom block-Thomas recursion
(``thomas_solve`` / ``newton_solve_rigid``).

This module rebuilds the *same* linear system out of autograd-tracked tensors
and solves it with ``torch.linalg.solve`` (never an explicit inverse), so
gradients propagate from ``P`` back into every entry of ``H`` and ``g`` -- and
hence into the weights ``U`` (from A.3), the displacements ``D`` (from A.2) and
ultimately the raster (A.4).  The Thomas recursion and a dense
``torch.linalg.solve`` of the full block-tridiagonal system give identical
answers; we use the dense solve because it is trivially differentiable.

Matrix algebra (mirrors the reference exactly):

    neg-Hessian likelihood term :  diag(U.sum(1) + U.sum(0)) - (U + Uᵀ)
    temporal prior (Laplacian)  :  laplacian(T, lambda_t, eps, wink, ridge_mask)
    right-hand side             :  g = (U⊙D).sum(1) - (U⊙D).sum(0)
    solve                       :  P = solve(L_t + negH, g)

For full 2-D motion the x- and y-least-squares share the **same** ``H`` (it only
depends on the weights ``U``), so we solve ``H @ [g_x, g_y] = [P_x, P_y]`` in one
shot -- a single extra right-hand side column, not a second system.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from .config import SolveConfig


def temporal_laplacian(
    T: int,
    *,
    lambd: float,
    eps: float,
    wink: bool = True,
    ridge_mask: Optional[torch.Tensor] = None,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Discrete 1-D Laplacian (smoothing prior) + ridge, matching ``laplacian()``.

    ``ridge_mask`` (per-time bool/float) adds ``eps`` only where there is data,
    exactly as in the reference (``diag = lambd + eps * ridge_mask``).
    """
    lap = torch.zeros(T, T, device=device, dtype=dtype)
    if ridge_mask is None:
        diag = torch.full((T,), lambd + eps, device=device, dtype=dtype)
    else:
        diag = lambd + eps * ridge_mask.to(device=device, dtype=dtype)
    lap += torch.diag(diag)
    if wink and T >= 2:
        lap[0, 0] -= 0.5 * lambd
        lap[-1, -1] -= 0.5 * lambd
    if T >= 2:
        off = -0.5 * lambd * torch.ones(T - 1, device=device, dtype=dtype)
        lap += torch.diag(off, 1) + torch.diag(off, -1)
    return lap


def neg_hessian_term(U: torch.Tensor) -> torch.Tensor:
    """``diag(U.sum(1) + U.sum(0)) - (U + Uᵀ)`` -- the weighted graph Laplacian.

    This is ``neg_hessian_likelihood_term`` from the reference, vectorized.
    """
    rowsum = U.sum(dim=1)
    colsum = U.sum(dim=0)
    return torch.diag(rowsum + colsum) - (U + U.t())


def newton_rhs(U: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """``g = (U⊙D).sum(1) - (U⊙D).sum(0)`` -- gradient of the cost at ``P = 0``."""
    UD = U * D
    return UD.sum(dim=1) - UD.sum(dim=0)


def _had_weights(Us: torch.Tensor) -> torch.Tensor:
    """Per-(window,time) mask of bins that participate (reference logic)."""
    had = (Us > 0).any(dim=2)                      # (B, T)
    dead_windows = ~had.any(dim=1)                 # windows with no data anywhere
    if dead_windows.any():
        had = had.clone()
        had[dead_windows] = True
    return had


def solve_displacement(
    Ds_per_dim: Dict[str, torch.Tensor],
    Us: torch.Tensor,
    cfg: SolveConfig,
    *,
    couple_windows: bool = False,
) -> torch.Tensor:
    """Solve for the motion trace(s).

    Parameters
    ----------
    Ds_per_dim : dict ``{dim: (B, T, T)}``
        Observed pairwise displacement matrices, one entry per motion dimension
        (``{'y': ...}`` for stock DREDge, ``{'x': ..., 'y': ...}`` for 2-D motion).
    Us : (B, T, T) tensor
        Per-window weight matrices (from A.3).  Shared across motion dimensions.
    cfg : SolveConfig
    couple_windows : bool
        If True and ``B > 1``, apply the spatial (cross-window) Laplacian prior,
        solving the full block-tridiagonal system.  Otherwise windows are solved
        independently (rigid, or ``lambda_s == 0``).

    Returns
    -------
    P : (B, T, n_dims) tensor -- displacement per window, time, motion-dimension
        (column order follows ``Ds_per_dim`` insertion order).
    """
    dims = list(Ds_per_dim.keys())
    B, T, _ = Us.shape
    device, dtype = Us.device, Us.dtype
    had = _had_weights(Us)

    L_t = torch.stack([
        temporal_laplacian(T, lambd=cfg.lambda_t, eps=cfg.eps, wink=cfg.wink,
                           ridge_mask=had[b], device=device, dtype=dtype)
        for b in range(B)
    ])                                                              # (B, T, T)
    negH = torch.stack([neg_hessian_term(Us[b]) for b in range(B)])  # (B, T, T)

    # right-hand side, one column per motion dim: (B, T, n_dims)
    G = torch.stack([
        torch.stack([newton_rhs(Us[b], Ds_per_dim[d][b]) for d in dims], dim=-1)
        for b in range(B)
    ])                                                              # (B, T, n_dims)

    if B == 1 or not couple_windows or cfg.lambda_s == 0:
        H = L_t + negH                                              # (B, T, T)
        P = torch.linalg.solve(H, G)                               # (B, T, n_dims)
        return P

    # --- spatially-coupled non-rigid case: full block-tridiagonal solve ------
    return _solve_block_tridiagonal(L_t, negH, G, had, cfg)


def _solve_block_tridiagonal(
    L_t: torch.Tensor,
    negH: torch.Tensor,
    G: torch.Tensor,
    had: torch.Tensor,
    cfg: SolveConfig,
) -> torch.Tensor:
    """Dense assembly + ``torch.linalg.solve`` of the (B*T) x (B*T) system.

    Equivalent (same solution) to the reference block-Thomas recursion, but fully
    differentiable.  Diagonal blocks carry the temporal prior + likelihood +
    spatial self-term; adjacent windows are coupled by ``Lambda_s_offdiag``.
    """
    B, T, _ = L_t.shape
    n_dims = G.shape[-1]
    device, dtype = L_t.device, L_t.dtype
    lam_s = cfg.lambda_s

    offdiag = temporal_laplacian(T, lambd=-lam_s / 2.0, eps=0.0, wink=cfg.wink,
                                 ridge_mask=None, device=device, dtype=dtype)

    def diag_block(b: int) -> torch.Tensor:
        scale = lam_s / 2.0 if (b == 0 or b == B - 1) else lam_s
        lam_diag = temporal_laplacian(T, lambd=scale, eps=cfg.eps, wink=cfg.wink,
                                      ridge_mask=had[b], device=device, dtype=dtype)
        return negH[b] + L_t[b] + lam_diag

    H = torch.zeros(B * T, B * T, device=device, dtype=dtype)
    for b in range(B):
        sl = slice(b * T, (b + 1) * T)
        H[sl, sl] = diag_block(b)
        if b + 1 < B:
            sl2 = slice((b + 1) * T, (b + 2) * T)
            H[sl, sl2] = offdiag
            H[sl2, sl] = offdiag

    G_flat = G.reshape(B * T, n_dims)
    P_flat = torch.linalg.solve(H, G_flat)
    return P_flat.reshape(B, T, n_dims)
