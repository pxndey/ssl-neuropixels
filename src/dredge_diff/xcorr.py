"""A.2 -- Normalized cross-correlation + differentiable displacement readout.

DREDge estimates the observed displacement between two time bins by computing a
normalized cross-correlation of their depth profiles and taking the
``argmax``-lag (``calc_corr_decent_pair`` -> ``torch.max``).  ``argmax`` is a step
function: zero gradient with respect to the correlation values, so nothing can be
learned through it.

Two pieces here:

1. :func:`normxcorr1d` / :func:`normxcorr2d` -- a clean-room, autograd-safe port
   of DREDge's ``normxcorr1d`` (same ``cov / sqrt(varX varT)`` formula, weighted
   and optionally centered).  The reference uses boolean in-place writes
   (``Nx[empty] = 1`` etc.) which break autograd; here those are ``torch.where``.

2. :func:`displacement_from_corr_1d` / ``_2d`` -- the differentiable readout:
     * ``soft``      : expected lag under ``softmax(corr / T)`` (soft-argmax).
                       ``T -> 0`` recovers the hard ``argmax``.
     * ``parabolic`` : sub-pixel peak from a 3-point parabola fit.
     * ``hard``      : the original ``argmax`` (for ablation / the limit tests).

The 2-D variants estimate a lateral+axial displacement ``(dx, dy)`` and are what
``motion_dims=('x','y')`` uses.  The 1-D variant is stock DREDge (``('y',)``).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .config import DisplacementConfig, XcorrConfig
from .windows import window_domains


# --------------------------------------------------------------------------- #
# Normalized cross-correlation
# --------------------------------------------------------------------------- #
def normxcorr1d(
    template: torch.Tensor,
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    centered: bool = True,
    normalized: bool = True,
    padding: int = 0,
) -> torch.Tensor:
    """Weighted normalized 1-D cross-correlation.

    Parameters
    ----------
    template : (nt, L) tensor
        Reference profiles (one per row, e.g. a time bin's depth profile).
    x : (nx, L) tensor
        Profiles to locate the templates within.
    weights : (L,) tensor, optional
        Per-position weights (the depth-window taper).
    padding : int
        Max lag searched; output has ``2 * padding + 1`` lags.

    Returns
    -------
    corr : (nx, nt, 2*padding+1) tensor
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    nt, Lt = template.shape
    nx, Lx = x.shape
    dtype, device = template.dtype, template.device

    onesx = torch.ones(1, 1, Lx, dtype=dtype, device=device)
    if weights is None:
        wk = torch.ones(1, 1, Lt, dtype=dtype, device=device)
        wt = template.unsqueeze(1)
    else:
        wk = weights.reshape(1, 1, -1)
        wt = (template * weights.reshape(1, -1)).unsqueeze(1)
    wx = x.unsqueeze(1)
    tmpl = template.unsqueeze(1)
    xin = x.unsqueeze(1)

    Nx = F.conv1d(onesx, wk, padding=padding)
    empty = Nx == 0
    Nx_safe = torch.where(empty, torch.ones_like(Nx), Nx)

    cov = F.conv1d(wx, wt, padding=padding) / Nx_safe
    if centered:
        Et = F.conv1d(onesx, wt, padding=padding) / Nx_safe
        Ex = F.conv1d(wx, wk, padding=padding) / Nx_safe
        cov = cov - Ex * Et

    if not normalized:
        return cov

    var_t = F.conv1d(onesx, wt * tmpl, padding=padding) / Nx_safe
    var_x = F.conv1d(wx * xin, wk, padding=padding) / Nx_safe
    if centered:
        var_t = var_t - Et ** 2
        var_x = var_x - Ex ** 2
    var_t = torch.where(var_t <= 0, torch.ones_like(var_t), var_t)
    var_x = torch.where(var_x <= 0, torch.ones_like(var_x), var_x)
    corr = cov / torch.sqrt(var_x) / torch.sqrt(var_t)
    corr = torch.where(empty.expand_as(corr), torch.zeros_like(corr), corr)
    return corr


def normxcorr2d(
    template: torch.Tensor,
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    *,
    centered: bool = True,
    normalized: bool = True,
    padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Weighted normalized 2-D cross-correlation.

    Parameters
    ----------
    template, x : (n, Hx, Hy) tensors
        2-D frames (e.g. an (x, y) activity image for a single time bin).
    weights : (Hx, Hy) tensor, optional
    padding : (Px, Py)
        Max lag along each axis.

    Returns
    -------
    corr : (nx, nt, 2*Px+1, 2*Py+1) tensor
    """
    nt, Hx, Hy = template.shape
    nx = x.shape[0]
    dtype, device = template.dtype, template.device

    onesx = torch.ones(1, 1, Hx, Hy, dtype=dtype, device=device)
    if weights is None:
        wk = torch.ones(1, 1, Hx, Hy, dtype=dtype, device=device)
        wt = template.unsqueeze(1)
    else:
        wk = weights.reshape(1, 1, Hx, Hy)
        wt = (template * weights.reshape(1, Hx, Hy)).unsqueeze(1)
    wx = x.unsqueeze(1)
    tmpl = template.unsqueeze(1)
    xin = x.unsqueeze(1)

    Nx = F.conv2d(onesx, wk, padding=padding)
    empty = Nx == 0
    Nx_safe = torch.where(empty, torch.ones_like(Nx), Nx)

    cov = F.conv2d(wx, wt, padding=padding) / Nx_safe
    if centered:
        Et = F.conv2d(onesx, wt, padding=padding) / Nx_safe
        Ex = F.conv2d(wx, wk, padding=padding) / Nx_safe
        cov = cov - Ex * Et

    if not normalized:
        return cov

    var_t = F.conv2d(onesx, wt * tmpl, padding=padding) / Nx_safe
    var_x = F.conv2d(wx * xin, wk, padding=padding) / Nx_safe
    if centered:
        var_t = var_t - Et ** 2
        var_x = var_x - Ex ** 2
    var_t = torch.where(var_t <= 0, torch.ones_like(var_t), var_t)
    var_x = torch.where(var_x <= 0, torch.ones_like(var_x), var_x)
    corr = cov / torch.sqrt(var_x) / torch.sqrt(var_t)
    corr = torch.where(empty.expand_as(corr), torch.zeros_like(corr), corr)
    return corr


# --------------------------------------------------------------------------- #
# Differentiable displacement readout (replaces argmax)
# --------------------------------------------------------------------------- #
def _gather_last(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return t.gather(-1, idx.unsqueeze(-1)).squeeze(-1)


def displacement_from_corr_1d(
    corr: torch.Tensor,
    disp_um: torch.Tensor,
    cfg: DisplacementConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map a correlation curve to ``(displacement, confidence)``.

    Parameters
    ----------
    corr : (..., n_lags) tensor
    disp_um : (n_lags,) tensor
        Physical displacement (microns) for each lag index.

    Returns
    -------
    D : (...) displacement
    C : (...) confidence (peak / expected correlation)
    """
    disp_um = disp_um.to(corr.dtype)
    if cfg.mode == "hard":
        C, idx = corr.max(dim=-1)
        D = disp_um[idx]
        return D, C

    if cfg.mode == "soft":
        p = torch.softmax(corr / cfg.temperature, dim=-1)
        D = (p * disp_um).sum(dim=-1)
        if cfg.confidence == "max":
            C = corr.max(dim=-1).values
        else:
            C = (p * corr).sum(dim=-1)
        return D, C

    if cfg.mode == "parabolic":
        C0, idx = corr.max(dim=-1)
        L = corr.shape[-1]
        idxm = (idx - 1).clamp(0, L - 1)
        idxp = (idx + 1).clamp(0, L - 1)
        cm = _gather_last(corr, idxm)
        cp = _gather_last(corr, idxp)
        denom = cm - 2 * C0 + cp
        delta = torch.where(denom.abs() > 1e-12, 0.5 * (cm - cp) / denom, torch.zeros_like(denom))
        delta = delta.clamp(-1.0, 1.0)
        spacing = (disp_um[1] - disp_um[0]) if disp_um.numel() > 1 else torch.ones((), dtype=corr.dtype, device=corr.device)
        D = disp_um[idx] + delta * spacing
        C = C0 - 0.25 * (cm - cp) * delta
        return D, C

    raise ValueError(f"unknown displacement mode {cfg.mode!r}")


def displacement_from_corr_2d(
    corr: torch.Tensor,
    disp_x_um: torch.Tensor,
    disp_y_um: torch.Tensor,
    cfg: DisplacementConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """2-D analogue: returns ``(Dx, Dy, C)`` via a 2-D soft-argmax."""
    *batch, Lx, Ly = corr.shape
    flat = corr.reshape(*batch, Lx * Ly)
    gx = disp_x_um.reshape(Lx, 1).expand(Lx, Ly).reshape(-1).to(corr.dtype)
    gy = disp_y_um.reshape(1, Ly).expand(Lx, Ly).reshape(-1).to(corr.dtype)

    if cfg.mode == "hard":
        C, idx = flat.max(dim=-1)
        return gx[idx], gy[idx], C

    # soft / parabolic both use the soft expectation in 2-D (parabola in 2-D is overkill)
    p = torch.softmax(flat / cfg.temperature, dim=-1)
    Dx = (p * gx).sum(dim=-1)
    Dy = (p * gy).sum(dim=-1)
    if cfg.confidence == "max":
        C = flat.max(dim=-1).values
    else:
        C = (p * flat).sum(dim=-1)
    return Dx, Dy, C


# --------------------------------------------------------------------------- #
# Drivers: time x time displacement/correlation matrices per window
# --------------------------------------------------------------------------- #
def _max_disp_bins(max_disp_um: Optional[float], bin_um: float, fallback_um: float) -> int:
    mdu = max_disp_um if max_disp_um is not None else fallback_um
    return max(int(mdu // bin_um), 1)


def cross_correlate_1d(
    raster: torch.Tensor,
    windows: torch.Tensor,
    bin_um: float,
    xcfg: XcorrConfig,
    dcfg: DisplacementConfig,
    *,
    max_disp_um: Optional[float] = None,
    fallback_disp_um: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Per-window time x time displacement (D) and correlation (C) matrices.

    Parameters
    ----------
    raster : (D, T) tensor (depth x time)
    windows : (B, D) tensor (depth tapers)

    Returns
    -------
    Ds : (B, T, T) -- ``Ds[b, a, c]`` ~ displacement of time ``a`` vs time ``c``
    Cs : (B, T, T) -- corresponding confidences
    max_disp_um : float actually used
    """
    Dd, T = raster.shape
    B = windows.shape[0]
    P = _max_disp_bins(max_disp_um, bin_um, fallback_disp_um)
    disp_um = (torch.arange(2 * P + 1, device=raster.device, dtype=raster.dtype) - P) * bin_um

    slices = window_domains(windows)
    Ds = raster.new_zeros((B, T, T))
    Cs = raster.new_zeros((B, T, T))
    bs = max(int(xcfg.batch_size), 1)
    for b in range(B):
        start, stop = slices[b]
        taper = windows[b, start:stop]               # (Lb,)
        profiles = raster[start:stop, :].t().contiguous()   # (T, Lb): per-time depth profiles

        if T <= bs:
            corr = normxcorr1d(
                profiles, profiles, weights=taper,
                centered=xcfg.centered, normalized=xcfg.normalized, padding=P,
            )                                        # (T_input, T_template, 2P+1)
            Dbt, Cbt = displacement_from_corr_1d(corr, disp_um, dcfg)  # (T_input, T_template)
            Ds[b] = Dbt.t()                          # D[a, c] uses corr[input=c, template=a]
            Cs[b] = Cbt.t()
        else:
            # Tile over time so the full (T, T, 2P+1) correlation is never
            # materialized (it is ~hundreds of GB for a long recording). This branch
            # is for inference over whole recordings; the single-shot branch above is
            # the one used during differentiable training (small T chunks).
            for i in range(0, T, bs):
                templ = profiles[i:i + bs]           # (ib, Lb)
                for j in range(0, T, bs):
                    inp = profiles[j:j + bs]         # (jb, Lb)
                    corr = normxcorr1d(
                        templ, inp, weights=taper,
                        centered=xcfg.centered, normalized=xcfg.normalized, padding=P,
                    )                                # (jb, ib, 2P+1)
                    d, c = displacement_from_corr_1d(corr, disp_um, dcfg)   # (jb, ib)
                    Ds[b, i:i + templ.shape[0], j:j + inp.shape[0]] = d.t()
                    Cs[b, i:i + templ.shape[0], j:j + inp.shape[0]] = c.t()
    return Ds, Cs, float(P * bin_um)


def cross_correlate_2d(
    raster: torch.Tensor,
    win_y: Optional[torch.Tensor],
    bin_um_x: float,
    bin_um_y: float,
    xcfg: XcorrConfig,
    dcfg: DisplacementConfig,
    *,
    max_disp_um_x: Optional[float] = None,
    max_disp_um_y: Optional[float] = None,
    fallback_disp_um: float = 100.0,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Tuple[float, float]]:
    """2-D motion: time x time displacement matrices for both x and y.

    Parameters
    ----------
    raster : (Dx, Dy, T) tensor
    win_y : (Dy,) optional taper along depth (broadcast over x). None => ones.

    Returns
    -------
    Ds : dict with keys ``'x'`` and ``'y'``, each (1, T, T)
    Cs : (1, T, T)
    (max_disp_um_x, max_disp_um_y)
    """
    Dx, Dy, T = raster.shape
    Px = _max_disp_bins(max_disp_um_x, bin_um_x, max(bin_um_x, fallback_disp_um / 4))
    Py = _max_disp_bins(max_disp_um_y, bin_um_y, max(bin_um_y, fallback_disp_um / 4))
    # the 2-D xcorr is a full-frame conv2d; cap the search to the frame extent so a
    # fine bin_um can't blow up memory (use a coarser bin_um / explicit max_disp_um
    # for large probes if you need a wider search).
    Px = min(Px, max(Dx - 1, 0)) if Dx > 1 else 0
    Py = min(Py, max(Dy - 1, 1))
    disp_x = (torch.arange(2 * Px + 1, device=raster.device, dtype=raster.dtype) - Px) * bin_um_x
    disp_y = (torch.arange(2 * Py + 1, device=raster.device, dtype=raster.dtype) - Py) * bin_um_y

    if win_y is None:
        taper2d = None
    else:
        taper2d = win_y.reshape(1, Dy).expand(Dx, Dy).contiguous()

    frames = raster.permute(2, 0, 1).contiguous()   # (T, Dx, Dy)
    corr = normxcorr2d(
        frames, frames, weights=taper2d,
        centered=xcfg.centered, normalized=xcfg.normalized, padding=(Px, Py),
    )                                               # (T_input, T_template, 2Px+1, 2Py+1)
    Dx_bt, Dy_bt, C_bt = displacement_from_corr_2d(corr, disp_x, disp_y, dcfg)
    Ds = {"x": Dx_bt.t().unsqueeze(0), "y": Dy_bt.t().unsqueeze(0)}
    Cs = C_bt.t().unsqueeze(0)
    return Ds, Cs, (float(Px * bin_um_x), float(Py * bin_um_y))
