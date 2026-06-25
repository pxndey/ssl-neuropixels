"""A.4 -- Differentiable soft-binning of continuous spike positions.

Stock DREDge builds its space-time activity image with ``numpy.histogramdd``
(``make_2d_motion_histogram``): every spike is dropped into exactly one
``(depth, time)`` bin, weighted by its amplitude.  Hard binning has zero
gradient with respect to the continuous spatial coordinates, so it blocks any
upstream network from being trained through the raster.

Here each spike instead *spreads* its amplitude over nearby spatial bins with a
smooth kernel (a localized Gaussian KDE, or a one-bin triangular / bilinear
kernel).  The contribution -- and therefore the whole raster -- is a smooth
function of the spike's continuous coordinates and of its (encoder-produced)
amplitude/feature, so gradients flow into both.

The implementation is **axis-generic**: it takes a list of per-axis bin centers,
so it supports a 1-D depth raster ``(D_y, T)`` (stock DREDge, ``motion_dims=('y',)``)
and a 2-D ``(D_x, D_y, T)`` raster (full 2-D motion, ``motion_dims=('x','y')``)
with identical code.  Differentiability with respect to ``x`` therefore comes for
free the moment ``x`` is added as a binning axis.

As ``bandwidth -> 0`` the Gaussian kernel collapses onto the nearest bin, so
``mode='gaussian'`` converges to ``mode='hard'`` (this is the limit asserted by
``tests/test_soft_binning.py``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from .config import SoftBinConfig

_EPS = 1e-12
_TINY = 1e-300   # guards 0/0 in the normalize branch without distorting tiny masses


def make_bin_centers(
    lo: float,
    hi: float,
    bin_size: float,
    *,
    device=None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Bin *centers* spanning ``[lo, hi]`` at spacing ``bin_size``.

    Centers (not edges) are used everywhere so that "nearest center" hard binning
    and the soft kernels share the same reference grid.
    """
    n = max(int(torch.floor(torch.tensor((hi - lo) / bin_size)).item()) + 1, 1)
    idx = torch.arange(n, device=device, dtype=dtype)
    return lo + (idx + 0.5) * bin_size


def soft_assign_1d(
    coords: torch.Tensor,
    centers: torch.Tensor,
    bandwidth: float,
    *,
    mode: str = "gaussian",
    normalize: bool = True,
    trunc: float = 4.0,
) -> torch.Tensor:
    """Soft assignment weights of ``coords`` onto ``centers`` along one axis.

    Parameters
    ----------
    coords : (n,) tensor
        Continuous positions (microns).  Gradients flow back to these.
    centers : (C,) tensor
        Bin centers along this axis (fixed geometry).
    bandwidth : float
        Gaussian sigma in microns (ignored for bilinear/hard).
    mode : {"gaussian", "bilinear", "hard"}
    normalize : bool
        If True each row sums to 1 (partition of unity) so a spike contributes
        exactly its amplitude -- matching the mass of a hard histogram count.
    trunc : float
        Zero out Gaussian weights beyond ``trunc * bandwidth`` for locality.

    Returns
    -------
    (n, C) tensor of non-negative weights.
    """
    coords = coords.reshape(-1, 1)
    centers = centers.reshape(1, -1)

    if mode == "gaussian":
        bw = max(float(bandwidth), 1e-6)
        z = (centers - coords) / bw
        w = torch.exp(-0.5 * z * z)
        if trunc and trunc > 0:
            w = torch.where(z.abs() <= trunc, w, torch.zeros_like(w))
    elif mode == "bilinear":
        if centers.shape[1] > 1:
            spacing = (centers[0, 1] - centers[0, 0]).abs()
        else:
            spacing = torch.ones((), device=centers.device, dtype=centers.dtype)
        d = (centers - coords).abs() / spacing.clamp_min(_EPS)
        w = (1.0 - d).clamp_min(0.0)
    elif mode == "hard":
        idx = (centers - coords).abs().argmin(dim=1)
        w = torch.zeros(coords.shape[0], centers.shape[1], device=coords.device, dtype=coords.dtype)
        w[torch.arange(coords.shape[0], device=coords.device), idx] = 1.0
    else:
        raise ValueError(f"unknown soft-bin mode {mode!r}")

    if normalize:
        # Only rows that are *exactly* zero (spike outside the grid / fully
        # truncated) are dropped. A legitimately tiny mass -- e.g. the nearest-bin
        # weight at very small bandwidth, exp(-128) ~ 1e-56 -- must still normalize
        # to ~1 (this is the bandwidth -> 0 hard-binning limit), so we must not
        # threshold the denominator at a "large" epsilon like 1e-12.
        denom = w.sum(dim=1, keepdim=True)
        w = torch.where(denom > 0, w / denom.clamp_min(_TINY), torch.zeros_like(w))
    return w


def soft_time_weights(
    times: torch.Tensor,
    time_centers: torch.Tensor,
    bandwidth_s: float,
    mode: str,
) -> torch.Tensor:
    """Soft assignment along the time axis (used only when ``time_mode != 'hard'``)."""
    return soft_assign_1d(times, time_centers, bandwidth_s, mode=mode, normalize=True, trunc=4.0)


def build_raster(
    spatial_coords: Sequence[torch.Tensor],
    spatial_centers: Sequence[torch.Tensor],
    weights: torch.Tensor,
    *,
    n_time: int,
    time_idx: Optional[torch.Tensor] = None,
    times: Optional[torch.Tensor] = None,
    time_centers: Optional[torch.Tensor] = None,
    cfg: Optional[SoftBinConfig] = None,
) -> torch.Tensor:
    """Build a ``(*spatial_dims, n_time)`` activity raster by soft-binning spikes.

    Parameters
    ----------
    spatial_coords : list of (n,) tensors
        One tensor per spatial axis (e.g. ``[y]`` or ``[x, y]``), microns.
    spatial_centers : list of (C_k,) tensors
        Bin centers for each spatial axis (same length/order as spatial_coords).
    weights : (n,) tensor
        Per-spike amplitude / cleaned feature.  Gradients flow back to these.
    n_time : int
        Number of time bins.
    time_idx : (n,) long tensor, optional
        Hard time-bin index per spike (used when ``cfg.time_mode == 'hard'``).
    times, time_centers : optional
        Continuous spike times and time-bin centers (used for soft time binning).
    cfg : SoftBinConfig

    Returns
    -------
    raster : tensor of shape ``(*[len(c) for c in spatial_centers], n_time)``.
        For the DREDge convention this is ``(depth, time)`` when there is a single
        spatial axis, or ``(x, y, time)`` for two.
    """
    cfg = cfg or SoftBinConfig()
    assert len(spatial_coords) == len(spatial_centers), "coords/centers axis count mismatch"
    n_axes = len(spatial_coords)
    if n_axes not in (1, 2):
        raise NotImplementedError("build_raster supports 1 or 2 spatial axes")

    device = weights.device
    dtype = weights.dtype

    # per-axis soft assignment matrices
    W = [
        soft_assign_1d(
            c.to(dtype), ctr.to(device=device, dtype=dtype),
            cfg.bandwidth_um, mode=cfg.mode, normalize=cfg.normalize, trunc=cfg.trunc,
        )
        for c, ctr in zip(spatial_coords, spatial_centers)
    ]
    dims = [w.shape[1] for w in W]

    # spatial contribution per spike, weighted by amplitude
    if n_axes == 1:
        contrib = W[0] * weights.reshape(-1, 1)                      # (n, Dy)
        flat = contrib                                              # (n, Dy)
    else:
        contrib = weights.reshape(-1, 1, 1) * W[0].unsqueeze(2) * W[1].unsqueeze(1)  # (n, Dx, Dy)
        flat = contrib.reshape(contrib.shape[0], -1)               # (n, Dx*Dy)

    n_space = flat.shape[1]

    if cfg.time_mode == "hard":
        if time_idx is None:
            raise ValueError("time_mode='hard' requires time_idx")
        time_idx = time_idx.to(torch.long)
        raster = torch.zeros(n_space, n_time, device=device, dtype=dtype)
        raster = raster.index_add(1, time_idx, flat.t())            # (n_space, n_time)
    else:
        if times is None or time_centers is None:
            raise ValueError("soft time binning requires `times` and `time_centers`")
        Wt = soft_time_weights(times.to(dtype), time_centers.to(device=device, dtype=dtype),
                               cfg.time_bandwidth_s, cfg.time_mode)   # (n, n_time)
        raster = flat.t() @ Wt                                      # (n_space, n_time)

    return raster.reshape(*dims, n_time)


def hard_histogram(
    spatial_coords: Sequence[torch.Tensor],
    spatial_centers: Sequence[torch.Tensor],
    weights: torch.Tensor,
    *,
    n_time: int,
    time_idx: torch.Tensor,
) -> torch.Tensor:
    """Reference non-differentiable raster (nearest-center hard assignment).

    Equivalent to ``numpy.histogramdd`` weighted by ``weights``; used by the
    soft-vs-hard agreement test as the ``bandwidth -> 0`` target.
    """
    cfg = SoftBinConfig(mode="hard", normalize=True, time_mode="hard")
    return build_raster(
        spatial_coords, spatial_centers, weights,
        n_time=n_time, time_idx=time_idx, cfg=cfg,
    )
