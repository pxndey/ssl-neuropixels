"""Non-rigid spatial windows (depth tapers), ported to torch.

A faithful re-implementation of spikeinterface's
``motion_utils.get_spatial_windows``.  The windows are a fixed function of probe
geometry (no gradient flows through them), but we build them as torch tensors so
they live on the same device/dtype as the raster.

* ``rigid=True``  -> a single flat (all-ones) window covering the probe.  Motion
  is then a single global trace per motion dimension -- the stock DREDge default
  for ``dredge_ap``/``dredge_lfp`` rigid mode.
* ``rigid=False`` -> overlapping ``gaussian`` / ``rect`` / ``triangle`` tapers
  spaced by ``win_step_um`` with width ``win_scale_um``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from .config import WindowConfig


def get_spatial_windows(
    contact_depths: torch.Tensor,
    spatial_bin_centers: torch.Tensor,
    cfg: WindowConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(windows, window_centers)``.

    Parameters
    ----------
    contact_depths : (n_channels,) tensor
        Electrode positions along the motion axis (used for the probe extent).
    spatial_bin_centers : (n_bins,) tensor
        Centers of the raster's spatial bins.
    cfg : WindowConfig

    Returns
    -------
    windows : (B, n_bins) tensor
        Per-window taper evaluated at each spatial bin center.
    window_centers : (B,) tensor
    """
    device = spatial_bin_centers.device
    dtype = spatial_bin_centers.dtype
    n = spatial_bin_centers.numel()

    if cfg.rigid:
        windows = torch.ones((1, n), device=device, dtype=dtype)
        centers = torch.tensor(
            [0.5 * (spatial_bin_centers[0].item() + spatial_bin_centers[-1].item())],
            device=device, dtype=dtype,
        )
        return windows, centers

    win_scale_um = float(cfg.win_scale_um)
    win_step_um = float(cfg.win_step_um)
    win_margin_um = cfg.win_margin_um
    if win_margin_um is None:
        win_margin_um = -win_scale_um / 2.0

    cmin = float(contact_depths.min().item())
    cmax = float(contact_depths.max().item())
    min_ = cmin - win_margin_um
    max_ = cmax + win_margin_um
    num_windows = int((max_ - min_) // win_step_um)
    if num_windows < 1:
        num_windows = 1
    border = ((max_ - min_) % win_step_um) / 2.0
    idx = torch.arange(num_windows + 1, device=device, dtype=dtype)
    window_centers = idx * win_step_um + min_ + border

    sbc = spatial_bin_centers.reshape(1, -1)
    wc = window_centers.reshape(-1, 1)
    if cfg.win_shape == "gaussian":
        windows = torch.exp(-((sbc - wc) ** 2) / (2 * win_scale_um ** 2))
    elif cfg.win_shape == "rect":
        windows = (torch.abs(sbc - wc) < (win_scale_um / 2.0)).to(dtype)
    elif cfg.win_shape == "triangle":
        center_dist = torch.abs(sbc - wc)
        in_window = center_dist <= (win_scale_um / 2.0)
        windows = torch.clamp(1.0 - center_dist / (win_scale_um / 2.0), min=0.0)
        windows = windows * in_window.to(dtype)
    else:
        raise ValueError(f"unknown win_shape {cfg.win_shape!r}")

    if cfg.zero_threshold is not None:
        windows = torch.where(windows < cfg.zero_threshold, torch.zeros_like(windows), windows)
        windows = windows / windows.sum(dim=1, keepdim=True).clamp_min(1e-12)

    return windows, window_centers


def window_domains(windows: torch.Tensor) -> list:
    """List of ``(start, stop)`` index ranges where each window is non-zero.

    Mirrors ``get_window_domains`` -- used to restrict the cross-correlation of a
    window to the depth band it actually covers.
    """
    slices = []
    n = windows.shape[1]
    for w in windows:
        nz = torch.nonzero(w, as_tuple=False).flatten()
        if nz.numel() == 0:
            slices.append((0, n))
        else:
            slices.append((int(nz[0].item()), int(nz[-1].item()) + 1))
    return slices
