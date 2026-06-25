"""DiffDredge -- the assembled, fully differentiable DREDge module.

Chains the four differentiable substitutions into one ``nn.Module``:

    spikes ->[A.4 soft-bin]-> raster ->[windows]
           ->[A.2 xcorr + soft-argmax]-> D, C
           ->[A.3 sigmoid gate]-> U
           ->[A.1 torch.linalg.solve]-> P (motion trace)

``forward`` accepts either a pre-built ``raster`` (depth x time, or x x y x time)
or raw spikes (continuous coordinates + per-spike features + time-bin indices),
in which case it soft-bins them first.  The latter is the seam where an upstream
encoder plugs in (see ``pipeline.py``): because every step is differentiable,
``P.sum().backward()`` reaches the per-spike coordinates and features.

Everything is config-driven (:class:`DredgeConfig`); each sub-step can be flipped
between its soft and hard variants for ablation, and ``motion_dims`` switches
between stock 1-D depth motion ``('y',)`` and full 2-D motion ``('x', 'y')``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from .config import DredgeConfig
from .soft_binning import build_raster, make_bin_centers
from .windows import get_spatial_windows
from .xcorr import cross_correlate_1d, cross_correlate_2d
from .threshold import threshold_correlation_matrix
from .solve import solve_displacement

_ORDER = ("x", "y", "z")


class DiffDredge(nn.Module):
    def __init__(self, cfg: Optional[DredgeConfig] = None):
        super().__init__()
        self.cfg = cfg or DredgeConfig()
        # spatial raster axes, always ordered x, y, z (y is the depth/window axis)
        self.spatial_axes = [d for d in _ORDER if d in self.cfg.motion_dims]
        self.is_2d = len(self.spatial_axes) == 2
        if self.is_2d and not self.cfg.window.rigid:
            raise NotImplementedError("2-D motion currently supports rigid windows only")

    # ------------------------------------------------------------------ #
    def _spatial_centers(
        self,
        spike_coords: Mapping[str, torch.Tensor],
        provided: Optional[Mapping[str, torch.Tensor]],
        device,
        dtype,
    ) -> Dict[str, torch.Tensor]:
        centers = {}
        for d in self.spatial_axes:
            if provided is not None and d in provided:
                centers[d] = provided[d].to(device=device, dtype=dtype)
                continue
            c = spike_coords[d]
            lo = float(c.min().item())
            hi = float(c.max().item())
            if hi <= lo:
                hi = lo + self.cfg.bin_um
            centers[d] = make_bin_centers(lo, hi, self.cfg.bin_um, device=device, dtype=dtype)
        return centers

    # ------------------------------------------------------------------ #
    def build_raster(
        self,
        spike_coords: Mapping[str, torch.Tensor],
        spike_features: torch.Tensor,
        spike_time_idx: torch.Tensor,
        n_time: int,
        spatial_centers: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple:
        """Soft-bin spikes into a raster. Returns ``(raster, centers)``."""
        device, dtype = spike_features.device, spike_features.dtype
        centers = self._spatial_centers(spike_coords, spatial_centers, device, dtype)
        coords = [spike_coords[d] for d in self.spatial_axes]
        ctrs = [centers[d] for d in self.spatial_axes]
        raster = build_raster(
            coords, ctrs, spike_features,
            n_time=n_time, time_idx=spike_time_idx, cfg=self.cfg.soft_bin,
        )
        return raster, centers

    # ------------------------------------------------------------------ #
    def forward(
        self,
        raster: Optional[torch.Tensor] = None,
        *,
        spike_coords: Optional[Mapping[str, torch.Tensor]] = None,
        spike_features: Optional[torch.Tensor] = None,
        spike_time_idx: Optional[torch.Tensor] = None,
        n_time: Optional[int] = None,
        spatial_centers: Optional[Mapping[str, torch.Tensor]] = None,
        contact_depths: Optional[torch.Tensor] = None,
        return_extras: bool = False,
    ):
        """Estimate the motion trace ``P``.

        Provide either ``raster`` directly, or (``spike_coords``,
        ``spike_features``, ``spike_time_idx``) to soft-bin on the fly.

        Returns
        -------
        P : (T, n_motion_dims) tensor (window dim squeezed when rigid/B==1),
            or (B, T, n_motion_dims) when non-rigid.
        extras : dict (only if ``return_extras``) with D, C, U, raster, centers, windows.
        """
        cfg = self.cfg
        centers = None

        if raster is None:
            if spike_coords is None or spike_features is None or spike_time_idx is None:
                raise ValueError("provide a raster, or spike_coords/features/time_idx to build one")
            if n_time is None:
                n_time = int(spike_time_idx.max().item()) + 1
            raster, centers = self.build_raster(
                spike_coords, spike_features, spike_time_idx, n_time, spatial_centers
            )

        device, dtype = raster.device, raster.dtype

        # spatial bin centers along y (the depth/window axis)
        if centers is not None:
            y_centers = centers["y"]
        elif spatial_centers is not None and "y" in spatial_centers:
            y_centers = spatial_centers["y"].to(device=device, dtype=dtype)
        else:
            n_y = raster.shape[-2] if self.is_2d else raster.shape[0]
            y_centers = make_bin_centers(0.0, (n_y - 1) * cfg.bin_um, cfg.bin_um, device=device, dtype=dtype)

        if contact_depths is None:
            contact_depths = y_centers
        else:
            contact_depths = contact_depths.to(device=device, dtype=dtype)

        windows, window_centers = get_spatial_windows(contact_depths, y_centers, cfg.window)

        # --- A.2 cross-correlation + differentiable displacement ---------- #
        if self.is_2d:
            win_y = windows[0]  # rigid -> ones over depth
            Ds, Cs, max_disp = cross_correlate_2d(
                raster, win_y, cfg.bin_um, cfg.bin_um, cfg.xcorr, cfg.disp,
                max_disp_um_x=cfg.xcorr.max_disp_um, max_disp_um_y=cfg.xcorr.max_disp_um,
                fallback_disp_um=cfg.window.win_scale_um,
            )
        else:
            fallback = (
                float(y_centers.max() - y_centers.min()) / 4.0
                if cfg.window.rigid else cfg.window.win_scale_um / 4.0
            )
            Ds_y, Cs, max_disp = cross_correlate_1d(
                raster, windows, cfg.bin_um, cfg.xcorr, cfg.disp,
                max_disp_um=cfg.xcorr.max_disp_um, fallback_disp_um=fallback,
            )
            Ds = {"y": Ds_y}

        # --- A.3 smooth gate -> weights U -------------------------------- #
        U = threshold_correlation_matrix(
            Cs, cfg.thresh, bin_s=cfg.bin_s, time_horizon_s=cfg.time_horizon_s,
        )

        # --- A.1 solve --------------------------------------------------- #
        couple = (not cfg.window.rigid) and (cfg.solve.lambda_s > 0)
        P = solve_displacement(Ds, U, cfg.solve, couple_windows=couple)   # (B, T, n_dims)

        B = P.shape[0]
        P_out = P[0] if B == 1 else P                                      # (T, n_dims) or (B, T, n_dims)

        if not return_extras:
            return P_out

        extras = {
            "D": Ds,
            "C": Cs,
            "U": U,
            "raster": raster,
            "y_centers": y_centers,
            "windows": windows,
            "window_centers": window_centers,
            "max_disp_um": max_disp,
            "motion_dims": list(Ds.keys()),
        }
        return P_out, extras
