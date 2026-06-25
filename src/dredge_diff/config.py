"""Configuration dataclasses for differentiable DREDge.

Every differentiable substitution exposes a ``mode`` that can be flipped back to
the original *hard* DREDge behaviour, so each step can be ablated independently
and the unit tests can assert ``soft -> hard`` agreement in the appropriate
limit (temperature -> 0, slope -> infinity, bandwidth -> 0).

Defaults follow the reference ``dredge_ap`` signature where one exists
(``bin_um=1.0``, ``bin_s=1.0``, ``mincorr=0.1``, ``win_step_um=400``,
``win_scale_um=450``, ``lambda_t=1.0``, ``eps=1e-3``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SoftBinConfig:
    """A.4 -- soft assignment of continuous (x, y) spike positions to the raster.

    ``mode='hard'`` reproduces ``numpy.histogramdd`` (nearest bin). ``'gaussian'``
    spreads each spike with a Gaussian KDE of width ``bandwidth_um``; in the
    ``bandwidth_um -> 0`` limit it collapses back to the hard assignment.
    ``'bilinear'`` uses a triangular kernel of one-bin support (linear interp).
    """

    mode: str = "gaussian"          # "gaussian" | "bilinear" | "hard"
    bandwidth_um: float = 5.0       # Gaussian sigma along spatial axes (um)
    trunc: float = 4.0              # truncate the Gaussian beyond trunc * sigma (locality); <=0 => dense
    normalize: bool = True          # conserve each spike's amplitude mass across bins (partition of unity)
    time_mode: str = "hard"         # "hard" | "gaussian" | "bilinear" along the time axis
    time_bandwidth_s: float = 0.5   # only used when time_mode != "hard"


@dataclass
class DisplacementConfig:
    """A.2 -- turning the cross-correlation curve into a displacement estimate.

    ``mode='hard'`` is the original ``argmax`` (non-differentiable in the lag).
    ``'soft'`` is the soft-argmax: expected lag under ``softmax(corr / T)``; as
    ``temperature -> 0`` it converges to the hard argmax. ``'parabolic'`` fits a
    parabola to the 3 samples around the peak for a sub-pixel, differentiable peak.
    """

    mode: str = "soft"              # "soft" | "parabolic" | "hard"
    temperature: float = 1.0        # soft-argmax temperature (smaller => sharper => closer to argmax)
    confidence: str = "expected"    # "expected" (sum p*corr) | "max" -- how C is read off in soft mode


@dataclass
class ThresholdConfig:
    """A.3 -- smooth replacement for the ``C >= mincorr`` reliability switch."""

    mode: str = "sigmoid"           # "sigmoid" | "hard"
    mincorr: float = 0.1            # threshold theta_C (reference default for dredge_ap)
    slope: float = 50.0             # sigmoid sharpness; slope -> inf recovers the hard step
    square: bool = True             # square the gated correlation (matches reference Ss = (gate*C)**2)


@dataclass
class SolveConfig:
    """A.1 -- the quadratic solve ``H P = g`` and its smoothing prior."""

    lambda_t: float = 1.0           # temporal smoothing prior strength (Laplacian)
    lambda_s: float = 1.0           # spatial (cross-window) prior strength; only matters when non-rigid
    eps: float = 1e-3               # ridge added to the prior diagonal for conditioning
    wink: bool = True               # Neumann ("winkler") boundary correction on the Laplacian endpoints


@dataclass
class XcorrConfig:
    """A.2 -- the normalized cross-correlation itself."""

    max_disp_um: Optional[float] = None   # search radius; None => derived from window scale (reference rule)
    centered: bool = True                 # subtract per-patch weighted means
    normalized: bool = True               # divide by per-patch weighted std
    batch_size: int = 512                 # time-bin tiling for the xcorr (memory control)


@dataclass
class WindowConfig:
    """Non-rigid spatial windows over depth (taper functions)."""

    rigid: bool = True              # one flat window covering the probe (scalar/global motion per dim)
    win_shape: str = "gaussian"     # "gaussian" | "rect" | "triangle"
    win_step_um: float = 400.0      # spacing between window centers
    win_scale_um: float = 450.0     # window width (sigma for gaussian)
    win_margin_um: Optional[float] = None  # None => -win_scale_um / 2 (reference rule)
    zero_threshold: float = 1e-5


@dataclass
class DredgeConfig:
    """Top-level configuration for :class:`dredge_diff.DiffDredge`."""

    # geometry / motion direction
    direction: str = "y"                       # primary depth axis label
    motion_dims: Tuple[str, ...] = ("y",)      # ("y",) = stock DREDge; ("x","y") = full 2D motion
    bin_um: float = 1.0                         # spatial bin size for the raster
    bin_s: float = 1.0                          # temporal bin size for the raster
    time_horizon_s: Optional[float] = None      # pairs of bins farther apart in time are not correlated

    # sub-step configs
    soft_bin: SoftBinConfig = field(default_factory=SoftBinConfig)
    disp: DisplacementConfig = field(default_factory=DisplacementConfig)
    thresh: ThresholdConfig = field(default_factory=ThresholdConfig)
    solve: SolveConfig = field(default_factory=SolveConfig)
    xcorr: XcorrConfig = field(default_factory=XcorrConfig)
    window: WindowConfig = field(default_factory=WindowConfig)

    def __post_init__(self):
        valid = {"x", "y", "z"}
        for d in self.motion_dims:
            if d not in valid:
                raise ValueError(f"motion_dims entries must be in {valid}, got {d!r}")
        if self.direction not in valid:
            raise ValueError(f"direction must be in {valid}, got {self.direction!r}")
