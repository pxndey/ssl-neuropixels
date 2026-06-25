"""Differentiable DREDge (clean-room).

A fully backpropagatable re-implementation of DREDge's motion-estimation
pipeline.  Every discrete step of stock DREDge has been replaced by a smooth,
autograd-friendly equivalent so that gradients from a downstream loss on the
motion trace ``P`` can flow all the way back to the raw inputs (and, in the
integrated pipeline, into an upstream waveform encoder):

    A.1  linear solve            ``solve.py``       (torch.linalg.solve, no inverse)
    A.2  displacement / argmax   ``xcorr.py``       (soft-argmax / parabolic)
    A.3  masking / threshold     ``threshold.py``   (sigmoid gate vs hard)
    A.4  input binning           ``soft_binning.py``(N-D soft histogram)

The numbers, default arguments and matrix algebra mirror the reference DREDge
implementation in spikeinterface
(``sortingcomponents/motion/dredge.py``) -- see ``README_diffdredge.md`` for the
exact correspondence -- but nothing here imports spikeinterface; this is a
self-contained clean-room build on top of plain PyTorch.
"""

from .config import (
    DredgeConfig,
    SoftBinConfig,
    DisplacementConfig,
    ThresholdConfig,
    SolveConfig,
    XcorrConfig,
    WindowConfig,
)
from .dredge import DiffDredge

__all__ = [
    "DredgeConfig",
    "SoftBinConfig",
    "DisplacementConfig",
    "ThresholdConfig",
    "SolveConfig",
    "XcorrConfig",
    "WindowConfig",
    "DiffDredge",
]
