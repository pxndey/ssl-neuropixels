"""Monopolar-triangulation localization pipeline.

Minimal version using SpikeInterface's node pipeline for everything.
Just two modules:
- core: config constants and preprocessing
- pipeline: localization runner
"""

from localizations.core import (
    preprocess_recording,
    get_peak_dtype,
    DEFAULT_FS,
    DEFAULT_FREQ_MIN,
    DEFAULT_FREQ_MAX,
    DEFAULT_MS_BEFORE_LOC,
    DEFAULT_MS_AFTER_LOC,
    DEFAULT_TPCA_N_COMPONENTS,
    DEFAULT_RADIUS_UM_LOC,
    DEFAULT_MAX_DISTANCE_UM,
    DEFAULT_FEATURE,
    DEFAULT_N_JOBS,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_MP_CONTEXT,
    SESSION_SAFE_END_SAMPLE,
)

from localize_mono.pipeline import (
    load_recording,
    localize_session,
    main,
)

__all__ = [
    # Core
    "preprocess_recording",
    "get_peak_dtype",
    "DEFAULT_FS",
    "DEFAULT_FREQ_MIN",
    "DEFAULT_FREQ_MAX",
    "DEFAULT_MS_BEFORE_LOC",
    "DEFAULT_MS_AFTER_LOC",
    "DEFAULT_TPCA_N_COMPONENTS",
    "DEFAULT_RADIUS_UM_LOC",
    "DEFAULT_MAX_DISTANCE_UM",
    "DEFAULT_FEATURE",
    "DEFAULT_N_JOBS",
    "DEFAULT_CHUNK_DURATION",
    "DEFAULT_MP_CONTEXT",
    "SESSION_SAFE_END_SAMPLE",
    # Pipeline
    "load_recording",
    "localize_session",
    "main",
]
