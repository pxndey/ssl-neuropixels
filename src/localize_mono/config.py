"""Configuration constants for preprocessing pipeline."""

from __future__ import annotations

# Signal processing
DEFAULT_FS = 30_000.0
DEFAULT_FREQ_MIN = 300.0
DEFAULT_FREQ_MAX = 6_000.0

# NN-input waveform extraction (90 samples = 3ms at 30kHz). This is the
# patch fed to TransformerSLN and is independent of the MP-target window below.
DEFAULT_MS_BEFORE = 1.5
DEFAULT_MS_AFTER = 1.5
DEFAULT_N_SAMPLES = 90

# MP-target localization window — mirrors neurips-week relocalize_clean.py
# (cfg.MS_BEFORE/MS_AFTER = 5.0/5.0). The TPCA basis is fit AND applied on this
# same dense-waveform window; they MUST match or the temporal-PCA basis is
# misaligned in time and the denoised localization is garbage.
DEFAULT_MS_BEFORE_LOC = 5.0
DEFAULT_MS_AFTER_LOC = 5.0

# Detection
DEFAULT_DETECT_THRESHOLD = 5.0
DEFAULT_PEAK_SIGN = "neg"
DEFAULT_EXCLUDE_SWEEP_MS = 1.0
DEFAULT_RADIUS_UM = 50.0

# Localization
DEFAULT_RADIUS_UM_LOC = 75.0
DEFAULT_MAX_DISTANCE_UM = 150.0
DEFAULT_FEATURE = "ptp"

# Neighborhood
DEFAULT_N_NEIGHBORS = 10
DEFAULT_N_ROWS_EACH_SIDE = 2

# Geometry
DEFAULT_SHANK_X_THRESHOLD = 100.0  # µm gap to identify new shank
DEFAULT_BAND_Y_THRESHOLD = 50.0

# Documented raw-data corruption (neurips-week 01_localization/config.py):
# AL032_2019-11-21's .cbin has a corrupt mtscomp chunk (#1769). Clip the
# recording to the first SAFE_END samples before that chunk so decompression
# never reaches it (drops ~10% of late spikes, keeps the session usable).
SESSION_SAFE_END_SAMPLE: dict[str, int] = {
    "AL032_2019-11-21": 52_800_000,
}

# TPCA
DEFAULT_TPCA_N_COMPONENTS = 7
DEFAULT_TPCA_TRAIN_SPIKES = 10_000
DEFAULT_TPCA_FIT_SECONDS = 300.0

# Parallel processing
DEFAULT_N_JOBS = 8
DEFAULT_CHUNK_DURATION = "1s"
DEFAULT_MP_CONTEXT = "spawn"

# Peak dtype for spike detection
PEAK_DTYPE = None  # Will be set at runtime to avoid numpy import issues

def get_peak_dtype():
    """Get peak dtype (lazy import to avoid circular deps)."""
    import numpy as np
    return np.dtype([
        ("sample_index", np.int64),
        ("channel_index", np.int64),
        ("amplitude", np.float32),
        ("segment_index", np.int64),
    ])
