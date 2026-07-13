"""Core configuration and preprocessing."""

from __future__ import annotations

import logging

import spikeinterface as si
import spikeinterface.preprocessing as sp

logger = logging.getLogger(__name__)

# Signal processing
DEFAULT_FS = 30_000.0
DEFAULT_FREQ_MIN = 300.0
DEFAULT_FREQ_MAX = 6_000.0

# NN-input waveform extraction (90 samples = 3ms at 30kHz)
DEFAULT_MS_BEFORE = 1.5
DEFAULT_MS_AFTER = 1.5
DEFAULT_N_SAMPLES = 90

# MP-target localization window
DEFAULT_MS_BEFORE_LOC = 5.0
DEFAULT_MS_AFTER_LOC = 5.0

# TPCA
DEFAULT_TPCA_N_COMPONENTS = 7

# Localization
DEFAULT_RADIUS_UM_LOC = 75.0
DEFAULT_MAX_DISTANCE_UM = 150.0
DEFAULT_FEATURE = "ptp"

# Parallel processing
DEFAULT_N_JOBS = 8
DEFAULT_CHUNK_DURATION = "1s"
DEFAULT_MP_CONTEXT = "spawn"

# Session-specific fixes
SESSION_SAFE_END_SAMPLE: dict[str, int] = {
    "AL032_2019-11-21": 52_800_000,
}


def get_peak_dtype():
    import numpy as np
    return np.dtype([
        ("sample_index", np.int64),
        ("channel_index", np.int64),
        ("amplitude", np.float32),
        ("segment_index", np.int64),
    ])


def preprocess_recording(
    recording: si.BaseRecording,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    dtype: str = "float32",
) -> si.BaseRecording:
    logger.info(f"Bandpass filtering ({freq_min}-{freq_max} Hz)...")
    recording = sp.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max, dtype=dtype)
    
    logger.info("Applying common median reference...")
    recording = sp.common_reference(recording, reference="global", operator="median")
    
    logger.info(f"Preprocessing complete: {recording.get_num_channels()} channels")
    return recording
