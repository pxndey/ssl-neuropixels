"""Signal preprocessing: filtering, CMR."""

from __future__ import annotations

import logging

import spikeinterface as si
import spikeinterface.preprocessing as sp

from .config import DEFAULT_FREQ_MIN, DEFAULT_FREQ_MAX

logger = logging.getLogger(__name__)


def preprocess_recording(
    recording: si.BaseRecording,
    freq_min: float = DEFAULT_FREQ_MIN,
    freq_max: float = DEFAULT_FREQ_MAX,
    dtype: str = "float32",
) -> si.BaseRecording:
    logger.info("Preprocessing recording...")

    logger.info(f"  Bandpass filtering ({freq_min}-{freq_max} Hz)...")
    recording = sp.bandpass_filter(
        recording,
        freq_min=freq_min,
        freq_max=freq_max,
        dtype=dtype,
    )

    logger.info("  Applying common median reference...")
    recording = sp.common_reference(
        recording,
        reference="global",
        operator="median",
    )

    logger.info(f"  Preprocessing complete: {recording.get_num_channels()} channels")
    return recording
