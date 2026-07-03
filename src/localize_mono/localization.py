"""Monopolar triangulation spike localization."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray

import spikeinterface as si
from spikeinterface.sortingcomponents.peak_localization.monopolar import (
    LocalizeMonopolarTriangulation,
)
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    PeakRetriever,
    ExtractDenseWaveforms,
)

try:
    from spikeinterface.sortingcomponents.waveforms.temporal_pca import (
        TemporalPCADenoising,
    )
    HAS_SI_TPCA = True
except ImportError:
    HAS_SI_TPCA = False

from .config import (
    DEFAULT_RADIUS_UM_LOC,
    DEFAULT_MAX_DISTANCE_UM,
    DEFAULT_MS_BEFORE_LOC,
    DEFAULT_MS_AFTER_LOC,
    DEFAULT_N_JOBS,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_MP_CONTEXT,
    DEFAULT_FEATURE,
)
from .tpca import apply_tpca

logger = logging.getLogger(__name__)


def localize_spikes_monopolar(
    recording: si.BaseRecording,
    peaks: NDArray,
    radius_um: float = DEFAULT_RADIUS_UM_LOC,
    max_distance_um: float = DEFAULT_MAX_DISTANCE_UM,
    ms_before: float = DEFAULT_MS_BEFORE_LOC,
    ms_after: float = DEFAULT_MS_AFTER_LOC,
    n_jobs: int = DEFAULT_N_JOBS,
    pca_model: Optional = None,
) -> NDArray:
    """
    Localize spikes using monopolar triangulation.
    
    Args:
        recording: Preprocessed recording
        peaks: Detected peaks structure
        radius_um: Localization radius
        max_distance_um: Maximum distance for localization
        ms_before, ms_after: Waveform window for localization
        n_jobs: Number of parallel jobs
        pca_model: Optional TPCA model for denoising
        
    Returns:
        locations: Structured array with (x, y, z, alpha)
    """
    logger.info(f"Localizing {len(peaks)} spikes "
                f"(radius={radius_um}µm, TPCA={pca_model is not None})")
    
    # Build pipeline
    peak_source = PeakRetriever(recording, peaks)
    dense_wf = ExtractDenseWaveforms(
        recording,
        ms_before=ms_before,
        ms_after=ms_after,
        parents=[peak_source],
        return_output=False,
    )
    
    if pca_model is not None and HAS_SI_TPCA:
        # Add TPCA denoising node
        tpca_node = TemporalPCADenoising(
            recording,
            parents=[peak_source, dense_wf],
            pca_model=pca_model,
            return_output=False,
        )
        localize = LocalizeMonopolarTriangulation(
            recording,
            parents=[peak_source, tpca_node],
            return_output=True,
            radius_um=radius_um,
            max_distance_um=max_distance_um,
            feature=DEFAULT_FEATURE,
        )
        nodes = [peak_source, dense_wf, tpca_node, localize]
    else:
        # Direct localization without TPCA
        localize = LocalizeMonopolarTriangulation(
            recording,
            parents=[peak_source, dense_wf],
            return_output=True,
            radius_um=radius_um,
            max_distance_um=max_distance_um,
            feature=DEFAULT_FEATURE,
        )
        nodes = [peak_source, dense_wf, localize]
    
    # Run pipeline
    import time
    t0 = time.time()
    locations = run_node_pipeline(
        recording,
        nodes=nodes,
        job_kwargs=dict(
            n_jobs=n_jobs,
            chunk_duration=DEFAULT_CHUNK_DURATION,
            progress_bar=True,
            mp_context=DEFAULT_MP_CONTEXT,
        ),
    )
    
    logger.info(f"Localized {len(locations)} spikes in {time.time() - t0:.1f}s")
    
    return locations
