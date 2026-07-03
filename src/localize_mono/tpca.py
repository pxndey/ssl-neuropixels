"""Temporal PCA denoising for waveforms.

Mirrors neurips-week 01_localization/relocalize_clean.py:
  - the basis is fit on a 5-minute slice of the (already bandpass+CMR'd)
    recording,
  - on a random subsample of the *already-detected* spikes (no re-detection),
  - over each spike's 10 neighbor channels stacked as per-channel temporal
    rows (n_spikes * 10, T),
  - on the SAME dense-waveform window the localizer uses (DEFAULT_MS_*_LOC),
    so SI's TemporalPCADenoising applies the basis with matching temporal
    alignment, and
  - without whitening (whitening would rescale the basis relative to the
    transform/inverse_transform reconstruction SI performs).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA

import spikeinterface as si

from .config import (
    DEFAULT_TPCA_N_COMPONENTS,
    DEFAULT_TPCA_TRAIN_SPIKES,
    DEFAULT_TPCA_FIT_SECONDS,
    DEFAULT_MS_BEFORE_LOC,
    DEFAULT_MS_AFTER_LOC,
)
from .extraction import extract_waveforms_streaming

logger = logging.getLogger(__name__)


def fit_tpca_model(
    waveforms: NDArray,
    n_components: int = DEFAULT_TPCA_N_COMPONENTS,
) -> PCA:
    """
    Fit a temporal-PCA basis on per-channel waveform rows.

    Args:
        waveforms: (n_spikes, n_neighbors, n_samples) array
        n_components: Number of PCA components

    Returns:
        Fitted sklearn PCA model (no whitening)
    """
    n_spikes, n_neighbors, n_samples = waveforms.shape

    # Per-channel temporal rows: (n_spikes * n_neighbors, n_samples).
    waveforms_2d = waveforms.reshape(-1, n_samples)

    logger.info(f"Fitting TPCA ({n_components} comp) on {waveforms_2d.shape[0]} "
                f"per-channel rows ({n_spikes} spikes x {n_neighbors} ch x "
                f"{n_samples} samp)...")

    pca = PCA(n_components=n_components, svd_solver="auto")
    pca.fit(waveforms_2d)

    logger.info(f"  Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    return pca


def apply_tpca(
    waveforms: NDArray,
    pca_model: PCA,
) -> NDArray:
    """
    Reconstruct waveforms through the TPCA basis (transform + inverse).

    Note: in the pipeline the denoising is actually performed by SI's
    TemporalPCADenoising node (see localization.py); this helper is kept for
    standalone/debug use.
    """
    original_shape = waveforms.shape
    n_samples = original_shape[-1]
    waveforms_2d = waveforms.reshape(-1, n_samples)
    components = pca_model.transform(waveforms_2d)
    waveforms_denoised = pca_model.inverse_transform(components)
    return waveforms_denoised.reshape(original_shape)


def fit_tpca_from_recording(
    recording: si.BaseRecording,
    peaks: NDArray,
    neighbor_table: NDArray,
    n_components: int = DEFAULT_TPCA_N_COMPONENTS,
    fit_seconds: float = DEFAULT_TPCA_FIT_SECONDS,
    n_train: int = DEFAULT_TPCA_TRAIN_SPIKES,
    ms_before: float = DEFAULT_MS_BEFORE_LOC,
    ms_after: float = DEFAULT_MS_AFTER_LOC,
    n_jobs: int = 1,
    seed: int = 42,
) -> PCA:
    """
    Fit a TPCA basis on a time-slice of the (already-preprocessed) recording.

    Reuses the already-detected `peaks` (no re-detection): all spikes in the
    first `fit_seconds` are randomly subsampled to `n_train`, each spike's
    10-neighbor waveforms are extracted on the localization window
    (`ms_before`/`ms_after`), and stacked as per-channel temporal rows for
    `sklearn.decomposition.PCA.fit`.

    Args:
        recording: Preprocessed (shank) recording
        peaks: Detected peaks structure (already boundary-filtered)
        neighbor_table: (n_channels, n_neighbors) channel -> neighbor lookup
        n_components: PCA components
        fit_seconds: Fit on the first N seconds of the recording
        n_train: Number of spikes to subsample for the fit
        ms_before/ms_after: Dense-waveform window — MUST match the localizer's
        n_jobs: Parallel jobs for extraction
        seed: RNG seed for the spike subsample

    Returns:
        Fitted sklearn PCA model
    """
    fs = recording.get_sampling_frequency()
    n_total = recording.get_num_samples()
    fit_end = min(int(fit_seconds * fs), n_total)

    sample_index = peaks["sample_index"]
    pool_idx = np.flatnonzero(sample_index < fit_end)
    if pool_idx.size == 0:
        raise ValueError(
            f"No spikes in the first {fit_seconds}s for TPCA fit — "
            f"pick a larger fit window"
        )

    rng = np.random.default_rng(seed)
    n_take = min(n_train, pool_idx.size)
    sel = np.sort(rng.choice(pool_idx, size=n_take, replace=False))
    logger.info(f"TPCA fit: pool={pool_idx.size} spikes in first {fit_seconds}s, "
                f"selected={n_take} (window={ms_before}/{ms_after}ms)")

    sel_times = peaks["sample_index"][sel]
    sel_peak_channels = peaks["channel_index"][sel].astype(np.int64)
    neighbor_ids = neighbor_table[sel_peak_channels]  # (n_take, n_neighbors)

    rec_slice = recording.frame_slice(start_frame=0, end_frame=fit_end)
    waveforms = extract_waveforms_streaming(
        rec_slice,
        sel_times,
        neighbor_ids,
        ms_before=ms_before,
        ms_after=ms_after,
        n_jobs=n_jobs,
    )  # (n_take, n_neighbors, n_samples)

    return fit_tpca_model(waveforms, n_components=n_components)
