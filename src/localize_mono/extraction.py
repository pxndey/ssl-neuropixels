"""Waveform extraction from recordings."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

import spikeinterface as si

from .config import (
    DEFAULT_MS_BEFORE,
    DEFAULT_MS_AFTER,
)

logger = logging.getLogger(__name__)


def extract_spike_waveforms(
    recording: si.BaseRecording,
    sample_index: NDArray,
    neighbor_ids: NDArray,
    ms_before: float = DEFAULT_MS_BEFORE,
    ms_after: float = DEFAULT_MS_AFTER,
) -> NDArray:
    """
    Extract spike waveforms from recording.
    
    Args:
        recording: Recording object
        sample_index: Sample indices of spikes
        neighbor_ids: Neighbor channel IDs for each spike (n_spikes, n_neighbors)
        ms_before, ms_after: Time window around spike
        
    Returns:
        waveforms: (n_spikes, n_neighbors, n_samples) array
    """
    fs = recording.get_sampling_frequency()
    n_before = int(ms_before * fs / 1000)
    n_after = int(ms_after * fs / 1000)
    n_samples = n_before + n_after
    
    n_spikes = len(sample_index)
    n_neighbors = neighbor_ids.shape[1]
    
    waveforms = np.zeros((n_spikes, n_neighbors, n_samples), dtype=np.float32)
    
    traces = recording.get_traces()
    n_total_samples = traces.shape[0]
    
    for i, spike_time in enumerate(sample_index):
        start = int(spike_time) - n_before
        end = int(spike_time) + n_after
        
        # Handle boundaries
        if start < 0 or end > n_total_samples:
            continue
        
        for j, ch in enumerate(neighbor_ids[i]):
            waveforms[i, j, :] = traces[start:end, ch]
    
    return waveforms


def extract_waveforms_streaming(
    recording: si.BaseRecording,
    spike_times: NDArray,
    neighbor_ids: NDArray,
    ms_before: float = DEFAULT_MS_BEFORE,
    ms_after: float = DEFAULT_MS_AFTER,
    chunk_duration: float = 10.0,  # seconds
    n_jobs: int = 1,
) -> NDArray:
    """
    Extract waveforms in a memory-efficient streaming fashion.
    
    Args:
        recording: Recording object
        spike_times: Sample indices of spikes
        neighbor_ids: Neighbor channel IDs for each spike
        ms_before, ms_after: Time window around spike
        chunk_duration: Duration of each chunk in seconds
        n_jobs: Number of parallel jobs (currently unused)
        
    Returns:
        waveforms: (n_spikes, n_neighbors, n_samples) array
    """
    fs = recording.get_sampling_frequency()
    n_before = int(ms_before * fs / 1000)
    n_after = int(ms_after * fs / 1000)
    n_samples = n_before + n_after
    
    n_spikes = len(spike_times)
    n_neighbors = neighbor_ids.shape[1]
    
    waveforms = np.zeros((n_spikes, n_neighbors, n_samples), dtype=np.float32)
    
    # Sort spikes by time for efficient chunking
    sort_indices = np.argsort(spike_times)
    spike_times_sorted = spike_times[sort_indices]
    
    chunk_samples = int(chunk_duration * fs)
    n_total_samples = recording.get_num_samples()
    n_chunks = int(np.ceil(n_total_samples / chunk_samples))
    
    logger.info(f"Extracting waveforms in {n_chunks} chunks...")
    
    spike_idx = 0
    for chunk_i in range(n_chunks):
        chunk_start = chunk_i * chunk_samples
        chunk_end = min(chunk_start + chunk_samples, n_total_samples)
        
        # Load chunk with margin for waveforms
        load_start = max(0, chunk_start - n_before)
        load_end = min(n_total_samples, chunk_end + n_after)
        
        chunk_traces = recording.get_traces(start_frame=load_start, end_frame=load_end)
        
        # Find spikes in this chunk
        chunk_spike_mask = (spike_times_sorted >= chunk_start) & (spike_times_sorted < chunk_end)
        chunk_spike_indices = np.where(chunk_spike_mask)[0]
        
        for idx_in_chunk in chunk_spike_indices:
            spike_time = spike_times_sorted[idx_in_chunk]
            original_idx = sort_indices[idx_in_chunk]
            
            start_in_chunk = int(spike_time) - n_before - load_start
            end_in_chunk = int(spike_time) + n_after - load_start
            
            if start_in_chunk >= 0 and end_in_chunk <= chunk_traces.shape[0]:
                for j, ch in enumerate(neighbor_ids[original_idx]):
                    waveforms[original_idx, j, :] = chunk_traces[start_in_chunk:end_in_chunk, ch]
        
        if (chunk_i + 1) % 10 == 0 or chunk_i == 0:
            logger.info(f"  Processed chunk {chunk_i + 1}/{n_chunks}")
    
    return waveforms
