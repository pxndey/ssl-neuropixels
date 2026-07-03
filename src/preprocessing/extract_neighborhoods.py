"""Extract neighborhood channels and centroids for each spike time using radius-based extraction."""

import argparse
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
from spikeinterface.core.node_pipeline import (
    PeakRetriever,
    ExtractSparseWaveforms,
    run_node_pipeline,
)
from pathlib import Path
import numpy as np


def load_recording(recording_path: str) -> si.BaseRecording:
    """Load recording based on file extension."""
    recording_path_obj = Path(recording_path)
    file_suffix = recording_path_obj.suffix.lower()
    
    if file_suffix == ".cbin":
        recording = se.read_cbin_ibl(
            file_path=recording_path,
            stream_name="ap"
        )
    elif file_suffix == ".nwb":
        recording = se.read_nwb_recording(
            file_path=recording_path,
            electrical_series_path="acquisition/ElectricalSeriesAP"
        )
    else:
        # SpikeGLX expects folder_path, but we pass file_path for consistency
        # Extract folder from the full file path
        recording_folder = str(Path(recording_path).parent)
        recording = se.read_spikeglx(
            folder_path=recording_folder,
            stream_id="imec0.ap"
        )
    
    return recording


def preprocess_recording(recording: si.BaseRecording) -> si.BaseRecording:
    """Apply standard preprocessing: bandpass filter and CMR."""
    recording = sp.bandpass_filter(recording, freq_min=300, freq_max=6000)
    recording = sp.common_reference(recording, reference="global", operator="median")
    return recording


def extract_neighborhoods(
    session_path: str,
    recording_path: str,
    radius_um: float = 48.0,
    ms_before: float = 1.5,
    ms_after: float = 1.5,
) -> None:
    """
    Extract neighborhood channels and centroids for each spike using radius-based extraction.
    
    Uses SpikeInterface's node_pipeline for efficient chunked parallel extraction.
    Allows variable number of neighbors per spike (for transformer-based models).
    
    Args:
        session_path: Path to session directory with spike_times.npy and spike_channels.npy
        recording_path: Path to recording file/directory
        radius_um: Radius in microns for neighborhood extraction (default 48.0 for NP1.0 footprint)
        ms_before: Time before spike in ms
        ms_after: Time after spike in ms
    """
    session_path = Path(session_path)
    session_id = session_path.name
    
    spike_times = np.load(session_path / "spike_times.npy")
    spike_channels = np.load(session_path / "spike_channels.npy")
    n_spikes = len(spike_times)
    
    print(f"Loaded {n_spikes} spikes from {session_id}")
    print(f"Loading recording from: {recording_path}")
    
    recording = load_recording(recording_path)
    recording = preprocess_recording(recording)
    
    fs = recording.get_sampling_frequency()
    n_before = int(ms_before * fs / 1000)
    n_after = int(ms_after * fs / 1000)
    n_samples = n_before + n_after
    
    channel_locations = recording.get_channel_locations()
    
    # Create peaks array for PeakRetriever
    # Need: sample_index, channel_index, and segment_index
    peaks = np.zeros(n_spikes, dtype=[('sample_index', np.int64), ('channel_index', np.int64), ('segment_index', np.int64)])
    peaks['sample_index'] = spike_times
    peaks['channel_index'] = spike_channels
    peaks['segment_index'] = 0
    
    print(f"Extracting waveforms with {radius_um}um radius...")
    print(f"ms_before: {ms_before}, ms_after: {ms_after}")
    print(f"Samples: {n_samples} ({n_before} before, {n_after} after)")
    
    # Build pipeline: PeakRetriever -> ExtractSparseWaveforms
    peak_retriever = PeakRetriever(recording, peaks)
    waveform_node = ExtractSparseWaveforms(
        recording=recording,
        ms_before=ms_before,
        ms_after=ms_after,
        parents=[peak_retriever],
        return_output=True,
        radius_um=radius_um,
    )
    
    # Run pipeline with memory gathering
    job_kwargs = dict(n_jobs=8, chunk_duration="1s", mp_context="spawn")
    waveforms = run_node_pipeline(
        recording=recording,
        nodes=[peak_retriever, waveform_node],
        job_kwargs=job_kwargs,
        gather_mode="memory",
        squeeze_output=True,
    )
    
    # waveforms shape: (n_spikes, n_samples, max_neighbors)
    # But with variable actual neighbors, need to get the actual channel count per spike
    print(f"Raw waveforms shape: {waveforms.shape}")
    
    # Get the neighbor mask used by ExtractSparseWaveforms
    neighbor_mask = waveform_node.neighbours_mask  # (n_channels, n_channels) bool
    
    # For each spike, determine which channels are in its neighborhood
    # and extract the actual (non-zero) waveforms
    print("Processing variable neighborhood sizes...")
    
    # First pass: determine max neighbors and actual counts
    neighbor_counts = np.array([neighbor_mask[ch].sum() for ch in spike_channels])
    max_neighbors = neighbor_counts.max()
    
    print(f"Neighbor count stats: min={neighbor_counts.min()}, max={max_neighbors}, mean={neighbor_counts.mean():.1f}")
    
    # Build dense arrays with padding
    # Shape: (n_spikes, max_neighbors, n_samples)
    waveforms_dense = np.zeros((n_spikes, max_neighbors, n_samples), dtype=np.float32)
    neighbor_ids = np.full((n_spikes, max_neighbors), -1, dtype=np.int64)  # -1 for padding
    local_coords = np.zeros((n_spikes, max_neighbors, 2), dtype=np.float32)
    centroids = np.zeros((n_spikes, 2), dtype=np.float32)
    actual_neighbor_counts = np.zeros(n_spikes, dtype=np.int64)
    
    for i in range(n_spikes):
        ch = spike_channels[i]
        nbrs = np.where(neighbor_mask[ch])[0]
        n_nbrs = len(nbrs)
        actual_neighbor_counts[i] = n_nbrs
        
        # Store neighbor IDs
        neighbor_ids[i, :n_nbrs] = nbrs
        
        # Get waveform for this spike (shape: n_samples, max_neighbors)
        wf = waveforms[i]  # This is the sparse waveform
        
        # Store only the actual channels (non-padded part)
        waveforms_dense[i, :n_nbrs, :] = wf[:, :n_nbrs].T
        
        # Compute centroid and local coordinates
        nbr_positions = channel_locations[nbrs]
        centroids[i] = nbr_positions.mean(axis=0)
        local_coords[i, :n_nbrs] = nbr_positions - centroids[i]
    
    print(f"Saving outputs to {session_path}...")
    np.save(session_path / "neighborhood_waveforms.npy", waveforms_dense)
    np.save(session_path / "neighbor_ids.npy", neighbor_ids)
    np.save(session_path / "centroids.npy", centroids)
    np.save(session_path / "local_coords.npy", local_coords)
    np.save(session_path / "neighbor_counts.npy", actual_neighbor_counts)
    
    print(f"Done! Saved:")
    print(f"  - neighborhood_waveforms.npy: {waveforms_dense.shape} (padded to max_neighbors={max_neighbors})")
    print(f"  - neighbor_ids.npy: {neighbor_ids.shape} (padded with -1)")
    print(f"  - centroids.npy: {centroids.shape}")
    print(f"  - local_coords.npy: {local_coords.shape}")
    print(f"  - neighbor_counts.npy: {actual_neighbor_counts.shape} (actual counts per spike)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract neighborhood channels and centroids for each spike time using radius-based extraction."
    )
    parser.add_argument("--session-path", type=str, required=True)
    parser.add_argument("--recording-path", type=str, required=True)
    parser.add_argument("--radius-um", type=float, default=48.0, help="Radius in microns for neighborhood (default: 48.0)")
    parser.add_argument("--ms-before", type=float, default=1.5)
    parser.add_argument("--ms-after", type=float, default=1.5)
    args = parser.parse_args()

    extract_neighborhoods(
        session_path=args.session_path,
        recording_path=args.recording_path,
        radius_um=args.radius_um,
        ms_before=args.ms_before,
        ms_after=args.ms_after,
    )
