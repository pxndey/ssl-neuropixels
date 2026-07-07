"""Localization pipeline using SpikeInterface node pipeline."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
    PeakRetriever,
    ExtractDenseWaveforms,
)
from spikeinterface.sortingcomponents.peak_localization.monopolar import (
    LocalizeMonopolarTriangulation,
)
from spikeinterface.sortingcomponents.waveforms.temporal_pca import (
    TemporalPCADenoising,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localize_mono.core import (
    preprocess_recording,
    get_peak_dtype,
    DEFAULT_MS_BEFORE_LOC,
    DEFAULT_MS_AFTER_LOC,
    DEFAULT_TPCA_N_COMPONENTS,
    DEFAULT_RADIUS_UM_LOC,
    DEFAULT_MAX_DISTANCE_UM,
    DEFAULT_FEATURE,
    DEFAULT_N_JOBS,
    DEFAULT_CHUNK_DURATION,
    DEFAULT_MP_CONTEXT,
)

logger = logging.getLogger(__name__)

REPO = Path("/scratch/ap7151/sln-v2")

SESSION_CONFIG = {
    "dataset1_p1": dict(
        recording="/scratch/ap7151/RAW_DATA/extra-motion/dataset1_p1/",
        fmt="spikeglx"
    ),
    "dataset1_p2": dict(
        recording="/scratch/ap7151/RAW_DATA/extra-motion/dataset1_p2/",
        fmt="spikeglx"
    ),
    "dandi_000957_sub-ZYE-0021_ses-1": dict(
        recording="/scratch/ap7151/pl2820-extramotion-npultra/sub-ZYE-0021/sub-ZYE-0021_ses-1_ecephys+image.nwb",
        fmt="nwb"
    ),
}


def load_recording(recording_path: str, fmt: str) -> si.BaseRecording:
    if fmt == "nwb":
        return se.read_nwb_recording(
            file_path=recording_path,
            electrical_series_path="acquisition/ElectricalSeriesAP"
        )
    if fmt == "spikeglx":
        return se.read_spikeglx(folder_path=recording_path, stream_id="imec0.ap")
    if fmt == "cbin":
        return se.read_cbin_ibl(recording_path, stream_name="ap")
    raise ValueError(f"unknown format {fmt!r}")


def localize_session(
    session_id: str,
    recording_path: str,
    fmt: str,
    n_jobs: int = DEFAULT_N_JOBS,
    limit: int | None = None,
    output: Path | None = None,
) -> None:
    """Run monopolar localization on a session."""
    session_path = REPO / "runs" / session_id
    
    spike_times = np.load(session_path / "spike_times.npy").astype(np.int64)
    spike_channels = np.load(session_path / "spike_channels.npy").astype(np.int64)
    
    if limit:
        spike_times = spike_times[:limit]
        spike_channels = spike_channels[:limit]
    
    n_all = len(spike_times)
    print(f"[localize] session={session_id} n_spikes={n_all} fmt={fmt} n_jobs={n_jobs}", flush=True)
    
    rec = load_recording(recording_path, fmt)
    rec = preprocess_recording(rec)
    
    fs = rec.get_sampling_frequency()
    n_total = rec.get_num_samples()
    n_before = int(DEFAULT_MS_BEFORE_LOC * fs / 1000)
    n_after = int(DEFAULT_MS_AFTER_LOC * fs / 1000)
    
    print(f"[rec] {rec.get_num_channels()} ch, {n_total} samp @ {fs:.0f} Hz", flush=True)
    
    in_bounds = (spike_times >= n_before) & (spike_times < n_total - n_after)
    n_valid = int(in_bounds.sum())
    print(f"[bounds] {n_valid}/{n_all} spikes in-bounds ({n_all - n_valid} dropped)", flush=True)
    
    peaks = np.zeros(n_valid, dtype=get_peak_dtype())
    peaks["sample_index"] = spike_times[in_bounds]
    peaks["channel_index"] = spike_channels[in_bounds]
    peaks["amplitude"] = 0.0
    peaks["segment_index"] = 0
    
    print("[pipeline] building nodes...", flush=True)
    
    # Build pipeline: PeakRetriever -> ExtractDenseWaveforms -> TemporalPCADenoising -> LocalizeMonopolarTriangulation
    peak_source = PeakRetriever(rec, peaks)
    
    dense_wf = ExtractDenseWaveforms(
        rec,
        ms_before=DEFAULT_MS_BEFORE_LOC,
        ms_after=DEFAULT_MS_AFTER_LOC,
        parents=[peak_source],
        return_output=False,
    )
    
    # TPCA denoising - fits model internally on the waveforms
    tpca_node = TemporalPCADenoising(
        rec,
        parents=[peak_source, dense_wf],
        return_output=False,
        n_components=DEFAULT_TPCA_N_COMPONENTS,
    )
    
    localize = LocalizeMonopolarTriangulation(
        rec,
        parents=[peak_source, tpca_node],
        return_output=True,
        radius_um=DEFAULT_RADIUS_UM_LOC,
        max_distance_um=DEFAULT_MAX_DISTANCE_UM,
        feature=DEFAULT_FEATURE,
    )
    
    print("[pipeline] running...", flush=True)
    import time
    t0 = time.time()
    
    locations = run_node_pipeline(
        rec,
        nodes=[peak_source, dense_wf, tpca_node, localize],
        job_kwargs=dict(
            n_jobs=n_jobs,
            chunk_duration=DEFAULT_CHUNK_DURATION,
            progress_bar=True,
            mp_context=DEFAULT_MP_CONTEXT,
        ),
    )
    
    print(f"[pipeline] localized {len(locations)} spikes in {time.time() - t0:.1f}s", flush=True)
    
    # Reconstruct full array with NaN for out-of-bounds spikes
    full = np.full((n_all, 4), np.nan, dtype=np.float64)
    full[in_bounds] = np.column_stack([locations["x"], locations["y"], locations["z"], locations["alpha"]])
    
    out = output or session_path / "monopolar_true" / "localizations.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, full)
    np.save(out.parent / "in_bounds.npy", in_bounds)
    
    print(f"[save] {out} shape={full.shape}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-id", required=True)
    p.add_argument("--recording-path", default=None)
    p.add_argument("--format", default=None, choices=["spikeglx", "nwb", "cbin"])
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()
    
    cfg = SESSION_CONFIG.get(args.session_id, {})
    recording_path = args.recording_path or cfg.get("recording")
    fmt = args.format or cfg.get("fmt")
    
    if not recording_path or not fmt:
        raise SystemExit(f"no recording/format for session {args.session_id}; pass --recording-path/--format")
    
    localize_session(
        session_id=args.session_id,
        recording_path=recording_path,
        fmt=fmt,
        n_jobs=args.n_jobs,
        limit=args.limit,
        output=Path(args.output) if args.output else None,
    )


if __name__ == "__main__":
    main()
