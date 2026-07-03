"""Monopolar-triangulation localization of OUR detected spikes.

Feeds our own runs/<session>/{spike_times,spike_channels}.npy as the peaks array
into the ported sln pipeline (bandpass + CMR -> TPCA denoise -> monopolar
triangulation), so the resulting (x,y,z,alpha) are on EXACTLY our spikes and can
be compared per-spike against the SLN model.

Output: runs/<session>/monopolar_true/localizations.npy, shape (n_all_spikes, 4)
= [x=lateral, y=depth, z=dist-from-probe, alpha], ALIGNED to the original spike
order (NaN rows for spikes dropped by the 5/5 ms boundary filter). Also saves
in_bounds.npy (bool mask).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import spikeinterface.extractors as se

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/ on path
from localize_mono.signals import preprocess_recording           # noqa: E402
from localize_mono.geometry import (                             # noqa: E402
    build_neighbor_table_sliding, build_neighbor_table_euclidean,
)
from localize_mono.tpca import fit_tpca_from_recording           # noqa: E402
from localize_mono.localization import localize_spikes_monopolar  # noqa: E402
from localize_mono.config import (                               # noqa: E402
    get_peak_dtype, DEFAULT_MS_BEFORE_LOC, DEFAULT_MS_AFTER_LOC,
)

REPO = Path("/scratch/ap7151/sln-v2")

SESSION_CONFIG = {
    "dataset1_p1": dict(recording="/scratch/ap7151/RAW_DATA/extra-motion/dataset1_p1/",
                        fmt="spikeglx", neighbor="sliding"),
    "dataset1_p2": dict(recording="/scratch/ap7151/RAW_DATA/extra-motion/dataset1_p2/",
                        fmt="spikeglx", neighbor="sliding"),
    "dandi_000957_sub-ZYE-0021_ses-1": dict(
        recording="/scratch/ap7151/pl2820-extramotion-npultra/sub-ZYE-0021/sub-ZYE-0021_ses-1_ecephys+image.nwb",
        fmt="nwb", neighbor="euclidean"),
}


def load_recording(recording_path, fmt):
    if fmt == "nwb":
        return se.read_nwb_recording(file_path=recording_path,
                                     electrical_series_path="acquisition/ElectricalSeriesAP")
    if fmt == "spikeglx":
        return se.read_spikeglx(folder_path=recording_path, stream_id="imec0.ap")
    if fmt == "cbin":
        return se.read_cbin_ibl(recording_path, stream_name="ap")
    raise ValueError(f"unknown format {fmt!r}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-id", required=True)
    p.add_argument("--recording-path", default=None)
    p.add_argument("--format", default=None, choices=["spikeglx", "nwb", "cbin"])
    p.add_argument("--neighbor-method", default=None, choices=["sliding", "euclidean"])
    p.add_argument("--n-jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--limit", type=int, default=None, help="localize only the first N spikes (debug)")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    cfg = SESSION_CONFIG.get(args.session_id, {})
    recording_path = args.recording_path or cfg.get("recording")
    fmt = args.format or cfg.get("fmt")
    neighbor_method = args.neighbor_method or cfg.get("neighbor", "sliding")
    if not recording_path or not fmt:
        raise SystemExit(f"no recording/format for session {args.session_id}; pass --recording-path/--format")

    session_path = REPO / "runs" / args.session_id
    spike_times = np.load(session_path / "spike_times.npy").astype(np.int64)
    spike_channels = np.load(session_path / "spike_channels.npy").astype(np.int64)
    if args.limit:
        spike_times = spike_times[:args.limit]
        spike_channels = spike_channels[:args.limit]
    n_all = len(spike_times)
    print(f"[localize] session={args.session_id} n_spikes={n_all} fmt={fmt} "
          f"neighbor={neighbor_method} n_jobs={args.n_jobs}", flush=True)

    rec = load_recording(recording_path, fmt)
    rec = preprocess_recording(rec)                                # bandpass + CMR
    fs = rec.get_sampling_frequency()
    n_total = rec.get_num_samples()
    n_before = int(DEFAULT_MS_BEFORE_LOC * fs / 1000)
    n_after = int(DEFAULT_MS_AFTER_LOC * fs / 1000)
    print(f"[rec] {rec.get_num_channels()} ch, {n_total} samp @ {fs:.0f} Hz; "
          f"loc window {n_before}/{n_after} samp", flush=True)

    in_bounds = (spike_times >= n_before) & (spike_times < n_total - n_after)
    n_valid = int(in_bounds.sum())
    print(f"[bounds] {n_valid}/{n_all} spikes in-bounds ({n_all - n_valid} dropped)", flush=True)

    peaks = np.zeros(n_valid, dtype=get_peak_dtype())
    peaks["sample_index"] = spike_times[in_bounds]
    peaks["channel_index"] = spike_channels[in_bounds]
    peaks["amplitude"] = 0.0
    peaks["segment_index"] = 0

    ch = rec.get_channel_locations()
    if neighbor_method == "sliding":
        neighbor_table = build_neighbor_table_sliding(ch[:, 0], ch[:, 1])
    else:
        neighbor_table = build_neighbor_table_euclidean(ch)

    print("[tpca] fitting basis...", flush=True)
    pca = fit_tpca_from_recording(rec, peaks, neighbor_table, n_jobs=args.n_jobs)

    print("[monopolar] localizing...", flush=True)
    locs = localize_spikes_monopolar(rec, peaks, n_jobs=args.n_jobs, pca_model=pca)

    full = np.full((n_all, 4), np.nan, dtype=np.float64)
    full[in_bounds] = np.column_stack([locs["x"], locs["y"], locs["z"], locs["alpha"]])

    out = Path(args.output) if args.output else session_path / "monopolar_true" / "localizations.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, full)
    np.save(out.parent / "in_bounds.npy", in_bounds)
    print(f"[save] {out}  shape={full.shape}  columns=[x_lateral, y_depth, z_perp, alpha]", flush=True)
    print(f"[save] {out.parent / 'in_bounds.npy'}", flush=True)


if __name__ == "__main__":
    main()
