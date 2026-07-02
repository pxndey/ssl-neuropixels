"""Run differentiable DREDge on an extracted session and save the motion trace.

This is the inference entry point that actually *produces* a motion trace ``P(t)``
-- the estimated probe drift over time -- from the outputs of
``extract_neighborhoods.py``:

    spike_times.npy   -> time bins
    centroids.npy     -> per-spike (x, y) localization  (y = depth)
    neighborhood_waveforms.npy + neighbor_ids/spike_channels -> per-spike amplitude

The raster is built by soft-binning (chunked over spikes so millions fit in
memory), then A.2->A.3->A.1 of :class:`dredge_diff.DiffDredge` solve for ``P``.
By default this is the plain DREDge estimate (raw detected amplitudes/positions,
hard argmax); pass ``--disp soft`` to use the differentiable soft-argmax, or wire
in the encoder's cleaned features via ``pipeline.py`` for the learned variant.

NOTE: ``centroids.npy`` is the neighborhood channel centroid (~peak channel),
a channel-resolution proxy, not a true sub-channel localization. For real
DREDge-AP resolution, feed amplitude-weighted center-of-mass localizations (or
the learned encoder positions) instead.

Run on a GPU via ``infer_motion.sbatch`` (uses ``singularity exec --nv``).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from dredge_diff import DiffDredge, DredgeConfig
from dredge_diff.config import (
    DisplacementConfig,
    SoftBinConfig,
    ThresholdConfig,
    WindowConfig,
    XcorrConfig,
)
from dredge_diff.soft_binning import build_raster, make_bin_centers


def peak_amplitudes(session: Path, chunk: int = 100_000) -> np.ndarray:
    """Per-spike amplitude = max |waveform| on the peak channel (memory-mapped)."""
    counts = np.load(session / "neighbor_counts.npy")
    neighbor_ids = np.load(session / "neighbor_ids.npy")
    spike_channels = np.load(session / "spike_channels.npy")
    match = neighbor_ids == spike_channels[:, None]
    peak_idx = np.where(match.any(axis=1), match.argmax(axis=1), 0)

    wf = np.load(session / "neighborhood_waveforms.npy", mmap_mode="r")   # (S, M, 90)
    S = wf.shape[0]
    amp = np.empty(S, dtype=np.float32)
    for i in range(0, S, chunk):
        sl = slice(i, min(i + chunk, S))
        w = np.asarray(wf[sl])                                            # (c, M, 90)
        peak_wf = w[np.arange(w.shape[0]), peak_idx[sl]]                  # (c, 90)
        amp[sl] = np.abs(peak_wf).max(axis=1)
    return amp


def main(args) -> None:
    session = Path(args.session)
    device = torch.device(args.device)

    if args.loc_npy:
        # real per-spike localizations (e.g. sln preprocessed: cols x, y(depth), z, alpha)
        loc = np.load(args.loc_npy)
        x_all = loc[:, args.x_col].astype(np.float64)
        y_all = loc[:, args.depth_col].astype(np.float64)
        amp = np.abs(loc[:, args.amp_col]).astype(np.float32)
        spike_times = np.load(args.times_npy).astype(np.int64)
        src = f"{args.loc_npy} (depth=col{args.depth_col}, amp=col{args.amp_col})"
    else:
        spike_times = np.load(session / "spike_times.npy").astype(np.int64)   # sample indices
        centroids = np.load(session / "centroids.npy").astype(np.float64)     # (S, 2) = (x, y)
        x_all, y_all = centroids[:, 0], centroids[:, 1]
        amp = peak_amplitudes(session)
        src = f"centroids in {session}"
    S = spike_times.shape[0]
    print(f"{S} spikes from {src} | depth range {y_all.min():.0f}..{y_all.max():.0f} um")

    samples_per_bin = args.bin_s * args.fs
    time_idx = (spike_times // samples_per_bin).astype(np.int64)
    n_time = int(time_idx.max()) + 1
    print(f"fs={args.fs} bin_s={args.bin_s} -> {n_time} time bins (~{n_time*args.bin_s:.0f} s)")

    y = torch.tensor(y_all, dtype=torch.float32)
    x = torch.tensor(x_all, dtype=torch.float32)
    feat = torch.tensor(amp, dtype=torch.float32)
    tidx = torch.tensor(time_idx, dtype=torch.long)

    two_d = args.motion_dims == "xy"
    axes = ["x", "y"] if two_d else ["y"]
    coord_t = {"x": x, "y": y}

    centers = {
        d: make_bin_centers(float(coord_t[d].min()), float(coord_t[d].max()),
                            args.bin_um, device=device, dtype=torch.float32)
        for d in axes
    }
    dims = [centers[d].numel() for d in axes]
    bin_cfg = SoftBinConfig(mode="gaussian", bandwidth_um=args.bandwidth_um, normalize=True, trunc=4.0)

    # --- build the raster, chunked over spikes (sum of per-chunk rasters) --------
    raster = torch.zeros(*dims, n_time, device=device)
    chunk = args.spike_chunk
    with torch.no_grad():
        for i in range(0, S, chunk):
            sl = slice(i, min(i + chunk, S))
            coords_chunk = [coord_t[d][sl].to(device) for d in axes]
            ctrs = [centers[d] for d in axes]
            raster = raster + build_raster(
                coords_chunk, ctrs, feat[sl].to(device),
                n_time=n_time, time_idx=tidx[sl].to(device), cfg=bin_cfg,
            )
    print(f"raster {tuple(raster.shape)} built")

    cfg = DredgeConfig(
        motion_dims=tuple(axes),
        bin_um=args.bin_um,
        bin_s=args.bin_s,
        window=WindowConfig(rigid=args.rigid, win_step_um=args.win_step_um, win_scale_um=args.win_scale_um),
        xcorr=XcorrConfig(max_disp_um=args.max_disp_um, batch_size=args.time_batch),
        disp=DisplacementConfig(mode=args.disp, temperature=args.temperature),
        thresh=ThresholdConfig(mode="sigmoid", mincorr=args.mincorr, slope=50.0),
    )
    dredge = DiffDredge(cfg).to(device)

    with torch.no_grad():
        P, extras = dredge(
            raster=raster, spatial_centers=centers, return_extras=True,
        )

    P = P.detach().cpu().numpy()                       # (T, n_dims) or (B, T, n_dims)
    time_s = (np.arange(n_time) + 0.5) * args.bin_s
    win_centers = extras["window_centers"].detach().cpu().numpy()

    np.save(session / f"motion_trace{args.tag}.npy", P)
    np.save(session / f"motion_time_s{args.tag}.npy", time_s)
    np.save(session / f"motion_window_centers_um{args.tag}.npy", win_centers)
    print(f"saved motion_trace.npy {P.shape}  (dims={extras['motion_dims']}, "
          f"max_disp={extras['max_disp_um']} um)")
    flat = P.reshape(-1, P.shape[-1])
    print(f"motion range per dim (um): "
          f"{np.round(flat.max(0) - flat.min(0), 2)}  std {np.round(flat.std(0), 2)}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=str, required=True, help="output directory (+ centroid fallback)")
    p.add_argument("--loc-npy", type=str, default="", help="real localizations .npy (cols include depth & amplitude)")
    p.add_argument("--times-npy", type=str, default="", help="spike_times .npy matching --loc-npy")
    p.add_argument("--depth-col", type=int, default=1, help="column of localizations giving depth (y)")
    p.add_argument("--x-col", type=int, default=0, help="column giving lateral x")
    p.add_argument("--amp-col", type=int, default=3, help="column giving amplitude (alpha)")
    p.add_argument("--tag", type=str, default="", help="suffix for saved motion_trace files")
    p.add_argument("--fs", type=float, default=30000.0, help="sampling rate of spike_times (Hz)")
    p.add_argument("--bin-s", type=float, default=1.0)
    p.add_argument("--bin-um", type=float, default=5.0)
    p.add_argument("--bandwidth-um", type=float, default=5.0)
    p.add_argument("--max-disp-um", type=float, default=100.0)
    p.add_argument("--mincorr", type=float, default=0.1)
    p.add_argument("--motion-dims", type=str, default="y", choices=["y", "xy"])
    p.add_argument("--disp", type=str, default="hard", choices=["hard", "soft", "parabolic"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--rigid", action="store_true", default=True)
    p.add_argument("--nonrigid", dest="rigid", action="store_false")
    p.add_argument("--win-step-um", type=float, default=400.0)
    p.add_argument("--win-scale-um", type=float, default=450.0)
    p.add_argument("--time-batch", type=int, default=512)
    p.add_argument("--spike-chunk", type=int, default=100_000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
