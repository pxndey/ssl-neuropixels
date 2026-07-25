"""Matched-spike depth-time raster across localization and drift treatments.

N stacked panels — depth z (µm) vs time (s), colored by log amplitude — the
standard drift-map view that shows whether motion correction straightens the
wavy depth bands. Style mirrors sln/src/plots/raster.py (scatter, viridis).

With ``--joint-run``, panels use that run's held-out split:
  1. raw SpikeInterface monopolar depth
  2. DREDge-corrected monopolar depth
  3. Approach 1 corrected transformer depth

CPU-only. Run inline via Singularity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/scratch/ap7151/sln-v2")
RAW_EXTRA_MOTION = Path("/scratch/ap7151/RAW_DATA/extra-motion")
FS = 30_000.0
JOINT_SPLITS = {
    "all-heldout": (1, 2),
    "train": (0,),
    "spike-heldout": (1,),
    "time-heldout": (2,),
}


def _load_manipulation_keyframes(session_path, raw_data_dir=None):
    session_name = Path(session_path).name
    probe_name = session_name.rsplit("_", 1)[-1]
    raw_dir = (
        Path(raw_data_dir)
        if raw_data_dir is not None
        else RAW_EXTRA_MOTION / session_name)
    timestamp_path = raw_dir / f"manip.timestamps_{probe_name}.npy"
    position_path = raw_dir / "manip.positions.npy"
    if not timestamp_path.exists() or not position_path.exists():
        return None
    times = np.asarray(np.load(timestamp_path), dtype=np.float64).reshape(-1)
    positions = np.asarray(
        np.load(position_path), dtype=np.float64).reshape(-1)
    if times.shape != positions.shape:
        raise ValueError(
            "manipulation timestamp and position keyframes differ in length")
    if (
            not np.isfinite(times).all()
            or not np.isfinite(positions).all()
            or np.any(np.diff(times) < 0)):
        raise ValueError("invalid manipulation position keyframes")
    return {
        "times_sec": times,
        "positions_um": positions - positions[0],
        "timestamp_path": timestamp_path,
        "position_path": position_path,
    }


def _load_raw(session_path, test_indices=None):
    """SI ground-truth monopolar localization. Columns are (x, y, z, alpha)
    where y is depth along the probe (col 1) and z is lateral offset (col 2).
    The model predicts z=depth (centroid_z + z_local), so we plot col 1 here
    to match the model's depth axis. If test_indices is given, only return
    those spikes so all panels show the same set."""
    p = Path(session_path)
    times = np.load(p / "spike_times.npy").astype(np.float64) / FS
    loc = np.load(p / "monopolar_true" / "localizations.npy")
    in_bounds = np.load(p / "monopolar_true" / "in_bounds.npy")
    depth = loc[:, 1].astype(np.float32)
    alpha = loc[:, 3].astype(np.float64)
    valid = in_bounds & np.isfinite(depth) & np.isfinite(alpha)
    if test_indices is not None:
        mask = np.zeros(len(times), dtype=bool)
        mask[test_indices] = True
        valid &= mask
    return times[valid], depth[valid], alpha[valid], np.where(valid)[0]


def _naive_drift(times, z, bin_s=10.0):
    t_max = times.max()
    edges = np.arange(0, t_max + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(times, edges) - 1, 0, len(centers) - 1)
    drift = np.zeros(len(centers))
    for i in range(len(centers)):
        m = bin_idx == i
        if m.sum() > 0:
            drift[i] = z[m].mean()
    drift = np.interp(times, centers, drift)
    return drift - drift[0]


def _load_ckpt_predictions(session_path):
    p = Path(session_path) / "inference_drift" / "predictions.npz"
    if p.exists():
        d = np.load(p)
        return d["times_sec"], d["z"], d["alpha"], d["drift"]
    return None, None, None, None


def _load_dredge_ap(session_path):
    """Load DREDge AP drift from <session>/dredge_ap/ and apply to raw depth."""
    p = Path(session_path) / "dredge_ap"
    if not (p / "drift_displacement.npy").exists():
        return None, None, None
    t_raw, z_raw, a_raw = _load_raw(session_path)
    times = np.load(p / "drift_times_s.npy")
    disp = np.load(p / "drift_displacement.npy")
    drift_interp = np.interp(t_raw, times, disp)
    z_corr = (z_raw - drift_interp).astype(np.float32)
    return t_raw, z_corr, a_raw


def _dredge_ref_path(session_path):
    """Per-session am15577 DREDge reference path."""
    sess = Path(session_path).name
    return Path(f"/scratch/am15577/UnitMatch/Post_Neurips/mp_ladder/results/Steinmetz/{sess}/mp_dredge")


def _load_dredge_ref(session_path, test_indices=None):
    """Apply pre-computed DREDge drift (am15577 reference) to OUR spikes.
    Reference path is session-aware (dataset1_p1 vs dataset1_p2 etc.).
    If test_indices is given, only return those spikes.
    """
    p = _dredge_ref_path(session_path)
    if not (p / "motion.npz").exists():
        return None, None, None, None
    m = np.load(p / "motion.npz")
    disp, t_anc, y_anc = m["disp"], m["t_anchors"], m["y_anchors"]

    t_raw, z_raw, a_raw, _ = _load_raw(session_path, test_indices=test_indices)
    win_idx = np.argmin(np.abs(z_raw[:, None] - y_anc[None, :]), axis=1)
    drift = np.zeros(len(t_raw), dtype=np.float32)
    for w in range(len(y_anc)):
        mask = win_idx == w
        if mask.any():
            drift[mask] = np.interp(t_raw[mask], t_anc, disp[:, w])
    z_corr = (z_raw - drift).astype(np.float32)
    return t_raw, z_raw, z_corr, a_raw


def _apply_dredge_ref(session_path, times, depths):
    p = _dredge_ref_path(session_path)
    if not (p / "motion.npz").exists():
        return None, None
    motion = np.load(p / "motion.npz")
    displacement = motion["disp"]
    time_anchors = motion["t_anchors"]
    depth_anchors = motion["y_anchors"]
    window_ids = np.argmin(
        np.abs(depths[:, None] - depth_anchors[None, :]), axis=1)
    drift = np.zeros(len(times), dtype=np.float32)
    for window in range(len(depth_anchors)):
        selected = window_ids == window
        if selected.any():
            drift[selected] = np.interp(
                times[selected], time_anchors,
                displacement[:, window])
    return (depths - drift).astype(np.float32), drift


def _load_joint_run(run_path, split_name):
    run_path = Path(run_path)
    cache = np.load(run_path / "localization_cache.npz", mmap_mode="r")
    traces = np.load(run_path / "traces.npz")
    with (run_path / "summary.json").open() as handle:
        summary = json.load(handle)
    split_codes = JOINT_SPLITS[split_name]
    positions = np.flatnonzero(np.isin(cache["split"], split_codes))
    spike_indices = np.asarray(cache["spike_index"][positions], dtype=np.int64)
    times = np.asarray(cache["times_sec"][positions], dtype=np.float64)
    z_probe = np.asarray(cache["z_probe"][positions], dtype=np.float32)
    raw_amplitude = np.asarray(
        cache["raw_peak_amplitude"][positions], dtype=np.float64)
    trace = np.asarray(traces["learned_drift"], dtype=np.float64)
    time_bin_sec = float(summary["resolved_config"]["time_bin_sec"])
    drift = np.interp(
        times / time_bin_sec,
        np.arange(trace.size, dtype=np.float64),
        trace,
    )
    return {
        "run_path": run_path,
        "split_name": split_name,
        "drift_fit_split": summary["resolved_config"].get(
            "drift_fit_split", "train"),
        "split_code": np.asarray(cache["split"][positions], dtype=np.int8),
        "spike_indices": spike_indices,
        "times_sec": times,
        "z_probe": z_probe,
        "z_brain": (z_probe - drift).astype(np.float32),
        "raw_amplitude": raw_amplitude,
        "drift_per_spike": drift.astype(np.float32),
        "drift_trace": trace,
        "drift_times": (
            np.arange(trace.size, dtype=np.float64) * time_bin_sec),
    }


def _match_joint_to_valid_raw(joint, raw_indices):
    order = np.argsort(joint["spike_indices"])
    sorted_indices = joint["spike_indices"][order]
    positions = np.searchsorted(sorted_indices, raw_indices)
    if (positions >= len(sorted_indices)).any():
        raise ValueError("raw spike index is absent from the joint split")
    matched = order[positions]
    if not np.array_equal(joint["spike_indices"][matched], raw_indices):
        raise ValueError("joint cache and raw localization indices do not match")
    return {
        key: value[matched]
        for key, value in joint.items()
        if isinstance(value, np.ndarray) and value.shape[:1] == (
            len(joint["spike_indices"]),)
    }


def _load_dredge_drift_line(session_path):
    """Return (t_anchors, mean_disp, disp_min, disp_max) for the DREDge reference.
    mean across nonrigid windows; min/max for shading.
    """
    p = _dredge_ref_path(session_path)
    if not (p / "motion.npz").exists():
        return None
    m = np.load(p / "motion.npz")
    disp, t_anc = m["disp"], m["t_anchors"]
    disp = disp - disp[0:1]
    return t_anc, disp.mean(axis=1), disp.min(axis=1), disp.max(axis=1)


def _model_drift_line(times_sec, drift_per_spike, bin_s=1.0):
    """Bin per-spike model D(t) into 1s bins, return (bin_centers, mean_drift)."""
    t_max = float(times_sec.max())
    edges = np.arange(0, t_max + bin_s, bin_s)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(times_sec, edges) - 1, 0, len(centers) - 1)
    out = np.full(len(centers), np.nan, dtype=np.float64)
    for i in range(len(centers)):
        m = bin_idx == i
        if m.sum() > 0:
            out[i] = drift_per_spike[m].mean()
    # forward-fill NaNs
    valid = np.where(np.isfinite(out))[0]
    if len(valid) > 0:
        out = np.interp(centers, centers[valid], out[valid])
    return centers, out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--session", default=str(REPO / "runs" / "dataset1_p1"))
    ap.add_argument(
        "--raw-data-dir", type=Path, default=None,
        help=(
            "raw session directory containing manip.timestamps_<probe>.npy "
            "and manip.positions.npy; inferred from --session by default"))
    ap.add_argument("--out", default=str(REPO / "plots" / "drift_raster.png"))
    ap.add_argument("--n-plot", type=int, default=900_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--joint-run", type=Path, default=None,
        help="Approach 1 output directory containing cache, trace, and summary")
    ap.add_argument(
        "--joint-split", choices=sorted(JOINT_SPLITS),
        default="all-heldout")
    ap.add_argument(
        "--export", type=Path, default=None,
        help="optional matched localization export (.npz)")
    ap.add_argument("--grayscale", action="store_true",
                    help="render amplitude in greyscale (gray_r) instead of viridis")
    ap.add_argument("--bin-s", type=float, default=10.0,
                   help="time bin (s) for the naive drift estimate")
    ap.add_argument("--t-range", type=float, nargs=2, default=None,
                   metavar=("T_LO", "T_HI"),
                   help="time range to plot (s), e.g. --t-range 750 1500")
    ap.add_argument("--y-range", type=float, nargs=2, default=None,
                   metavar=("Y_LO", "Y_HI"),
                   help="depth range to plot (µm), e.g. --y-range 250 1000")
    args = ap.parse_args()

    sess = Path(args.session).name
    manipulation = _load_manipulation_keyframes(
        args.session, args.raw_data_dir)
    if manipulation is not None:
        print(
            f"[manipulation] {len(manipulation['times_sec'])} keyframes, "
            f"position [{manipulation['positions_um'].min():.1f}, "
            f"{manipulation['positions_um'].max():.1f}] µm, "
            f"timestamps={manipulation['timestamp_path']}",
            flush=True)
    else:
        print("[warn] no manipulation ground truth found", flush=True)
    joint = (
        _load_joint_run(args.joint_run, args.joint_split)
        if args.joint_run is not None else None)
    if joint is not None:
        test_indices = joint["spike_indices"]
        print(
            f"[match] using {len(test_indices)} Approach 1 "
            f"{args.joint_split} spikes for all panels",
            flush=True)
    else:
        test_idx_path = Path(args.session) / "inference" / "test_indices.npy"
        test_indices = (
            np.load(test_idx_path) if test_idx_path.exists() else None)
        if test_indices is not None:
            print(
                f"[match] using {len(test_indices)} legacy test-split spikes "
                "for all panels",
                flush=True)

    t_raw, z_raw, a_raw, raw_indices = _load_raw(
        args.session, test_indices=test_indices)
    N_raw = t_raw.shape[0]
    print(f"[data] {N_raw} valid raw spikes, t_max={t_raw.max():.1f}s, "
          f"z [{z_raw.min():.0f}, {z_raw.max():.0f}] µm", flush=True)

    t_ck = z_ck = a_ck = d_ck = None
    ckpt_panels = []
    joint_panel = None
    joint_drift_line = None
    if joint is not None:
        matched_joint = _match_joint_to_valid_raw(joint, raw_indices)
        if not np.allclose(
                matched_joint["times_sec"], t_raw, rtol=0, atol=1e-4):
            raise ValueError("joint and raw spike times do not match")
        a_raw = matched_joint["raw_amplitude"]
        joint_panel = (
            "Approach 1 corrected transformer depth",
            t_raw,
            matched_joint["z_brain"],
            a_raw,
        )
        joint_drift_line = (
            joint["drift_times"], joint["drift_trace"])
        print(
            f"[joint] {len(t_raw)} matched spikes, "
            f"drift [{joint['drift_trace'].min():.1f}, "
            f"{joint['drift_trace'].max():.1f}] µm, "
            f"z_brain [{matched_joint['z_brain'].min():.0f}, "
            f"{matched_joint['z_brain'].max():.0f}] µm",
            flush=True)
    else:
        t_ck, z_ck, a_ck, d_ck = _load_ckpt_predictions(args.session)
    if t_ck is not None:
        z_corr = (z_ck - d_ck).astype(np.float32)
        ckpt_panels.append(("drift model (z - D(t))", t_ck, z_corr, a_ck))
        print(f"[ckpt] {len(t_ck)} spikes, drift [{d_ck.min():.1f}, {d_ck.max():.1f}] µm, "
              f"z_corr [{z_corr.min():.0f}, {z_corr.max():.0f}] µm", flush=True)
    elif joint is None:
        print("[warn] no inference_drift/predictions.npz found; skipping model panels",
              flush=True)

    dredge_panels = []
    if joint is not None:
        d_corr, dredge_drift = _apply_dredge_ref(
            args.session, t_raw, z_raw)
        if d_corr is not None:
            dredge_panels.append((
                "DREDge-corrected monopolar depth",
                t_raw,
                d_corr,
                a_raw,
            ))
            print(
                f"[dredge] {len(t_raw)} matched spikes, corrected "
                f"[{d_corr.min():.0f}, {d_corr.max():.0f}] µm",
                flush=True)
    else:
        dref = _load_dredge_ref(
            args.session, test_indices=test_indices)
        d_corr = dredge_drift = None
        if dref[0] is not None:
            t_d, d_raw, d_corr, a_d = dref
            dredge_panels.append((
                "DREDge corrected (motion.npz)",
                t_d,
                d_corr,
                a_d,
            ))
            print(
                f"[dredge] {len(t_d)} spikes, corrected "
                f"[{d_corr.min():.0f}, {d_corr.max():.0f}] µm",
                flush=True)
    if not dredge_panels:
        d_corr = dredge_drift = None
        print("[warn] no DREDge reference found at am15577 path", flush=True)

    if joint is not None:
        panels = [
            ("raw monopolar depth", t_raw, z_raw, a_raw),
        ] + dredge_panels + [joint_panel]
    else:
        panels = [
            ("raw recording (monopolar)", t_raw, z_raw, a_raw),
        ] + ckpt_panels + dredge_panels

    if args.export is not None and joint is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        export_values = {
            "spike_index": raw_indices,
            "times_sec": t_raw,
            "raw_peak_amplitude": a_raw,
            "monopolar_depth": z_raw,
            "joint_probe_depth": matched_joint["z_probe"],
            "joint_brain_depth": matched_joint["z_brain"],
            "joint_drift": matched_joint["drift_per_spike"],
            "split_code": matched_joint["split_code"],
        }
        if d_corr is not None:
            export_values["dredge_brain_depth"] = d_corr
            export_values["dredge_drift"] = dredge_drift
        if manipulation is not None:
            export_values["manipulation_times_sec"] = (
                manipulation["times_sec"])
            export_values["manipulation_positions_um"] = (
                manipulation["positions_um"])
            export_values["manipulation_interpolation"] = np.asarray(
                "linear_between_keyframes")
        np.savez(args.export, **export_values)
        print(f"[export] wrote {args.export}", flush=True)

    # compute crop mask once on the raw panel, apply to all so the same
    # spikes survive in every panel
    crop_mask = np.ones(len(panels[0][1]), dtype=bool)
    if args.t_range is not None:
        crop_mask &= (panels[0][1] >= args.t_range[0]) & (panels[0][1] < args.t_range[1])
    if args.y_range is not None:
        crop_mask &= (panels[0][2] >= args.y_range[0]) & (panels[0][2] < args.y_range[1])
    if not crop_mask.all():
        panels = [(name, t[crop_mask], z[crop_mask], a[crop_mask])
                  for name, t, z, a in panels]
        if args.t_range is not None:
            print(f"[crop] time range [{args.t_range[0]}, {args.t_range[1]}) s",
                  flush=True)
        if args.y_range is not None:
            print(f"[crop] depth range [{args.y_range[0]}, {args.y_range[1]}) µm "
                  f"(on raw depth)", flush=True)

    n_raster = len(panels)
    rng = np.random.default_rng(args.seed)

    # all panels now have the same spike count — sample once, reuse everywhere
    N = panels[0][1].shape[0]
    idx = np.arange(N) if args.n_plot >= N else np.sort(
        rng.choice(N, args.n_plot, replace=False))
    idx = idx[np.argsort(panels[0][3][idx])]

    # D(t) line panel is appended below the rasters if we have either source.
    drift_lines = []  # list of (label, t, d, optional (t_lo, d_lo, t_hi, d_hi) for shading)
    if joint_drift_line is not None:
        drift_lines.append((
            "Approach 1 D(t)",
            joint_drift_line[0],
            joint_drift_line[1],
            None,
        ))
    elif t_ck is not None and d_ck is not None:
        ct, cd = _model_drift_line(t_ck, d_ck, bin_s=args.bin_s)
        drift_lines.append(("model D(t)", ct, cd, None))
    dref_line = _load_dredge_drift_line(args.session)
    if dref_line is not None:
        t_anc, d_mean, d_min, d_max = dref_line
        shade = (t_anc, d_min, t_anc, d_max)
        drift_lines.append(("DREDge D(t) (mean over windows)", t_anc, d_mean, shade))
    has_line_panel = len(drift_lines) > 0 or manipulation is not None

    n_rows = n_raster + (1 if has_line_panel else 0)
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(14, 3.5 * n_raster + 2.0),
                             sharex=False, constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    raster_axes = axes[:n_raster]
    line_ax = axes[-1] if has_line_panel else None

    cmap = "gray_r" if args.grayscale else "magma"
    sc = None
    for ax, (name, t, z, alpha) in zip(raster_axes, panels):
        ti = t[idx]
        c = np.log10(np.clip(alpha[idx], 1e-6, None))
        cmin, cmax = np.percentile(c, [1, 99])
        sc = ax.scatter(ti, z[idx], c=c, cmap=cmap, vmin=cmin, vmax=cmax,
                        s=2.0, alpha=0.3, linewidths=0, rasterized=True)
        ax.set_title(name, fontsize=13, loc="left")
        ax.set_ylabel("depth y [µm]", fontsize=12)
        ax.set_facecolor("white")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    raster_axes[-1].set_xlabel("time [s]", fontsize=12)

    cbar = fig.colorbar(sc, ax=raster_axes, orientation="vertical", shrink=0.5, pad=0.01)
    cbar.set_label("log₁₀ amplitude", fontsize=11)

    if has_line_panel:
        colors = {
            "model D(t)": "#1f77b4",
            "Approach 1 D(t)": "#1f77b4",
            "DREDge D(t) (mean over windows)": "#d62728",
        }
        for label, tl, dl, shade in drift_lines:
            line_color = colors.get(label, "k")
            line_ax.plot(tl, dl, color=line_color, lw=1.2, label=label)
            if shade is not None:
                t_lo, d_lo, t_hi, d_hi = shade
                line_ax.fill_between(t_lo, d_lo, d_hi, color=line_color, alpha=0.15,
                                     linewidths=0, label=f"{label} range")
        if manipulation is not None:
            manip_times = manipulation["times_sec"]
            manip_positions = manipulation["positions_um"]
            trace_end = max(float(panel[1].max()) for panel in panels)
            if manip_times[-1] < trace_end:
                manip_times = np.append(manip_times, trace_end)
                manip_positions = np.append(
                    manip_positions, manip_positions[-1])
            line_ax.plot(
                manip_times, manip_positions, color="black", lw=1.2,
                ls="--", marker="o", markersize=2.5,
                label=(
                    "manipulator position keyframes "
                    "(linear interpolation)"))
        line_ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
        line_ax.set_xlabel("time [s]", fontsize=12)
        line_ax.set_ylabel("D(t) [µm]", fontsize=12)
        line_ax.set_title("drift trace", fontsize=13, loc="left")
        line_ax.legend(fontsize=10, loc="best", frameon=False)
        for sp in ("top", "right"):
            line_ax.spines[sp].set_visible(False)

    split_text = ""
    if joint is not None:
        split_text = (
            f", localizer={args.joint_split}, "
            f"drift-fit={joint['drift_fit_split']}")
    fig.suptitle(
        f"{sess}: matched spike depth raster{split_text} "
        f"({min(args.n_plot, N_raw):,} spikes)",
        fontsize=14)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=800, format="png")
    fig.savefig(args.out.replace(".png", ".svg"), format="svg")
    plt.close(fig)
    print(f"[{sess}] wrote raster -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
