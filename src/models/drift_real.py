"""Estimate rigid drift from all localized spikes in a real recording.

The optimizer sees only the recording's time-depth-amplitude raster. The
DREDge motion file is loaded after optimization for evaluation and is never
used by the loss or initialization.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from drift_population import (
    DriftStageConfig,
    PopulationRegistrationConfig,
    evaluate_rigid_drift,
    gaussian_smooth_depth,
    optimize_rigid_drift_stage,
)


FS = 30_000.0


def bin_real_spikes(times_sec, depths, amplitudes, time_bin_sec, spatial_min,
                    spatial_max, spatial_bin_um):
    num_time_bins = int(math.floor(times_sec.max() / time_bin_sec)) + 1
    num_depth_bins = int(math.ceil((spatial_max - spatial_min) / spatial_bin_um))
    time_ids = np.floor(times_sec / time_bin_sec).astype(np.int64)
    depth_position = (depths - spatial_min) / spatial_bin_um
    left = np.floor(depth_position).astype(np.int64)
    fraction = depth_position - left

    valid = (
        (time_ids >= 0) & (time_ids < num_time_bins)
        & (left >= 0) & (left < num_depth_bins)
        & np.isfinite(amplitudes)
    )
    time_ids = time_ids[valid]
    left = left[valid]
    fraction = fraction[valid]
    weights = np.log1p(np.maximum(amplitudes[valid], 0.0))

    flat_size = num_time_bins * num_depth_bins
    flat_left = time_ids * num_depth_bins + left
    raster = np.bincount(
        flat_left, weights=weights * (1.0 - fraction),
        minlength=flat_size).astype(np.float32)

    right_valid = left + 1 < num_depth_bins
    flat_right = time_ids[right_valid] * num_depth_bins + left[right_valid] + 1
    raster += np.bincount(
        flat_right, weights=weights[right_valid] * fraction[right_valid],
        minlength=flat_size).astype(np.float32)
    raster = raster.reshape(num_time_bins, num_depth_bins).T.copy()
    counts = np.bincount(time_ids, minlength=num_time_bins).astype(np.int64)
    centers = (np.arange(num_time_bins, dtype=np.float64) + 0.5) * time_bin_sec
    return raster, counts, centers, int(valid.sum())


def trace_metrics(estimate, reference):
    estimate = estimate - estimate[0]
    reference = reference - reference[0]
    valid = np.isfinite(estimate) & np.isfinite(reference)
    estimate = estimate[valid]
    reference = reference[valid]
    if estimate.size < 3:
        return {"correlation": float("nan"), "rmse_um": float("nan"),
                "mae_um": float("nan")}
    return {
        "correlation": float(np.corrcoef(estimate, reference)[0, 1]),
        "rmse_um": float(np.sqrt(np.mean((estimate - reference) ** 2))),
        "mae_um": float(np.mean(np.abs(estimate - reference))),
    }


def load_dredge_reference(path, target_times):
    motion = np.load(path)
    displacement = np.asarray(motion["disp"], dtype=np.float64)
    anchors = np.asarray(motion["t_anchors"], dtype=np.float64)
    depth_windows = np.asarray(motion["y_anchors"], dtype=np.float64)
    interpolated = np.column_stack([
        np.interp(target_times, anchors, displacement[:, column])
        for column in range(displacement.shape[1])
    ])
    interpolated -= interpolated[0:1]
    return {
        "windows": interpolated,
        "median": np.nanmedian(interpolated, axis=1),
        "mean": np.nanmean(interpolated, axis=1),
        "depth_windows_um": depth_windows,
    }


def initialization_trace(name, num_time_bins, amplitude, device):
    if name == "zero":
        return torch.zeros(num_time_bins, device=device)
    if name == "linear":
        trace = torch.linspace(
            -amplitude, amplitude, num_time_bins, device=device)
        return trace - trace[0]
    raise ValueError(f"unknown initialization: {name}")


def repository_state():
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
        dirty = None
    return {"revision": revision, "dirty": dirty}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-path", type=Path,
        default=Path("/scratch/ap7151/sln-v2/runs/dataset1_p1"))
    parser.add_argument(
        "--dredge-motion", type=Path,
        default=Path(
            "/scratch/am15577/UnitMatch/Post_Neurips/mp_ladder/results/"
            "Steinmetz/dataset1_p1/mp_dredge/motion.npz"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/scratch/ap7151/sln-v2/outputs/drift_real_p1"))
    parser.add_argument("--time-bin-sec", type=float, default=2.0)
    parser.add_argument("--spatial-min", type=float, default=-200.0)
    parser.add_argument("--spatial-max", type=float, default=4100.0)
    parser.add_argument("--spatial-bin-um", type=float, default=2.0)
    parser.add_argument(
        "--lag-sec", nargs="+", type=float,
        default=[2.0, 10.0, 30.0, 60.0, 120.0, 300.0])
    parser.add_argument("--coarse-sigma-um", type=float, default=8.0)
    parser.add_argument("--fine-sigma-um", type=float, default=4.0)
    parser.add_argument("--coarse-knots", type=int, default=32)
    parser.add_argument("--fine-knots", type=int, default=128)
    parser.add_argument("--coarse-steps", type=int, default=200)
    parser.add_argument("--fine-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.25)
    parser.add_argument("--coarse-smoothness", type=float, default=0.002)
    parser.add_argument("--fine-smoothness", type=float, default=0.0002)
    parser.add_argument("--template-weight", type=float, default=0.5)
    parser.add_argument("--max-abs-knot-um", type=float, default=100.0)
    parser.add_argument(
        "--initializations", nargs="+", choices=["zero", "linear"],
        default=["zero", "linear"])
    parser.add_argument("--initialization-amplitude", type=float, default=5.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; submit src/models/drift_real_p1.sbatch")
    if not args.dredge_motion.exists():
        raise SystemExit(f"DREDge reference not found: {args.dredge_motion}")

    session = args.session_path
    spike_times = np.load(session / "spike_times.npy", mmap_mode="r")
    locations = np.load(
        session / "monopolar_true" / "localizations.npy", mmap_mode="r")
    in_bounds = np.load(
        session / "monopolar_true" / "in_bounds.npy", mmap_mode="r")
    times_sec = np.asarray(spike_times, dtype=np.float64) / FS
    depths = np.asarray(locations[:, 1], dtype=np.float64)
    amplitudes = np.asarray(locations[:, 3], dtype=np.float64)
    valid = (
        np.asarray(in_bounds, dtype=bool) & np.isfinite(depths)
        & np.isfinite(amplitudes)
    )
    raster_np, counts, time_centers, num_used = bin_real_spikes(
        times_sec[valid], depths[valid], amplitudes[valid],
        args.time_bin_sec, args.spatial_min, args.spatial_max,
        args.spatial_bin_um)

    device = torch.device("cuda")
    raw_raster = torch.from_numpy(raster_np).to(device)
    coarse_raster = gaussian_smooth_depth(
        raw_raster, args.coarse_sigma_um / args.spatial_bin_um)
    fine_raster = gaussian_smooth_depth(
        raw_raster, args.fine_sigma_um / args.spatial_bin_um)
    lag_bins = sorted({
        max(int(round(lag / args.time_bin_sec)), 1)
        for lag in args.lag_sec
        if lag < time_centers[-1]
    })
    registration_config = PopulationRegistrationConfig(
        spatial_bin_um=args.spatial_bin_um,
        lag_bins=tuple(lag_bins),
        template_weight=args.template_weight,
        max_abs_knot_um=args.max_abs_knot_um,
    )
    coarse_stage = DriftStageConfig(
        num_knots=args.coarse_knots,
        steps=args.coarse_steps,
        learning_rate=args.learning_rate,
        smoothness=args.coarse_smoothness,
    )
    fine_stage = DriftStageConfig(
        num_knots=args.fine_knots,
        steps=args.fine_steps,
        learning_rate=args.learning_rate,
        smoothness=args.fine_smoothness,
    )

    traces = {}
    histories = {}
    population_evaluations = {}
    for initialization in args.initializations:
        initial = initialization_trace(
            initialization, raw_raster.shape[1],
            args.initialization_amplitude, device)
        initial_evaluation = evaluate_rigid_drift(
            fine_raster, initial, registration_config)
        coarse_trace, coarse_history = optimize_rigid_drift_stage(
            initial, coarse_raster, registration_config, coarse_stage)
        final_trace, fine_history = optimize_rigid_drift_stage(
            coarse_trace, fine_raster, registration_config, fine_stage)
        final_evaluation = evaluate_rigid_drift(
            fine_raster, final_trace, registration_config)
        trace = final_trace.cpu().numpy().astype(np.float64)
        trace -= trace[0]
        traces[initialization] = trace
        histories[initialization] = {
            "coarse": coarse_history,
            "fine": fine_history,
        }
        population_evaluations[initialization] = {
            "initial": initial_evaluation,
            "final": final_evaluation,
        }

    reference = load_dredge_reference(args.dredge_motion, time_centers)
    results = []
    for initialization in args.initializations:
        trace = traces[initialization]
        per_window = [
            trace_metrics(trace, reference["windows"][:, column])
            for column in range(reference["windows"].shape[1])
        ]
        results.append({
            "initialization": initialization,
            "drift_min_um": float(trace.min()),
            "drift_max_um": float(trace.max()),
            "drift_span_um": float(np.ptp(trace)),
            "dredge_median": trace_metrics(trace, reference["median"]),
            "dredge_mean": trace_metrics(trace, reference["mean"]),
            "dredge_windows": per_window,
            "population_objective": population_evaluations[initialization],
        })

    pairwise_initialization = None
    if len(traces) > 1:
        names = list(traces)
        pairwise_initialization = {
            "first": names[0],
            "second": names[1],
            **trace_metrics(traces[names[0]], traces[names[1]]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "traces.npz",
        times_sec=time_centers,
        spike_counts=counts,
        dredge_median=reference["median"],
        dredge_mean=reference["mean"],
        dredge_windows=reference["windows"],
        dredge_depth_windows_um=reference["depth_windows_um"],
        **{f"learned_{name}": trace for name, trace in traces.items()},
    )
    for name, history in histories.items():
        np.savez(
            args.output_dir / f"history_{name}.npz",
            **{
                f"{stage}_{metric}": np.asarray(values)
                for stage, stage_history in history.items()
                for metric, values in stage_history.items()
            })
    summary = {
        "method": "fixed_localization_population_stationarity_v1",
        "method_scope": "rigid drift from fixed localizations; no waveform encoder",
        "session": str(session),
        "input_files": {
            "spike_times": str(session / "spike_times.npy"),
            "localizations": str(
                session / "monopolar_true" / "localizations.npy"),
            "in_bounds": str(
                session / "monopolar_true" / "in_bounds.npy"),
        },
        "localization_source": "SpikeInterface monopolar triangulation",
        "raster_weight": "log1p(monopolar_alpha)",
        "dredge_motion": str(args.dredge_motion),
        "dredge_role": "evaluation_only_loaded_after_optimization",
        "repository": repository_state(),
        "num_input_spikes": int(len(spike_times)),
        "num_used_spikes": num_used,
        "num_time_bins": int(raster_np.shape[1]),
        "num_depth_bins": int(raster_np.shape[0]),
        "spikes_per_time_bin": {
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "maximum": int(counts.max()),
        },
        "lag_bins": lag_bins,
        "resolved_config": vars(args),
        "results": results,
        "initialization_agreement": pairwise_initialization,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    for result in results:
        print("[real-drift] " + json.dumps(result, sort_keys=True), flush=True)
    print(f"[real-drift] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
