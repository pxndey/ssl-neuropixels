"""Train the alternating dense-population joint drift localizer from PLAN_3."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drift_joint import JointDriftLocalizer
from drift_population import (
    PopulationRegistrationConfig,
    RigidSplineDrift,
    drift_curvature_loss,
    gaussian_smooth_depth,
    linear_splat_depth,
    normalized_population_template,
    population_feedback_loss,
    population_stationarity_loss,
    shift_raster_to_brain,
)
from drift_real import load_dredge_reference, trace_metrics
from FAIL_drift_train import SpikeDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLING_RATE_HZ = 30_000.0
SPLIT_TRAIN = np.int8(0)
SPLIT_SPIKE_HELDOUT = np.int8(1)
SPLIT_TIME_HELDOUT = np.int8(2)


@dataclass
class LocalizationCache:
    spike_index: np.ndarray
    times_sec: np.ndarray
    z_probe: np.ndarray
    raw_peak_amplitude: np.ndarray
    split: np.ndarray


def repository_state():
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True,
            capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
        dirty = None
    return {"revision": revision, "dirty": dirty}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_sha256(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_count(parameters):
    return sum(parameter.numel() for parameter in parameters)


def gradient_norm(parameters):
    squared = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().square().sum()
        squared = value if squared is None else squared + value
    return 0.0 if squared is None else float(squared.sqrt().item())


def gradients_norm(gradients):
    values = [gradient.detach().square().sum()
              for gradient in gradients if gradient is not None]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt().item())


def update_norm(before, parameters):
    values = [
        (parameter.detach() - previous).square().sum()
        for previous, parameter in zip(before, parameters)
    ]
    return 0.0 if not values else float(torch.stack(values).sum().sqrt().item())


def load_spike_batch(dataset, indices, device):
    indices = np.asarray(indices, dtype=np.int64)
    fixed_n = dataset.fixed_n
    used_channels = min(dataset.M, fixed_n)
    waveforms = np.zeros(
        (len(indices), fixed_n, dataset.n_samples), dtype=np.float32)
    coords = np.zeros((len(indices), fixed_n, 2), dtype=np.float32)
    raw = np.asarray(
        dataset.waveforms[indices, :used_channels, :], dtype=np.float32)
    raw_coords = np.asarray(
        dataset.local_coords[indices, :used_channels, :], dtype=np.float32)
    counts = np.minimum(
        np.asarray(dataset.neighbor_counts[indices], dtype=np.int64),
        used_channels)
    mask = np.arange(fixed_n)[None, :] < counts[:, None]
    waveforms[:, :used_channels] = raw
    coords[:, :used_channels] = raw_coords
    waveforms *= mask[:, :, None]
    coords *= mask[:, :, None]
    ptp = np.ptp(waveforms, axis=2)
    raw_amplitude = ptp.max(axis=1).astype(np.float32)
    if dataset.normalize:
        scale = np.where(raw_amplitude > 1e-6, raw_amplitude, 1.0)
        waveforms /= scale[:, None, None]
    centroids = np.asarray(dataset.centroids[indices], dtype=np.float32)
    times_sec = np.asarray(dataset.times_sec[indices], dtype=np.float32)
    return {
        "indices": indices,
        "waveforms": torch.from_numpy(waveforms).to(device),
        "coords": torch.from_numpy(coords).to(device),
        "mask": torch.from_numpy(mask).to(device),
        "centroids": torch.from_numpy(centroids).to(device),
        "times_sec": torch.from_numpy(times_sec).to(device),
        "raw_amplitude": torch.from_numpy(raw_amplitude).to(device),
    }


def make_splits(times_sec, spike_heldout_fraction, time_heldout_fraction,
                time_heldout_start_fraction, seed):
    if spike_heldout_fraction < 0 or time_heldout_fraction < 0:
        raise ValueError("held-out fractions must be nonnegative")
    if spike_heldout_fraction + time_heldout_fraction >= 1:
        raise ValueError("held-out fractions leave no training data")
    split = np.full(len(times_sec), SPLIT_TRAIN, dtype=np.int8)
    duration = float(times_sec.max())
    start = duration * time_heldout_start_fraction
    stop = min(start + duration * time_heldout_fraction, duration)
    time_heldout = (times_sec >= start) & (times_sec < stop)
    split[time_heldout] = SPLIT_TIME_HELDOUT
    candidates = np.flatnonzero(~time_heldout)
    rng = np.random.default_rng(seed)
    count = int(round(spike_heldout_fraction * len(candidates)))
    if count:
        split[rng.choice(candidates, size=count, replace=False)] = (
            SPLIT_SPIKE_HELDOUT)
    return split, {"start_sec": start, "stop_sec": stop}


def cache_localizations(model, dataset, source_indices, split, device,
                        batch_size):
    model.eval()
    z_probe = np.empty(len(source_indices), dtype=np.float32)
    raw_amplitude = np.empty(len(source_indices), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(source_indices), batch_size):
            stop = min(start + batch_size, len(source_indices))
            batch = load_spike_batch(
                dataset, source_indices[start:stop], device)
            outputs = model.encode(
                batch["waveforms"], batch["coords"], batch["mask"],
                batch["centroids"])
            z_probe[start:stop] = outputs["z_probe"].cpu().numpy()
            raw_amplitude[start:stop] = (
                batch["raw_amplitude"].cpu().numpy())
            if start == 0 or stop == len(source_indices) or (
                    start // batch_size) % 100 == 0:
                print(
                    f"[cache] localized {stop}/{len(source_indices)} spikes",
                    flush=True)
    return LocalizationCache(
        spike_index=np.asarray(source_indices, dtype=np.int64).copy(),
        times_sec=np.asarray(
            dataset.times_sec[source_indices], dtype=np.float32),
        z_probe=z_probe,
        raw_peak_amplitude=raw_amplitude,
        split=np.asarray(split, dtype=np.int8).copy(),
    )


def save_cache(cache, output_dir, checkpoint_path, checkpoint_config, model,
               time_bin_sec, cycle):
    cache_path = output_dir / "localization_cache.npz"
    temporary = output_dir / "localization_cache.tmp.npz"
    np.savez(
        temporary,
        spike_index=cache.spike_index,
        times_sec=cache.times_sec,
        z_probe=cache.z_probe,
        raw_peak_amplitude=cache.raw_peak_amplitude,
        split=cache.split,
    )
    temporary.replace(cache_path)
    time_ids = np.floor(cache.times_sec / time_bin_sec).astype(np.int64)
    num_bins = int(time_ids.max()) + 1
    counts = np.bincount(time_ids, minlength=num_bins)
    manifest = {
        "cycle": cycle,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_config": checkpoint_config,
        "encoder_state_sha256": module_sha256(model.encoder),
        "num_spikes": int(len(cache.spike_index)),
        "split_counts": {
            "train": int((cache.split == SPLIT_TRAIN).sum()),
            "spike_heldout": int(
                (cache.split == SPLIT_SPIKE_HELDOUT).sum()),
            "time_heldout": int(
                (cache.split == SPLIT_TIME_HELDOUT).sum()),
        },
        "split_encoding": {
            "0": "training",
            "1": "held-out spikes within populated time bins",
            "2": "held-out contiguous time block",
        },
        "raw_amplitude_quantiles": np.quantile(
            cache.raw_peak_amplitude,
            [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]).tolist(),
        "time_bin_occupancy": {
            "num_bins": num_bins,
            "empty_fraction": float((counts == 0).mean()),
            "minimum": int(counts.min()),
            "median": float(np.median(counts)),
            "maximum": int(counts.max()),
        },
        "repository": repository_state(),
    }
    with (output_dir / "localization_cache_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    print(f"[cache] wrote {cache_path}", flush=True)


def registration_weights(raw_amplitude, method):
    if method == "count":
        return torch.ones_like(raw_amplitude)
    if method == "raw":
        return raw_amplitude.clamp_min(0)
    if method == "log1p":
        return torch.log1p(raw_amplitude.clamp_min(0))
    raise ValueError(f"unknown registration weight: {method}")


def time_bin_ids(times_sec, time_bin_sec, num_time_bins):
    return np.clip(
        np.floor(times_sec / time_bin_sec).astype(np.int64),
        0, num_time_bins - 1)


def build_cached_raster(cache, selected, registration_times, args, device,
                        num_time_bins):
    selected = np.asarray(selected, dtype=bool)
    depths = torch.from_numpy(cache.z_probe[selected]).to(device)
    amplitudes = torch.from_numpy(
        cache.raw_peak_amplitude[selected]).to(device)
    weights = registration_weights(amplitudes, args.registration_weight)
    ids_np = time_bin_ids(
        registration_times[selected], args.time_bin_sec, num_time_bins)
    ids = torch.from_numpy(ids_np).to(device)
    raster = linear_splat_depth(
        depths, weights, ids, num_time_bins, args.spatial_min,
        args.spatial_max, args.spatial_bin_um)
    counts = torch.bincount(ids, minlength=num_time_bins)
    return raster, counts


def optimize_drift_module(drift_module, num_time_bins, raster,
                          valid_time_bins,
                          registration_config, steps, learning_rate,
                          smoothness, max_abs_um, label):
    if steps < 1:
        return [], drift_module(num_time_bins).detach()
    optimizer = torch.optim.Adam(
        drift_module.parameters(), lr=learning_rate)
    history = []
    for step in range(steps):
        before = [parameter.detach().clone()
                  for parameter in drift_module.parameters()]
        trace = drift_module(num_time_bins)
        corrected, support = shift_raster_to_brain(
            raster, trace, registration_config.spatial_bin_um,
            return_support=True)
        stationarity = population_stationarity_loss(
            corrected, registration_config.lag_bins,
            registration_config.template_weight,
            valid_time_bins=valid_time_bins, support=support)
        curvature = drift_curvature_loss(drift_module.knots)
        loss = stationarity["loss"] + smoothness * curvature
        optimizer.zero_grad()
        loss.backward()
        grad = gradient_norm(drift_module.parameters())
        optimizer.step()
        drift_module.clamp_(max_abs_um)
        update = update_norm(before, drift_module.parameters())
        values = {
            "step": step,
            "loss": float(loss.item()),
            "pair_loss": float(stationarity["pair_loss"].item()),
            "template_loss": float(stationarity["template_loss"].item()),
            "curvature": float(curvature.item()),
            "gradient_norm": grad,
            "update_norm": update,
            "drift_span_um": float(
                (drift_module(num_time_bins).max()
                 - drift_module(num_time_bins).min()).item()),
            "boundary_mass_fraction": float(
                stationarity["boundary_mass_fraction"].item()),
            "valid_pair_fraction": float(
                stationarity["valid_pair_fraction"].item()),
            "per_lag_cosine": {
                str(lag): float(value.item())
                for lag, value in stationarity["per_lag_cosine"].items()
            },
        }
        history.append(values)
        interval = max(steps // 5, 1)
        if step == 0 or step + 1 == steps or (step + 1) % interval == 0:
            print(
                f"[{label} {step + 1:04d}/{steps}] "
                f"loss={values['loss']:.6f} "
                f"span={values['drift_span_um']:.2f}um "
                f"grad={grad:.6f} update={update:.6f}",
                flush=True)
    return history, drift_module(num_time_bins).detach()


class FeedbackSampler:
    def __init__(self, cache, registration_times, time_bin_sec,
                 num_time_bins, lag_bins, min_spikes_per_bin):
        self.train_indices = np.flatnonzero(
            cache.split == SPLIT_TRAIN)
        self.registration_times = registration_times
        self.time_bin_sec = time_bin_sec
        self.num_time_bins = num_time_bins
        ids = time_bin_ids(
            registration_times[self.train_indices], time_bin_sec,
            num_time_bins)
        order = np.argsort(ids, kind="stable")
        self.sorted_indices = self.train_indices[order]
        counts = np.bincount(ids, minlength=num_time_bins)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        populated = counts >= min_spikes_per_bin
        pairs = []
        for lag in lag_bins:
            starts = np.flatnonzero(
                populated[:-lag] & populated[lag:])
            pairs.extend((int(start), int(start + lag)) for start in starts)
        if not pairs:
            raise ValueError("no populated time-bin pairs for feedback")
        self.pairs = np.asarray(pairs, dtype=np.int64)

    def sample(self, rng, num_pairs, spikes_per_bin):
        choices = rng.choice(
            len(self.pairs), size=num_pairs,
            replace=len(self.pairs) < num_pairs)
        pairs = self.pairs[choices]
        bins = np.unique(pairs)
        bin_to_column = {
            int(value): column for column, value in enumerate(bins)}
        indices = []
        time_column_ids = []
        for value in bins:
            members = self.sorted_indices[
                self.offsets[value]:self.offsets[value + 1]]
            sampled = rng.choice(
                members, size=spikes_per_bin,
                replace=len(members) < spikes_per_bin)
            indices.append(sampled)
            time_column_ids.append(
                np.full(spikes_per_bin, bin_to_column[int(value)]))
        pair_columns = np.asarray([
            [bin_to_column[int(first)], bin_to_column[int(second)]]
            for first, second in pairs
        ], dtype=np.int64)
        return (
            np.concatenate(indices).astype(np.int64),
            np.concatenate(time_column_ids).astype(np.int64),
            pair_columns,
        )


def run_local_phase(model, dataset, train_spike_indices, optimizer, args,
                    device, rng):
    model.train()
    model.set_encoder_trainable(True)
    model.set_drift_trainable(False)
    parameters = list(model.encoder.parameters())
    print(
        f"[trainable] phase=local encoder={parameter_count(parameters)} "
        "drift=0",
        flush=True)
    history = []
    for step in range(args.local_steps_per_cycle):
        indices = rng.choice(
            train_spike_indices, size=args.local_batch_size,
            replace=len(train_spike_indices) < args.local_batch_size)
        batch = load_spike_batch(dataset, indices, device)
        before = [parameter.detach().clone() for parameter in parameters]
        loss, _ = model.local_reconstruction_loss(
            batch["waveforms"], batch["coords"], batch["mask"],
            batch["centroids"])
        optimizer.zero_grad()
        loss.backward()
        grad = gradient_norm(parameters)
        if args.max_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, args.max_gradient_norm)
        optimizer.step()
        update = update_norm(before, parameters)
        history.append({
            "step": step,
            "loss": float(loss.item()),
            "gradient_norm": grad,
            "update_norm": update,
        })
    if history:
        print(
            f"[local] steps={len(history)} "
            f"loss={np.mean([item['loss'] for item in history]):.6f} "
            f"grad={np.mean([item['gradient_norm'] for item in history]):.6f} "
            f"update={np.mean([item['update_norm'] for item in history]):.6f}",
            flush=True)
    return history


def build_feedback_template(model, fine_raster, valid_time_bins,
                            registration_config):
    with torch.no_grad():
        corrected = shift_raster_to_brain(
            fine_raster, model.drift_trace(),
            registration_config.spatial_bin_um)
        return normalized_population_template(
            corrected, valid_time_bins=valid_time_bins, detach=True)


def run_feedback_phase(model, dataset, cache, registration_times, sampler,
                       template, optimizer, args, device, rng, cycle):
    model.train()
    model.set_encoder_trainable(True)
    model.set_drift_trainable(False)
    parameters = list(model.encoder.parameters())
    print(
        f"[trainable] phase=feedback encoder={parameter_count(parameters)} "
        "drift=0",
        flush=True)
    teacher_weight = args.teacher_weight
    if args.teacher_decay_cycles > 0:
        teacher_weight *= max(
            0.0, 1.0 - cycle / args.teacher_decay_cycles)
    history = []
    for step in range(args.joint_steps_per_cycle):
        indices, time_column_ids_np, pair_columns_np = sampler.sample(
            rng, args.feedback_pairs_per_step,
            args.feedback_spikes_per_bin)
        batch = load_spike_batch(dataset, cache.spike_index[indices], device)
        local_loss, outputs = model.local_reconstruction_loss(
            batch["waveforms"], batch["coords"], batch["mask"],
            batch["centroids"])
        feedback_times = torch.from_numpy(
            registration_times[indices].astype(np.float32)).to(device)
        z_brain = model.correct_depth(outputs["z_probe"], feedback_times)
        time_column_ids = torch.from_numpy(time_column_ids_np).to(device)
        pair_columns = torch.from_numpy(pair_columns_np).to(device)
        weights = registration_weights(
            batch["raw_amplitude"], args.registration_weight)
        raster = linear_splat_depth(
            z_brain, weights, time_column_ids,
            int(time_column_ids.max().item()) + 1,
            args.spatial_min, args.spatial_max, args.spatial_bin_um)
        raster = gaussian_smooth_depth(
            raster, args.fine_sigma_um / args.spatial_bin_um)
        feedback = population_feedback_loss(
            raster, pair_columns, template, args.template_weight)
        teacher = torch.mean(
            (outputs["z_probe"] - torch.from_numpy(
                cache.z_probe[indices]).to(device)).square())
        low = torch.relu(args.spatial_min - outputs["z_probe"])
        high = torch.relu(outputs["z_probe"] - args.spatial_max)
        range_loss = ((low.square() + high.square()) / 10_000.0).mean()
        total = (
            local_loss
            + args.population_weight * feedback["loss"]
            + teacher_weight * teacher
            + args.range_weight * range_loss
        )
        population_gradients = torch.autograd.grad(
            feedback["loss"], parameters, retain_graph=True,
            allow_unused=True)
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer.zero_grad()
        total.backward()
        grad = gradient_norm(parameters)
        population_grad = gradients_norm(population_gradients)
        if args.max_gradient_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                parameters, args.max_gradient_norm)
        optimizer.step()
        update = update_norm(before, parameters)
        values = {
            "step": step,
            "total_loss": float(total.item()),
            "local_loss": float(local_loss.item()),
            "population_loss": float(feedback["loss"].item()),
            "pair_loss": float(feedback["pair_loss"].item()),
            "template_loss": float(feedback["template_loss"].item()),
            "teacher_loss": float(teacher.item()),
            "range_loss": float(range_loss.item()),
            "teacher_weight": teacher_weight,
            "population_gradient_norm": population_grad,
            "total_gradient_norm": grad,
            "encoder_update_norm": update,
            "boundary_mass_fraction": float(
                feedback["boundary_mass_fraction"].item()),
        }
        history.append(values)
        print(
            f"[feedback {step + 1:03d}/{args.joint_steps_per_cycle}] "
            f"total={values['total_loss']:.6f} "
            f"population={values['population_loss']:.6f} "
            f"pop_grad={population_grad:.6f} update={update:.6f}",
            flush=True)
    return history


def evaluate_local_reconstruction(model, dataset, spike_indices, args,
                                  device, rng):
    if len(spike_indices) == 0:
        return float("nan")
    if len(spike_indices) > args.evaluation_spikes:
        spike_indices = rng.choice(
            spike_indices, size=args.evaluation_spikes, replace=False)
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(spike_indices), args.inference_batch_size):
            indices = spike_indices[
                start:start + args.inference_batch_size]
            batch = load_spike_batch(dataset, indices, device)
            loss, _ = model.local_reconstruction_loss(
                batch["waveforms"], batch["coords"], batch["mask"],
                batch["centroids"])
            total += float(loss.item()) * len(indices)
            count += len(indices)
    return total / max(count, 1)


def evaluate_population(model, cache, selected, times, args, device,
                        num_time_bins, registration_config):
    raster, counts = build_cached_raster(
        cache, selected, times, args, device, num_time_bins)
    raster = gaussian_smooth_depth(
        raster, args.fine_sigma_um / args.spatial_bin_um)
    valid_time_bins = counts >= args.min_spikes_per_bin
    with torch.no_grad():
        corrected, support = shift_raster_to_brain(
            raster, model.drift_trace(), args.spatial_bin_um,
            return_support=True)
        try:
            stationarity = population_stationarity_loss(
                corrected, registration_config.lag_bins,
                registration_config.template_weight,
                valid_time_bins=valid_time_bins, support=support)
        except ValueError as error:
            return {"valid": False, "reason": str(error)}
        input_mass = raster.sum().clamp_min(1e-8)
        return {
            "valid": True,
            "loss": float(stationarity["loss"].item()),
            "pair_loss": float(stationarity["pair_loss"].item()),
            "template_loss": float(
                stationarity["template_loss"].item()),
            "raster_occupancy": float(
                stationarity["raster_occupancy"].item()),
            "valid_pair_fraction": float(
                stationarity["valid_pair_fraction"].item()),
            "mass_ratio": float(
                (corrected.sum() / input_mass).item()),
            "boundary_mass_fraction": float(
                stationarity["boundary_mass_fraction"].item()),
            "per_lag_cosine": {
                str(lag): float(value.item())
                for lag, value in stationarity["per_lag_cosine"].items()
            },
        }


def evaluate_cycle(model, dataset, cache, registration_times, args, device,
                   num_time_bins, registration_config, rng):
    masks = {
        "train_registration_time": cache.split == SPLIT_TRAIN,
        "spike_heldout_real_time": cache.split == SPLIT_SPIKE_HELDOUT,
        "time_heldout_real_time": cache.split == SPLIT_TIME_HELDOUT,
    }
    population = {}
    for name, selected in masks.items():
        times = (
            registration_times
            if name == "train_registration_time" else cache.times_sec)
        population[name] = evaluate_population(
            model, cache, selected, times, args, device, num_time_bins,
            registration_config)
    spike_heldout_indices = cache.spike_index[
        cache.split == SPLIT_SPIKE_HELDOUT]
    local_loss = evaluate_local_reconstruction(
        model, dataset, spike_heldout_indices, args, device, rng)
    trace = model.drift_trace().detach().cpu().numpy()
    coordinates = cache.times_sec / args.time_bin_sec
    sampled = np.interp(
        coordinates, np.arange(num_time_bins, dtype=np.float32), trace)
    corrected = cache.z_probe - sampled
    return {
        "local_reconstruction": local_loss,
        "population": population,
        "drift_min_um": float(trace.min()),
        "drift_max_um": float(trace.max()),
        "drift_span_um": float(np.ptp(trace)),
        "drift_curvature": float(np.mean(np.diff(trace, n=2) ** 2)),
        "corrected_depth_variance": float(np.var(corrected)),
        "invalid_depth_fraction": float(
            ((cache.z_probe < args.spatial_min)
             | (cache.z_probe >= args.spatial_max)).mean()),
    }


def ablation_phases(name):
    return {
        "localizer-only": (True, False, False, False),
        "frozen-drift": (False, True, False, False),
        "alternating": (True, True, False, False),
        "full": (True, True, True, False),
        "shuffled-times": (True, True, True, True),
    }[name]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-path", type=Path,
        default=REPO_ROOT / "runs/dataset1_p1")
    parser.add_argument(
        "--localizer-checkpoint", type=Path,
        default=REPO_ROOT / "checkpoints/dataset1_p1/localizer.pt")
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPO_ROOT / "outputs/drift_joint_p1/full_zero")
    parser.add_argument(
        "--dredge-motion", type=Path,
        default=Path(
            "/scratch/am15577/UnitMatch/Post_Neurips/mp_ladder/results/"
            "Steinmetz/dataset1_p1/mp_dredge/motion.npz"))
    parser.add_argument(
        "--ablation",
        choices=[
            "localizer-only", "frozen-drift", "alternating", "full",
            "shuffled-times"],
        default="full")
    parser.add_argument(
        "--drift-initialization", choices=["zero", "linear"],
        default="zero")
    parser.add_argument(
        "--drift-fit-split", choices=["train", "all"], default="train",
        help=(
            "spikes used to fit the unsupervised drift module; localizer and "
            "feedback updates remain restricted to the training split"))
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--local-steps-per-cycle", type=int, default=200)
    parser.add_argument("--drift-steps-per-cycle", type=int, default=400)
    parser.add_argument("--joint-steps-per-cycle", type=int, default=20)
    parser.add_argument("--cache-refresh-cycles", type=int, default=1)
    parser.add_argument("--encoder-lr", type=float, default=3e-4)
    parser.add_argument("--drift-lr", type=float, default=0.25)
    parser.add_argument("--joint-encoder-lr", type=float, default=3e-5)
    parser.add_argument("--population-weight", type=float, default=0.1)
    parser.add_argument("--teacher-weight", type=float, default=1e-3)
    parser.add_argument("--teacher-decay-cycles", type=float, default=2.0)
    parser.add_argument("--range-weight", type=float, default=1e-3)
    parser.add_argument("--local-batch-size", type=int, default=512)
    parser.add_argument("--inference-batch-size", type=int, default=4096)
    parser.add_argument("--feedback-pairs-per-step", type=int, default=8)
    parser.add_argument("--feedback-spikes-per-bin", type=int, default=256)
    parser.add_argument("--evaluation-spikes", type=int, default=20_000)
    parser.add_argument("--max-spikes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spike-heldout-fraction", type=float, default=0.1)
    parser.add_argument("--time-heldout-fraction", type=float, default=0.1)
    parser.add_argument(
        "--time-heldout-start-fraction", type=float, default=0.45)
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
    parser.add_argument("--coarse-smoothness", type=float, default=0.002)
    parser.add_argument("--fine-smoothness", type=float, default=0.0002)
    parser.add_argument("--template-weight", type=float, default=0.5)
    parser.add_argument("--max-abs-knot-um", type=float, default=100.0)
    parser.add_argument("--initialization-amplitude", type=float, default=5.0)
    parser.add_argument("--min-spikes-per-bin", type=int, default=32)
    parser.add_argument(
        "--registration-weight", choices=["count", "raw", "log1p"],
        default="log1p")
    parser.add_argument("--max-gradient-norm", type=float, default=10.0)
    parser.add_argument("--no-save-cache", action="store_true")
    return parser


def validate_args(args):
    if args.cycles < 1 or args.cache_refresh_cycles < 1:
        raise ValueError("cycles and cache refresh interval must be positive")
    for name in (
            "local_steps_per_cycle", "drift_steps_per_cycle",
            "joint_steps_per_cycle", "coarse_steps"):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be nonnegative")
    if args.inference_batch_size < 1 or args.local_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if args.feedback_pairs_per_step < 1 or args.feedback_spikes_per_bin < 1:
        raise ValueError("feedback sampling values must be positive")
    if not 0 <= args.time_heldout_start_fraction < 1:
        raise ValueError("invalid time held-out start")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required; submit src/models/drift_joint_p1.sbatch")
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(
        args.localizer_checkpoint, map_location="cpu", weights_only=False)
    localizer_config = dict(checkpoint["cfg"])
    localizer_config.setdefault("recon_feature", "ptp")
    localizer_config.setdefault("loss_type", "mse")
    dataset = SpikeDataset(
        args.session_path, fixed_n=localizer_config["n_channels"],
        normalize=localizer_config.get("normalize", False))
    source_indices = np.arange(dataset.n_spikes, dtype=np.int64)
    if args.max_spikes is not None:
        source_indices = source_indices[:args.max_spikes]
    source_times = np.asarray(
        dataset.times_sec[source_indices], dtype=np.float32)
    split, block = make_splits(
        source_times, args.spike_heldout_fraction,
        args.time_heldout_fraction, args.time_heldout_start_fraction,
        args.seed)
    num_time_bins = int(math.floor(
        float(source_times.max()) / args.time_bin_sec)) + 1
    lag_bins = tuple(sorted({
        max(int(round(lag / args.time_bin_sec)), 1)
        for lag in args.lag_sec
        if lag < float(source_times.max())
    }))
    registration_config = PopulationRegistrationConfig(
        spatial_bin_um=args.spatial_bin_um,
        lag_bins=lag_bins,
        template_weight=args.template_weight,
        max_abs_knot_um=args.max_abs_knot_um,
    )
    if args.drift_initialization == "zero":
        initial_trace = torch.zeros(num_time_bins)
    else:
        initial_trace = torch.linspace(
            -args.initialization_amplitude,
            args.initialization_amplitude, num_time_bins)
        initial_trace -= initial_trace[0]
    model = JointDriftLocalizer(
        localizer_config, num_time_bins, args.fine_knots,
        args.time_bin_sec, initial_drift=initial_trace).to(device)
    model.encoder.load_state_dict(checkpoint["model_state_dict"])
    local_optimizer = torch.optim.Adam(
        model.encoder.parameters(), lr=args.encoder_lr)
    joint_optimizer = torch.optim.Adam(
        model.encoder.parameters(), lr=args.joint_encoder_lr)
    local_phase, drift_phase, feedback_phase, shuffle_times = (
        ablation_phases(args.ablation))
    rng = np.random.default_rng(args.seed)

    print(
        f"[model] class={model.__class__.__name__} "
        f"encoder={model.encoder.__class__.__name__} "
        f"encoder_parameters={parameter_count(model.encoder.parameters())} "
        f"drift_parameters={parameter_count(model.drift.parameters())}",
        flush=True)
    print(
        f"[phases] local={local_phase} drift={drift_phase} "
        f"feedback={feedback_phase} shuffled_times={shuffle_times}",
        flush=True)
    print(
        f"[data] session={args.session_path} spikes={len(source_indices)} "
        f"time_bins={num_time_bins} heldout_block={block}",
        flush=True)
    print(
        "[data-roles] localizer=train feedback=train "
        f"drift={args.drift_fit_split}",
        flush=True)
    print(
        "[config] " + json.dumps(
            {**vars(args), "localizer_config": localizer_config},
            default=str, sort_keys=True),
        flush=True)

    cache = cache_localizations(
        model, dataset, source_indices, split, device,
        args.inference_batch_size)
    baseline_z_probe = cache.z_probe.copy()
    if not args.no_save_cache:
        save_cache(
            cache, args.output_dir, args.localizer_checkpoint,
            localizer_config, model, args.time_bin_sec, cycle=-1)
    registration_times = cache.times_sec.copy()
    if shuffle_times:
        registration_times = rng.permutation(registration_times)
    train_spike_indices = cache.spike_index[
        cache.split == SPLIT_TRAIN]

    history = {"cycles": []}
    initial_evaluation = evaluate_cycle(
        model, dataset, cache, registration_times, args, device,
        num_time_bins, registration_config, rng)
    coarse_complete = False
    encoder_changed_since_cache = False

    for cycle in range(args.cycles):
        print(f"[cycle {cycle:03d}] starting", flush=True)
        local_history = []
        drift_history = {"coarse": [], "fine": []}
        feedback_history = []
        if local_phase and args.local_steps_per_cycle:
            local_history = run_local_phase(
                model, dataset, train_spike_indices, local_optimizer,
                args, device, rng)
            encoder_changed_since_cache = True

        refresh_due = cycle % args.cache_refresh_cycles == 0
        if encoder_changed_since_cache and refresh_due:
            cache = cache_localizations(
                model, dataset, source_indices, split, device,
                args.inference_batch_size)
            encoder_changed_since_cache = False
            if not args.no_save_cache:
                save_cache(
                    cache, args.output_dir, args.localizer_checkpoint,
                    localizer_config, model, args.time_bin_sec, cycle=cycle)
            if not shuffle_times:
                registration_times = cache.times_sec.copy()

        train_mask = cache.split == SPLIT_TRAIN
        train_raster, train_counts = build_cached_raster(
            cache, train_mask, registration_times, args, device,
            num_time_bins)
        if args.drift_fit_split == "all":
            drift_mask = np.ones(len(cache.split), dtype=bool)
            drift_raster, drift_counts = build_cached_raster(
                cache, drift_mask, registration_times, args, device,
                num_time_bins)
        else:
            drift_raster, drift_counts = train_raster, train_counts
        drift_valid_time_bins = (
            drift_counts >= args.min_spikes_per_bin)
        coarse_drift_raster = gaussian_smooth_depth(
            drift_raster, args.coarse_sigma_um / args.spatial_bin_um)
        fine_drift_raster = gaussian_smooth_depth(
            drift_raster, args.fine_sigma_um / args.spatial_bin_um)

        if drift_phase:
            model.set_encoder_trainable(False)
            model.set_drift_trainable(True)
            print(
                f"[trainable] phase=drift encoder=0 "
                f"drift={parameter_count(model.drift.parameters())}",
                flush=True)
            if not coarse_complete and args.coarse_steps:
                coarse = RigidSplineDrift(
                    model.drift_trace().detach(),
                    args.coarse_knots).to(device)
                drift_history["coarse"], coarse_trace = (
                    optimize_drift_module(
                        coarse, num_time_bins, coarse_drift_raster,
                        drift_valid_time_bins,
                        registration_config, args.coarse_steps,
                        args.drift_lr, args.coarse_smoothness,
                        args.max_abs_knot_um, "drift-coarse"))
                model.set_drift_trace_(coarse_trace)
                coarse_complete = True
            drift_history["fine"], _ = optimize_drift_module(
                model.drift, num_time_bins, fine_drift_raster,
                drift_valid_time_bins,
                registration_config, args.drift_steps_per_cycle,
                args.drift_lr, args.fine_smoothness,
                args.max_abs_knot_um, "drift-fine")

        if feedback_phase and args.joint_steps_per_cycle:
            feedback_valid_time_bins = (
                train_counts >= args.min_spikes_per_bin)
            feedback_raster = gaussian_smooth_depth(
                train_raster, args.fine_sigma_um / args.spatial_bin_um)
            template = build_feedback_template(
                model, feedback_raster, feedback_valid_time_bins,
                registration_config)
            sampler = FeedbackSampler(
                cache, registration_times, args.time_bin_sec,
                num_time_bins, lag_bins, args.min_spikes_per_bin)
            feedback_history = run_feedback_phase(
                model, dataset, cache, registration_times, sampler,
                template, joint_optimizer, args, device, rng, cycle)
            encoder_changed_since_cache = True

        evaluation = evaluate_cycle(
            model, dataset, cache, registration_times, args, device,
            num_time_bins, registration_config, rng)
        cycle_result = {
            "cycle": cycle,
            "local_phase": local_history,
            "drift_phase": drift_history,
            "feedback_phase": feedback_history,
            "evaluation": evaluation,
            "cache_stale_after_feedback": encoder_changed_since_cache,
        }
        history["cycles"].append(cycle_result)
        with (args.output_dir / "history.json").open("w") as handle:
            json.dump(history, handle, indent=2, sort_keys=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "localizer_config": localizer_config,
            "trainer_config": vars(args),
            "cycle": cycle,
            "local_optimizer_state_dict": local_optimizer.state_dict(),
            "joint_optimizer_state_dict": joint_optimizer.state_dict(),
        }, args.output_dir / "joint_localizer.pt")
        print(
            f"[cycle {cycle:03d}] "
            f"local={evaluation['local_reconstruction']:.6f} "
            f"drift_span={evaluation['drift_span_um']:.2f}um",
            flush=True)

    if encoder_changed_since_cache:
        cache = cache_localizations(
            model, dataset, source_indices, split, device,
            args.inference_batch_size)
        if not args.no_save_cache:
            save_cache(
                cache, args.output_dir, args.localizer_checkpoint,
                localizer_config, model, args.time_bin_sec,
                cycle=args.cycles)
        if not shuffle_times:
            registration_times = cache.times_sec.copy()

    final_evaluation = evaluate_cycle(
        model, dataset, cache, registration_times, args, device,
        num_time_bins, registration_config, rng)
    trace = model.drift_trace().detach().cpu().numpy().astype(np.float64)
    time_centers = (
        np.arange(num_time_bins, dtype=np.float64) + 0.5
    ) * args.time_bin_sec
    dredge_evaluation = None
    trace_output = {
        "times_sec": time_centers,
        "learned_drift": trace,
    }
    if args.dredge_motion.exists():
        reference = load_dredge_reference(
            args.dredge_motion, time_centers)
        per_window = [
            trace_metrics(trace, reference["windows"][:, column])
            for column in range(reference["windows"].shape[1])
        ]
        dredge_evaluation = {
            "mean": trace_metrics(trace, reference["mean"]),
            "median": trace_metrics(trace, reference["median"]),
            "windows": per_window,
        }
        trace_output.update({
            "dredge_mean": reference["mean"],
            "dredge_median": reference["median"],
            "dredge_windows": reference["windows"],
            "dredge_depth_windows_um": reference["depth_windows_um"],
        })
    else:
        print(
            f"[evaluation] DREDge reference not found: {args.dredge_motion}",
            flush=True)
    np.savez(args.output_dir / "traces.npz", **trace_output)

    encoder_change = cache.z_probe - baseline_z_probe
    summary = {
        "method": "alternating_dense_population_joint_v1",
        "model_class": model.__class__.__name__,
        "ablation": args.ablation,
        "drift_initialization": args.drift_initialization,
        "session": str(args.session_path),
        "localizer_checkpoint": str(args.localizer_checkpoint),
        "localizer_checkpoint_sha256": file_sha256(
            args.localizer_checkpoint),
        "dredge_role": "evaluation_only_loaded_after_training",
        "data_roles": {
            "localizer_fit": "training split only",
            "encoder_feedback_fit": "training split only",
            "drift_fit": (
                "entire recording"
                if args.drift_fit_split == "all"
                else "training split only"),
            "heldout_interpretation": (
                "localizer-held-out but drift-in-sample"
                if args.drift_fit_split == "all"
                else "localizer-and-drift-held-out"),
        },
        "repository": repository_state(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "resolved_config": {
            **vars(args), "localizer_config": localizer_config,
            "lag_bins": lag_bins,
        },
        "split_counts": {
            "train": int((cache.split == SPLIT_TRAIN).sum()),
            "spike_heldout": int(
                (cache.split == SPLIT_SPIKE_HELDOUT).sum()),
            "time_heldout": int(
                (cache.split == SPLIT_TIME_HELDOUT).sum()),
        },
        "initial_evaluation": initial_evaluation,
        "final_evaluation": final_evaluation,
        "encoder_depth_change": {
            "rmse_um": float(np.sqrt(np.mean(encoder_change ** 2))),
            "mae_um": float(np.mean(np.abs(encoder_change))),
            "maximum_abs_um": float(np.max(np.abs(encoder_change))),
        },
        "dredge_evaluation": dredge_evaluation,
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    torch.save({
        "model_state_dict": model.state_dict(),
        "localizer_config": localizer_config,
        "trainer_config": vars(args),
        "cycle": args.cycles - 1,
        "local_optimizer_state_dict": local_optimizer.state_dict(),
        "joint_optimizer_state_dict": joint_optimizer.state_dict(),
        "summary": summary,
    }, args.output_dir / "joint_localizer.pt")
    print(
        "[final] " + json.dumps({
            "local_reconstruction": final_evaluation[
                "local_reconstruction"],
            "drift_span_um": final_evaluation["drift_span_um"],
            "encoder_depth_rmse_um": summary[
                "encoder_depth_change"]["rmse_um"],
            "dredge": dredge_evaluation,
        }, sort_keys=True),
        flush=True)
    print(f"[final] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
