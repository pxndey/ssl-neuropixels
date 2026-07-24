"""Differentiable rigid-drift registration primitives.

Coordinates entering this module are always probe-frame localizations corrected
to ``z_brain = z_probe - D(t)``.  The registration objective therefore has its
zero-shift optimum when the sampled drift ``D(t)`` removes probe motion.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RegistrationConfig:
    num_time_bins: int
    bin_width_sec: float
    spatial_min: float = 0.0
    spatial_max: float = 3840.0
    spatial_grid_step: float = 1.0
    sigma: float = 2.0
    temporal_window_bins: int = 20
    max_shift_bins: int = 30
    beta: float = 3.0

    @property
    def num_shifts(self):
        return 2 * self.max_shift_bins + 1


def sample_drift(trace, times_sec, bin_width_sec):
    """Linearly sample the gauge-fixed drift trace at probe times.

    The stored trace has a free additive gauge.  Subtracting its first element
    makes a constant offset unobservable without detaching it from gradients.
    """
    if trace.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if trace.numel() == 0:
        raise ValueError("trace must not be empty")
    if bin_width_sec <= 0:
        raise ValueError("bin_width_sec must be positive")
    coordinates = (times_sec / bin_width_sec).clamp(0, trace.numel() - 1)
    left = coordinates.floor().to(torch.long)
    right = (left + 1).clamp(max=trace.numel() - 1)
    fraction = coordinates - left.to(coordinates.dtype)
    gauge_fixed = trace - trace[0]
    return gauge_fixed[left] + fraction * (gauge_fixed[right] - gauge_fixed[left])


def time_bin_counts(times_sec, config):
    """Return non-differentiable event counts for each registration time bin."""
    if times_sec.ndim != 1:
        raise ValueError("times_sec must be one-dimensional")
    bin_ids = (times_sec / config.bin_width_sec).to(torch.long).clamp(
        0, config.num_time_bins - 1)
    return torch.bincount(bin_ids, minlength=config.num_time_bins)


def soft_rasterize(z_brain, weights, times_sec, config):
    """Rasterize brain-frame localizations into differentiable depth rows."""
    if z_brain.ndim != 1 or weights.ndim != 1 or times_sec.ndim != 1:
        raise ValueError("z_brain, weights, and times_sec must be one-dimensional")
    if not (z_brain.numel() == weights.numel() == times_sec.numel()):
        raise ValueError("z_brain, weights, and times_sec must have equal length")
    if config.num_time_bins < 1 or config.spatial_grid_step <= 0 or config.sigma <= 0:
        raise ValueError("registration configuration has invalid raster dimensions")
    grid = torch.arange(
        config.spatial_min, config.spatial_max, config.spatial_grid_step,
        device=z_brain.device, dtype=z_brain.dtype)
    if grid.numel() == 0:
        raise ValueError("registration spatial grid must not be empty")
    raster = z_brain.new_zeros((config.num_time_bins, grid.numel()))
    if z_brain.numel() == 0:
        return raster
    bin_ids = (times_sec / config.bin_width_sec).to(torch.long).clamp(
        0, config.num_time_bins - 1)
    squared_distance = (grid.unsqueeze(0) - z_brain.unsqueeze(1)).square()
    bumps = weights.unsqueeze(1) * torch.exp(
        -squared_distance / (2.0 * config.sigma**2))
    raster.index_add_(0, bin_ids, bumps)
    return raster


def registration_diagnostics(correlations, config, valid_pair_fraction, raster,
                             spike_counts=None):
    """Summarize confidence for raw normalized correlations and their logits."""
    zero = raster.new_tensor(0.0)
    occupancy = raster.norm(dim=1).gt(1e-8)
    occupied = occupancy.sum()
    if spike_counts is None:
        spikes_per_occupied = zero
    else:
        if spike_counts.shape != (raster.shape[0],):
            raise ValueError("spike_counts must contain one entry per raster row")
        spikes_per_occupied = (
            spike_counts[occupancy].to(raster.dtype).sum()
            / occupied.clamp_min(1).to(raster.dtype))
    if correlations.numel() == 0:
        return {
            "valid": False,
            "zero_shift_correlation": zero,
            "zero_shift_probability": zero,
            "correlation_entropy": zero,
            "peak_probability": zero,
            "boundary_hit_fraction": raster.new_tensor(1.0),
            "valid_pair_fraction": raster.new_tensor(valid_pair_fraction),
            "raster_occupancy": occupancy.to(raster.dtype).mean(),
            "spikes_per_occupied_time_bin": spikes_per_occupied,
        }

    logits = correlations * config.beta
    probabilities = torch.softmax(logits, dim=-1)
    center = config.max_shift_bins
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
    peaks = probabilities.argmax(dim=-1)
    return {
        "valid": True,
        "zero_shift_correlation": correlations[:, center].mean(),
        "zero_shift_probability": probabilities[:, center].mean(),
        "correlation_entropy": entropy.mean(),
        "peak_probability": probabilities.max(dim=-1).values.mean(),
        "boundary_hit_fraction": (
            (peaks == 0) | (peaks == config.num_shifts - 1)
        ).to(raster.dtype).mean(),
        "valid_pair_fraction": raster.new_tensor(valid_pair_fraction),
        "raster_occupancy": occupancy.to(raster.dtype).mean(),
        "spikes_per_occupied_time_bin": spikes_per_occupied,
    }


def zero_shift_nll(raster, config, spike_counts=None):
    """Return zero-shift NLL for valid temporal pairs and confidence metrics.

    Each pair contributes ``-log p(shift=0)``.  A uniform correlation vector
    therefore costs ``log(num_shifts)`` rather than incorrectly looking like a
    successful zero displacement.  Empty and unoccupied rasters get an even
    larger finite penalty and an explicit invalid diagnostic.
    """
    if raster.ndim != 2:
        raise ValueError("raster must be two-dimensional")
    if raster.shape[0] != config.num_time_bins:
        raise ValueError("raster time dimension does not match configuration")
    if config.max_shift_bins < 0 or config.temporal_window_bins < 1:
        raise ValueError("registration shift and temporal windows must be positive")
    if raster.shape[1] < 1:
        raise ValueError("raster must contain at least one spatial bin")

    occupied = raster.norm(dim=1).gt(1e-8)
    normalized = raster / raster.norm(dim=1, keepdim=True).clamp_min(1e-8)
    correlations_by_lag = []
    possible_pairs = 0
    valid_pairs = 0
    max_lag = min(config.temporal_window_bins, raster.shape[0] - 1)
    for lag in range(1, max_lag + 1):
        possible_pairs += raster.shape[0] - lag
        valid = occupied[:-lag] & occupied[lag:]
        if not valid.any():
            continue
        correlations = _normalized_shift_correlations(
            normalized[:-lag][valid], normalized[lag:][valid],
            config.max_shift_bins)
        correlations_by_lag.append(correlations)
        valid_pairs += correlations.shape[0]

    valid_pair_fraction = valid_pairs / max(possible_pairs, 1)
    if not correlations_by_lag:
        penalty = raster.new_tensor(math.log(config.num_shifts) + 1.0)
        return penalty, registration_diagnostics(
            raster.new_empty((0, config.num_shifts)), config,
            valid_pair_fraction, raster, spike_counts)

    correlations = torch.cat(correlations_by_lag, dim=0)
    target = torch.full(
        (correlations.shape[0],), config.max_shift_bins,
        dtype=torch.long, device=raster.device)
    loss = F.cross_entropy(correlations * config.beta, target)
    return loss, registration_diagnostics(
        correlations, config, valid_pair_fraction, raster, spike_counts)


def _normalized_shift_correlations(first, second, max_shift):
    """Correlate rows for shifts from ``-max_shift`` through ``+max_shift``."""
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ValueError("first and second must be equally shaped two-dimensional rows")
    grid_len = first.shape[1]
    padded_second = F.pad(second, (max_shift, max_shift))
    second_windows = padded_second.unfold(1, grid_len, 1)
    overlap = F.pad(first.new_ones((1, grid_len)), (max_shift, max_shift)).unfold(
        1, grid_len, 1)
    numerator = (first.unsqueeze(1) * second_windows).sum(dim=2)
    first_norm = (first.square().unsqueeze(1) * overlap).sum(dim=2).sqrt()
    second_norm = second_windows.square().sum(dim=2).sqrt()
    return numerator / (first_norm * second_norm).clamp_min(1e-8)
