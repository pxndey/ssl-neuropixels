"""Reusable dense-population rigid-drift optimization.

This module operates on a depth-by-time raster built from fixed probe-frame
localizations. Every column is one time bin. It contains no waveform encoder
and no reference-motion input.
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PopulationRegistrationConfig:
    spatial_bin_um: float
    lag_bins: tuple[int, ...]
    template_weight: float = 0.5
    max_abs_knot_um: float = 100.0

    def __post_init__(self):
        if self.spatial_bin_um <= 0:
            raise ValueError("spatial_bin_um must be positive")
        if not self.lag_bins or any(lag < 1 for lag in self.lag_bins):
            raise ValueError("lag_bins must contain positive integers")
        if self.template_weight < 0:
            raise ValueError("template_weight must be nonnegative")
        if self.max_abs_knot_um <= 0:
            raise ValueError("max_abs_knot_um must be positive")


@dataclass(frozen=True)
class DriftStageConfig:
    num_knots: int
    steps: int
    learning_rate: float
    smoothness: float

    def __post_init__(self):
        if self.num_knots < 2:
            raise ValueError("num_knots must be at least two")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.smoothness < 0:
            raise ValueError("smoothness must be nonnegative")


def interpolate_knots(knots, num_time_bins):
    if knots.ndim != 1 or knots.numel() < 2:
        raise ValueError("knots must be a one-dimensional tensor of length >= 2")
    if num_time_bins < 2:
        raise ValueError("num_time_bins must be at least two")
    trace = F.interpolate(
        knots.reshape(1, 1, -1), size=num_time_bins,
        mode="linear", align_corners=True).reshape(-1)
    return trace - trace[0]


def resample_trace(trace, num_knots):
    if trace.ndim != 1 or trace.numel() < 2:
        raise ValueError("trace must be a one-dimensional tensor of length >= 2")
    if num_knots < 2:
        raise ValueError("num_knots must be at least two")
    return F.interpolate(
        trace.reshape(1, 1, -1), size=num_knots,
        mode="linear", align_corners=True).reshape(-1)


class RigidSplineDrift(nn.Module):
    """Gauge-fixed rigid drift represented by linearly interpolated knots."""

    def __init__(self, initial_trace, num_knots):
        super().__init__()
        knots = resample_trace(initial_trace, num_knots).detach().clone()
        self.knots = nn.Parameter(knots)

    def forward(self, num_time_bins):
        return interpolate_knots(self.knots, num_time_bins)

    def clamp_(self, max_abs_um):
        with torch.no_grad():
            self.knots.clamp_(-max_abs_um, max_abs_um)


def gaussian_smooth_depth(raster, sigma_bins):
    if raster.ndim != 2:
        raise ValueError("raster must have shape (depth, time)")
    if sigma_bins < 0:
        raise ValueError("sigma_bins must be nonnegative")
    if sigma_bins == 0:
        return raster
    radius = max(int(math.ceil(4.0 * sigma_bins)), 1)
    offsets = torch.arange(
        -radius, radius + 1, device=raster.device, dtype=raster.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma_bins).square())
    kernel = (kernel / kernel.sum()).reshape(1, 1, -1)
    return F.conv1d(
        raster.T.unsqueeze(1), kernel, padding=radius).squeeze(1).T


def linear_splat_depth(z, weights, time_ids, num_time_bins, spatial_min,
                       spatial_max, spatial_bin_um):
    """Linearly splat weighted depths into a depth-by-time raster."""
    if z.ndim != 1 or weights.ndim != 1 or time_ids.ndim != 1:
        raise ValueError("z, weights, and time_ids must be one-dimensional")
    if not (z.numel() == weights.numel() == time_ids.numel()):
        raise ValueError("z, weights, and time_ids must have equal length")
    if num_time_bins < 1 or spatial_max <= spatial_min or spatial_bin_um <= 0:
        raise ValueError("invalid raster dimensions")
    num_depth_bins = int(math.ceil(
        (spatial_max - spatial_min) / spatial_bin_um))
    raster = z.new_zeros((num_depth_bins, num_time_bins))
    if z.numel() == 0:
        return raster

    position = (z - spatial_min) / spatial_bin_um
    left = position.floor().to(torch.long)
    fraction = position - left.to(position.dtype)
    flat = raster.reshape(-1)
    time_valid = (time_ids >= 0) & (time_ids < num_time_bins)
    finite = torch.isfinite(z) & torch.isfinite(weights)
    for depth_ids, contribution in (
            (left, weights * (1.0 - fraction)),
            (left + 1, weights * fraction)):
        valid = (
            time_valid & finite & (depth_ids >= 0)
            & (depth_ids < num_depth_bins))
        if valid.any():
            flat_ids = (
                depth_ids[valid] * num_time_bins
                + time_ids[valid].to(torch.long))
            flat = flat.scatter_add(0, flat_ids, contribution[valid])
    return flat.reshape(num_depth_bins, num_time_bins)


def shift_raster_to_brain(raster, drift_um, spatial_bin_um,
                          return_support=False):
    """Return the raster in coordinates ``z_brain = z_probe - D(t)``.

    An output sample at brain depth ``z`` reads the probe-frame raster at
    ``z + D(t)``.
    """
    if raster.ndim != 2:
        raise ValueError("raster must have shape (depth, time)")
    if drift_um.ndim != 1 or drift_um.shape[0] != raster.shape[1]:
        raise ValueError("drift must contain one value per raster column")
    if spatial_bin_um <= 0:
        raise ValueError("spatial_bin_um must be positive")
    num_depth_bins, num_time_bins = raster.shape
    x_base = torch.linspace(
        -1.0, 1.0, num_depth_bins, device=raster.device, dtype=raster.dtype)
    y_base = torch.linspace(
        -1.0, 1.0, num_time_bins, device=raster.device, dtype=raster.dtype)
    shift_bins = drift_um / spatial_bin_um
    shift_normalized = (
        2.0 * shift_bins / max(num_depth_bins - 1, 1)).unsqueeze(1)
    x_grid = x_base.unsqueeze(0) + shift_normalized
    y_grid = y_base.unsqueeze(1).expand(-1, num_depth_bins)
    grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(0)
    shifted = F.grid_sample(
        raster.T.unsqueeze(0).unsqueeze(0), grid, mode="bilinear",
        padding_mode="zeros", align_corners=True)
    corrected = shifted.squeeze(0).squeeze(0).T
    if not return_support:
        return corrected
    support = (
        (x_grid >= -1.0) & (x_grid <= 1.0)
        & (y_grid >= -1.0) & (y_grid <= 1.0))
    return corrected, support.T


def normalized_population_template(raster, valid_time_bins=None,
                                   detach=False):
    if raster.ndim != 2:
        raise ValueError("raster must have shape (depth, time)")
    occupied = raster.norm(dim=0).gt(1e-8)
    if valid_time_bins is not None:
        if valid_time_bins.shape != (raster.shape[1],):
            raise ValueError(
                "valid_time_bins must contain one value per raster column")
        occupied = occupied & valid_time_bins.to(torch.bool)
    if not occupied.any():
        raise ValueError("the raster contains no valid occupied columns")
    normalized = raster[:, occupied] / raster[:, occupied].norm(
        dim=0, keepdim=True).clamp_min(1e-8)
    template = normalized.mean(dim=1)
    template = template / template.norm().clamp_min(1e-8)
    return template.detach() if detach else template


def raster_boundary_mass(raster, boundary_bins=4):
    if raster.ndim != 2:
        raise ValueError("raster must have shape (depth, time)")
    if boundary_bins < 1:
        raise ValueError("boundary_bins must be positive")
    width = min(boundary_bins, raster.shape[0])
    total = raster.sum().clamp_min(1e-8)
    if 2 * width >= raster.shape[0]:
        boundary = total
    else:
        boundary = raster[:width, :].sum() + raster[-width:, :].sum()
    return boundary / total


def population_stationarity_loss(corrected_raster, lag_bins, template_weight,
                                 detach_template=False, valid_time_bins=None,
                                 support=None, template=None):
    """Measure stationarity across columns of a corrected depth-by-time raster."""
    if corrected_raster.ndim != 2:
        raise ValueError("corrected_raster must have shape (depth, time)")
    valid_lags = sorted({
        int(lag) for lag in lag_bins
        if 0 < int(lag) < corrected_raster.shape[1]
    })
    if not valid_lags:
        raise ValueError("no lag is valid for the raster duration")
    if template_weight < 0:
        raise ValueError("template_weight must be nonnegative")

    column_norm = corrected_raster.norm(dim=0)
    occupied = column_norm.gt(1e-8)
    if valid_time_bins is not None:
        if valid_time_bins.shape != (corrected_raster.shape[1],):
            raise ValueError(
                "valid_time_bins must contain one value per raster column")
        occupied = occupied & valid_time_bins.to(torch.bool)
    if support is not None:
        if support.shape != corrected_raster.shape:
            raise ValueError("support must have the same shape as the raster")
        support = support.to(torch.bool)
    normalized = corrected_raster / column_norm.unsqueeze(0).clamp_min(1e-8)
    pair_losses = []
    pair_weights = []
    per_lag_cosine = {}
    valid_pair_count = corrected_raster.new_zeros(())
    possible_pair_count = 0
    max_lag = max(valid_lags)
    for lag in valid_lags:
        valid = occupied[:-lag] & occupied[lag:]
        possible_pair_count += valid.numel()
        if not valid.any():
            continue
        if support is None:
            similarity = (
                normalized[:, :-lag][:, valid]
                * normalized[:, lag:][:, valid]).sum(dim=0)
        else:
            common = (
                support[:, :-lag][:, valid]
                & support[:, lag:][:, valid])
            first = corrected_raster[:, :-lag][:, valid] * common
            second = corrected_raster[:, lag:][:, valid] * common
            first_norm = first.norm(dim=0)
            second_norm = second.norm(dim=0)
            usable = first_norm.gt(1e-8) & second_norm.gt(1e-8)
            if not usable.any():
                continue
            similarity = (
                first[:, usable] * second[:, usable]).sum(dim=0) / (
                    first_norm[usable]
                    * second_norm[usable]).clamp_min(1e-8)
        weight = math.exp(-float(lag) / max_lag)
        pair_losses.append((1.0 - similarity).mean() * weight)
        pair_weights.append(weight)
        valid_pair_count = valid_pair_count + similarity.numel()
        per_lag_cosine[lag] = similarity.mean()
    if not pair_losses:
        raise ValueError("the raster contains no occupied temporal pairs")
    pair_loss = torch.stack(pair_losses).sum() / sum(pair_weights)

    if template is None:
        template = normalized_population_template(
            corrected_raster, occupied, detach=detach_template)
    elif template.ndim != 1 or template.shape[0] != corrected_raster.shape[0]:
        raise ValueError("template must contain one value per spatial bin")
    elif detach_template:
        template = template.detach()
    if support is None:
        template_similarity = (
            normalized[:, occupied] * template.unsqueeze(1)).sum(dim=0)
    else:
        columns = corrected_raster[:, occupied] * support[:, occupied]
        targets = template.unsqueeze(1) * support[:, occupied]
        template_similarity = (columns * targets).sum(dim=0) / (
            columns.norm(dim=0) * targets.norm(dim=0)).clamp_min(1e-8)
    template_loss = (1.0 - template_similarity).mean()
    total = pair_loss + template_weight * template_loss
    return {
        "loss": total,
        "pair_loss": pair_loss,
        "template_loss": template_loss,
        "raster_occupancy": occupied.to(corrected_raster.dtype).mean(),
        "valid_pair_fraction": (
            valid_pair_count / max(possible_pair_count, 1)),
        "per_lag_cosine": per_lag_cosine,
        "boundary_mass_fraction": raster_boundary_mass(corrected_raster),
    }


def population_feedback_loss(raster, pair_indices, template, template_weight):
    """Compare sampled graph-bearing time columns and a fixed depth template."""
    if raster.ndim != 2:
        raise ValueError("raster must have shape (depth, time)")
    if pair_indices.ndim != 2 or pair_indices.shape[1] != 2:
        raise ValueError("pair_indices must have shape (pairs, 2)")
    if template.ndim != 1 or template.shape[0] != raster.shape[0]:
        raise ValueError("template must contain one value per spatial bin")
    if template_weight < 0:
        raise ValueError("template_weight must be nonnegative")
    normalized = raster / raster.norm(dim=0, keepdim=True).clamp_min(1e-8)
    first = normalized[:, pair_indices[:, 0]]
    second = normalized[:, pair_indices[:, 1]]
    pair_similarity = (first * second).sum(dim=0)
    template_similarity = (
        normalized * template.detach().unsqueeze(1)).sum(dim=0)
    pair_loss = (1.0 - pair_similarity).mean()
    template_loss = (1.0 - template_similarity).mean()
    return {
        "loss": pair_loss + template_weight * template_loss,
        "pair_loss": pair_loss,
        "template_loss": template_loss,
        "pair_cosine": pair_similarity.mean(),
        "template_cosine": template_similarity.mean(),
        "raster_occupancy": raster.norm(dim=0).gt(1e-8).to(
            raster.dtype).mean(),
        "boundary_mass_fraction": raster_boundary_mass(raster),
    }


def drift_curvature_loss(knots):
    curvature = torch.diff(knots, n=2)
    return (
        curvature.square().mean()
        if curvature.numel() else knots.new_zeros(())
    )


def optimize_rigid_drift_stage(initial_trace, raster, registration_config,
                               stage_config):
    """Optimize one coarse or fine rigid-drift stage."""
    model = RigidSplineDrift(initial_trace, stage_config.num_knots)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=stage_config.learning_rate)
    history = {
        "loss": [],
        "pair_loss": [],
        "template_loss": [],
        "curvature": [],
        "drift_span_um": [],
        "raster_occupancy": [],
        "valid_pair_fraction": [],
        "mass_ratio": [],
        "boundary_mass_fraction": [],
    }
    input_mass = raster.sum().detach().clamp_min(1e-8)
    for _ in range(stage_config.steps):
        drift = model(raster.shape[1])
        corrected, support = shift_raster_to_brain(
            raster, drift, registration_config.spatial_bin_um,
            return_support=True)
        stationarity = population_stationarity_loss(
            corrected, registration_config.lag_bins,
            registration_config.template_weight, support=support)
        curvature = drift_curvature_loss(model.knots)
        loss = stationarity["loss"] + stage_config.smoothness * curvature
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.clamp_(registration_config.max_abs_knot_um)
        with torch.no_grad():
            current = model(raster.shape[1])
            mass_ratio = corrected.sum() / input_mass
        history["loss"].append(loss.item())
        history["pair_loss"].append(stationarity["pair_loss"].item())
        history["template_loss"].append(
            stationarity["template_loss"].item())
        history["curvature"].append(curvature.item())
        history["drift_span_um"].append(
            (current.max() - current.min()).item())
        history["raster_occupancy"].append(
            stationarity["raster_occupancy"].item())
        history["valid_pair_fraction"].append(
            stationarity["valid_pair_fraction"].item())
        history["mass_ratio"].append(mass_ratio.item())
        history["boundary_mass_fraction"].append(
            stationarity["boundary_mass_fraction"].item())
        for lag, cosine in stationarity["per_lag_cosine"].items():
            history.setdefault(f"lag_{lag}_cosine", []).append(cosine.item())
    return model(raster.shape[1]).detach(), history


def evaluate_rigid_drift(raster, drift, registration_config,
                         valid_time_bins=None):
    with torch.no_grad():
        corrected, support = shift_raster_to_brain(
            raster, drift, registration_config.spatial_bin_um,
            return_support=True)
        stationarity = population_stationarity_loss(
            corrected, registration_config.lag_bins,
            registration_config.template_weight,
            valid_time_bins=valid_time_bins, support=support)
        input_mass = raster.sum().clamp_min(1e-8)
        return {
            "loss": stationarity["loss"].item(),
            "pair_loss": stationarity["pair_loss"].item(),
            "template_loss": stationarity["template_loss"].item(),
            "raster_occupancy": stationarity["raster_occupancy"].item(),
            "valid_pair_fraction": stationarity["valid_pair_fraction"].item(),
            "mass_ratio": (corrected.sum() / input_mass).item(),
            "boundary_mass_fraction": stationarity[
                "boundary_mass_fraction"].item(),
            "per_lag_cosine": {
                str(lag): value.item()
                for lag, value in stationarity["per_lag_cosine"].items()
            },
        }
