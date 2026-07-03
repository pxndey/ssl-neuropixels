"""
Shank-aware neighborhood lookup for NP 2.0 4-shank probes.

Original: /scratch/ap7151/pre-summer-archive/neurips-week/01_localization/code/utils/neighbors.py
Extended with band-awareness for multi-bank recordings (e.g. mishi-dataset),
where a single shank's active channels are pulled from disjoint y-bands.

Neighborhood builder:
  - build_neighbor_table_sliding   : window slides off-center at edges; always
                                     full; constrained to the channel's
                                     (shank, band).

Backward compatibility: probes with a single contiguous y-band per shank
(AL032/AL036, Steinmetz NP1.0, Steinmetz NP2.0 single-shank) yield exactly
one band per shank, so the band split is a no-op and the output is identical
to the pre-band code.
"""

from __future__ import annotations
import logging
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

import spikeinterface as si

from .config import DEFAULT_N_NEIGHBORS, DEFAULT_N_ROWS_EACH_SIDE
from .config import DEFAULT_SHANK_X_THRESHOLD, DEFAULT_BAND_Y_THRESHOLD

logger = logging.getLogger(__name__)


def build_shank_layout(
    channel_x: np.ndarray,
    channel_y: np.ndarray,
    shank_x_threshold: float = 100.0,
    band_y_threshold: float = 50.0,
):
    """Return per-shank channel groupings, plus per-band sub-groupings.

    Returns a tuple:
      shank_of                : (n_ch,) int32   — shank id per channel
      shank_rows              : list[shank][row][channels]  rows sorted by y, channels by x
      shank_unique_ys         : list[shank][np.ndarray of unique y values]
      channel_row_in_shank    : (n_ch,) int32   — row index within shank
      shank_bands             : list[shank][band][row][channels]  rows split into bands
      channel_band_in_shank   : (n_ch,) int32   — band id within the channel's shank
      channel_row_in_band     : (n_ch,) int32   — row index within the channel's band

    Band detection: within a shank, consecutive unique_ys with gap >
    `band_y_threshold` start a new band. For probes with one contiguous strip
    per shank (15/20 µm row pitch, ≤ band_y_threshold), each shank has a single
    band and the band-aware fields collapse to the row-aware ones.
    """
    n_ch = len(channel_x)

    unique_xs   = np.sort(np.unique(channel_x))
    x_gaps      = np.diff(unique_xs)
    boundaries  = np.where(x_gaps > shank_x_threshold)[0] + 1
    shank_x_groups = np.split(unique_xs, boundaries)
    n_shanks    = len(shank_x_groups)

    shank_of = np.full(n_ch, -1, dtype=np.int32)
    for sid, x_group in enumerate(shank_x_groups):
        for xv in x_group:
            shank_of[np.isclose(channel_x, xv)] = sid
    assert (shank_of >= 0).all(), "some channels were not assigned to a shank"

    shank_rows:      list[list[list[int]]]      = []
    shank_unique_ys: list[np.ndarray]           = []
    shank_bands:     list[list[list[list[int]]]] = []
    shank_band_lens: list[list[int]]            = []

    for sid in range(n_shanks):
        ch_idx = np.where(shank_of == sid)[0]
        ys     = np.unique(channel_y[ch_idx])
        rows: list[list[int]] = []
        for y in ys:
            in_row = ch_idx[np.isclose(channel_y[ch_idx], y)]
            in_row = in_row[np.argsort(channel_x[in_row])]
            rows.append(in_row.tolist())
        shank_rows.append(rows)
        shank_unique_ys.append(ys)

        if len(ys) <= 1:
            band_splits: list[int] = []
        else:
            y_gaps      = np.diff(ys)
            band_splits = (np.where(y_gaps > band_y_threshold)[0] + 1).tolist()

        bands: list[list[list[int]]] = []
        band_lens: list[int] = []
        prev = 0
        for sp in band_splits:
            bands.append(rows[prev:sp])
            band_lens.append(sp - prev)
            prev = sp
        bands.append(rows[prev:])
        band_lens.append(len(rows) - prev)
        shank_bands.append(bands)
        shank_band_lens.append(band_lens)

    channel_row_in_shank  = np.full(n_ch, -1, dtype=np.int32)
    channel_band_in_shank = np.full(n_ch, -1, dtype=np.int32)
    channel_row_in_band   = np.full(n_ch, -1, dtype=np.int32)

    for sid in range(n_shanks):
        ys        = shank_unique_ys[sid]
        band_lens = shank_band_lens[sid]
        band_of_row = np.empty(len(ys), dtype=np.int32)
        row_in_band = np.empty(len(ys), dtype=np.int32)
        cursor = 0
        for bid, blen in enumerate(band_lens):
            band_of_row[cursor:cursor + blen] = bid
            row_in_band[cursor:cursor + blen] = np.arange(blen, dtype=np.int32)
            cursor += blen
        for c in np.where(shank_of == sid)[0]:
            r = int(np.searchsorted(ys, channel_y[c]))
            channel_row_in_shank[c]  = r
            channel_band_in_shank[c] = int(band_of_row[r])
            channel_row_in_band[c]   = int(row_in_band[r])

    return (shank_of, shank_rows, shank_unique_ys, channel_row_in_shank,
            shank_bands, channel_band_in_shank, channel_row_in_band)


def build_neighbor_table_sliding(
    channel_x: np.ndarray,
    channel_y: np.ndarray,
    n_rows_each_side: int = DEFAULT_N_ROWS_EACH_SIDE,
    shank_x_threshold: float = DEFAULT_SHANK_X_THRESHOLD,
    band_y_threshold: float = DEFAULT_BAND_Y_THRESHOLD,
) -> np.ndarray:
    """
    Sliding-window neighborhood within each (shank, band). Every channel gets
    the same number of real neighbors: 2*(2*n_rows_each_side + 1). At
    band tip/back the window slides toward the band interior so it always
    spans (2*n_rows_each_side + 1) rows from the same band.

    For probes with a single band per shank (AL032/AL036, Steinmetz), this is
    identical to the pre-band code.
    """
    n_ch        = len(channel_x)
    n_neighbors = (2 * n_rows_each_side + 1) * 2
    win_rows    = 2 * n_rows_each_side + 1

    (shank_of, _, _, _,
     shank_bands, channel_band_in_shank, channel_row_in_band) = build_shank_layout(
        channel_x, channel_y, shank_x_threshold, band_y_threshold
    )

    neighbor_ids = np.full((n_ch, n_neighbors), -1, dtype=np.int32)

    for c in range(n_ch):
        sid    = int(shank_of[c])
        bid    = int(channel_band_in_shank[c])
        r      = int(channel_row_in_band[c])
        rows   = shank_bands[sid][bid]
        n_rows = len(rows)
        assert n_rows >= win_rows, (
            f"shank {sid} band {bid} has only {n_rows} rows; need at least "
            f"{win_rows}. Channel {c} at y={float(channel_y[c]):.1f} cannot "
            f"build a {win_rows}-row neighborhood."
        )

        r_lo = r - n_rows_each_side
        r_hi = r + n_rows_each_side
        if r_lo < 0:
            r_lo, r_hi = 0, win_rows - 1
        elif r_hi > n_rows - 1:
            r_lo, r_hi = n_rows - win_rows, n_rows - 1

        neighbors: list[int] = []
        for ri in range(r_lo, r_hi + 1):
            neighbors.extend(rows[ri])

        assert len(neighbors) == n_neighbors, (
            f"ch {c}: got {len(neighbors)} neighbors, expected {n_neighbors}. "
            f"Likely a row with the wrong column count in shank {sid} band {bid}."
        )
        neighbor_ids[c] = np.array(neighbors, dtype=np.int32)

    return neighbor_ids


def build_neighbor_table_euclidean(
    channel_positions: NDArray,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
) -> NDArray:
    n_channels = len(channel_positions)
    neighbor_table = np.zeros((n_channels, n_neighbors), dtype=np.int32)

    positions = np.asarray(channel_positions)
    for i in range(n_channels):
        distances = np.sum((positions - positions[i]) ** 2, axis=1)
        neighbor_indices = np.argsort(distances)[:n_neighbors]
        neighbor_table[i] = neighbor_indices

    return neighbor_table


def split_recording_by_shank(
    recording: si.BaseRecording,
    shank_x_threshold: float = DEFAULT_SHANK_X_THRESHOLD,
) -> dict[int, si.BaseRecording]:
    channel_locations = recording.get_channel_locations()
    channel_x = channel_locations[:, 0]
    channel_y = channel_locations[:, 1]

    shank_of, _, _, _, _, _, _ = build_shank_layout(
        channel_x, channel_y, shank_x_threshold
    )

    n_shanks = int(shank_of.max() + 1)
    logger.info(f"Detected {n_shanks} shanks")

    shank_recordings = {}
    for shank_id in range(n_shanks):
        shank_mask = shank_of == shank_id
        shank_channel_ids = recording.channel_ids[shank_mask]
        shank_recordings[shank_id] = recording.select_channels(shank_channel_ids)

    return shank_recordings


def compute_spike_geometry(
    recording: si.BaseRecording,
    peaks: NDArray,
    neighbor_table: NDArray,
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    channel_locations = recording.get_channel_locations()
    n_spikes = len(peaks)
    n_neighbors = neighbor_table.shape[1]

    peak_channels = peaks["channel_index"]
    neighbor_ids = neighbor_table[peak_channels]

    centroids = np.zeros((n_spikes, 2))
    local_coords = np.zeros((n_spikes, n_neighbors, 2))

    for i in range(n_spikes):
        nbrs = neighbor_ids[i]
        nbr_positions = channel_locations[nbrs]
        centroid = nbr_positions.mean(axis=0)
        centroids[i] = centroid
        local_coords[i] = nbr_positions - centroid

    return centroids, local_coords, neighbor_ids, peak_channels
