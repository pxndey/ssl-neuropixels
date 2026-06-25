"""Dataset + collate for the masked waveform encoder.

Each sample is one spike's neighborhood:

    waveforms : (N, 90)   per-channel AP-band snippets (N = present channels)
    coords    : (N, 2)    (dx, dy) microns *relative to the peak channel*
    peak_idx  : int       index of the peak channel within this neighborhood

Coordinates are kept in **real microns** (not normalized by probe pitch) on
purpose: waveform attenuation depends on physical distance, so a micron-based
position embedding is what lets a model trained on one probe geometry transfer to
another.

The block-contiguous mask (``masking.block_contiguous_mask``) is drawn per sample
at load time; ``collate_masked`` pads the variable ``N`` up to ``N_max`` and emits
the two masks the model needs:

    padding_mask : (B, N_max)  True = slot does not exist (geometry padding)
    content_mask : (B, N_max)  True = real channel whose waveform is hidden
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .masking import block_contiguous_mask


class WaveformMaskedDataset(Dataset):
    """Holds per-spike neighborhoods and draws a fresh block mask each __getitem__.

    Parameters
    ----------
    waveforms : (S, M, 90) float array (zero-padded to M channels)
    coords    : (S, M, 2) float array, microns relative to peak (padded)
    counts    : (S,) int array, number of real channels per spike
    peak_idx  : (S,) int array, peak channel index within the neighborhood
    mask_frac : float
    seed      : optional int for reproducible masks
    """

    def __init__(
        self,
        waveforms: np.ndarray,
        coords: np.ndarray,
        counts: np.ndarray,
        peak_idx: np.ndarray,
        mask_frac: float = 0.30,
        seed: Optional[int] = None,
    ):
        assert waveforms.shape[0] == coords.shape[0] == counts.shape[0] == peak_idx.shape[0]
        # asarray avoids duplicating the (multi-GB) waveform array when it already
        # has the right dtype
        self.waveforms = np.asarray(waveforms, dtype=np.float32)
        self.coords = np.asarray(coords, dtype=np.float32)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.peak_idx = np.asarray(peak_idx, dtype=np.int64)
        self.mask_frac = mask_frac
        self._base_seed = seed

    def __len__(self) -> int:
        return self.waveforms.shape[0]

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        n = int(self.counts[i])
        wf = self.waveforms[i, :n]                 # (n, 90)
        coords = self.coords[i, :n]                # (n, 2)
        peak = int(self.peak_idx[i])
        peak = min(max(peak, 0), n - 1)

        rng = np.random.default_rng(None if self._base_seed is None else self._base_seed + i)
        content_mask = block_contiguous_mask(coords, peak, self.mask_frac, rng=rng)

        return {
            "waveforms": torch.from_numpy(np.ascontiguousarray(wf)),
            "coords": torch.from_numpy(np.ascontiguousarray(coords)),
            "content_mask": torch.from_numpy(content_mask),
            "peak_idx": torch.tensor(peak, dtype=torch.long),
            "n_present": torch.tensor(n, dtype=torch.long),
        }


def collate_masked(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Pad a batch of variable-N neighborhoods to ``N_max`` and build both masks."""
    B = len(batch)
    n_max = max(int(b["n_present"]) for b in batch)
    L = batch[0]["waveforms"].shape[1]

    waveforms = torch.zeros(B, n_max, L, dtype=torch.float32)
    coords = torch.zeros(B, n_max, 2, dtype=torch.float32)
    padding_mask = torch.ones(B, n_max, dtype=torch.bool)     # True = padding (does not exist)
    content_mask = torch.zeros(B, n_max, dtype=torch.bool)    # True = real & hidden
    peak_idx = torch.zeros(B, dtype=torch.long)

    for b, s in enumerate(batch):
        n = int(s["n_present"])
        waveforms[b, :n] = s["waveforms"]
        coords[b, :n] = s["coords"]
        padding_mask[b, :n] = False
        content_mask[b, :n] = s["content_mask"]
        peak_idx[b] = s["peak_idx"]

    return {
        "waveforms": waveforms,
        "coords": coords,
        "padding_mask": padding_mask,
        "content_mask": content_mask,
        "peak_idx": peak_idx,
    }


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def make_synthetic_dataset(
    n_spikes: int = 256,
    max_neighbors: int = 12,
    n_samples: int = 90,
    grid: tuple = (2, 8),
    pitch_um: tuple = (32.0, 20.0),
    mask_frac: float = 0.30,
    seed: int = 0,
) -> WaveformMaskedDataset:
    """A small physically-plausible dataset for tests and dry-run training.

    Channels live on a regular ``grid`` (columns x rows) with the given pitch; a
    biphasic template is attenuated by distance from a random source near the
    peak channel, so masked-block reconstruction is a meaningful task.
    """
    rng = np.random.default_rng(seed)
    n_cols, n_rows = grid
    xs, ys = np.meshgrid(np.arange(n_cols) * pitch_um[0], np.arange(n_rows) * pitch_um[1], indexing="ij")
    all_coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)  # (M, 2)
    M = all_coords.shape[0]
    max_neighbors = min(max_neighbors, M)

    t = np.linspace(0, 1, n_samples)
    template = (np.sin(2 * np.pi * t) * np.exp(-((t - 0.35) ** 2) / 0.02)).astype(np.float32)

    waveforms = np.zeros((n_spikes, max_neighbors, n_samples), dtype=np.float32)
    coords = np.zeros((n_spikes, max_neighbors, 2), dtype=np.float32)
    counts = np.zeros(n_spikes, dtype=np.int64)
    peak_idx = np.zeros(n_spikes, dtype=np.int64)

    for i in range(n_spikes):
        n = int(rng.integers(max(4, max_neighbors // 2), max_neighbors + 1))
        sel = rng.choice(M, size=n, replace=False)
        csel = all_coords[sel]
        peak = int(np.argmax(-(csel ** 2).sum(1)))   # channel nearest origin-ish; arbitrary but real
        src = csel[peak] + rng.normal(0, 5, size=2)
        amp = rng.uniform(20, 80)
        d = np.sqrt(((csel - src) ** 2).sum(1))
        atten = amp / (1.0 + (d / 20.0) ** 2)        # distance attenuation
        noise = rng.normal(0, 1.0, size=(n, n_samples)).astype(np.float32)
        waveforms[i, :n] = atten[:, None] * template[None, :] + noise
        coords[i, :n] = csel - csel[peak]            # relative to peak
        counts[i] = n
        peak_idx[i] = peak

    return WaveformMaskedDataset(waveforms, coords, counts, peak_idx, mask_frac=mask_frac, seed=seed)


def from_extracted(session_path: str, mask_frac: float = 0.30, seed: Optional[int] = None) -> WaveformMaskedDataset:
    """Build a dataset from ``extract_neighborhoods.py`` outputs.

    Reads ``neighborhood_waveforms.npy``, ``local_coords.npy`` (relative to the
    neighborhood centroid), ``neighbor_ids.npy``, ``neighbor_counts.npy`` and
    ``spike_channels.npy``; re-references the coordinates to the **peak channel**
    (the spike's own detected channel) as the encoder expects.
    """
    sp = Path(session_path)
    waveforms = np.load(sp / "neighborhood_waveforms.npy")    # (S, M, 90)
    local_coords = np.load(sp / "local_coords.npy")           # (S, M, 2) rel centroid
    neighbor_ids = np.load(sp / "neighbor_ids.npy")           # (S, M) channel ids, -1 pad
    counts = np.load(sp / "neighbor_counts.npy")              # (S,)
    spike_channels = np.load(sp / "spike_channels.npy")       # (S,)

    S, M = neighbor_ids.shape
    # peak channel = the neighbor whose id equals the spike's own detected channel
    # (vectorized over all S spikes; falls back to index 0 if somehow absent)
    match = neighbor_ids == spike_channels[:, None]               # (S, M) bool
    peak_idx = np.where(match.any(axis=1), match.argmax(axis=1), 0).astype(np.int64)
    peak_coord = local_coords[np.arange(S), peak_idx]             # (S, 2)
    coords = (local_coords - peak_coord[:, None, :]).astype(np.float32)   # re-reference to peak
    # (padding slots beyond neighbor_counts[i] are never read by __getitem__)

    return WaveformMaskedDataset(waveforms, coords, counts, peak_idx, mask_frac=mask_frac, seed=seed)
