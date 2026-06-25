"""Block-contiguous channel masking for the self-supervised encoder.

The masking is applied per spike at data-loading time and defines the
reconstruction target.  It is deliberately a **contiguous spatial block**, not a
scattered set: if hidden channels were scattered, the model could reconstruct
each one by trivially averaging its still-visible immediate neighbors, learning
nothing about waveform propagation.  Masking a connected block forces the model
to extrapolate across distance.

Algorithm (exactly as specified):

    n_present = number of real channels
    n_mask    = floor(0.30 * n_present)
    candidates = all real channels except the peak channel
    seed       = random.choice(candidates)
    masked_set = grow a contiguous block from seed by nearest-spatial-neighbor
                 expansion until len(masked_set) == n_mask
                 (the peak channel is never added)

The peak channel carries the spike's defining signal and is *always* kept
visible (never padded, never masked).
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def pairwise_distances(coords: np.ndarray) -> np.ndarray:
    """(N, N) Euclidean distance matrix from (N, 2) micron coordinates."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def grow_block_mask(
    coords: np.ndarray,
    peak_idx: int,
    n_mask: int,
    *,
    rng: Optional[np.random.Generator] = None,
    dist: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Grow a contiguous masked block of size ``n_mask`` from a random seed.

    Returns a boolean array ``(N,)`` -- ``True`` where the channel is masked.
    The peak channel is guaranteed ``False``.
    """
    N = coords.shape[0]
    mask = np.zeros(N, dtype=bool)
    if n_mask <= 0 or N <= 1:
        return mask
    rng = rng if rng is not None else np.random.default_rng()
    if dist is None:
        dist = pairwise_distances(coords)

    candidates = [i for i in range(N) if i != peak_idx]
    if not candidates:
        return mask
    n_mask = min(n_mask, len(candidates))

    seed = int(rng.choice(candidates))
    mask[seed] = True
    masked = [seed]

    INF = np.inf
    while len(masked) < n_mask:
        # distance from every channel to the current masked set
        d_to_set = dist[:, masked].min(axis=1)
        d_to_set[mask] = INF            # already in the block
        d_to_set[peak_idx] = INF        # never mask the peak
        nxt = int(np.argmin(d_to_set))
        if not np.isfinite(d_to_set[nxt]):
            break                       # no reachable candidate left
        mask[nxt] = True
        masked.append(nxt)

    return mask


def block_contiguous_mask(
    coords: np.ndarray,
    peak_idx: int,
    mask_frac: float = 0.30,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Convenience wrapper computing ``n_mask = floor(mask_frac * n_present)``."""
    N = coords.shape[0]
    n_mask = int(np.floor(mask_frac * N))
    return grow_block_mask(coords, peak_idx, n_mask, rng=rng)


# --------------------------------------------------------------------------- #
# Inspection helpers (used by the masking unit test)
# --------------------------------------------------------------------------- #
def knn_adjacency(coords: np.ndarray, k: int = 8) -> np.ndarray:
    """Symmetric boolean adjacency: ``i ~ j`` if ``j`` is among ``i``'s ``k`` nearest."""
    N = coords.shape[0]
    k = min(k, N - 1)
    dist = pairwise_distances(coords)
    adj = np.zeros((N, N), dtype=bool)
    order = np.argsort(dist, axis=1)
    for i in range(N):
        for j in order[i, 1:k + 1]:
            adj[i, j] = True
            adj[j, i] = True
    return adj


def is_connected_subset(mask: np.ndarray, adjacency: np.ndarray) -> bool:
    """True if the channels selected by ``mask`` form one connected component."""
    nodes = np.flatnonzero(mask)
    if nodes.size <= 1:
        return True
    seen = {int(nodes[0])}
    stack = [int(nodes[0])]
    node_set = set(int(n) for n in nodes)
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(adjacency[u]):
            v = int(v)
            if v in node_set and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen == node_set
