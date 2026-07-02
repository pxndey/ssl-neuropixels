"""Spike depth raster (depth vs time, colored by log amplitude).

Same inputs as the diff-DREDge run: the monopolar localizations
(depth = col 1, alpha = col 3) + spike_times. Single panel, in the style of
sln/src/plots/raster.py (subsample, big-amplitude spikes on top, viridis).
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(a):
    loc = np.load(a.loc)
    st = np.load(a.times).astype(np.float64)
    depth = loc[:, a.depth_col].astype(np.float32)
    alpha = np.abs(loc[:, a.amp_col]).astype(np.float64)
    t = st / a.fs
    N = t.shape[0]

    rng = np.random.default_rng(a.seed)
    idx = np.arange(N) if a.n_plot >= N else np.sort(rng.choice(N, a.n_plot, replace=False))
    idx = idx[np.argsort(alpha[idx])]                 # draw large-amplitude spikes last (on top)
    c = np.log10(np.clip(alpha[idx], 1e-6, None))
    cmin, cmax = np.percentile(c, [1, 99])

    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    sc = ax.scatter(t[idx], depth[idx], c=c, cmap="viridis", vmin=cmin, vmax=cmax,
                    s=0.3, alpha=0.4, linewidths=0, rasterized=True)
    ax.set_xlabel("time [s]", fontsize=13)
    ax.set_ylabel("depth y [µm]", fontsize=13)
    ax.set_title(f"dataset1_p1: monopolar localization raster ({idx.size:,} spikes)",
                 fontsize=14, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    cb = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("log₁₀ amplitude", fontsize=12)

    fig.savefig(a.out, dpi=a.dpi)
    print("saved ->", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    LOC = "/scratch/ap7151/sln/runs/extra-motion/dataset1_p1/preprocessed/shank_0"
    ap.add_argument("--loc", default=f"{LOC}/localizations.npy")
    ap.add_argument("--times", default=f"{LOC}/spike_times.npy")
    ap.add_argument("--depth-col", type=int, default=1)
    ap.add_argument("--amp-col", type=int, default=3)
    ap.add_argument("--fs", type=float, default=30000.0)
    ap.add_argument("--n-plot", type=int, default=900_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--out", default="/scratch/ap7151/sln-fixed/plots/raster_dataset1_p1.png")
    main(ap.parse_args())
