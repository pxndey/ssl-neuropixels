"""3x2 recon-feature comparison grid: rows = reconstruction target, cols = probe.

For each xy/yz/zx projection, one figure whose rows are the reconstruction
features (peak_to_trough, first_half, second_half) and columns are the two np12
probes (dataset1_p1, dataset1_p2). Each panel shows that (feature, probe) model's
test-set localizations (from runs/<probe>/inference_<feature>/), colored by spike
time, equal-aspect. Axis limits are shared per column (probe) across features.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_localizations import (  # noqa: E402
    GREEN_RED, PROJECTIONS, AXIS_LABEL, robust_limits, sample, norm_time, REPO,
)

FEATURES = ["peak_to_trough", "first_half", "second_half"]
PROBES = ["dataset1_p1", "dataset1_p2"]
# projection -> figure size (equal-aspect grid; roughly matches each projection's shape)
FIGSIZE = {"xy": (9.0, 11.0), "yz": (7.0, 13.0), "zx": (13.0, 8.5)}


def load_feat(probe, feature):
    d = REPO / "runs" / probe / f"inference_{feature}"
    loc = np.load(d / "localizations.npy")                     # (N,4) [lat, perp, depth, alpha]
    idx = np.load(d / "test_indices.npy")
    st = np.load(REPO / "runs" / probe / "spike_times.npy", mmap_mode="r")
    t = np.asarray(st[idx], dtype=np.float64)
    return loc[:, [0, 1, 2]].astype(np.float64), t


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sample", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "plots" / "feature_grid"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load + subsample once per (probe, feature)
    data = {}
    for probe in PROBES:
        for feat in FEATURES:
            xyz, t = load_feat(probe, feat)
            si = sample(len(xyz), args.n_sample, args.seed)
            data[(probe, feat)] = (xyz[si], norm_time(t[si]))

    for proj, (ai, aj) in PROJECTIONS.items():
        collim = {}
        for probe in PROBES:
            xs = [data[(probe, f)][0][:, ai] for f in FEATURES]
            ys = [data[(probe, f)][0][:, aj] for f in FEATURES]
            collim[probe] = (robust_limits(xs), robust_limits(ys))

        fig, axes = plt.subplots(len(FEATURES), len(PROBES),
                                 figsize=FIGSIZE.get(proj, (10.0, 11.0)),
                                 constrained_layout=True, squeeze=False)
        sc = None
        for r, feat in enumerate(FEATURES):
            for c, probe in enumerate(PROBES):
                ax = axes[r][c]
                xyz, tn = data[(probe, feat)]
                xlim, ylim = collim[probe]
                sc = ax.scatter(xyz[:, ai], xyz[:, aj], c=tn, cmap=GREEN_RED,
                                norm=Normalize(0.0, 1.0), s=0.5, alpha=0.35,
                                linewidths=0, rasterized=True)
                ax.set_xlim(*xlim)
                ax.set_ylim(*ylim)
                ax.set_aspect("equal")
                ax.tick_params(labelsize=8)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                if r == 0:
                    ax.set_title(probe, fontsize=12, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{feat}\n{AXIS_LABEL[aj]}", fontsize=9)
                if r == len(FEATURES) - 1:
                    ax.set_xlabel(AXIS_LABEL[ai], fontsize=9)
        fig.suptitle(f"recon-feature comparison — {proj} localizations", fontsize=13)
        cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.5, pad=0.01)
        cbar.set_label("spike time (early → late)", fontsize=9)
        cbar.set_ticks([])

        out = out_dir / f"{proj}.png"
        fig.savefig(out, dpi=800)
        plt.close(fig)
        print(f"[grid] {proj}: {len(FEATURES)}x{len(PROBES)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
