"""Visualize drift correction before/after a joint-loss fine-tune.

Reads a `predictions.npz` produced by `train.py --joint-loss --save-predictions`
(or `infer.py` on a joint checkpoint). That file contains:
    x, z           : observed absolute probe-frame coordinates (pre-correction)
    x_bar, z_bar   : drift-corrected coordinates (observed - drift(t))
    t_sec          : spike time in seconds
    drift_table    : the learned per-bin drift trajectory

Renders three panels:
    (1) z vs t scatter, colored by t, before correction  (smear visible)
    (2) z vs t scatter, colored by t, after correction   (should form sharp bands)
    (3) the learned drift trace Delta_z(t) (and Delta_x(t) when present)

Usage:
    python plot_drift_correction.py --npz runs/<session>/inference_joint/predictions.npz \
        --out plots/joint_<session>.png
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/scratch/ap7151/sln-v2")


def _scatter(ax, t, pos, title, cmap="viridis", s=1, alpha=0.4):
    sc = ax.scatter(t, pos, c=t, cmap=cmap, s=s, alpha=alpha, rasterized=True)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("z (um)")
    ax.set_title(title)
    return sc


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=str, required=True,
                   help="predictions.npz from a joint-loss run")
    p.add_argument("--out", type=str, default=None,
                   help="output png path (default: alongside the npz)")
    p.add_argument("--max-points", type=int, default=200000,
                   help="subsample for plot density")
    args = p.parse_args()

    d = np.load(args.npz)
    t = d["t_sec"]
    z = d["z"]
    z_bar = d["z_bar"] if "z_bar" in d.files else None
    x_bar = d["x_bar"] if "x_bar" in d.files else None
    drift = d["drift_table"] if "drift_table" in d.files else None

    if len(t) > args.max_points:
        idx = np.random.default_rng(0).choice(len(t), args.max_points, replace=False)
        idx.sort()
        t = t[idx]; z = z[idx]
        if z_bar is not None:
            z_bar = z_bar[idx]
        if x_bar is not None:
            x_bar = x_bar[idx]

    n_panels = 2 + (drift is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5),
                             constrained_layout=True)

    sc = _scatter(axes[0], t, z, "z vs t  (observed, pre-correction)")
    fig.colorbar(sc, ax=axes[0], label="t (s)", shrink=0.8)

    if z_bar is not None:
        sc = _scatter(axes[1], t, z_bar,
                      "z vs t  (drift-corrected, z_bar = z - dz(t))")
        fig.colorbar(sc, ax=axes[1], label="t (s)", shrink=0.8)
    else:
        axes[1].text(0.5, 0.5, "no z_bar in npz", ha="center", va="center",
                     transform=axes[1].transAxes)
        axes[1].set_title("(no drift-corrected z)")

    if drift is not None:
        ax = axes[2]
        n_bins = drift.shape[0]
        t_bins = np.linspace(t.min(), t.max(), n_bins)
        if drift.shape[1] >= 2:
            ax.plot(t_bins, drift[:, 1], label="dz(t)", lw=1.5)
            ax.plot(t_bins, drift[:, 0], label="dx(t)", lw=1.5, alpha=0.7)
            ax.legend()
        else:
            ax.plot(t_bins, drift[:, 0], label="dz(t)", lw=1.5)
            ax.legend()
        ax.set_xlabel("time (s)")
        ax.set_ylabel("drift (um)")
        ax.set_title("learned drift trace")
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)

    out = Path(args.out) if args.out else Path(args.npz).with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=800)
    print(f"saved {out} ({len(t)} points, drift bins={drift.shape[0] if drift is not None else 'n/a'})")


if __name__ == "__main__":
    main()
