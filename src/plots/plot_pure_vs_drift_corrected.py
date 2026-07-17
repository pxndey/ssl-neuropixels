"""Compare pure localizer vs drift-corrected localizations (2x1 plot).

Reads:
    - runs/<session>/inference/localizations.npy (pure model)
    - runs/<session>/inference_joint/predictions.npz (drift-corrected model)

Renders 2x1 subplot:
    Left:  Pure localization (no drift correction)
    Right: Drift-corrected localization (z_bar = z - dz(t))

Both panels show x vs z scatter colored by spike time (green=early -> red=late).
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path("/scratch/ap7151/sln-v2")
SESSIONS = ["dataset1_p1", "dataset1_p2"]

GREEN_RED = LinearSegmentedColormap.from_list("green_red", ["#1a9850", "#d73027"])


def load_pure_localizations(session):
    """Load pure model localizations."""
    d = REPO / "runs" / session / "inference"
    loc = np.load(d / "localizations.npy")
    idx = np.load(d / "test_indices.npy")
    stimes = np.load(REPO / "runs" / session / "spike_times.npy", mmap_mode="r")
    t = np.asarray(stimes[idx], dtype=np.float64)
    return loc[:, [0, 1, 2]].astype(np.float64), t


def load_drift_corrected(session):
    """Load drift-corrected predictions."""
    d = REPO / "runs" / session / "inference_joint" / "predictions.npz"
    data = np.load(d)
    x_bar = data["x_bar"] if "x_bar" in data.files else data["x"]
    z_bar = data["z_bar"] if "z_bar" in data.files else data["z"]
    t = data["t_sec"]
    xyz = np.stack([x_bar, np.zeros_like(x_bar), z_bar], axis=1)
    return xyz.astype(np.float64), t.astype(np.float64)


def sample(n, k, seed):
    if k >= n:
        return np.arange(n)
    return np.random.default_rng(seed).choice(n, size=k, replace=False)


def norm_time(t):
    return (t - t.min()) / max(t.max() - t.min(), 1e-9)


def robust_limits(arrays, lo=0.5, hi=99.5, margin=0.05):
    a = np.concatenate(arrays)
    p0, p1 = np.percentile(a, [lo, hi])
    if p1 <= p0:
        p1 = p0 + 1.0
    pad = (p1 - p0) * margin
    return p0 - pad, p1 + pad


def panel(ax, X, Z, t_norm, title, xlim, zlim):
    sc = ax.scatter(X, Z, c=t_norm, cmap=GREEN_RED, norm=Normalize(0.0, 1.0),
                    s=0.5, alpha=0.35, linewidths=0, rasterized=True, zorder=2)
    ax.set_title(title, fontsize=14, loc="left", fontweight="bold")
    ax.set_xlabel("x — lateral (µm)", fontsize=11)
    ax.set_ylabel("z — depth (µm)", fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*zlim)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return sc


def render_comparison(session, n_sample, seed, out_dir):
    pure_xyz, pure_t = load_pure_localizations(session)
    drift_xyz, drift_t = load_drift_corrected(session)

    mi = sample(len(pure_xyz), n_sample, seed)
    di = sample(len(drift_xyz), n_sample, seed)

    pure_xyz, pure_t = pure_xyz[mi], norm_time(pure_t[mi])
    drift_xyz, drift_t = drift_xyz[di], norm_time(drift_t[di])

    xlim = robust_limits([pure_xyz[:, 0], drift_xyz[:, 0]])
    zlim = robust_limits([pure_xyz[:, 2], drift_xyz[:, 2]])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    sc = panel(axes[0], pure_xyz[:, 0], pure_xyz[:, 2], pure_t,
               "Pure localizer (no drift correction)", xlim, zlim)
    sc = panel(axes[1], drift_xyz[:, 0], drift_xyz[:, 2], drift_t,
               "Drift-corrected (joint fine-tune)", xlim, zlim)

    fig.suptitle(f"{session} — x vs z localizations", fontsize=13)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.6, pad=0.02)
    cbar.set_label("spike time (early → late)", fontsize=9)
    cbar.set_ticks([])

    out_dir = out_dir / session
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pure_vs_drift_corrected.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[plot] {session}: saved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--n-sample", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "plots" / "drift_comparison"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for session in args.sessions:
        render_comparison(session, args.n_sample, args.seed, out_dir)


if __name__ == "__main__":
    main()
