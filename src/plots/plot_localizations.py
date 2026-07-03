"""Plot xy / yz / zx localization scatters for the trained localizer models.

For each session, three projections of the self-supervised (SLNv2) test-set
localizations (runs/<session>/inference/localizations.npy, probe-global). For the
`dataset*` sessions we also load the raw monopolar-triangulation localizations
from the sibling `sln` project and render a 1x2 comparison (SLNv2 | MP) per
projection; the `dandi` (NP Ultra) session has no monopolar reference, so it gets
a single panel per projection.

Unified coordinate convention (both methods mapped into it):
    x = lateral (probe width, um)
    y = distance from probe (perpendicular, um)
    z = depth (probe long axis, um)

    mine  localizations.npy (N,4) = [lateral, perp, depth, alpha]  -> [:, [0,1,2]]
    mono  peak_locations.npy (N,3) = [lateral, depth, perp]        -> [:, [0,2,1]]

Points are colored by spike time (green=early -> red=late), min-max normalized
per method. Style follows 09_compares/code/plots/localization_vertical.py.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path("/scratch/ap7151/sln-v2")
SLN_EXTRA = Path("/scratch/ap7151/sln/runs/extra-motion")

SESSIONS = ["dataset1_p1", "dataset1_p2", "dandi_000957_sub-ZYE-0021_ses-1"]

GREEN_RED = LinearSegmentedColormap.from_list("green_red", ["#1a9850", "#d73027"])

# (axis_i, axis_j) into the unified (x=lateral, y=perp, z=depth) space
PROJECTIONS = {"xy": (0, 1), "yz": (1, 2), "zx": (2, 0)}
AXIS_LABEL = {
    0: "x — lateral (µm)",
    1: "y — dist. from probe (µm)",
    2: "z — depth (µm)",
}
# projections kept side-by-side (1xN); all others stack vertically (Nx1).
# yz is dist-from-probe x depth (very tall/narrow) -> side-by-side reads better.
HORIZONTAL_PROJ = {"yz"}


def load_mine(session):
    d = REPO / "runs" / session / "inference"
    loc = np.load(d / "localizations.npy")               # (N,4) [lateral, perp, depth, alpha]
    idx = np.load(d / "test_indices.npy")
    stimes = np.load(REPO / "runs" / session / "spike_times.npy", mmap_mode="r")
    t = np.asarray(stimes[idx], dtype=np.float64)
    return loc[:, [0, 1, 2]].astype(np.float64), t


def load_mono(session):
    d = SLN_EXTRA / session / "compare_loc" / "monopolar"
    if not (d / "peak_locations.npy").exists():
        return None, None
    loc = np.load(d / "peak_locations.npy")              # (N,3) [lateral, depth, perp]
    t = np.load(d / "peak_sample_index.npy").astype(np.float64)
    return loc[:, [0, 2, 1]].astype(np.float64), t       # -> [lateral, perp, depth]


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


def panel(ax, X, Y, t_norm, title, xlabel, ylabel, xlim, ylim):
    sc = ax.scatter(X, Y, c=t_norm, cmap=GREEN_RED, norm=Normalize(0.0, 1.0),
                    s=0.5, alpha=0.35, linewidths=0, rasterized=True, zorder=2)
    ax.set_title(title, fontsize=15, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")           # 1 um on x == 1 um on y (true spatial scale)
    ax.tick_params(labelsize=9)
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return sc


def fig_size(n_panels, xlim, ylim, vertical, long_in=8.0):
    """Figure size (inches); per-panel box aspect matches the data extent (longer
    data axis spans `long_in`), plus label/colorbar/title padding.
    vertical=True -> n_panels x 1 stack; vertical=False -> 1 x n_panels row."""
    w = xlim[1] - xlim[0]
    h = ylim[1] - ylim[0]
    ipu = long_in / max(w, h)        # inches per um
    pw, ph = w * ipu, h * ipu
    if vertical:
        return (pw + 2.6, n_panels * ph + 1.4 + 0.7 * n_panels)
    return (n_panels * pw + 2.6, ph + 1.9)


def render_session(session, n_sample, seed, out_dir):
    mine_xyz, mine_t = load_mine(session)
    mono_xyz, mono_t = load_mono(session)

    mi = sample(len(mine_xyz), n_sample, seed)
    methods = [("SLNv2", mine_xyz[mi], norm_time(mine_t[mi]))]
    if mono_xyz is not None:
        oi = sample(len(mono_xyz), n_sample, seed)
        methods.append(("MP", mono_xyz[oi], norm_time(mono_t[oi])))

    out_dir = out_dir / session
    out_dir.mkdir(parents=True, exist_ok=True)

    for proj, (ai, aj) in PROJECTIONS.items():
        xlim = robust_limits([m[1][:, ai] for m in methods])
        ylim = robust_limits([m[1][:, aj] for m in methods])
        n = len(methods)
        vertical = proj not in HORIZONTAL_PROJ
        if vertical:
            fig, axes = plt.subplots(n, 1, figsize=fig_size(n, xlim, ylim, True),
                                     constrained_layout=True, squeeze=False)
            axlist = axes[:, 0]
        else:
            fig, axes = plt.subplots(1, n, figsize=fig_size(n, xlim, ylim, False),
                                     constrained_layout=True, squeeze=False)
            axlist = axes[0]
        sc = None
        for ax, (label, xyz, t) in zip(axlist, methods):
            sc = panel(ax, xyz[:, ai], xyz[:, aj], t, label,
                       AXIS_LABEL[ai], AXIS_LABEL[aj], xlim, ylim)
        fig.suptitle(f"{session}  —  {proj} localizations", fontsize=13)
        cbar = fig.colorbar(sc, ax=axlist.tolist(), shrink=0.6, pad=0.01)
        cbar.set_label("spike time (early → late)", fontsize=9)
        cbar.set_ticks([])

        out = out_dir / f"{proj}.png"
        fig.savefig(out, dpi=800)
        plt.close(fig)
        print(f"[plot] {session}: {proj} ({n} panel{'s' if n > 1 else ''}) -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--n-sample", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "plots" / "localizations"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    for session in args.sessions:
        render_session(session, args.n_sample, args.seed, out_dir)


if __name__ == "__main__":
    main()
