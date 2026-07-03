"""SLNv2 (model) vs monopolar ("true") per-spike localization agreement, y=x.

Grid: rows = session, cols = physical axis (lateral / depth / dist-from-probe).
Each panel scatters the SLNv2 coordinate (x-axis) vs the monopolar coordinate
(y-axis) over the held-out TEST spikes both methods localized, colored by spike
time (green = early -> red = late, alpha 0.4), with a black y=x line + Pearson r.

Column order differs between the two: SLNv2 localizations.npy = [lateral, perp,
depth, alpha]; monopolar_true localizations.npy = [lateral, depth, perp, alpha]
(SpikeInterface order). Both are mapped to the same physical axis below.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

REPO = Path("/scratch/ap7151/sln-v2")
SESSIONS = ["dataset1_p1", "dataset1_p2", "dandi_000957_sub-ZYE-0021_ses-1"]
# (physical axis label, SLNv2 column, monopolar column)
AXES = [("lateral", 0, 0), ("depth", 2, 1), ("dist-from-probe", 1, 2)]
GREEN_RED = LinearSegmentedColormap.from_list("green_red", ["#1a9850", "#d73027"])


def robust_lim(a, b, lo=0.5, hi=99.5, margin=0.05):
    v = np.concatenate([a, b])
    p0, p1 = np.percentile(v, [lo, hi])
    if p1 <= p0:
        p1 = p0 + 1.0
    pad = (p1 - p0) * margin
    return p0 - pad, p1 + pad


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inference-name", default="inference",
                    help="runs/<session>/<inference-name>/ holding the SLNv2 localizations")
    ap.add_argument("--sessions", nargs="*", default=SESSIONS)
    ap.add_argument("--n-sample", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "plots" / "true_vs_model.png"))
    args = ap.parse_args()
    sessions = args.sessions

    fig, axes = plt.subplots(len(sessions), len(AXES), figsize=(13.5, 4.5 * len(sessions)),
                             constrained_layout=True, squeeze=False)
    for r, sess in enumerate(sessions):
        inf = REPO / "runs" / sess / args.inference_name / "localizations.npy"
        tix = REPO / "runs" / sess / args.inference_name / "test_indices.npy"
        mon = REPO / "runs" / sess / "monopolar_true" / "localizations.npy"
        have = inf.exists() and tix.exists() and mon.exists()
        if have:
            sln = np.load(inf).astype(np.float64)
            test_idx = np.load(tix)
            mono = np.load(mon)[test_idx].astype(np.float64)
            st = np.load(REPO / "runs" / sess / "spike_times.npy", mmap_mode="r")
            t = np.asarray(st[test_idx], dtype=np.float64)
            valid = np.flatnonzero(np.isfinite(mono).all(axis=1))    # boundary-valid spikes
            if valid.size > args.n_sample:
                sel = np.sort(np.random.default_rng(args.seed).choice(valid, args.n_sample, replace=False))
            else:
                sel = valid
            ts = t[sel]
            t_norm = (ts - ts.min()) / max(ts.max() - ts.min(), 1e-9)
        for c, (name, sc, mc) in enumerate(AXES):
            ax = axes[r][c]
            if not have:
                ax.text(0.5, 0.5, "(missing)", ha="center", va="center", fontsize=11)
                ax.set_axis_off()
                continue
            xf, yf = sln[valid, sc], mono[valid, mc]
            rp = np.corrcoef(xf, yf)[0, 1] if valid.size > 2 else float("nan")
            lim = robust_lim(xf, yf)
            ax.scatter(sln[sel, sc], mono[sel, mc], c=t_norm, cmap=GREEN_RED,
                       norm=Normalize(0.0, 1.0), s=2.0, alpha=0.4, linewidths=0, rasterized=True)
            ax.plot(lim, lim, "k--", lw=1.0, zorder=3)
            ax.set_xlim(*lim)
            ax.set_ylim(*lim)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8)
            ax.set_title(f"{name}   r={rp:.2f}  (n={valid.size:,})", fontsize=10)
            if r == len(sessions) - 1:
                ax.set_xlabel("SLNv2 (µm)", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{sess}\nmonopolar (µm)", fontsize=9)

    fig.suptitle(f"SLNv2 [{args.inference_name}] vs monopolar (\"true\") — per-spike, test set",
                 fontsize=13)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=800)
    print(f"[true-vs-model] -> {out}", flush=True)


if __name__ == "__main__":
    main()
