"""1x3 spike depth rasters: raw monopolar | MP+DREDGE | MP+diff-DREDGE.

Same spikes/order/colormap in every panel; only the per-spike depth differs:
  1. raw       -- monopolar localization depth (localizations col 1)
  2. MP+DREDGE -- sln glcache/mp_dredge.npy depth (their dredge_ap correction)
  3. MP+diff-DREDGE -- raw depth minus our diff-DREDge Delta-y(t) (sign-aligned to mp)
"""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FS = 30000.0


def main(a):
    sid = a.session_id
    locdir = f"/scratch/ap7151/sln/runs/extra-motion/{sid}/preprocessed/shank_0"
    gldir = f"/scratch/ap7151/sln/runs/extra-motion/{sid}/glcache"
    mpz = f"/scratch/ap7151/sln/runs/extra-motion/{sid}/mp_dredge/motion.npz"
    ours_npy = f"/scratch/ap7151/sln-fixed/runs/{sid}/motion_trace_mp.npy"
    out = a.out or f"/scratch/ap7151/sln-fixed/plots/raster3_{sid}.png"

    loc = np.load(f"{locdir}/localizations.npy")
    st = np.load(f"{locdir}/spike_times.npy").astype(np.float64)
    raw_depth = loc[:, 1].astype(np.float32)
    alpha = np.abs(loc[:, 3]).astype(np.float64)
    t = st / FS

    mp_depth = np.load(f"{gldir}/mp_dredge.npy")[:, 1].astype(np.float32)

    # our rigid diff-DREDge trace, applied per time bin, sign-aligned to mp_dredge
    P = np.load(ours_npy).reshape(-1)
    mp_mean = np.asarray(np.load(mpz)["disp"]).mean(axis=1)
    s = -1.0 if np.corrcoef(P[:len(mp_mean)], mp_mean)[0, 1] < 0 else 1.0
    tbin = np.clip((st // (a.bin_s * FS)).astype(np.int64), 0, len(P) - 1)
    ours_depth = (raw_depth - s * P[tbin]).astype(np.float32)

    glraw = np.load(f"{gldir}/raw.npy")[:, 1].astype(np.float32)
    if not np.allclose(glraw, raw_depth, atol=1e-3):
        print("WARNING: glcache/raw depth != localizations depth (order mismatch?)")

    N = t.shape[0]
    rng = np.random.default_rng(a.seed)
    idx = np.arange(N) if a.n_plot >= N else np.sort(rng.choice(N, a.n_plot, replace=False))
    idx = idx[np.argsort(alpha[idx])]                     # big-amplitude on top
    c = np.log10(np.clip(alpha[idx], 1e-6, None))
    cmin, cmax = np.percentile(c, [1, 99])
    ti = t[idx]

    panels = [
        (f"raw monopolar (corr sign s={'+' if s > 0 else '-'})", raw_depth),
        ("monopolar + DREDGE (mp_dredge)", mp_depth),
        ("monopolar + diff-DREDGE (ours)", ours_depth),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), sharey=True, constrained_layout=True)
    sc = None
    for ax, (name, depth) in zip(axes, panels):
        sc = ax.scatter(ti, depth[idx], c=c, cmap="viridis", vmin=cmin, vmax=cmax,
                        s=0.3, alpha=0.4, linewidths=0, rasterized=True)
        ax.set_title(name, fontsize=13, loc="left")
        ax.set_xlabel("time [s]", fontsize=12)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("depth y [µm]", fontsize=12)
    cb = fig.colorbar(sc, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label("log₁₀ amplitude", fontsize=11)
    fig.suptitle(f"{sid}: spike depth raster ({idx.size:,} spikes)", fontsize=15)

    fig.savefig(out, dpi=a.dpi)
    print("saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", default="dataset1_p1")
    ap.add_argument("--bin-s", type=float, default=1.0)
    ap.add_argument("--n-plot", type=int, default=900_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--out", default="")
    main(ap.parse_args())
