"""Two-line plot: our diff-DREDge vs mp_dredge, on the SAME monopolar localizations."""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(a):
    sid = a.session_id
    run = a.run or f"/scratch/ap7151/sln-fixed/runs/{sid}"
    ref = a.ref or f"/scratch/ap7151/sln/runs/extra-motion/{sid}/mp_dredge/motion.npz"
    out = a.out or f"/scratch/ap7151/sln-fixed/plots/diffdredge_vs_mpdredge_{sid}.png"

    ours = np.load(f"{run}/motion_trace_mp.npy").reshape(-1)
    ts = np.load(f"{run}/motion_time_s_mp.npy").reshape(-1)
    z = np.load(ref)
    mp = np.asarray(z["disp"]).mean(axis=1)              # mp_dredge: mean of its nonrigid windows

    n = min(len(ours), len(mp), len(ts))
    ours, mp, ts = ours[:n], mp[:n], ts[:n]

    ours_c = ours - ours.mean()
    mp_c = mp - mp.mean()
    s = -1.0 if np.corrcoef(ours_c, mp_c)[0, 1] < 0 else 1.0   # DREDge sign is arbitrary
    ours_a = s * ours_c
    r = float(np.corrcoef(ours_a, mp_c)[0, 1])
    rmse = float(np.sqrt(np.mean((ours_a - mp_c) ** 2)))
    print(f"[{sid}] diff-dredge vs mp_dredge: r={r:.4f} (raw sign {'-' if s < 0 else '+'}), RMSE={rmse:.2f} um")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ts, mp_c, color="C0", lw=2.0, label="mp_dredge")
    ax.plot(ts, ours_a, color="C1", lw=1.4,
            label=f"diff-dredge (ours, sign-aligned)   r={r:.3f}, RMSE={rmse:.1f} um")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y drift (um, mean-subtracted)")
    ax.set_title(f"{sid}: diff-DREDge vs mp_dredge on the same monopolar localizations")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--session-id", default="dataset1_p1")
    ap.add_argument("--run", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--out", default="")
    main(ap.parse_args())
