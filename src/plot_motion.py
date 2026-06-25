"""Compare the DREDge motion estimate against the manipulator ground truth.

The "extra-motion" recordings impose a known probe displacement with a
micromanipulator: ``manip.positions.npy`` (microns) sampled at
``manip.timestamps_p<partition>.npy`` (seconds) -- a step function. We overlay
that on our estimated trace ``motion_trace.npy`` / ``motion_time_s.npy``.

DREDge motion is relative (zero-referenced) and its sign convention can be
opposite to the manipulator's, so both traces are mean-subtracted and the
estimate is sign-aligned to the ground truth (the chosen sign is reported).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(args) -> None:
    run = Path(args.session)
    raw = Path(args.raw)

    est = np.load(run / "motion_trace.npy").reshape(-1)          # (T,) rigid -> single trace
    ts = np.load(run / "motion_time_s.npy").reshape(-1)          # (T,)

    pos = np.load(raw / "manip.positions.npy").reshape(-1)       # (K,) microns
    tstamp = np.load(raw / f"manip.timestamps_p{args.partition}.npy").reshape(-1)

    # ground-truth step function evaluated on our time grid
    idx = np.clip(np.searchsorted(tstamp, ts, side="right") - 1, 0, len(pos) - 1)
    gt = pos[idx]

    gt_c = gt - gt.mean()
    est_c = est - est.mean()
    r_raw = float(np.corrcoef(est_c, gt_c)[0, 1])
    sign = -1.0 if r_raw < 0 else 1.0
    est_a = sign * est_c
    r = float(np.corrcoef(est_a, gt_c)[0, 1])
    rmse = float(np.sqrt(np.mean((est_a - gt_c) ** 2)))

    print(f"correlation (sign-aligned) r = {r:.3f}  [raw sign {'-' if sign<0 else '+'}]")
    print(f"RMSE = {rmse:.2f} um   imposed step = {pos.max()-pos.min():.0f} um")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ts, gt_c, color="k", lw=2.0, drawstyle="steps-post",
            label="manipulator ground truth (imposed)")
    ax.plot(ts, est_a, color="C1", lw=1.2, alpha=0.9,
            label=f"DREDge estimate (sign-aligned, r={r:.2f}, RMSE={rmse:.1f} um)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("y displacement (um, mean-subtracted)")
    ax.set_title(f"{run.name}: imposed vs estimated probe drift (y)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = run / args.out
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=str, default="/scratch/ap7151/sln-fixed/runs/dataset1_p1")
    p.add_argument("--raw", type=str, default="/scratch/ap7151/RAW_DATA/extra-motion/dataset1_p1")
    p.add_argument("--partition", type=str, default="1")
    p.add_argument("--out", type=str, default="motion_compare.png")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
