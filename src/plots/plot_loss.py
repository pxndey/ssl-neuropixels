"""Plot train/val loss trends parsed from train_localizer slurm logs.

Scans slurm_logs/train_localizer_*.out, reads the per-epoch `[epoch ...]` lines
and the `[final] ... test_loss=` line, and renders one panel per session
(session name taken from the `[data] session=...` line inside each log). Losses
are raw-PTP MSE, so per-session y-scales differ wildly -> one panel each.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/scratch/ap7151/sln-v2")
LOGS = REPO / "slurm_logs"

RE_SESSION = re.compile(r"\[data\] session=(\S+)")
RE_EPOCH = re.compile(r"\[epoch (\d+)\] train_loss=([\d.eE+-]+) val_loss=([\d.eE+-]+)")
RE_FINAL = re.compile(r"\[final\].*test_loss=([\d.eE+-]+)")


def parse_log(path):
    text = path.read_text(errors="ignore")
    ms = RE_SESSION.search(text)
    session = ms.group(1) if ms else path.stem
    epochs, tr, va = [], [], []
    for m in RE_EPOCH.finditer(text):
        epochs.append(int(m.group(1)))
        tr.append(float(m.group(2)))
        va.append(float(m.group(3)))
    mf = RE_FINAL.search(text)
    test = float(mf.group(1)) if mf else None
    return session, epochs, tr, va, test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-glob", default="train_localizer_*.out")
    ap.add_argument("--out", default=str(REPO / "plots" / "loss_trends.png"))
    args = ap.parse_args()

    runs = []
    for p in sorted(LOGS.glob(args.logs_glob)):
        session, epochs, tr, va, test = parse_log(p)
        if epochs:
            runs.append((session, epochs, tr, va, test))
    # de-duplicate by session (keep the most-recent / longest run)
    by_session = {}
    for r in runs:
        by_session[r[0]] = r
    runs = [by_session[k] for k in sorted(by_session)]
    if not runs:
        print("No parseable train_localizer logs found.")
        return

    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.2), constrained_layout=True,
                             squeeze=False)
    for ax, (session, epochs, tr, va, test) in zip(axes[0], runs):
        ax.plot(epochs, tr, "-o", ms=3, lw=1.6, color="#1f77b4", label="train")
        ax.plot(epochs, va, "-o", ms=3, lw=1.6, color="#d62728", label="val")
        if test is not None:
            ax.axhline(test, ls="--", lw=1.0, color="#555555",
                       label=f"test={test:.1f}")
        ax.set_title(session, fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("epoch")
        ax.set_ylabel("recon loss (raw-PTP MSE)")
        ax.legend(fontsize=9, frameon=False)
        ax.tick_params(labelsize=9)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[loss] {n} session(s) -> {out}")
    for session, epochs, tr, va, test in runs:
        print(f"  {session}: {len(epochs)} epochs, "
              f"train {tr[0]:.1f}->{tr[-1]:.1f}, val {va[0]:.1f}->{va[-1]:.1f}, test={test}")


if __name__ == "__main__":
    main()
