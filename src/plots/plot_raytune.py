"""Generate all Ray Tune sweep analysis plots from hpo_runs/ and hpo_scratch/.

Uses per-trial sweep_analysis.csv and per-trial progress.csv files produced by
the ASHA sweep (src/models/train.py --sweep). No model or data needed.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path("/scratch/ap7151/sln-v2")

def _hp_cols(rows):
    """Dynamically discover config/* columns from the first CSV row."""
    return sorted([k for k in rows[0].keys() if k.startswith("config/")])

LABEL_MAP = {
    "config/lr": "lr",
    "config/weight_decay": "weight_decay",
    "config/feat_dim": "feat_dim",
    "config/hidden": "hidden",
    "config/num_heads": "num_heads",
    "config/pos_dim": "pos_dim",
    "config/b": "b",
    "config/knn_k": "knn_k",
    "config/gamma_1": "gamma_1 (dredge)",
    "config/gamma_2": "gamma_2 (smooth)",
    "config/temporal_window_bins": "window_bins",
    "config/sigma": "sigma (um)",
    "config/beta": "beta (softmax T)",
    "config/max_shift_bins": "max_shift_bins",
    "config/raster_subsample": "raster_subsample",
    "config/bin_width_sec": "bin_width (s)",
}


def _all_sweep_dirs():
    """Return list of (sweep_name, analysis_csv_path) found in hpo_runs."""
    found = []
    for analysis_csv in sorted((REPO / "hpo_runs").glob("*/sweep_analysis.csv")):
        found.append((analysis_csv.parent.name, analysis_csv))
    return found


def _read_sweep_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _read_progress(progress_path):
    with open(progress_path) as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _val(row):
    return float(row["val_loss"])


# ---------------------------------------------------------------------------
# Figure 1: final val_loss per trial (bar, sorted)
# ---------------------------------------------------------------------------
def plot_val_loss_per_sweep(sweep_name, rows, out_dir):
    vals = [_val(r) for r in rows]
    ids = [r["trial_id"] for r in rows]
    ids_short = [s.split("_")[-1] if "_" in s else s for s in ids]
    idx = np.argsort(vals)
    vals_s = np.array(vals)[idx]
    ids_s = [ids_short[i] for i in idx]
    colors = ["#1a9850" if v == vals_s[0] else "#4575b4" for v in vals_s]

    fig, ax = plt.subplots(figsize=(max(4.5, 0.55 * len(vals)), 3.5), constrained_layout=True)
    bars = ax.barh(range(len(vals_s)), vals_s, color=colors, edgecolor="white", linewidth=0.5)
    for bar in bars:
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                f" {bar.get_width():.4f} ", ha="left", va="center",
                fontsize=7, color="#333333")
    ax.set_yticks(range(len(vals_s)))
    ax.set_yticklabels(ids_s, fontsize=8)
    ax.set_xlabel("final val_loss (lower = better)", fontsize=10)
    ax.set_title(f"{sweep_name} — final val_loss per trial", fontsize=11, loc="left", fontweight="bold")
    ax.invert_yaxis()
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    out = out_dir / f"{sweep_name}_val_loss_per_trial.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[raytune] {out}")


# ---------------------------------------------------------------------------
# Figure 2: overlaid learning curves (train + val) per trial
# ---------------------------------------------------------------------------
def plot_learning_curves(sweep_name, rows, hpo_scratch, out_dir):
    """Overlay all trial progress.csv curves, highlight best."""
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    best_val = min(_val(r) for r in rows)
    best_tid = next(r["trial_id"] for r in rows if _val(r) == best_val)

    labeled_best = False
    for row in rows:
        tid = row["trial_id"]
        trial_dir = list(hpo_scratch.glob(f"{sweep_name}_sweep/train*{tid}*"))
        if not trial_dir:
            continue
        prog = _read_progress(trial_dir[0] / "progress.csv")
        epochs = [int(p["epoch"]) for p in prog]
        train = [float(p["train_loss"]) for p in prog]
        val = [float(p["val_loss"]) for p in prog]
        if tid == best_tid:
            ax.plot(epochs, train, "-", lw=1.6, color="#1a9850", alpha=0.9, label="train (best)")
            ax.plot(epochs, val, "-", lw=1.6, color="#d73027", alpha=0.9, label="val (best)")
        else:
            ax.plot(epochs, train, "-", lw=0.7, color="#a6dba0", alpha=0.5)
            ax.plot(epochs, val, "-", lw=0.7, color="#f4a582", alpha=0.5)

    ax.set_xlabel("epoch", fontsize=10)
    ax.set_ylabel("recon loss (raw-PTP MSE)", fontsize=10)
    ax.set_title(f"{sweep_name} — training curves", fontsize=11, loc="left", fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    out = out_dir / f"{sweep_name}_learning_curves.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[raytune] {out}")


# ---------------------------------------------------------------------------
# Figure 3: ASHA stopping histogram (completed iterations)
# ---------------------------------------------------------------------------
def plot_asha_stopping(sweep_name, rows, out_dir):
    iterations = [int(r["training_iteration"]) for r in rows]
    fig, ax = plt.subplots(figsize=(5, 3.2), constrained_layout=True)
    _, _, patches = ax.hist(iterations, bins=np.arange(1, max(iterations) + 3) - 0.5,
                              color="#4575b4", edgecolor="white")
    for patch in patches:
        h = patch.get_height()
        if h > 0:
            ax.text(patch.get_x() + patch.get_width() / 2., h,
                    f"{int(h)}", ha="center", va="bottom", fontsize=9, color="#333333")
    ax.set_xlabel("training iterations completed", fontsize=10)
    ax.set_ylabel("trials", fontsize=10)
    ax.set_title(f"{sweep_name} — ASHA early-stopping histogram", fontsize=11, loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    out = out_dir / f"{sweep_name}_asha_histogram.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[raytune] {out}")


# ---------------------------------------------------------------------------
# Figure 4: hyperparameter vs. val_loss scatter matrix
# ---------------------------------------------------------------------------
def plot_hp_vs_loss(sweep_name, rows, out_dir):
    cols = _hp_cols(rows)
    n = len(cols)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.0 * (n + 1) // 2, 5.5), constrained_layout=True)
    axes = np.array(axes).flatten()
    vals = np.array([_val(r) for r in rows])
    cmin, cmax = vals.min(), vals.max()

    for ax, col in zip(axes, cols):
        xs = np.array([float(r.get(col, np.nan)) for r in rows])
        finite = np.isfinite(xs)
        ax.scatter(xs[finite], vals[finite], s=45, c=vals[finite], cmap="viridis_r",
                   edgecolors="white", linewidths=0.6, vmin=cmin, vmax=cmax)
        ax.set_xlabel(LABEL_MAP.get(col, col), fontsize=9)
        ax.set_ylabel("val_loss", fontsize=9)
        # annotate each point with trial id
        for fi in np.where(finite)[0]:
            tid = rows[fi]["trial_id"].split("_")[-1]
            ax.annotate(tid, (xs[fi], vals[fi]), fontsize=5.5, color="#333333",
                        textcoords="offset points", xytext=(3, 3), alpha=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
        # Log scale for lr / weight_decay if range is wide
        if "lr" in col or "decay" in col:
            if xs[finite].max() / max(xs[finite].min(), 1e-12) > 50:
                ax.set_xscale("log")

    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{sweep_name} — hyperparameter sensitivity", fontsize=12, fontweight="bold")
    out = out_dir / f"{sweep_name}_hp_vs_loss.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[raytune] {out}")


# ---------------------------------------------------------------------------
# Figure 5: parallel coordinates
# ---------------------------------------------------------------------------
def _normalize(v, lo=None, hi=None):
    lo = lo if lo is not None else v.min()
    hi = hi if hi is not None else v.max()
    return (v - lo) / max(hi - lo, 1e-9)


def plot_parallel_coords(sweep_name, rows, out_dir):
    cols = _hp_cols(rows)
    n = len(cols) + 1  # +1 for val_loss axis
    n_trials = len(rows)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * n), 4.2), constrained_layout=True)
    vals = np.array([_val(r) for r in rows])
    losses_norm = _normalize(vals)

    xs = []
    for i, row in enumerate(rows):
        line = []
        for j, col in enumerate(cols):
            v = float(row.get(col, np.nan))
            col_vals = np.array([float(r.get(col, np.nan)) for r in rows])
            line.append(_normalize(np.array([v]), col_vals.min(), col_vals.max())[0])
        line.append(losses_norm[i])
        xs.append(line)
        color = plt.cm.RdYlGn_r(losses_norm[i])
        ax.plot(range(n), line, "-", lw=1.0, alpha=0.6, color=color, zorder=1)

    for j, col in enumerate(cols):
        col_vals = np.array([float(r.get(col, np.nan)) for r in rows])
        lo, hi = col_vals.min(), col_vals.max()
        ax.text(j, -0.08, f"{lo:.1e}\n{LABEL_MAP.get(col, col)}\n{hi:.1e}",
                ha="center", fontsize=7, color="#333333")
    ax.text(n - 1, -0.08, f"{vals.min():.4f}\nval_loss\n{vals.max():.4f}",
            ha="center", fontsize=7, color="#333333")

    ax.set_xticks(range(n))
    ax.set_xticklabels([LABEL_MAP.get(c, c) for c in cols] + ["val_loss"], fontsize=8)
    ax.set_ylim(-0.18, 1.05)
    ax.set_title(f"{sweep_name} — parallel coordinates", fontsize=11, loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    out = out_dir / f"{sweep_name}_parallel_coords.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[raytune] {out}")


# ---------------------------------------------------------------------------
# Figure 6: efficiency frontier (time_total_s vs val_loss)
# ---------------------------------------------------------------------------
def plot_efficiency(sweep_name, rows, out_dir):
    times = np.array([float(r["time_total_s"]) for r in rows])
    vals = np.array([_val(r) for r in rows])
    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    best_idx = vals.argmin()
    colors = ["#d73027" if i == best_idx else "#4575b4" for i in range(len(vals))]
    ax.scatter(times, vals, s=60, c=colors, edgecolors="white", linewidths=0.6)
    for i, (t, v) in enumerate(zip(times, vals)):
        tid = rows[i]["trial_id"].split("_")[-1]
        ax.annotate(tid, (t, v), fontsize=6, color="#333333",
                    textcoords="offset points", xytext=(3, 3), alpha=0.7)
    ax.set_xlabel("time_total_s", fontsize=10)
    ax.set_ylabel("val_loss", fontsize=10)
    ax.set_title(f"{sweep_name} — efficiency frontier", fontsize=11, loc="left", fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=9)
    out = out_dir / f"{sweep_name}_efficiency.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[raytune] {out}")


# ---------------------------------------------------------------------------
# Figure 7: cross-sweep best config comparison
# ---------------------------------------------------------------------------
def plot_best_config_comparison(out_dir):
    configs = {}
    for p in sorted((REPO / "hpo_runs").glob("*/best_config.json")):
        with open(p) as f:
            configs[p.parent.name] = json.load(f)
    if not configs:
        return

    all_keys = sorted({k for c in configs.values() for k in c.keys()})
    vals = []
    names = []
    for name in sorted(configs):
        names.append(name)
        vals.append([configs[name].get(k, np.nan) for k in all_keys])
    vals = np.array(vals)
    # normalize numeric keys
    for j, k in enumerate(all_keys):
        col = vals[:, j]
        finite = np.isfinite(col)
        if finite.any():
            lo, hi = col[finite].min(), col[finite].max()
            vals[finite, j] = _normalize(col[finite], lo, hi)

    fig, ax = plt.subplots(figsize=(0.8 * len(all_keys) + 2, 0.5 * len(names) + 1.5), constrained_layout=True)
    im = ax.imshow(vals, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    for i, name in enumerate(sorted(configs)):
        for j, k in enumerate(all_keys):
            raw_v = configs[name].get(k)
            if isinstance(raw_v, (int, float)) and not np.isnan(raw_v):
                txt = f"{raw_v:.1e}" if abs(raw_v) < 0.01 or abs(raw_v) > 100 else f"{raw_v:.2g}"
            else:
                txt = "—" if raw_v is None else str(raw_v)
            text_color = "w" if vals[i, j] < 0.5 else "#222222"
            ax.text(j, i, txt, ha="center", va="center", color=text_color, fontsize=6)
    ax.set_xticks(range(len(all_keys)))
    ax.set_xticklabels(all_keys, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title("Best hyperparameters across sweeps (normalized)", fontsize=11, loc="left", fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.5)
    out = out_dir / "best_config_comparison.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[raytune] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(REPO / "plots" / "raytune"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep_info = _all_sweep_dirs()
    if not sweep_info:
        print("[warn] no sweep_analysis.csv found under hpo_runs/")
        return

    for sweep_name, analysis_csv in sweep_info:
        rows = _read_sweep_csv(analysis_csv)
        if not rows:
            continue
        plot_val_loss_per_sweep(sweep_name, rows, out_dir)
        plot_learning_curves(sweep_name, rows, REPO / "hpo_scratch", out_dir)
        plot_asha_stopping(sweep_name, rows, out_dir)
        plot_hp_vs_loss(sweep_name, rows, out_dir)
        plot_parallel_coords(sweep_name, rows, out_dir)
        plot_efficiency(sweep_name, rows, out_dir)

    plot_best_config_comparison(out_dir)
    print(f"[raytune] all plots -> {out_dir}")


if __name__ == "__main__":
    main()
