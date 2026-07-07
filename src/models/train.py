"""Train the self-supervised monopolar-triangulation localizer, with optional Ray Tune sweep.

Exposes `run_training` (the shared train/val/test loop, importable by sweep mode) and
`split_train_val_test_indices`, plus an argparse CLI whose hyperparameters follow the precedence:

    built-in preset defaults  <  --config-json (a sweep's best_config.json)  <  explicit CLI flags

Sweep mode (--sweep) runs ~8 trials on a truncated session, selecting on val_loss, then writes
to <repo>/hpo_runs/<model_type>/best_config.json (and best_result.json, sweep_analysis.csv).
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (
    SetLocalizer,
    NP12_CONFIG,
    NPULTRA_CONFIG,
    fourier_positional_embedding,
    build_knn_attention_mask,
    physics_forward,
    compute_feature,
    masked_recon_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"np12": NP12_CONFIG, "npultra": NPULTRA_CONFIG}

REPRESENTATIVE_SESSION = {
    "np12": "runs/dataset1_p1",
    "npultra": "runs/dandi_000957_sub-ZYE-0021_ses-1",
}


class SpikeNeighborhoodDataset(Dataset):
    """Wraps one session's mmap'd `runs/<session>/*.npy` arrays.

    `__getitem__(i)` slices the `n = neighbor_counts[i]` real channels into a
    fixed-length-`fixed_n` waveform/coords tensor (real `[0:n)`, zero pad
    `[n:fixed_n)`), and a boolean `mask` (True = real). Works whether the
    shortfall comes from a session's file-level `M < fixed_n` or a spike's own
    `neighbor_counts < M`.
    """

    def __init__(self, session_path, fixed_n, normalize=False):
        self.session_path = Path(session_path)
        self.fixed_n = int(fixed_n)
        self.normalize = bool(normalize)
        self.waveforms = np.load(self.session_path / "neighborhood_waveforms.npy", mmap_mode="r")
        self.local_coords = np.load(self.session_path / "local_coords.npy", mmap_mode="r")
        self.neighbor_counts = np.load(self.session_path / "neighbor_counts.npy", mmap_mode="r")
        self.centroids = np.load(self.session_path / "centroids.npy", mmap_mode="r")
        self.n_spikes = self.waveforms.shape[0]
        self.M = self.waveforms.shape[1]
        self.n_samples = self.waveforms.shape[2]
        if self.M > self.fixed_n:
            print(f"[warn] {self.session_path.name}: file M={self.M} > fixed_n="
                  f"{self.fixed_n}; channels beyond {self.fixed_n} are truncated",
                  flush=True)

    def __len__(self):
        return self.n_spikes

    def __getitem__(self, i):
        fn = self.fixed_n
        m = min(int(self.neighbor_counts[i]), self.M, fn)
        wf = np.zeros((fn, self.n_samples), dtype=np.float32)
        coords = np.zeros((fn, 2), dtype=np.float32)
        wf[:m] = self.waveforms[i, :m, :]
        coords[:m] = self.local_coords[i, :m, :]
        mask = np.zeros(fn, dtype=bool)
        mask[:m] = True
        if self.normalize:
            ptp = wf.max(axis=1) - wf.min(axis=1)
            scale = float(ptp.max())
            wf = wf / (scale if scale > 1e-6 else 1.0)
        centroid = np.array(self.centroids[i], dtype=np.float32)
        return (
            torch.from_numpy(wf),
            torch.from_numpy(coords),
            torch.from_numpy(mask),
            torch.from_numpy(centroid),
        )


def collate_fn(batch):
    wf = torch.stack([b[0] for b in batch])
    coords = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    centroids = torch.stack([b[3] for b in batch])
    return wf, coords, mask, centroids


def split_train_val_test_indices(n_spikes, val_frac, test_frac, seed, max_spikes=None):
    """Seeded per-spike 80/10/10-style split. `max_spikes` truncates the
    permutation *before* splitting, so a truncated sweep keeps the fractions."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_spikes)
    if max_spikes is not None and max_spikes < n_spikes:
        perm = perm[:max_spikes]
    n_val = int(len(perm) * val_frac)
    n_test = int(len(perm) * test_frac)
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]


def _forward_batch(model, cfg, wf, coords, mask, device):
    wf = wf.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    xc, zc = coords[..., 0], coords[..., 1]
    pos_emb = fourier_positional_embedding(xc, zc, cfg["pos_dim"], cfg["max_freq"])
    knn = build_knn_attention_mask(xc, zc, mask, k=cfg.get("knn_k", 16)) if cfg["use_knn"] else None
    x, y, z, alpha = model(wf, pos_emb, mask, knn_allowed=knn)
    ptp_pred = physics_forward(x, y, z, alpha, xc, zc, cfg["b"])
    ptp_true = compute_feature(wf, cfg.get("recon_feature", "ptp"))
    loss = masked_recon_loss(ptp_true, ptp_pred, mask)
    return loss, (x, y, z, alpha)


def _run_epoch(model, cfg, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, count = 0.0, 0
    torch.set_grad_enabled(train)
    for wf, coords, mask, _centroids in loader:
        loss, _ = _forward_batch(model, cfg, wf, coords, mask, device)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        bs = wf.shape[0]
        total += loss.item() * bs
        count += bs
    torch.set_grad_enabled(True)
    return total / max(count, 1)


def run_training(cfg, session_path, model_type, epochs, device, val_frac,
                 test_frac, seed, batch_size, max_spikes=None, num_workers=4,
                 checkpoint_path=None, save_predictions_path=None, report_fn=None):
    """Shared train/val/test loop. Returns {train_loss, val_loss, test_loss}.

    `report_fn`, if given, is called once per epoch with a metrics dict (the
    ASHA hook used by the sweep); `None` in plain CLI mode.
    """
    device = torch.device(device)
    ds = SpikeNeighborhoodDataset(session_path, fixed_n=cfg["n_channels"],
                                  normalize=cfg.get("normalize", False))
    train_idx, val_idx, test_idx = split_train_val_test_indices(
        len(ds), val_frac, test_frac, seed, max_spikes=max_spikes)

    def loader(indices, shuffle):
        return DataLoader(
            Subset(ds, indices), batch_size=batch_size, shuffle=shuffle,
            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True,
            persistent_workers=(num_workers > 0), drop_last=False)

    train_loader = loader(train_idx, True)
    val_loader = loader(val_idx, False)
    test_loader = loader(test_idx, False)

    print(f"[data] session={Path(session_path).name} M={ds.M} fixed_n={cfg['n_channels']} "
          f"n_spikes={len(ds)} train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
          flush=True)
    print(f"[cfg] {json.dumps(cfg, sort_keys=True)}", flush=True)

    model = SetLocalizer(
        n_channels=cfg["n_channels"], n_samples=cfg["n_samples"],
        pos_dim=cfg["pos_dim"], feat_dim=cfg["feat_dim"], hidden=cfg["hidden"],
        num_heads=cfg["num_heads"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])

    train_loss = val_loss = float("nan")
    for epoch in range(epochs):
        train_loss = _run_epoch(model, cfg, train_loader, device, optimizer)
        val_loss = _run_epoch(model, cfg, val_loader, device)
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}",
              flush=True)
        if report_fn is not None:
            report_fn({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "cfg": cfg,
                    "model_type": model_type, "session_id": Path(session_path).name},
                   checkpoint_path)
        print(f"[checkpoint] saved {checkpoint_path}", flush=True)

    if save_predictions_path is not None:
        _save_predictions(model, cfg, ds, batch_size, num_workers, device,
                          save_predictions_path)

    test_loss = _run_epoch(model, cfg, test_loader, device)
    print(f"[final] train_loss={train_loss:.6f} val_loss={val_loss:.6f} "
          f"test_loss={test_loss:.6f}", flush=True)
    if report_fn is not None:
        report_fn({"train_loss": train_loss, "val_loss": val_loss,
                   "test_loss": test_loss, "epoch": epochs})

    return {"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}


def _save_predictions(model, cfg, ds, batch_size, num_workers, device, out_path):
    """Post-hoc pass over the full session; adds `centroids` back to report
    absolute probe-frame (x, y, z, alpha)."""
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=num_workers,
                        pin_memory=True)
    xs, ys, zs, alphas = [], [], [], []
    torch.set_grad_enabled(False)
    for wf, coords, mask, centroids in loader:
        _, (x, y, z, alpha) = _forward_batch(model, cfg, wf, coords, mask, device)
        xs.append((x.cpu() + centroids[:, 0]).numpy())
        zs.append((z.cpu() + centroids[:, 1]).numpy())
        ys.append(y.cpu().numpy())
        alphas.append(alpha.cpu().numpy())
    torch.set_grad_enabled(True)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, x=np.concatenate(xs), y=np.concatenate(ys),
             z=np.concatenate(zs), alpha=np.concatenate(alphas))
    print(f"[predictions] saved {out_path} ({sum(len(a) for a in xs)} spikes)", flush=True)


# -----------------------------------------------------------------------------
# Sweep mode (Ray Tune ASHA)
# -----------------------------------------------------------------------------

COMMON_SEARCH_SPACE = {
    "lr": None,  # filled by --choices flags
    "weight_decay": None,
    "feat_dim": None,
    "hidden": None,
    "num_heads": None,
    "pos_dim": None,
    "b": None,
}


def _sweep_trainable(config, base_cfg=None, models_dir=None, session_path=None,
                     model_type=None, epochs=None, val_frac=None, test_frac=None,
                     seed=None, batch_size=None, max_spikes=None, num_workers=None):
    import sys as _sys
    if models_dir and models_dir not in _sys.path:
        _sys.path.insert(0, models_dir)
    import torch as _torch
    from train import run_training  # self-import in spawned worker

    def report_fn(metrics):
        from ray import tune as _raytune
        _raytune.report(metrics)

    cfg = dict(base_cfg)
    cfg.update(config)
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    run_training(
        cfg=cfg, session_path=session_path, model_type=model_type, epochs=epochs,
        device=device, val_frac=val_frac, test_frac=test_frac, seed=seed,
        batch_size=batch_size, max_spikes=max_spikes, num_workers=num_workers,
        report_fn=report_fn)


def _run_sweep(args):
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    model_type = args.model_type
    session_rel = args.session_path or REPRESENTATIVE_SESSION[model_type]
    session_path = str(session_rel if os.path.isabs(session_rel) else REPO_ROOT / session_rel)
    n_gpus = torch.cuda.device_count()

    ray.init(num_cpus=args.cpus_per_trial, num_gpus=max(n_gpus, 1),
             ignore_reinit_error=True, include_dashboard=False)

    scheduler = ASHAScheduler(
        time_attr="training_iteration", metric="val_loss", mode="min",
        max_t=args.epochs, grace_period=args.grace_period, reduction_factor=2)

    search_space = dict(COMMON_SEARCH_SPACE)
    search_space["lr"] = tune.choice(args.lr_choices or [1e-4, 3e-4, 1e-3, 3e-3])
    search_space["weight_decay"] = tune.choice(args.weight_decay_choices or [0.0, 1e-5, 1e-4])
    search_space["feat_dim"] = tune.choice(args.feat_dim_choices or [16, 32, 64])
    search_space["hidden"] = tune.choice(args.hidden_choices or [64, 128, 256])
    search_space["num_heads"] = tune.choice(args.num_heads_choices or [2, 4])
    search_space["pos_dim"] = tune.choice(args.pos_dim_choices or [4, 8, 16])
    search_space["b"] = tune.choice(args.b_choices or [0.5, 1.0, 2.0])
    if model_type == "npultra":
        search_space["knn_k"] = tune.choice(args.knn_k_choices or [8, 16, 32])

    trainable_with_params = tune.with_parameters(
        _sweep_trainable, base_cfg=CONFIGS[model_type], models_dir=str(Path(__file__).parent),
        session_path=session_path, model_type=model_type, epochs=args.epochs,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
        batch_size=args.batch_size, max_spikes=args.max_spikes, num_workers=args.num_workers)

    storage_path = str(REPO_ROOT / "hpo_scratch")
    print(f"[sweep] model_type={model_type} session={session_path} "
          f"num_samples={args.num_samples} max_spikes={args.max_spikes} "
          f"gpus={n_gpus} storage={storage_path}", flush=True)

    analysis = tune.run(
        trainable_with_params,
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": 1},
        storage_path=storage_path,
        name=f"{model_type}_sweep",
        verbose=1,
    )

    best_config = analysis.get_best_config(metric="val_loss", mode="min", scope="all")
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
    ma = best_trial.metric_analysis

    out_dir = REPO_ROOT / "hpo_runs" / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, sort_keys=True)

    best_result = {
        "val_loss": ma.get("val_loss", {}).get("min"),
        "train_loss": ma.get("train_loss", {}).get("last"),
        "test_loss": ma.get("test_loss", {}).get("last"),
        "trial_id": best_trial.trial_id,
        "session": session_path,
        "model_type": model_type,
        "seed": args.seed,
        "max_spikes": args.max_spikes,
        "num_samples": args.num_samples,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "best_result.json", "w") as f:
        json.dump(best_result, f, indent=2)

    try:
        analysis.results_df.to_csv(out_dir / "sweep_analysis.csv")
    except Exception as e:
        print(f"[warn] could not write sweep_analysis.csv: {e}", flush=True)

    print(f"[sweep] best_config={json.dumps(best_config, sort_keys=True)}", flush=True)
    print(f"[sweep] best_result={json.dumps(best_result)}", flush=True)
    print(f"[sweep] wrote {out_dir}/best_config.json, best_result.json, sweep_analysis.csv",
          flush=True)
    ray.shutdown()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_cfg(model_type, args):
    cfg = dict(CONFIGS[model_type])
    if args.config_json:
        with open(args.config_json) as f:
            cfg.update(json.load(f))
    overrides = {
        "lr": args.lr, "weight_decay": args.weight_decay, "feat_dim": args.feat_dim,
        "hidden": args.hidden, "num_heads": args.num_heads, "pos_dim": args.pos_dim,
        "b": args.b, "knn_k": args.knn_k,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val
    if args.no_normalize:
        cfg["normalize"] = False
    if args.recon_feature is not None:
        cfg["recon_feature"] = args.recon_feature
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=["np12", "npultra"], required=True)
    p.add_argument("--session-path", type=str, default=None)
    p.add_argument("--config-json", type=str, default=None,
                   help="best_config.json from a sweep (overrides preset defaults)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-spikes", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint-path", type=str, default=None,
                   help="default: <repo>/checkpoints/<session_id>/localizer.pt")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--no-normalize", action="store_true",
                   help="disable per-spike PTP normalization (preset default: on)")
    p.add_argument("--recon-feature",
                   choices=["ptp", "peak_to_trough", "first_half", "second_half"],
                   default=None, help="per-channel reconstruction target (preset default: ptp)")
    p.add_argument("--save-predictions", type=str, default=None,
                   help="path to write an absolute-frame predictions .npz")

    # hyperparameter overrides (default None -> fall back to preset/config-json)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--feat-dim", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--pos-dim", type=int, default=None)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--knn-k", type=int, default=None)

    # sweep mode flags
    p.add_argument("--sweep", action="store_true", help="run Ray Tune HPO sweep instead of single training")
    p.add_argument("--num-samples", type=int, default=8, help="number of trials for sweep")
    p.add_argument("--grace-period", type=int, default=2, help="ASHA grace period")
    p.add_argument("--cpus-per-trial", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))

    # sweep search space choices (opt-in to override defaults)
    p.add_argument("--lr-choices", type=float, nargs="+", default=None)
    p.add_argument("--weight-decay-choices", type=float, nargs="+", default=None)
    p.add_argument("--feat-dim-choices", type=int, nargs="+", default=None)
    p.add_argument("--hidden-choices", type=int, nargs="+", default=None)
    p.add_argument("--num-heads-choices", type=int, nargs="+", default=None)
    p.add_argument("--pos-dim-choices", type=int, nargs="+", default=None)
    p.add_argument("--b-choices", type=float, nargs="+", default=None)
    p.add_argument("--knn-k-choices", type=int, nargs="+", default=None)

    args = p.parse_args()

    if args.sweep:
        _run_sweep(args)
        return

    if not args.session_path:
        p.error("--session-path is required unless using --sweep")

    cfg = build_cfg(args.model_type, args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    session_id = Path(args.session_path).name

    if args.no_checkpoint:
        checkpoint_path = None
    elif args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
    else:
        checkpoint_path = REPO_ROOT / "checkpoints" / session_id / "localizer.pt"

    print(f"[run] model_type={args.model_type} device={device} epochs={args.epochs} "
          f"batch_size={args.batch_size} seed={args.seed} "
          f"max_spikes={args.max_spikes}", flush=True)

    run_training(
        cfg=cfg, session_path=args.session_path, model_type=args.model_type,
        epochs=args.epochs, device=device, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed, batch_size=args.batch_size,
        max_spikes=args.max_spikes, num_workers=args.num_workers,
        checkpoint_path=checkpoint_path, save_predictions_path=args.save_predictions)


if __name__ == "__main__":
    main()
