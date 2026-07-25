"""Train the unified joint drift + localization model, with optional Ray Tune sweep.

Two-phase per-epoch training:
  Phase 1 (monopole): standard shuffled mini-batches over all spikes for the
  per-spike PTP reconstruction loss. This trains the encoder + drift + insulation.
  Phase 2 (dredge): a time-stratified subsample of spikes is run through the
  encoder to build a full-recording raster R (num_bins × grid_len), then the
  Diff-DREDge normalized cross-correlation loss + smoothness penalty are
  computed on R. This trains the drift field to absorb global probe motion and
  the encoder to produce drift-invariant brain coordinates.

The raster must span the full recording so that 1s time bins separated by
``window_bins`` seconds can be cross-correlated. Using a subsample (default
50k spikes) keeps the computational graph for the dredge loss manageable.

Sweep mode (--sweep) runs ASHA over (lambda_smooth, window_bins, lr, sigma,
beta, max_shift_bins) selecting on val_dredge.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NP12_CONFIG, NPULTRA_CONFIG
from FAIL_drift_model import UnifiedDriftLocalizer

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"np12": NP12_CONFIG, "NPULTRA_CONFIG": NPULTRA_CONFIG}

REPRESENTATIVE_SESSION = {
    "np12": "runs/dataset1_p1",
    "npultra": "runs/dandi_000957_sub-ZYE-0021_ses-1",
}

SAMPLING_RATE_HZ = 30_000.0

DRIFT_PRESET = {
    "np12": {
        "n_channels": 12, "n_samples": 90, "use_knn": False, "knn_k": 16,
        "normalize": True, "recon_feature": "ptp", "loss_type": "mse",
        "max_freq": 0.1, "lr": 1e-3, "weight_decay": 0.0,
        "feat_dim": 32, "hidden": 128, "num_heads": 4, "pos_dim": 8,
        "b": 1.0, "max_z": 3840.0, "bin_width_sec": 1.0,
        "temporal_window_bins": 30, "max_shift_bins": 30, "beta": 15.0,
        "sigma": 2.0, "gamma_1": 1.0, "gamma_2": 1.0,
        "raster_subsample": 50000, "mono_batch_size": 512,
    },
    "npultra": {
        "n_channels": 120, "n_samples": 90, "use_knn": True, "knn_k": 16,
        "normalize": True, "recon_feature": "ptp", "loss_type": "mse",
        "max_freq": 0.1, "lr": 1e-3, "weight_decay": 0.0,
        "feat_dim": 32, "hidden": 128, "num_heads": 4, "pos_dim": 8,
        "b": 1.0, "max_z": 3840.0, "bin_width_sec": 1.0,
        "temporal_window_bins": 30, "max_shift_bins": 30, "beta": 15.0,
        "sigma": 2.0, "gamma_1": 1.0, "gamma_2": 1.0,
        "raster_subsample": 50000, "mono_batch_size": 256,
    },
}


class SpikeDataset(Dataset):
    """Per-spike dataset for the monopole phase. Same padding as train.py."""

    def __init__(self, session_path, fixed_n, normalize=False):
        self.session_path = Path(session_path)
        self.fixed_n = int(fixed_n)
        self.normalize = bool(normalize)
        self.waveforms = np.load(self.session_path / "neighborhood_waveforms.npy", mmap_mode="r")
        self.local_coords = np.load(self.session_path / "local_coords.npy", mmap_mode="r")
        self.neighbor_counts = np.load(self.session_path / "neighbor_counts.npy", mmap_mode="r")
        self.centroids = np.load(self.session_path / "centroids.npy", mmap_mode="r")
        self.spike_times = np.load(self.session_path / "spike_times.npy", mmap_mode="r")
        self.n_spikes = self.waveforms.shape[0]
        self.M = self.waveforms.shape[1]
        self.n_samples = self.waveforms.shape[2]
        self.times_sec = (self.spike_times.astype(np.float64) / SAMPLING_RATE_HZ)
        self.total_duration = float(self.times_sec.max())
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
        times_sec = np.float32(self.times_sec[i])
        return (torch.from_numpy(wf), torch.from_numpy(coords),
                torch.from_numpy(mask), torch.from_numpy(centroid),
                torch.tensor(times_sec))


def collate_fn(batch):
    wf = torch.stack([b[0] for b in batch])
    coords = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    centroids = torch.stack([b[3] for b in batch])
    times = torch.stack([b[4] for b in batch])
    return wf, coords, mask, centroids, times


def split_train_val_test_indices(n_spikes, val_frac, test_frac, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_spikes)
    n_val = int(len(perm) * val_frac)
    n_test = int(len(perm) * test_frac)
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]


def _to_device(batch, device):
    wf, coords, mask, centroid, times_sec = batch
    return (wf.to(device, non_blocking=True),
            coords.to(device, non_blocking=True),
            mask.to(device, non_blocking=True),
            centroid.to(device, non_blocking=True),
            times_sec.to(device, non_blocking=True))


def _sample_raster_indices(ds, train_idx, n_raster, seed):
    """Time-stratified subsample: pick spikes spread evenly across the recording."""
    rng = np.random.default_rng(seed)
    times = ds.times_sec[train_idx]
    order = np.argsort(times, kind="stable")
    n = len(order)
    if n <= n_raster:
        return train_idx[order]
    step = n / n_raster
    sel_pos = (np.arange(n_raster) * step).astype(int)
    return train_idx[order[sel_pos]]


def _load_raster_batch(ds, indices, device):
    """Load a batch of spikes for the raster phase (all at once)."""
    fn = ds.fixed_n
    n = len(indices)
    wf = np.zeros((n, fn, ds.n_samples), dtype=np.float32)
    coords = np.zeros((n, fn, 2), dtype=np.float32)
    mask = np.zeros((n, fn), dtype=bool)
    for j, idx in enumerate(indices):
        m = min(int(ds.neighbor_counts[idx]), ds.M, fn)
        wf[j, :m] = ds.waveforms[idx, :m, :]
        coords[j, :m] = ds.local_coords[idx, :m, :]
        mask[j, :m] = True
    if ds.normalize:
        ptp = wf.max(axis=2) - wf.min(axis=2)
        scale = ptp.max(axis=1, keepdims=True)[:, :, None]
        wf = wf / np.where(scale > 1e-6, scale, 1.0)
    centroid = ds.centroids[indices].astype(np.float32)
    times_sec = ds.times_sec[indices].astype(np.float32)
    return (torch.from_numpy(wf).to(device),
            torch.from_numpy(coords).to(device),
            torch.from_numpy(mask).to(device),
            torch.from_numpy(centroid).to(device),
            torch.from_numpy(times_sec).to(device))


def _run_monopole_epoch(model, cfg, ds, indices, device, optimizer=None):
    """Phase 1: shuffled mini-batch monopole loss."""
    train = optimizer is not None
    model.train(train)
    loader = DataLoader(
        Dataset.__new__(Dataset) if True else None,
        batch_size=cfg["mono_batch_size"], shuffle=train,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
        persistent_workers=False, drop_last=False) if False else None

    from torch.utils.data import Subset
    loader = DataLoader(Subset(ds, indices), batch_size=cfg["mono_batch_size"],
                        shuffle=train, collate_fn=collate_fn, num_workers=4,
                        pin_memory=True, drop_last=False)

    total, count = 0.0, 0
    torch.set_grad_enabled(train)
    for wf, coords, mask, centroid, times_sec in loader:
        wf, coords, mask, centroid, times_sec = _to_device(
            (wf, coords, mask, centroid, times_sec), device)
        l_mono, _, _, _ = model(wf, coords, mask, centroid, times_sec,
                                 phase="monopole")
        if train:
            optimizer.zero_grad()
            l_mono.backward()
            optimizer.step()
        n = wf.shape[0]
        total += l_mono.item() * n
        count += n
    torch.set_grad_enabled(True)
    return total / max(count, 1)


def _run_dredge_epoch(model, cfg, ds, raster_indices, device, optimizer=None):
    """Phase 2: full-recording raster + Diff-DREDge loss."""
    train = optimizer is not None
    model.train(train)
    wf, coords, mask, centroid, times_sec = _load_raster_batch(
        ds, raster_indices, device)

    torch.set_grad_enabled(train)
    _, l_dredge, l_smooth, _ = model(wf, coords, mask, centroid, times_sec,
                                      phase="dredge")
    total = cfg["gamma_1"] * l_dredge + cfg["gamma_2"] * l_smooth
    if train:
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    torch.set_grad_enabled(True)
    return l_dredge.item(), l_smooth.item()


def _eval_monopole(model, cfg, ds, indices, device):
    if len(indices) == 0:
        return float("nan")
    model.eval()
    from torch.utils.data import Subset
    loader = DataLoader(Subset(ds, indices), batch_size=cfg["mono_batch_size"],
                        shuffle=False, collate_fn=collate_fn, num_workers=4,
                        pin_memory=True, drop_last=False)
    total, count = 0.0, 0
    torch.set_grad_enabled(False)
    for wf, coords, mask, centroid, times_sec in loader:
        wf, coords, mask, centroid, times_sec = _to_device(
            (wf, coords, mask, centroid, times_sec), device)
        l_mono, _, _, _ = model(wf, coords, mask, centroid, times_sec,
                                 phase="monopole")
        n = wf.shape[0]
        total += l_mono.item() * n
        count += n
    torch.set_grad_enabled(True)
    return total / max(count, 1)


def _eval_dredge(model, cfg, ds, raster_indices, device):
    if len(raster_indices) == 0:
        return float("nan"), float("nan")
    model.eval()
    wf, coords, mask, centroid, times_sec = _load_raster_batch(
        ds, raster_indices, device)
    torch.set_grad_enabled(False)
    _, l_dredge, l_smooth, _ = model(wf, coords, mask, centroid, times_sec,
                                      phase="dredge")
    torch.set_grad_enabled(True)
    return l_dredge.item(), l_smooth.item()


def run_drift_training(cfg, session_path, model_type, epochs, device, val_frac,
                      test_frac, seed, checkpoint_path=None,
                      save_predictions_path=None, report_fn=None,
                      checkpoint_every=5, resume=False, no_split=False,
                      raster_all_spikes=False):
    """Two-phase train/val/test loop. Returns metrics dict.

    ``checkpoint_every``: save checkpoint + loss_history every N epochs (and at
    the end). ``resume=True`` loads ``checkpoint_path`` (and the optimizer state
    if present) and continues from the saved epoch. The saved checkpoint also
    includes the best-so-far state keyed by val_loss, so a diverging run can be
    inspected at its best epoch via ``<checkpoint>.best.pt``.

    ``no_split=True``: skip the train/val/test split entirely. Both the monopole
    and dredge phases use all spikes, and no val/test metrics are reported (they
    are set to NaN). Useful for unsupervised drift estimation where the dredge
    loss is the only signal and we want maximum data density.
    """
    device = torch.device(device)
    ds = SpikeDataset(session_path, fixed_n=cfg["n_channels"],
                      normalize=cfg.get("normalize", False))

    if no_split:
        all_idx = np.arange(ds.n_spikes)
        train_idx = all_idx
        val_idx = np.array([], dtype=int)
        test_idx = np.array([], dtype=int)
        raster_train = _sample_raster_indices(ds, all_idx, cfg["raster_subsample"], seed)
        raster_val = np.array([], dtype=int)
        raster_test = np.array([], dtype=int)
    else:
        train_idx, val_idx, test_idx = split_train_val_test_indices(
            ds.n_spikes, val_frac, test_frac, seed)
        raster_source = np.arange(ds.n_spikes) if raster_all_spikes else train_idx
        raster_train = _sample_raster_indices(ds, raster_source, cfg["raster_subsample"], seed)
        raster_val = _sample_raster_indices(ds, val_idx, max(cfg["raster_subsample"] // 5, 1000), seed)
        raster_test = _sample_raster_indices(ds, test_idx, max(cfg["raster_subsample"] // 5, 1000), seed)

    print(f"[data] session={Path(session_path).name} M={ds.M} fixed_n={cfg['n_channels']} "
          f"n_spikes={ds.n_spikes} dur={ds.total_duration:.1f}s "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
          f"raster_train={len(raster_train)} no_split={no_split} "
          f"raster_all_spikes={raster_all_spikes}", flush=True)
    print(f"[cfg] {json.dumps(cfg, sort_keys=True)}", flush=True)

    model = UnifiedDriftLocalizer(
        n_channels=cfg["n_channels"], n_samples=cfg["n_samples"],
        total_recording_duration_sec=ds.total_duration,
        pos_dim=cfg["pos_dim"], feat_dim=cfg["feat_dim"], hidden=cfg["hidden"],
        num_heads=cfg["num_heads"], max_freq=cfg["max_freq"],
        use_knn=cfg["use_knn"], knn_k=cfg.get("knn_k", 16), b=cfg["b"],
        loss_type=cfg.get("loss_type", "mse"),
        recon_feature=cfg.get("recon_feature", "ptp"),
        bin_width_sec=cfg["bin_width_sec"], max_z=cfg.get("max_z", 3840.0),
        sigma=cfg.get("sigma", 2.0),
        temporal_window_bins=cfg["temporal_window_bins"],
        max_shift_bins=cfg.get("max_shift_bins", 30), beta=cfg.get("beta", 15.0),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                weight_decay=cfg["weight_decay"])

    start_epoch = 0
    best_val = float("inf")
    best_state = None
    history = {
        "epoch": [], "train_loss": [], "val_loss": [],
        "train_monopole": [], "train_dredge": [], "train_smooth": [],
        "val_monopole": [], "val_dredge": [], "val_smooth": [],
    }

    if resume and checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val_loss", float("inf")))
        if "loss_history" in ckpt:
            for k in history:
                history[k] = list(ckpt["loss_history"].get(k, []))
        print(f"[resume] loaded {checkpoint_path} at epoch {start_epoch - 1}, "
              f"best_val={best_val:.6f}", flush=True)

    metrics = {"train_loss": float("nan"), "val_loss": float("nan")}
    track_metric = "train_loss" if no_split else "val_loss"
    for epoch in range(start_epoch, epochs):
        mono_train = _run_monopole_epoch(model, cfg, ds, train_idx, device, optimizer)
        dredge_train, smooth_train = _run_dredge_epoch(
            model, cfg, ds, raster_train, device, optimizer)

        mono_val = _eval_monopole(model, cfg, ds, val_idx, device)
        dredge_val, smooth_val = _eval_dredge(model, cfg, ds, raster_val, device)

        train_loss = mono_train + cfg["gamma_1"] * dredge_train + cfg["gamma_2"] * smooth_train
        val_loss = mono_val + cfg["gamma_1"] * dredge_val + cfg["gamma_2"] * smooth_val
        metrics = {
            "train_loss": train_loss, "val_loss": val_loss,
            "train_monopole": mono_train, "train_dredge": dredge_train,
            "train_smooth": smooth_train,
            "val_monopole": mono_val, "val_dredge": dredge_val,
            "val_smooth": smooth_val,
            "epoch": epoch,
        }
        history["epoch"].append(epoch)
        for k in ("train_loss", "val_loss", "train_monopole", "train_dredge",
                  "train_smooth", "val_monopole", "val_dredge", "val_smooth"):
            history[k].append(float(metrics[k]))
        print(f"[epoch {epoch:03d}] train_loss={train_loss:.6f} "
              f"val_loss={val_loss:.6f} | "
              f"train_mono={mono_train:.4f} "
              f"train_dredge={dredge_train:.6f} "
              f"train_smooth={smooth_train:.6f} | "
              f"val_mono={mono_val:.4f} "
              f"val_dredge={dredge_val:.6f} "
              f"val_smooth={smooth_val:.6f}", flush=True)
        if report_fn is not None:
            report_fn(metrics)

        track_val = train_loss if no_split else val_loss
        if track_val < best_val:
            best_val = track_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model_state_dict": best_state, "cfg": cfg,
                            "model_type": model_type, "session_id": Path(session_path).name,
                            "epoch": epoch, "val_loss": best_val, "best": True},
                           str(checkpoint_path) + ".best.pt")
                print(f"[best] epoch {epoch} {track_metric}={best_val:.6f} -> {checkpoint_path}.best.pt",
                      flush=True)

        if checkpoint_path is not None and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg,
                        "model_type": model_type, "session_id": Path(session_path).name,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch, "best_val_loss": best_val,
                        "loss_history": history},
                       checkpoint_path)
            epoch_ckpt = checkpoint_path.parent / f"drift_localizer_epoch{epoch:03d}.pt"
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg,
                        "model_type": model_type, "session_id": Path(session_path).name,
                        "epoch": epoch, "val_loss": val_loss, "best_val_loss": best_val},
                       epoch_ckpt)
            hist_path = checkpoint_path.parent / "loss_history.npz"
            np.savez(hist_path, **{k: np.asarray(v, dtype=np.float32 if k != "epoch" else np.int32)
                                   for k, v in history.items()})
            print(f"[checkpoint] epoch {epoch} -> {checkpoint_path.name} + {epoch_ckpt.name} "
                  f"+ {hist_path.name} (best_val={best_val:.6f})", flush=True)

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "cfg": cfg,
                    "model_type": model_type, "session_id": Path(session_path).name,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epochs - 1, "best_val_loss": best_val,
                    "loss_history": history},
                   checkpoint_path)
        hist_path = checkpoint_path.parent / "loss_history.npz"
        np.savez(hist_path, **{k: np.asarray(v, dtype=np.float32 if k != "epoch" else np.int32)
                               for k, v in history.items()})
        print(f"[checkpoint] final -> {checkpoint_path.name} + {hist_path.name} "
              f"(best_val={best_val:.6f})", flush=True)
        if best_state is not None:
            torch.save({"model_state_dict": best_state, "cfg": cfg,
                        "model_type": model_type, "session_id": Path(session_path).name,
                        "epoch": -1, "val_loss": best_val, "best": True},
                       str(checkpoint_path) + ".best.pt")

    if save_predictions_path is not None:
        pred_idx = test_idx if len(test_idx) > 0 else np.arange(ds.n_spikes)
        _save_predictions(model, cfg, ds, pred_idx, device, save_predictions_path)

    mono_test = _eval_monopole(model, cfg, ds, test_idx, device)
    dredge_test, smooth_test = _eval_dredge(model, cfg, ds, raster_test, device)
    test_loss = mono_test + cfg["gamma_1"] * dredge_test + cfg["gamma_2"] * smooth_test
    print(f"[final] train_loss={metrics['train_loss']:.6f} "
          f"val_loss={metrics['val_loss']:.6f} test_loss={test_loss:.6f}", flush=True)
    metrics["test_loss"] = test_loss
    if report_fn is not None:
        report_fn({"train_loss": metrics["train_loss"], "val_loss": metrics["val_loss"],
                   "test_loss": test_loss, "epoch": epochs})
    return metrics


def _save_predictions(model, cfg, ds, pred_idx, device, out_path):
    """Save probe-frame and drift-corrected predictions without registration."""
    model.eval()
    from torch.utils.data import Subset
    loader = DataLoader(Subset(ds, pred_idx), batch_size=cfg["mono_batch_size"],
                        shuffle=False, collate_fn=collate_fn, num_workers=4,
                        pin_memory=True, drop_last=False)
    all_x, all_y, all_z_probe, all_z_brain = [], [], [], []
    all_alpha, all_t, all_drift = [], [], []
    torch.set_grad_enabled(False)
    for wf, coords, mask, centroid, times_sec in loader:
        wf, coords, mask, centroid, times_sec = _to_device(
            (wf, coords, mask, centroid, times_sec), device)
        _, _, _, extras = model(wf, coords, mask, centroid, times_sec,
                                 phase="inference")
        all_x.append(extras["x_probe"].cpu().numpy())
        all_z_probe.append(extras["z_probe"].cpu().numpy())
        all_z_brain.append(extras["z_brain"].cpu().numpy())
        all_y.append(extras["y"].cpu().numpy())
        all_alpha.append(extras["alpha"].cpu().numpy())
        all_t.append(times_sec.cpu().numpy())
        all_drift.append(extras["drift"].cpu().numpy())
    torch.set_grad_enabled(True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path,
             x_probe=np.concatenate(all_x), y=np.concatenate(all_y),
             z_probe=np.concatenate(all_z_probe),
             z_brain=np.concatenate(all_z_brain),
             alpha=np.concatenate(all_alpha),
             times_sec=np.concatenate(all_t),
             sampled_drift=np.concatenate(all_drift))
    print(f"[predictions] saved {out_path} ({sum(len(a) for a in all_x)} spikes)", flush=True)


# -----------------------------------------------------------------------------
# Sweep mode (Ray Tune ASHA)
# -----------------------------------------------------------------------------

def _sweep_trainable(config, base_cfg=None, models_dir=None, session_path=None,
                     model_type=None, epochs=None, val_frac=None, test_frac=None,
                     seed=None):
    import sys as _sys
    if models_dir and models_dir not in _sys.path:
        _sys.path.insert(0, models_dir)
    from FAIL_drift_train import run_drift_training

    def report_fn(metrics):
        from ray import tune as _raytune
        _raytune.report(metrics)

    cfg = dict(base_cfg)
    cfg.update(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_drift_training(
        cfg=cfg, session_path=session_path, model_type=model_type, epochs=epochs,
        device=device, val_frac=val_frac, test_frac=test_frac, seed=seed,
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

    search_space = {
        "lr": tune.choice(args.lr_choices or [1e-4, 3e-4, 1e-3, 3e-3]),
        "gamma_1": tune.uniform(0.1, 5.0),
        "gamma_2": tune.loguniform(1e-2, 1e2),
        "temporal_window_bins": tune.choice(args.window_bins_choices or [10, 30, 60, 120, 300, 600]),
        "sigma": tune.choice([1.0, 2.0, 4.0]),
        "beta": tune.choice([5.0, 15.0, 30.0]),
        "max_shift_bins": tune.choice([15, 30, 50]),
        "raster_subsample": tune.choice([20000, 50000, 100000]),
    }

    trainable_with_params = tune.with_parameters(
        _sweep_trainable, base_cfg=DRIFT_PRESET[model_type],
        models_dir=str(Path(__file__).parent), session_path=session_path,
        model_type=model_type, epochs=args.epochs, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed)

    storage_path = str(REPO_ROOT / "hpo_scratch")
    resume = bool(args.resume)
    print(f"[sweep] model_type={model_type}_drift session={session_path} "
          f"num_samples={args.num_samples} gpus={n_gpus} storage={storage_path} "
          f"resume={resume}", flush=True)

    analysis = tune.run(
        trainable_with_params,
        config=search_space,
        num_samples=args.num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": 1},
        storage_path=storage_path,
        name=f"{model_type}_drift_sweep",
        verbose=1,
        resume=resume,
    )

    best_config = analysis.get_best_config(metric="val_loss", mode="min", scope="all")
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
    ma = best_trial.metric_analysis

    out_dir = REPO_ROOT / "hpo_runs" / f"{model_type}_drift"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, sort_keys=True)

    best_result = {
        "val_loss": ma.get("val_loss", {}).get("min"),
        "train_loss": ma.get("train_loss", {}).get("last"),
        "test_loss": ma.get("test_loss", {}).get("last"),
        "trial_id": best_trial.trial_id,
        "session": session_path, "model_type": model_type, "seed": args.seed,
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
    print(f"[sweep] wrote {out_dir}/best_config.json, best_result.json, sweep_analysis.csv",
          flush=True)
    ray.shutdown()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def build_cfg(model_type, args):
    cfg = dict(DRIFT_PRESET[model_type])
    if args.config_json:
        with open(args.config_json) as f:
            cfg.update(json.load(f))
    overrides = {
        "lr": args.lr, "weight_decay": args.weight_decay, "feat_dim": args.feat_dim,
        "hidden": args.hidden, "num_heads": args.num_heads, "pos_dim": args.pos_dim,
        "b": args.b, "knn_k": args.knn_k,
        "gamma_1": args.gamma_1, "gamma_2": args.gamma_2,
        "bin_width_sec": args.bin_width, "temporal_window_bins": args.window_bins,
        "sigma": args.sigma, "beta": args.beta,
        "max_shift_bins": args.max_shift_bins,
        "raster_subsample": args.raster_subsample,
        "mono_batch_size": args.mono_batch_size,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val
    if args.no_normalize:
        cfg["normalize"] = False
    if args.recon_feature is not None:
        cfg["recon_feature"] = args.recon_feature
    if args.loss_type is not None:
        cfg["loss_type"] = args.loss_type
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=["np12", "npultra"], required=True)
    p.add_argument("--session-path", type=str, default=None)
    p.add_argument("--config-json", type=str, default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=5,
                   help="save checkpoint + loss_history every N epochs (default: 5)")
    p.add_argument("--resume", action="store_true",
                   help="resume: training loads --checkpoint-path; sweep reuses completed trials")
    p.add_argument("--no-split", action="store_true",
                   help="train on all spikes without any train/val/test split (unsupervised drift mode)")
    p.add_argument("--raster-all-spikes", action="store_true",
                   help="build the dredge raster from all spikes (train+val+test) instead of just the train split; monopole still uses train split")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--recon-feature",
                   choices=["ptp", "peak_to_trough", "first_half", "second_half"],
                   default=None)
    p.add_argument("--loss-type", choices=["mse", "mae", "rmse"], default=None)
    p.add_argument("--save-predictions", type=str, default=None)

    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--feat-dim", type=int, default=None)
    p.add_argument("--hidden", type=int, default=None)
    p.add_argument("--num-heads", type=int, default=None)
    p.add_argument("--pos-dim", type=int, default=None)
    p.add_argument("--b", type=float, default=None)
    p.add_argument("--knn-k", type=int, default=None)

    p.add_argument("--gamma-1", type=float, default=None)
    p.add_argument("--gamma-2", type=float, default=None)
    p.add_argument("--bin-width", type=float, default=None)
    p.add_argument("--window-bins", type=int, default=None)
    p.add_argument("--sigma", type=float, default=None)
    p.add_argument("--beta", type=float, default=None)
    p.add_argument("--max-shift-bins", type=int, default=None)
    p.add_argument("--raster-subsample", type=int, default=None)
    p.add_argument("--mono-batch-size", type=int, default=None)

    p.add_argument("--sweep", action="store_true")
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--grace-period", type=int, default=2)
    p.add_argument("--cpus-per-trial", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--lr-choices", type=float, nargs="+", default=None)
    p.add_argument("--window-bins-choices", type=int, nargs="+", default=None)

    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA is required but not available"

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
        checkpoint_path = REPO_ROOT / "checkpoints" / session_id / "drift_localizer.pt"

    print(f"[run] model_type={args.model_type} device={device} epochs={args.epochs} "
          f"seed={args.seed} window_bins={cfg['temporal_window_bins']} "
          f"raster_subsample={cfg['raster_subsample']} "
          f"resume={args.resume} checkpoint_every={args.checkpoint_every} "
          f"no_split={args.no_split} raster_all_spikes={args.raster_all_spikes}",
          flush=True)

    run_drift_training(
        cfg=cfg, session_path=args.session_path, model_type=args.model_type,
        epochs=args.epochs, device=device, val_frac=args.val_frac,
        test_frac=args.test_frac, seed=args.seed,
        checkpoint_path=checkpoint_path, save_predictions_path=args.save_predictions,
        checkpoint_every=args.checkpoint_every, resume=args.resume,
        no_split=args.no_split, raster_all_spikes=args.raster_all_spikes)


if __name__ == "__main__":
    main()
