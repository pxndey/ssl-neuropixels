"""Train the self-supervised monopolar-triangulation localizer on one session.

Exposes `run_training` (the one shared train/val/test loop, importable by the
Ray Tune sweep) and `split_train_val_test_indices`, plus an argparse CLI whose
hyperparameters follow the precedence:

    built-in preset defaults  <  --config-json (a sweep's best_config.json)  <  explicit CLI flags
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localizer import (  # noqa: E402
    SetLocalizer,
    NP12_CONFIG,
    NPULTRA_CONFIG,
    fourier_positional_embedding,
    build_knn_attention_mask,
    physics_forward,
    compute_ptp,
    masked_recon_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"np12": NP12_CONFIG, "npultra": NPULTRA_CONFIG}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class SpikeNeighborhoodDataset(Dataset):
    """Wraps one session's mmap'd `runs/<session>/*.npy` arrays.

    `__getitem__(i)` slices the `n = neighbor_counts[i]` real channels into a
    fixed-length-`fixed_n` waveform/coords tensor (real `[0:n)`, zero pad
    `[n:fixed_n)`), and a boolean `mask` (True = real). Works whether the
    shortfall comes from a session's file-level `M < fixed_n` or a spike's own
    `neighbor_counts < M`.
    """

    def __init__(self, session_path, fixed_n):
        self.session_path = Path(session_path)
        self.fixed_n = int(fixed_n)
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
        centroid = np.array(self.centroids[i], dtype=np.float32)  # copy: mmap is read-only
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


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _forward_batch(model, cfg, wf, coords, mask, device):
    wf = wf.to(device, non_blocking=True)
    coords = coords.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)
    xc, zc = coords[..., 0], coords[..., 1]
    pos_emb = fourier_positional_embedding(xc, zc, cfg["pos_dim"], cfg["max_freq"])
    knn = build_knn_attention_mask(xc, zc, mask, k=cfg.get("knn_k", 16)) if cfg["use_knn"] else None
    x, y, z, alpha = model(wf, pos_emb, mask, knn_allowed=knn)
    ptp_pred = physics_forward(x, y, z, alpha, xc, zc, cfg["b"])
    ptp_true = compute_ptp(wf)
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
    ds = SpikeNeighborhoodDataset(session_path, fixed_n=cfg["n_channels"])
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
    return cfg


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=["np12", "npultra"], required=True)
    p.add_argument("--session-path", type=str, required=True)
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
    args = p.parse_args()

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
