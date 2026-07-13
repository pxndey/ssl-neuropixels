"""Inference and cross-probe evaluation for the localizer.

Modes:
  --mode single (default): One no-grad inference pass over a session's held-out TEST split.
                           Inputs: --checkpoint, --session-path. Outputs: localizations.npy

  --mode cross-probe:      Cross-probe generalization test for np12 models.
                           Loads trained np12 checkpoints (dataset1_p1, dataset1_p2) and
                           evaluates each model's reconstruction loss on the other's test split.
                           Outputs: 2x2 loss matrix + localizations in inference_xprobe_from_<model>/

All modes use the same seed/val_frac/test_frac defaults as training, so "test set"
is exactly the spikes training never saw. Predicted (x, z) are offset by each spike's
centroid to report absolute probe-frame coordinates; y and alpha are frame-independent.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SetLocalizer  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt["cfg"], ckpt["model_state_dict"]


def _build_model(cfg, state_dict, device):
    m = SetLocalizer(
        n_channels=cfg["n_channels"], n_samples=cfg["n_samples"],
        pos_dim=cfg["pos_dim"], feat_dim=cfg["feat_dim"],
        hidden=cfg["hidden"], num_heads=cfg["num_heads"]
    ).to(device)
    m.load_state_dict(state_dict)
    m.eval()
    return m


def _forward_batch(model, cfg, wf, coords, mask, device):
    from model import fourier_positional_embedding, build_knn_attention_mask, physics_forward, compute_feature, masked_recon_loss
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


class _SpikeNeighborhoodDataset:
    """Minimal dataset loader for inference."""
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


def _collate_fn(batch):
    wf = torch.stack([b[0] for b in batch])
    coords = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    centroids = torch.stack([b[3] for b in batch])
    return wf, coords, mask, centroids


def _split_train_val_test_indices(n_spikes, val_frac, test_frac, seed, max_spikes=None):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_spikes)
    if max_spikes is not None and max_spikes < n_spikes:
        perm = perm[:max_spikes]
    n_val = int(len(perm) * val_frac)
    n_test = int(len(perm) * test_frac)
    return perm[n_val + n_test:], perm[:n_val], perm[n_val:n_val + n_test]


def _eval_session(model, cfg, ds, test_idx, device, batch_size, num_workers, save_loc=None):
    """Run inference on test split, optionally save localizations."""
    loader = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False,
                        collate_fn=_collate_fn, num_workers=num_workers, pin_memory=True)
    total, count = 0.0, 0
    xs, ys, zs, als = [], [], [], []
    torch.set_grad_enabled(False)
    for wf, coords, mask, centroids in loader:
        loss, (x, y, z, alpha) = _forward_batch(model, cfg, wf, coords, mask, device)
        n = wf.shape[0]
        total += loss.item() * n
        count += n
        if save_loc:
            xs.append((x.cpu() + centroids[:, 0]).numpy())
            zs.append((z.cpu() + centroids[:, 1]).numpy())
            ys.append(y.cpu().numpy())
            als.append(alpha.cpu().numpy())
    torch.set_grad_enabled(True)

    if save_loc:
        loc = np.stack([np.concatenate(xs), np.concatenate(ys),
                        np.concatenate(zs), np.concatenate(als)], axis=1).astype(np.float32)
        save_loc.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_loc, loc)
        np.save(save_loc.parent / "test_indices.npy", np.asarray(test_idx, dtype=np.int64))

    return total / max(count, 1)


def _mode_single(args):
    """Single-session inference on test split."""
    session_path = Path(args.session_path)
    session_id = session_path.name
    ckpt_path = Path(args.checkpoint) if args.checkpoint else REPO_ROOT / "checkpoints" / session_id / "localizer.pt"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg, state = _load_checkpoint(ckpt_path, device)
    model = _build_model(cfg, state, device)
    print(f"[infer] session={session_id} ckpt={ckpt_path} device={device}", flush=True)
    print(f"[infer] cfg keys={list(cfg.keys())}", flush=True)

    ds = _SpikeNeighborhoodDataset(session_path, fixed_n=cfg["n_channels"], normalize=cfg.get("normalize", False))
    _, _, test_idx = _split_train_val_test_indices(len(ds), args.val_frac, args.test_frac, args.seed)
    print(f"[infer] n_spikes={len(ds)} test_spikes={len(test_idx)}", flush=True)

    out = Path(args.output) if args.output else session_path / "inference" / "localizations.npy"
    loss = _eval_session(model, cfg, ds, test_idx, device, args.batch_size, args.num_workers, save_loc=out)
    print(f"[infer] test_recon_loss={loss:.6f}", flush=True)
    print(f"[infer] saved {out}", flush=True)


def _mode_cross_probe(args):
    """Cross-probe evaluation for np12 models."""
    sessions = {"p1": "dataset1_p1", "p2": "dataset1_p2"}
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpts, models = {}, {}
    for k, v in sessions.items():
        c = torch.load(REPO_ROOT / "checkpoints" / v / "localizer.pt",
                       map_location=device, weights_only=False)
        ckpts[k], models[k] = c, _build_model(c["cfg"], c["model_state_dict"], device)
        print(f"[load] model {k} <- {v}  recon_feature={c['cfg'].get('recon_feature')} "
              f"normalize={c['cfg'].get('normalize')}", flush=True)

    matrix = {}
    for dk, dv in sessions.items():
        ds = _SpikeNeighborhoodDataset(REPO_ROOT / "runs" / dv, fixed_n=12, normalize=True)
        _, _, test_idx = _split_train_val_test_indices(len(ds), args.val_frac, args.test_frac, args.seed)
        print(f"[data] {dv}: n_spikes={len(ds)} test={len(test_idx)}", flush=True)
        for mk in sessions:
            want = (mk != dk)
            out = REPO_ROOT / "runs" / dv / f"inference_xprobe_from_{mk}" / "localizations.npy" if want else None
            loss = _eval_session(models[mk], ckpts[mk]["cfg"], ds, test_idx, device,
                                 args.batch_size, args.num_workers, save_loc=out)
            matrix[(mk, dk)] = loss
            tag = "within" if mk == dk else "CROSS"
            print(f"[eval] model={mk} data={dk} ({tag}): test_recon_loss={loss:.6f}", flush=True)

    print("\n=== cross-probe recon-loss matrix (rows=model, cols=data) ===", flush=True)
    print(f"{'':10s}{'data=p1':>14s}{'data=p2':>14s}")
    for mk in sessions:
        print(f"model={mk:4s}" + "".join(f"{matrix[(mk, dk)]:>14.6f}" for dk in sessions))
    for dk in sessions:
        within = matrix[(dk, dk)]
        for mk in sessions:
            if mk != dk:
                print(f"  data={dk}: CROSS(model={mk}) / within = "
                      f"{matrix[(mk, dk)] / max(within, 1e-9):.2f}x", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["single", "cross-probe"], default="single",
                   help="inference mode: single session or cross-probe evaluation")
    p.add_argument("--session-path", type=str, default=None,
                   help="session path (required for single mode)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="checkpoint path (default: <repo>/checkpoints/<session_id>/localizer.pt)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=str, default=None,
                   help="output path (single mode, default: <session_path>/inference/localizations.npy)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    assert torch.cuda.is_available(), "CUDA is required but not available"

    if args.mode == "single":
        if not args.session_path:
            p.error("--session-path is required for single mode")
        _mode_single(args)
    else:
        _mode_cross_probe(args)


if __name__ == "__main__":
    main()
