"""One no-grad inference pass over a session's held-out TEST split, saving
probe-global localizations.

The test split is reconstructed with the SAME seed / val_frac / test_frac
defaults as `train_localizer`, so "test set" here is exactly the spikes training
never saw. Predicted (x, z) are offset by each spike's `centroid` to report
absolute probe-frame coordinates; y (depth) and alpha (amplitude) are
frame-independent. All model/config comes from the checkpoint.

Outputs (under runs/<session>/inference/):
    localizations.npy  (N_test, 4) float32  columns [x_global, y, z_global, alpha]
    test_indices.npy   (N_test,)   int64    spike indices into the session arrays
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_localizer import (  # noqa: E402
    SpikeNeighborhoodDataset,
    split_train_val_test_indices,
    collate_fn,
    _forward_batch,
    REPO_ROOT,
)
from localizer import SetLocalizer  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-path", type=str, required=True)
    p.add_argument("--checkpoint-path", type=str, default=None,
                   help="default: <repo>/checkpoints/<session_id>/localizer.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output", type=str, default=None,
                   help="default: <session_path>/inference/localizations.npy")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    session_path = Path(args.session_path)
    session_id = session_path.name
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else \
        REPO_ROOT / "checkpoints" / session_id / "localizer.pt"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    print(f"[infer] session={session_id} ckpt={ckpt_path} device={device}", flush=True)
    print(f"[infer] cfg={cfg}", flush=True)

    model = SetLocalizer(
        n_channels=cfg["n_channels"], n_samples=cfg["n_samples"], pos_dim=cfg["pos_dim"],
        feat_dim=cfg["feat_dim"], hidden=cfg["hidden"], num_heads=cfg["num_heads"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = SpikeNeighborhoodDataset(session_path, fixed_n=cfg["n_channels"],
                                  normalize=cfg.get("normalize", False))
    _, _, test_idx = split_train_val_test_indices(len(ds), args.val_frac, args.test_frac, args.seed)
    print(f"[infer] n_spikes={len(ds)} test_spikes={len(test_idx)}", flush=True)

    loader = DataLoader(Subset(ds, test_idx), batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)

    xs, ys, zs, als = [], [], [], []
    torch.set_grad_enabled(False)
    for wf, coords, mask, centroids in loader:
        _, (x, y, z, alpha) = _forward_batch(model, cfg, wf, coords, mask, device)
        xs.append((x.cpu() + centroids[:, 0]).numpy())   # local x + centroid -> probe-global
        zs.append((z.cpu() + centroids[:, 1]).numpy())
        ys.append(y.cpu().numpy())
        als.append(alpha.cpu().numpy())
    torch.set_grad_enabled(True)

    loc = np.stack([np.concatenate(xs), np.concatenate(ys),
                    np.concatenate(zs), np.concatenate(als)], axis=1).astype(np.float32)

    out = Path(args.output) if args.output else session_path / "inference" / "localizations.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, loc)
    np.save(out.parent / "test_indices.npy", np.asarray(test_idx, dtype=np.int64))
    print(f"[infer] saved {loc.shape} -> {out}", flush=True)
    print(f"[infer] columns: [x_global, y, z_global, alpha]; also wrote test_indices.npy", flush=True)


if __name__ == "__main__":
    main()
