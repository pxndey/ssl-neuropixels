"""Cross-probe generalization test for the np12 models.

Loads the two trained np12 checkpoints (dataset1_p1, dataset1_p2) and, for each
probe's held-out TEST split (same seed/fracs as training), evaluates the
reconstruction loss under BOTH models -> a 2x2 (model x data) matrix. The
diagonal is within-probe (matches training's test loss); the off-diagonal is
cross-probe. Off-diagonal localizations are saved to
runs/<data>/inference_xprobe_from_<model>/ for plotting.

Both np12 models share architecture + per-spike normalization + recon_feature,
so the only thing that differs across the matrix is which probe the weights were
trained on -> the off/within ratio isolates cross-probe weight generalization.
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
    SpikeNeighborhoodDataset, split_train_val_test_indices, collate_fn,
    _forward_batch, REPO_ROOT,
)
from localizer import SetLocalizer  # noqa: E402

SESSIONS = {"p1": "dataset1_p1", "p2": "dataset1_p2"}


def build_model(cfg, state, device):
    m = SetLocalizer(n_channels=cfg["n_channels"], n_samples=cfg["n_samples"],
                     pos_dim=cfg["pos_dim"], feat_dim=cfg["feat_dim"],
                     hidden=cfg["hidden"], num_heads=cfg["num_heads"]).to(device)
    m.load_state_dict(state)
    m.eval()
    return m


def eval_on(model, cfg, ds, test_idx, device, bs, nw, want_loc):
    loader = DataLoader(Subset(ds, test_idx), batch_size=bs, shuffle=False,
                        collate_fn=collate_fn, num_workers=nw, pin_memory=True)
    total, count = 0.0, 0
    xs, ys, zs, als = [], [], [], []
    torch.set_grad_enabled(False)
    for wf, coords, mask, cen in loader:
        loss, (x, y, z, alpha) = _forward_batch(model, cfg, wf, coords, mask, device)
        n = wf.shape[0]
        total += loss.item() * n
        count += n
        if want_loc:
            xs.append((x.cpu() + cen[:, 0]).numpy())
            zs.append((z.cpu() + cen[:, 1]).numpy())
            ys.append(y.cpu().numpy())
            als.append(alpha.cpu().numpy())
    torch.set_grad_enabled(True)
    loc = None
    if want_loc:
        loc = np.stack([np.concatenate(xs), np.concatenate(ys),
                        np.concatenate(zs), np.concatenate(als)], axis=1).astype(np.float32)
    return total / max(count, 1), loc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpts, models = {}, {}
    for k, v in SESSIONS.items():
        c = torch.load(REPO_ROOT / "checkpoints" / v / "localizer.pt",
                       map_location=device, weights_only=False)
        ckpts[k], models[k] = c, build_model(c["cfg"], c["model_state_dict"], device)
        print(f"[load] model {k} <- {v}  recon_feature={c['cfg'].get('recon_feature')} "
              f"normalize={c['cfg'].get('normalize')}", flush=True)

    matrix = {}
    for dk, dv in SESSIONS.items():
        ds = SpikeNeighborhoodDataset(REPO_ROOT / "runs" / dv, fixed_n=12, normalize=True)
        _, _, test_idx = split_train_val_test_indices(len(ds), args.val_frac, args.test_frac, args.seed)
        print(f"[data] {dv}: n_spikes={len(ds)} test={len(test_idx)}", flush=True)
        for mk in SESSIONS:
            want = (mk != dk)
            loss, loc = eval_on(models[mk], ckpts[mk]["cfg"], ds, test_idx,
                                device, args.batch_size, args.num_workers, want)
            matrix[(mk, dk)] = loss
            tag = "within" if mk == dk else "CROSS"
            print(f"[eval] model={mk} data={dk} ({tag}): test_recon_loss={loss:.6f}", flush=True)
            if want:
                out = REPO_ROOT / "runs" / dv / f"inference_xprobe_from_{mk}" / "localizations.npy"
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, loc)
                np.save(out.parent / "test_indices.npy", np.asarray(test_idx, dtype=np.int64))

    print("\n=== cross-probe recon-loss matrix (rows=model, cols=data) ===", flush=True)
    print(f"{'':10s}{'data=p1':>14s}{'data=p2':>14s}")
    for mk in SESSIONS:
        print(f"model={mk:4s}" + "".join(f"{matrix[(mk, dk)]:>14.6f}" for dk in SESSIONS))
    for dk in SESSIONS:
        within = matrix[(dk, dk)]
        for mk in SESSIONS:
            if mk != dk:
                print(f"  data={dk}: CROSS(model={mk}) / within = "
                      f"{matrix[(mk, dk)] / max(within, 1e-9):.2f}x", flush=True)


if __name__ == "__main__":
    main()
