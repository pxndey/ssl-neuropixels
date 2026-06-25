"""Standalone self-supervised pretraining loop for the waveform encoder.

Trains :class:`WaveformMAE` with :func:`dredge_loss` on masked channels only.
This is step (a) of the two-stage plan in the spec: pretrain Part B on its own
masked-reconstruction objective; later, ``pipeline.py`` fine-tunes it end-to-end
with a motion-correction loss flowing through differentiable DREDge.

Runs on synthetic data out of the box (``--data synthetic``) or on the outputs of
``extract_neighborhoods.py`` (``--data /path/to/runs/<session>``).

Inside the Singularity container:

    python -m waveform_encoder.train --data synthetic --epochs 3 --device cpu
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import collate_masked, from_extracted, make_synthetic_dataset
from .loss import dredge_loss, target_mask
from .model import EncoderConfig, WaveformMAE


def build_dataset(spec: str, mask_frac: float, seed: int):
    if spec == "synthetic":
        return make_synthetic_dataset(n_spikes=512, mask_frac=mask_frac, seed=seed)
    return from_extracted(spec, mask_frac=mask_frac, seed=seed)


def train(args) -> float:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ds = build_dataset(args.data, args.mask_frac, args.seed)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_masked, num_workers=args.num_workers, drop_last=False,
    )

    cfg = EncoderConfig(d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers)
    model = WaveformMAE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    last = float("nan")
    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for batch in loader:
            wf = batch["waveforms"].to(device)
            coords = batch["coords"].to(device)
            content = batch["content_mask"].to(device)
            padding = batch["padding_mask"].to(device)

            out = model(wf, coords, content, padding)
            tmask = target_mask(content, padding)
            loss = dredge_loss(out["recon"], wf, mask=tmask, kind=args.loss)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            running += float(loss.item())
            n_batches += 1
        last = running / max(n_batches, 1)
        print(f"epoch {epoch + 1}/{args.epochs}  dredge_loss={last:.4f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, out_path)
        print(f"saved checkpoint -> {out_path}")
    return last


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, default="synthetic",
                   help="'synthetic' or a path to an extracted session directory")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mask-frac", type=float, default=0.30)
    p.add_argument("--loss", type=str, default="mse", choices=["mse", "huber"])
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="")
    return p


if __name__ == "__main__":
    train(build_argparser().parse_args())
