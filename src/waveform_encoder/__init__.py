"""Masked-transformer waveform encoder (Part B).

A self-supervised feature extractor that denoises per-channel AP-band waveforms
and learns geometry-agnostic representations (relative micron coordinates), used
upstream of differentiable DREDge.

    masking.py  -- block-contiguous channel masking (peak never masked)
    dataset.py  -- Dataset + collate (padding_mask, content_mask); synthetic + extracted
    model.py    -- WaveformMAE (conv encoder, pos MLP, mask token, transformer, decoder)
    loss.py     -- dredge_loss (masked reconstruction)
    train.py    -- standalone SSL training loop
"""

from .model import EncoderConfig, WaveformMAE
from .loss import dredge_loss, target_mask
from .masking import block_contiguous_mask, grow_block_mask
from .dataset import (
    WaveformMaskedDataset,
    collate_masked,
    make_synthetic_dataset,
    from_extracted,
)

__all__ = [
    "EncoderConfig",
    "WaveformMAE",
    "dredge_loss",
    "target_mask",
    "block_contiguous_mask",
    "grow_block_mask",
    "WaveformMaskedDataset",
    "collate_masked",
    "make_synthetic_dataset",
    "from_extracted",
]
