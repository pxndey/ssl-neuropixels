"""Masked-transformer waveform encoder (Part B).

A self-supervised, per-channel waveform encoder + spatial transformer that learns
clean waveform representations and generalizes across probe geometries.  It sits
*upstream* of differentiable DREDge: its per-channel reconstructions / cleaned
features feed the soft-binning seam (see ``pipeline.py``).

Two masks, two jobs (never conflate them):

* ``padding_mask`` -> ``src_key_padding_mask``: blocks attention to/from
  non-existent (geometry-padding) slots entirely.
* ``content_mask`` -> only swaps a real channel's embedding for the learned
  ``mask_token`` *before* the position add.  The transformer still treats it as a
  normal valid token (it is a real channel, just hidden).  It must never go into
  ``src_key_padding_mask``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    n_samples: int = 90
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 4
    dim_feedforward: Optional[int] = None   # default 4 * d_model
    dropout: float = 0.1
    pos_hidden: int = 64
    decoder_hidden: int = 256

    def ff(self) -> int:
        return self.dim_feedforward if self.dim_feedforward is not None else 4 * self.d_model


class WaveformConvEncoder(nn.Module):
    """Per-channel 1-D conv stack capturing local waveform shape (depol/repol)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),     # pool over time -> (.,128,1)
        )
        self.proj = nn.Linear(128, d_model)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        # waveforms: (B, N, L) -> (B*N, 1, L)
        B, N, L = waveforms.shape
        h = waveforms.reshape(B * N, 1, L)
        h = self.net(h).squeeze(-1)      # (B*N, 128)
        h = self.proj(h)                 # (B*N, d_model)
        return h.reshape(B, N, -1)


class WaveformMAE(nn.Module):
    """Masked auto-encoder over a spike's channel neighborhood."""

    def __init__(self, cfg: Optional[EncoderConfig] = None):
        super().__init__()
        self.cfg = cfg or EncoderConfig()
        d = self.cfg.d_model

        self.waveform_encoder = WaveformConvEncoder(d)
        self.pos_mlp = nn.Sequential(
            nn.Linear(2, self.cfg.pos_hidden),
            nn.ReLU(),
            nn.Linear(self.cfg.pos_hidden, d),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=self.cfg.nhead, dim_feedforward=self.cfg.ff(),
            dropout=self.cfg.dropout, batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.num_layers)

        self.decoder_mlp = nn.Sequential(
            nn.Linear(d, self.cfg.decoder_hidden),
            nn.ReLU(),
            nn.Linear(self.cfg.decoder_hidden, self.cfg.n_samples),
        )

        # heads used by the integrated pipeline (cleaned per-spike feature + position refinement)
        self.amp_head = nn.Linear(d, 1)
        self.pos_head = nn.Linear(d, 2)

    # ------------------------------------------------------------------ #
    def encode(
        self,
        waveforms: torch.Tensor,
        coords: torch.Tensor,
        content_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run encoder + transformer, returning per-channel tokens ``(B, N, d)``."""
        x = self.waveform_encoder(waveforms)                 # (B, N, d)
        pos = self.pos_mlp(coords)                           # (B, N, d)

        mask_tok = self.mask_token.expand_as(x)
        x = torch.where(content_mask.unsqueeze(-1), mask_tok, x)
        x = x + pos

        out = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return out

    def forward(
        self,
        waveforms: torch.Tensor,
        coords: torch.Tensor,
        content_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        peak_idx: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        out = self.encode(waveforms, coords, content_mask, padding_mask)
        recon = self.decoder_mlp(out)                        # (B, N, 90)

        result = {"tokens": out, "recon": recon}
        if peak_idx is not None:
            peak_tok = self.peak_token(out, peak_idx)        # (B, d)
            result["amplitude"] = self.amp_head(peak_tok).squeeze(-1)   # (B,)
            result["pos_offset"] = self.pos_head(peak_tok)              # (B, 2)
        return result

    # ------------------------------------------------------------------ #
    @staticmethod
    def peak_token(tokens: torch.Tensor, peak_idx: torch.Tensor) -> torch.Tensor:
        """Gather the peak channel's token for each sample: ``(B, N, d) -> (B, d)``."""
        B = tokens.shape[0]
        return tokens[torch.arange(B, device=tokens.device), peak_idx]
