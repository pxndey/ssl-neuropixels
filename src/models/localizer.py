"""Self-supervised monopolar-triangulation spike localizer (shared library).

A CNN + attention set-encoder reads a spike's neighborhood channels (each an
electrode's waveform + its centroid-relative coordinate), predicts a point-source
(x, y, z, alpha), and decodes it through a fixed differentiable monopole physics
model into a per-channel peak-to-peak (PTP) amplitude. Training reconstructs the
*observed* PTP, so no labels are needed.

One `SetLocalizer` class serves both probe regimes via two config presets:
  - NP12_CONFIG    : NP1/2 sessions, fixed N=12, full dense attention.
  - NPULTRA_CONFIG : NP Ultra (dandi) sessions, fixed N=120, k-NN-restricted
                     attention.

Adaptation vs. the original sample script: a single real/pad `mask` (our data has
no dead-but-real channels), and a learned `mask_token` of shape (1, 1, token_dim)
substituted for the whole (content + position) token of any non-real slot before
self-attention. Masked-mean pooling and the physics loss still use `mask` to
exclude pad slots, so the mask token only changes what the encoder *sees*, never
what the decoder is scored against.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Physics decoder + losses (unchanged from the sample script)
# ---------------------------------------------------------------------------

def physics_forward(x, y, z, alpha, xc, zc, b):
    """Predict per-channel PTP from point-source params via the monopole model.

    x, y, z, alpha : (B,)        predicted per-spike source params
    xc, zc         : (B, N)      per-sample channel coordinates (padded)
    b              : scalar      fixed softening offset

    returns: (B, N) predicted PTP amplitude per channel
    """
    dx = x.unsqueeze(-1) - xc
    dz = z.unsqueeze(-1) - zc
    r2 = dx ** 2 + dz ** 2 + y.unsqueeze(-1) ** 2 + b ** 2
    return alpha.unsqueeze(-1) / torch.sqrt(r2)


def masked_recon_loss(ptp_true, ptp_pred, mask):
    """MSE reconstruction loss, ignoring padded (nonexistent) channels."""
    diff2 = (ptp_true - ptp_pred) ** 2
    diff2 = diff2 * mask.float()
    return diff2.sum() / mask.float().sum().clamp(min=1)


def compute_ptp(wf):
    """wf: (..., n_samples) raw waveform -> peak-to-peak amplitude."""
    return wf.amax(dim=-1) - wf.amin(dim=-1)


# ---------------------------------------------------------------------------
# Fixed Fourier positional embedding of 2D channel coordinates
# ---------------------------------------------------------------------------

def fourier_positional_embedding(xc, zc, pos_dim=8, max_freq=0.1):
    """Fixed sinusoidal ("Fourier feature") embedding of 2D channel coordinates.

    xc, zc  : (B, N) raw channel coordinates (um), zero-padded
    pos_dim : total embedding size, must be divisible by 4 (sin/cos x x/z)
    max_freq: highest frequency; 1/max_freq ~ smallest meaningful pitch

    returns : (B, N, pos_dim)
    """
    assert pos_dim % 4 == 0, "pos_dim must be divisible by 4 (sin/cos * x/z)"
    n_freqs = pos_dim // 4
    device = xc.device

    freqs = torch.logspace(
        start=0, end=torch.log10(torch.tensor(max_freq)).item(),
        steps=n_freqs, base=10.0, device=device,
    )

    def embed_1d(coord):
        angles = coord.unsqueeze(-1) * freqs.view(1, 1, -1) * 2 * torch.pi
        return torch.sin(angles), torch.cos(angles)

    sx, cx = embed_1d(xc)
    sz, cz = embed_1d(zc)
    return torch.cat([sx, cx, sz, cz], dim=-1)


# ---------------------------------------------------------------------------
# k-NN restricted attention mask (unchanged from the sample script)
# ---------------------------------------------------------------------------

def build_knn_attention_mask(xc, zc, mask, k=16):
    """Per-electrode k nearest-neighbor attention mask from channel geometry.

    xc, zc : (B, N) channel coordinates (padded)
    mask   : (B, N) bool, True = real electrode
    k      : neighbors each electrode may attend to

    returns: (B, N, N) bool, True = ALLOWED to attend (pad keys never allowed).
    """
    B, N = xc.shape
    coords = torch.stack([xc, zc], dim=-1)
    dist = torch.cdist(coords, coords)

    big = torch.finfo(dist.dtype).max
    pad_mask_2d = ~mask.unsqueeze(1)                    # (B, 1, N) True = pad key
    dist = dist.masked_fill(pad_mask_2d, big)

    k_eff = min(k, N)
    _, nn_idx = torch.topk(dist, k_eff, dim=-1, largest=False)

    allowed = torch.zeros(B, N, N, dtype=torch.bool, device=xc.device)
    allowed.scatter_(-1, nn_idx, True)
    allowed = allowed & mask.unsqueeze(1)               # never attend to pad keys
    return allowed


# ---------------------------------------------------------------------------
# Set/attention encoder (adapted: single mask + learned mask token)
# ---------------------------------------------------------------------------

class SetLocalizer(nn.Module):
    """Fixed-N set encoder over a spike's neighborhood channels.

    A shared Conv1d temporal encoder produces a per-electrode feature; the
    Fourier positional embedding is concatenated to form a token. For any
    non-real (pad) slot the whole token is replaced by a learned `mask_token`
    before self-attention. Attention is full-dense unless a k-NN allow-mask is
    provided. A masked mean pool (over real slots only) yields a permutation-
    invariant summary decoded by an MLP head into (x, y, z, alpha).
    """

    def __init__(self, n_channels, n_samples=90, pos_dim=8, feat_dim=32,
                 hidden=128, num_heads=4):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(16, feat_dim, kernel_size=5, padding=2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.token_dim = feat_dim + pos_dim
        assert self.token_dim % num_heads == 0, (
            f"token_dim={self.token_dim} (feat_dim+pos_dim) must be divisible "
            f"by num_heads={num_heads}")
        # learned placeholder token for any structurally-absent (pad) slot
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.token_dim) * 0.02)
        self.attn = nn.MultiheadAttention(self.token_dim, num_heads=num_heads,
                                          batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.token_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 4),   # raw x, y, z, alpha
        )

    def forward(self, wf, pos_emb, mask, knn_allowed=None):
        """
        wf:          (B, N, n_samples)  waveforms (zero where padded)
        pos_emb:     (B, N, pos_dim)    positional embeddings
        mask:        (B, N) bool        True = real electrode
        knn_allowed: (B, N, N) bool     True = allowed to attend (optional)
        """
        B, N, T = wf.shape
        feat = self.temporal_encoder(wf.reshape(B * N, 1, T)).squeeze(-1)
        feat = feat.reshape(B, N, -1)

        tokens = torch.cat([feat, pos_emb], dim=-1)          # (B, N, token_dim)
        mask_token = self.mask_token.expand(B, N, -1)
        tokens = torch.where(mask.unsqueeze(-1), tokens, mask_token)

        if knn_allowed is not None:
            attn_mask = ~knn_allowed
            attn_mask = attn_mask.repeat_interleave(self.attn.num_heads, dim=0)
            attended, _ = self.attn(tokens, tokens, tokens, attn_mask=attn_mask)
        else:
            attended, _ = self.attn(tokens, tokens, tokens)

        mask_f = mask.unsqueeze(-1).float()
        pooled = (attended * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        raw = self.head(pooled)
        x, y_raw, z, alpha_raw = raw.unbind(-1)
        y = F.softplus(y_raw)          # y >= 0 (sign unidentifiable)
        alpha = F.softplus(alpha_raw)  # alpha > 0
        return x, y, z, alpha


# ---------------------------------------------------------------------------
# Config presets (built-in defaults; HPO / CLI flags override these)
# ---------------------------------------------------------------------------

NP12_CONFIG = {
    "n_channels": 12,
    "n_samples": 90,
    "use_knn": False,
    "knn_k": 16,
    "normalize": True,   # per-spike PTP normalization (scale-invariant)
    "max_freq": 0.1,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "feat_dim": 32,
    "hidden": 128,
    "num_heads": 4,
    "pos_dim": 8,
    "b": 1.0,
}

NPULTRA_CONFIG = {
    "n_channels": 120,
    "n_samples": 90,
    "use_knn": True,
    "knn_k": 16,
    "normalize": True,   # per-spike PTP normalization (scale-invariant)
    "max_freq": 0.1,
    "lr": 1e-3,
    "weight_decay": 0.0,
    "feat_dim": 32,
    "hidden": 128,
    "num_heads": 4,
    "pos_dim": 8,
    "b": 1.0,
}


# ---------------------------------------------------------------------------
# Synthetic wiring check (not part of the training pipeline)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for name, cfg in [("np12", NP12_CONFIG), ("npultra", NPULTRA_CONFIG)]:
        B, N = 4, cfg["n_channels"]
        wf = torch.randn(B, N, cfg["n_samples"])
        xc = torch.randn(B, N) * 30
        zc = torch.randn(B, N) * 30
        n_real = torch.randint(1, N + 1, (B,))
        mask = torch.arange(N).unsqueeze(0) < n_real.unsqueeze(1)
        wf = wf * mask.unsqueeze(-1)
        pos = fourier_positional_embedding(xc, zc, cfg["pos_dim"], cfg["max_freq"])
        knn = build_knn_attention_mask(xc, zc, mask, k=cfg["knn_k"]) if cfg["use_knn"] else None
        model = SetLocalizer(N, cfg["n_samples"], cfg["pos_dim"], cfg["feat_dim"],
                             cfg["hidden"], cfg["num_heads"])
        x, y, z, alpha = model(wf, pos, mask, knn_allowed=knn)
        ptp_pred = physics_forward(x, y, z, alpha, xc, zc, cfg["b"])
        ptp_true = compute_ptp(wf)
        loss = masked_recon_loss(ptp_true, ptp_pred, mask)
        print(f"{name}: x{tuple(x.shape)} ptp{tuple(ptp_pred.shape)} "
              f"loss={loss.item():.4f}")
