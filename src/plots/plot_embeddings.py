"""PCA visualization of CNN and transformer intermediate embeddings
from a trained localizer checkpoint.

Loads a .pt checkpoint, runs the session's test split through the model
with forward hooks on the temporal encoder and self-attention, collects
per-channel features, and renders 2-D PCA projections colored by channel
index and real/pad mask.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
except ImportError:
    PCA = _SimplePCA

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

REPO = Path("/scratch/ap7151/sln-v2")

# Map long session names to short display names used in titles
DISPLAY_NAME = {
    "dandi_000957_sub-ZYE-0021_ses-1": "NPUltra",
}

# avoid Agg warning on import
plt.rcParams["figure.dpi"] = 100

# ---------------------------------------------------------------------------
# imports from sibling models/
# ---------------------------------------------------------------------------
_models_dir = str(Path(__file__).resolve().parent.parent / "models")
sys.path.insert(0, _models_dir)
from model import (  # noqa: E402
    SetLocalizer,
    fourier_positional_embedding,
    build_knn_attention_mask,
    compute_feature,
)


# ---------------------------------------------------------------------------
# minimal dataset (mirrors infer.py/train.py)
# ---------------------------------------------------------------------------
class _SpikeDataset:
    def __init__(self, session_path, fixed_n, normalize=False):
        self.session_path = Path(session_path)
        self.fixed_n = int(fixed_n)
        self.normalize = bool(normalize)
        self.waveforms = np.load(self.session_path / "neighborhood_waveforms.npy",
                                  mmap_mode="r")
        self.local_coords = np.load(self.session_path / "local_coords.npy",
                                     mmap_mode="r")
        self.neighbor_counts = np.load(self.session_path / "neighbor_counts.npy",
                                        mmap_mode="r")
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
        return (
            torch.from_numpy(wf),
            torch.from_numpy(coords),
            torch.from_numpy(mask),
        )


def _collate(batch):
    wf = torch.stack([b[0] for b in batch])
    coords = torch.stack([b[1] for b in batch])
    mask = torch.stack([b[2] for b in batch])
    return wf, coords, mask


def _split_indices(n_spikes, val_frac=0.1, test_frac=0.1, seed=0, max_spikes=None):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_spikes)
    if max_spikes is not None and max_spikes < n_spikes:
        perm = perm[:max_spikes]
    n_val = int(len(perm) * val_frac)
    n_test = int(len(perm) * test_frac)
    return perm[n_val + n_test:], perm[:n_val], perm[n_val: n_val + n_test]


# ---------------------------------------------------------------------------
# hook helpers
# ---------------------------------------------------------------------------
class _FeatureHooks:
    def __init__(self, model):
        self.model = model
        self.cnn_out = []
        self.attn_out = []
        self._handles = []

    def __enter__(self):
        # hook on the whole temporal encoder Sequential -> output (B*N, feat_dim, 1)
        h1 = self.model.temporal_encoder.register_forward_hook(self._hook_cnn)
        # hook on MultiheadAttention -> returns tuple (attended, weights)
        h2 = self.model.attn.register_forward_hook(self._hook_attn)
        self._handles = [h1, h2]
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()

    def _hook_cnn(self, module, inp, out):
        # out shape: (B*N, feat_dim, 1)
        self.cnn_out.append(out.squeeze(-1).detach().cpu())  # (B*N, feat_dim)

    def _hook_attn(self, module, inp, out):
        # out[0] shape: (B, N, token_dim)
        self.attn_out.append(out[0].detach().cpu())


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------
def extract_features(checkpoint_path, session_path, max_spikes=None, batch_size=256,
                     num_workers=4, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]

    model = SetLocalizer(
        n_channels=cfg["n_channels"],
        n_samples=cfg["n_samples"],
        pos_dim=cfg["pos_dim"],
        feat_dim=cfg["feat_dim"],
        hidden=cfg["hidden"],
        num_heads=cfg["num_heads"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = _SpikeDataset(session_path, fixed_n=cfg["n_channels"],
                        normalize=cfg.get("normalize", False))

    # use the held-out test split for an unbiased embedding view
    _, _, test_idx = _split_indices(len(ds), max_spikes=max_spikes)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, test_idx),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_cnn = []
    all_attn = []
    all_masks = []
    all_idx = []
    all_ptp = []

    with torch.no_grad(), _FeatureHooks(model) as hooks:
        for wf, coords, mask in loader:
            B, N, T = wf.shape
            wf = wf.to(device, non_blocking=True)
            coords = coords.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            xc, zc = coords[..., 0], coords[..., 1]
            pos_emb = fourier_positional_embedding(xc, zc, cfg["pos_dim"], cfg["max_freq"])
            knn = build_knn_attention_mask(xc, zc, mask, k=cfg.get("knn_k", 16)) \
                if cfg.get("use_knn", False) else None

            _ = model(wf, pos_emb, mask, knn_allowed=knn)

            # cnn_out from hook was squeezed to (B*N, feat_dim)
            cnn = hooks.cnn_out.pop(0).view(B, N, -1)
            attn = hooks.attn_out.pop(0)

            # per-channel ptp for coloring
            ptp = compute_feature(wf, cfg.get("recon_feature", "ptp")).cpu()

            all_cnn.append(cnn)
            all_attn.append(attn)
            all_masks.append(mask.cpu())
            # channel index per sample: [[0,1,...,N-1], [0,1,...,N-1], ...]
            all_idx.append(torch.arange(N).unsqueeze(0).expand(B, -1).cpu())
            all_ptp.append(ptp)

    cnn = torch.cat(all_cnn, dim=0).numpy()            # (S, N, feat_dim)
    attn = torch.cat(all_attn, dim=0).numpy()          # (S, N, token_dim)
    masks = torch.cat(all_masks, dim=0).numpy()        # (S, N)
    ch_idx = torch.cat(all_idx, dim=0).numpy()         # (S, N)
    ptps = torch.cat(all_ptp, dim=0).numpy()           # (S, N)

    return cnn, attn, masks, ch_idx, ptps


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _scatter_panel(ax, xy, color, title, cmap, vmin=None, vmax=None,
                   point_size=1.2, alpha=0.15, cbar_label=""):
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=point_size, c=color, cmap=cmap,
                    alpha=alpha, linewidths=0, rasterized=True,
                    vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if cbar_label:
        cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.03)
        cbar.set_label(cbar_label, fontsize=8)
    return sc


def plot_embedding_pca(features, masks, ch_idx, ptps, out_dir, tag="", N=12):
    """features: (S, N, D); render PCA panels."""
    S, Nt, D = features.shape
    assert Nt == N

    flat = features.reshape(-1, D)                 # (S*N, D)
    mask_flat = masks.reshape(-1)
    idx_flat = ch_idx.reshape(-1)
    ptp_flat = ptps.reshape(-1)

    # variance
    pca_full = PCA(n_components=min(10, D))
    pca_full.fit(flat)
    evr = pca_full.explained_variance_ratio_

    pca2 = PCA(n_components=2)
    xy = pca2.fit_transform(flat)                 # (S*N, 2)

    # -------- CNN / ATTENTION figures --------
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5), constrained_layout=True)

    # channel index (discrete; cap at 12/20 colors)
    n_colors = min(N, 12)
    cmap_idx = plt.cm.get_cmap("tab10" if N <= 10 else "tab20")
    ax = axes[0, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=0.8, c=idx_flat,
                     cmap=cmap_idx, vmin=0, vmax=n_colors - 1,
                     alpha=0.15, linewidths=0, rasterized=True)
    ax.set_title(f"{tag} — colored by channel index", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.03)
    cbar.set_label("channel index", fontsize=8)
    cbar.set_ticks(range(0, N, max(1, N // 5)))

    # real / pad
    ax = axes[0, 1]
    sc = ax.scatter(xy[mask_flat, 0], xy[mask_flat, 1], s=0.8,
                     c="#1a9850", alpha=0.15, linewidths=0, rasterized=True,
                     label=f"real ({mask_flat.sum()})")
    ax.scatter(xy[~mask_flat, 0], xy[~mask_flat, 1], s=0.8,
               c="#d73027", alpha=0.10, linewidths=0, rasterized=True,
               label=f"pad ({(~mask_flat).sum()})")
    ax.set_title(f"{tag} — real vs. pad channels", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # amplitude (log-ptp)
    ptp_safe = np.clip(ptp_flat, 1e-9, ptp_flat.max())
    log_ptp = np.log10(ptp_safe)
    ax = axes[1, 0]
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=0.8, c=log_ptp, cmap="viridis",
                    alpha=0.15, linewidths=0, rasterized=True)
    ax.set_title(f"{tag} — colored by log₁₀(PTP)", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    cbar = plt.colorbar(sc, ax=ax, shrink=0.55, pad=0.03)
    cbar.set_label("log₁₀(PTP)", fontsize=8)

    # variance bar
    ax = axes[1, 1]
    ypos = np.arange(1, len(evr) + 1)
    ax.barh(ypos, evr * 100, color="#4575b4", edgecolor="white", linewidth=0.4)
    ax.set_xlabel("explained variance (%)", fontsize=9)
    ax.set_ylabel("PC", fontsize=9)
    ax.set_title(f"{tag} — variance explained", fontsize=10, loc="left", fontweight="bold")
    ax.set_xlim(0, min(100, (evr.max() * 100) + 5))
    ax.invert_yaxis()
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    out = out_dir / f"embeddings_pca_{tag.lower().replace(' ', '_')}.png"
    fig.savefig(out, dpi=800)
    plt.close(fig)
    print(f"[embed] {out}")


# ---------------------------------------------------------------------------
# Plotly 3D interactive scatter
# ---------------------------------------------------------------------------
def plot_embedding_pca_3d(features, masks, ch_idx, ptps, out_dir, tag="", N=12,
                            max_points=20000):
    if not HAS_PLOTLY:
        print("[embed] plotly not available, skipping 3D")
        return
    S, Nt, D = features.shape
    assert Nt == N
    flat = features.reshape(-1, D)
    mask_flat = masks.reshape(-1)
    idx_flat = ch_idx.reshape(-1)
    ptp_flat = ptps.reshape(-1)

    rng = np.random.default_rng(0)
    n_total = flat.shape[0]
    if n_total > max_points:
        sel = rng.choice(n_total, max_points, replace=False)
        flat_sub = flat[sel]
        mask_sub = mask_flat[sel]
        idx_sub = idx_flat[sel]
        ptp_sub = ptp_flat[sel]
    else:
        flat_sub = flat
        mask_sub = mask_flat
        idx_sub = idx_flat
        ptp_sub = ptp_flat

    pca3 = PCA(n_components=3)
    xyz = pca3.fit_transform(flat_sub)

    # channel index
    fig = px.scatter_3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        color=idx_sub.astype(str),
        opacity=0.4,
        title=f"{tag} 3D PCA — channel index",
        labels={"color": "channel", "x": "PC 1", "y": "PC 2", "z": "PC 3"},
        template="simple_white",
    )
    fig.update_traces(marker=dict(size=2))
    fig.write_html(str(out_dir / f"embeddings_3d_{tag.lower().replace(' ', '_')}_channel.html"))

    # mask
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=xyz[mask_sub, 0], y=xyz[mask_sub, 1], z=xyz[mask_sub, 2],
        mode="markers", marker=dict(size=2, color="#1a9850", opacity=0.5),
        name=f"real ({mask_sub.sum()})",
    ))
    fig.add_trace(go.Scatter3d(
        x=xyz[~mask_sub, 0], y=xyz[~mask_sub, 1], z=xyz[~mask_sub, 2],
        mode="markers", marker=dict(size=2, color="#d73027", opacity=0.3),
        name=f"pad ({(~mask_sub).sum()})",
    ))
    fig.update_layout(
        title=f"{tag} 3D PCA — real vs pad",
        scene=dict(xaxis_title="PC 1", yaxis_title="PC 2", zaxis_title="PC 3"),
        template="simple_white",
    )
    fig.write_html(str(out_dir / f"embeddings_3d_{tag.lower().replace(' ', '_')}_mask.html"))

    # amplitude
    ptp_safe = np.clip(ptp_sub, 1e-9, ptp_sub.max())
    log_ptp = np.log10(ptp_safe)
    fig = px.scatter_3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        color=log_ptp,
        opacity=0.4,
        title=f"{tag} 3D PCA — log₁₀(PTP)",
        labels={"color": "log₁₀(PTP)", "x": "PC 1", "y": "PC 2", "z": "PC 3"},
        template="simple_white",
        color_continuous_scale="Viridis",
    )
    fig.update_traces(marker=dict(size=2))
    fig.write_html(str(out_dir / f"embeddings_3d_{tag.lower().replace(' ', '_')}_amplitude.html"))
    print(f"[embed] 3D {tag.lower().replace(' ', '_')} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True,
                    help="Path to .pt checkpoint (e.g. checkpoints/dataset1_p1/localizer.pt)")
    ap.add_argument("--session-path", type=Path, default=None,
                    help="Session directory (defaults to runs/<session_id> from checkpoint)")
    ap.add_argument("--max-spikes", type=int, default=None,
                    help="Max spikes to process (default: all test spikes)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default=str(REPO / "plots" / "embeddings"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    session_id = ckpt.get("session_id", args.checkpoint.parent.name)
    cfg = ckpt["cfg"]

    if args.session_path is None:
        session_path = REPO / "runs" / session_id
    else:
        session_path = Path(args.session_path)
    if not (session_path / "neighborhood_waveforms.npy").exists():
        print(f"[error] session data not found: {session_path}")
        return

    N = cfg["n_channels"]
    display_name = DISPLAY_NAME.get(session_id, session_id)
    tag = f"{display_name}_n{N}"

    print(f"[embed] session={session_id} display={display_name} checkpoint={args.checkpoint}")
    cnn, attn, masks, ch_idx, ptps = extract_features(
        args.checkpoint, session_path, max_spikes=args.max_spikes,
        batch_size=args.batch_size, num_workers=args.num_workers,
        device=args.device,
    )
    print(f"[embed] shapes: cnn={cnn.shape}, attn={attn.shape}, samples={len(cnn)}")

    plot_embedding_pca(cnn, masks, ch_idx, ptps, out_dir, tag=f"{tag}_cnn", N=N)
    plot_embedding_pca_3d(cnn, masks, ch_idx, ptps, out_dir, tag=f"{tag}_cnn", N=N)
    plot_embedding_pca(attn, masks, ch_idx, ptps, out_dir, tag=f"{tag}_transformer", N=N)
    plot_embedding_pca_3d(attn, masks, ch_idx, ptps, out_dir, tag=f"{tag}_transformer", N=N)
    print(f"[embed] done -> {out_dir}")


if __name__ == "__main__":
    main()
