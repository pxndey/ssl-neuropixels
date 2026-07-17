# PLAN — Self-Supervised Joint-Loss Drift Correction

Branch: `motion` (off `main`). Adds a DREDge-free self-supervised drift-correction
loss on top of the existing monopole-PTP localizer, via **contrastive spatial
coherence** + **grid density minimization**. Implemented as opt-in flags on the
existing `train.py`/`infer.py`; the legacy monopole-only baseline is untouched.

---

## Objective

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Monopole PTP}} + \gamma_1\, \mathcal{L}_{\text{contrastive}} + \gamma_2\, \mathcal{L}_{\text{entropy}}$$

1. **L_Monopole** (existing): per-spike geometric localization via the differentiable monopole physics decoder.
2. **L_contrastive** (new, InfoNCE / NT-Xent): forces the neural "fingerprint" `[waveform_emb | y | alpha]` to be invariant across time frames regardless of physical drift. Positives = spikes close in time AND close in drift-corrected spatial position; negatives = all other in-batch spikes.
3. **L_entropy** (new, grid density): a per-bin learnable drift table `Δz(t)` (and optionally `Δx(t)`) is optimized so that, after subtracting drift from the predicted absolute coordinates, all spikes over the recording snap into sharp discrete spatial bands (low-entropy soft histogram). A smoothness penalty on the drift table prevents overfitting.

No DREDge teacher: drift is discovered purely by gradient descent on these two self-supervised objectives.

---

## Architecture changes

### `src/models/model.py` — `SetLocalizer.forward`
- New `return_embedding=False` kwarg. When `True`, returns a 5-tuple adding the
  masked-mean-pooled 32-dim waveform embedding `feat` (output of the Conv1d
  temporal encoder, *before* positional concat and self-attention). This is the
  `e_j` used in the neural fingerprint. Default `False` preserves the legacy
  4-tuple, so `infer.py` (non-joint path) and the Ray sweep are unchanged.

### `src/models/ssl_losses.py` (new, ~230 LOC)
- `DriftCorrection(nn.Module)`: per-bin lookup table along recording time.
  - `__init__(n_bins, axis=("z",), init_scale=0.0)`. `axis` selects which probe
    axes to correct (`("x",)`, `("z",)`, or `("x","z")`); output channel order
    follows `axis`. `init_scale=0.0` means the table starts at exact zeros, so
    a freshly-loaded pretrained model is a strict no-op at step 0 (preserves
    the baseline).
  - `forward(t_norm)`: linear interpolation between adjacent bin centers,
    endpoints clipped. `t_norm = spike_time / rec_duration ∈ [0,1]`.
  - `smoothness_penalty()`: `Σ_axes ‖Δ_k − Δ_{k−1}‖²` — the regularizer.
- `neural_fingerprint(feat, y, alpha)`: concat + L2-normalize for cosine sim.
- `infonce_loss(fp, t, pos_corr, tau, pos_dt, pos_dz)`: NT-Xent over the batch.
  Positives: `|Δt| < pos_dt` AND `‖Δpos_corr‖ < pos_dz`. Diagonal excluded.
  Anchors with no positive contribute zero (not NaN). Numerically stabilized
  via max-subtraction and `clamp_min(1e-12)`.
- `grid_entropy_loss(coords, bin_um, sigma, coord_range)`: soft 1-D Gaussian
  histogram per axis, returns mean over axes of `−Σ_b P(b) log P(b)`.
- `apply_drift_correction(pred_pos, drift, axis)`: `pred_pos − drift` (subtract).
- `__main__`: synthetic-batch smoke check of every piece (shapes, grad-flow,
  zero-init no-op, stop-grad through drift).

### `src/models/train.py`
- `SpikeNeighborhoodDataset`: new `return_time` flag + `sample_rate`; loads
  `spike_times.npy` and converts to seconds. `__getitem__` returns a 5-tuple
  (with `spike_time` float32, seconds) when `return_time=True`, else the legacy
  4-tuple. `collate_fn` handles both.
- `TimeWindowSampler`: yields batches of temporally-adjacent spikes (sorted by
  `spike_time`, chunked at `batch_size`, split if a chunk spans > `window_sec`).
  Batch order shuffled each epoch via `set_epoch`. Yields positions into the
  `Subset` (not raw dataset indices) so it composes with `batch_sampler=`.
- `_forward_batch(..., return_extra=False)`: optionally also returns the
  embedding (via `model(..., return_embedding=True)`).
- `_lin_schedule(epoch, start_epoch, n_epochs, start_val, end_val)`: piecewise-
  linear scalar schedule for gamma warmups.
- `_run_joint_epoch`: assembles `L = L_mono + γ1(epoch)·L_contrastive + γ2(epoch)·(L_entropy + lam_smooth·L_smooth)`.
  - `γ1` ramps `gamma1_start → gamma1_end` over `gamma1_warmup_epochs` from epoch 0.
  - `γ2` is `gamma2_start` until `gamma2_start_epoch`, then ramps `→ gamma2_end`
    over `gamma2_warmup_epochs`.
  - **Stop-gradient trick**: `z_bar = (z + centroid_z).detach() − Δz(t)` (and
    likewise for `x_bar`). The entropy loss therefore trains *only* the drift
    table; the encoder keeps learning from `L_mono` + `L_contrastive`.
  - Two optimizer param groups: `model.head` (scaled by `coord_head_lr_scale`,
    phase-1 freeze) and the rest + drift (base LR). Switched to 1.0 at phase 2.
  - Logs per-epoch: `total, mono, con, ent, sm, |dz|, g1, g2`.
- `run_training`: branches on `cfg["joint"]`.
  - Auto-infers `rec_duration` from `spike_times.max()`, `n_drift_bins` from
    `drift_bin_sec`, and `entropy_coord_range` from `centroids.min/max` (per
    selected axis) when not given.
  - Loads architecture dims (`feat_dim`, `pos_dim`, `hidden`, ...) from the init
    checkpoint's `cfg` so a pretrained model with a non-default arch loads.
  - Uses `TimeWindowSampler` + `batch_sampler=` for the train loader; val/test
    stay `shuffle=False`.
  - Checkpoint saves `drift_state_dict` + the full joint `cfg`.
  - `_save_predictions`: when drift is present, also writes `x_bar, z_bar,
    t_sec, drift_table` so the saved arrays reflect static physical positions.
- CLI: 17 new `--joint-loss` flags (see `train.py --help`). All opt-in; without
  `--joint-loss` behavior is identical to before. `--init-from-checkpoint` is
  required with `--joint-loss`. `--drift {x,z,x+z}` selects axes.

### `src/models/infer.py`
- `_load_checkpoint` also returns `drift_state_dict`. `_build_drift` reconstructs
  the `DriftCorrection` from a joint checkpoint. `_SpikeNeighborhoodDataset` +
  `_collate_fn` mirror `train.py` (5-tuple when `return_time`). `_eval_session`
  applies drift correction and saves `x_bar, z_bar, spike_times_sec` when a
  drift module is loaded. Legacy 4-tuple path unchanged.

### `src/models/train_joint.sbatch`
Thin SLURM wrapper: `sbatch train_joint.sbatch <session> <model_type> <init_ckpt> [epochs] [batch] [lr] [drift]`.

### `src/plots/plot_drift_correction.py`
Renders 3 panels: z-vs-t before correction, z-vs-t after correction (should
form sharp horizontal bands), and the learned `Δz(t)` trace. `dpi=800`.

---

## Staged fine-tuning strategy (defaults)

| Phase | Epochs | `γ1` | `γ2` | coord-head LR | What trains |
|---|---|---|---|---|---|
| 1 (warmup contrastive) | 0–5 | 0.1→1.0 | 0 | `base_lr × 0.0` (frozen) | waveform embedding `e_j` |
| 2 (joint) | 6+ | 1.0 | 0→0.5 | `base_lr × 1.0` | everything + drift table |

The coord-head freeze in phase 1 prevents the encoder from warping spatial
predictions to satisfy the contrastive constraint before the embedding is
stable. The `γ2=0` start in phase 1 prevents the trivial "collapse all z to a
constant" entropy-minimum.

---

## CLI defaults

| Flag | Default | Meaning |
|---|---|---|
| `--window-sec` | 10 | max time span of one training batch |
| `--pos-dt-sec` | 2 | positive-pair time half-window |
| `--pos-dz-um` | 10 | positive-pair spatial radius (drift-corrected) |
| `--tau` | 0.1 | InfoNCE temperature |
| `--drift-bin-sec` | 1.0 | drift-table time resolution |
| `--drift` | `z` | axes: `x`, `z`, or `x+z` |
| `--gamma1-start/end` | 0.1 / 1.0 | contrastive weight ramp |
| `--gamma1-warmup-epochs` | 5 | ramp duration |
| `--gamma2-start/end` | 0.0 / 0.5 | entropy weight ramp |
| `--gamma2-start-epoch` | 6 | when phase 2 begins |
| `--gamma2-warmup-epochs` | 5 | ramp duration |
| `--lam-smooth` | 1.0 | drift smoothness penalty weight |
| `--entropy-bin-um` | 1.0 | soft-histogram bin width |
| `--entropy-sigma` | 1.0 | Gaussian kernel std |
| `--coord-head-lr-scale` | 0.0 | phase-1 coord-head LR multiplier |
| `--sample-rate` | 30000 | Hz, for `spike_times.npy → seconds` |

---

## Run

```bash
# 1. Pretrain the monopole localizer (existing baseline, if not already done)
sbatch src/models/train.sbatch dataset1_p1 np12

# 2. Fine-tune with the joint loss
sbatch src/models/train_joint.sbatch dataset1_p1 np12 \
    checkpoints/dataset1_p1/localizer.pt 20 256 1e-4 z

# 3. Inference (loads the joint checkpoint + drift table automatically)
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python src/models/infer.py \
        --session-path runs/dataset1_p1 \
        --checkpoint checkpoints/dataset1_p1/localizer_joint.pt"

# 4. Visualize
python src/plots/plot_drift_correction.py \
    --npz runs/dataset1_p1/inference_joint/predictions.npz \
    --out plots/joint_dataset1_p1.png
```

---

## Verification done

- `ssl_losses.py` `__main__` smoke check: fingerprint normalization, InfoNCE
  grad-flow, zero-init drift no-op, entropy grads reach the drift table — all pass.
- End-to-end CPU smoke: 2 epochs, 3000 spikes, `dataset1_p1`, init from
  pretrained checkpoint. Observed:
  - `γ1` 0.1→1.0 ramp applied; `γ2` stays 0 (phase 1).
  - `|dz|=0` (correct: no gradient to drift while `γ2=0`).
  - `mono=0.008` (pretrained preserved), `con=1.43`, `ent=3.99` — finite, no NaN.
  - Architecture auto-loaded from checkpoint (`feat_dim=64 pos_dim=16 hidden=64`).

## Not yet run (needs GPU)

- Full multi-epoch GPU run to confirm `Δz(t)` actually learns a non-trivial
  drift trace and the corrected-z scatter forms bands.
- NP Ultra (`npultra`) joint run (k-NN attention path).

---

## Files changed on `motion`

| File | Change |
|---|---|
| `src/models/ssl_losses.py` | **new** — drift table + InfoNCE + grid entropy |
| `src/models/model.py` | `return_embedding` flag on `SetLocalizer.forward` |
| `src/models/train.py` | dataset `return_time`, `TimeWindowSampler`, `_run_joint_epoch`, joint branch in `run_training`, 17 CLI flags |
| `src/models/infer.py` | 5-tuple dataset, drift load/apply, corrected-coordinate outputs |
| `src/models/train_joint.sbatch` | **new** — SLURM wrapper |
| `src/plots/plot_drift_correction.py` | **new** — before/after visualization |
| `PLAN.md` | **this file** |
