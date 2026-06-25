# End-to-End Differentiable DREDge + Learned Waveform Encoder

A single trainable PyTorch pipeline coupling two parts:

- **Part A — Differentiable DREDge** (`src/dredge_diff/`): a clean-room,
  fully backpropagatable re-implementation of DREDge's motion-estimation
  pipeline, exposed as an `nn.Module`.
- **Part B — Masked-Transformer Waveform Encoder** (`src/waveform_encoder/`): a
  self-supervised, per-channel waveform encoder that sits **upstream** of DREDge
  and produces cleaned per-spike features (and optional position refinements)
  feeding DREDge's (now differentiable) soft-binning step.

`src/pipeline.py` wires B → A so a downstream loss on the motion trace `P`
backpropagates all the way into the encoder weights.

This is a **clean-room** build: nothing imports spikeinterface or the original
DREDge. The reference implementation
(`reference/spikeinterface/.../sortingcomponents/motion/dredge.py`) was used only
to match the algorithm, matrix algebra, argument names and default values.

---

## Drop-in replacements vs. net-new

| Piece | Reference (non-differentiable) | Here (differentiable) | Status |
|---|---|---|---|
| **A.4** input binning | `make_2d_motion_histogram` (`np.histogramdd`) | `soft_binning.build_raster` (Gaussian/​bilinear KDE) | drop-in replacement |
| **A.2** displacement | `calc_corr_decent_pair` → `torch.max` (argmax) | `xcorr.displacement_from_corr_1d/2d` (soft-argmax / parabolic) | drop-in replacement |
| **A.2** correlation | `normxcorr1d` (boolean in-place) | `xcorr.normxcorr1d/2d` (autograd-safe `torch.where`) | drop-in replacement |
| **A.3** thresholding | `threshold_correlation_matrix` (`C >= mincorr`) | `threshold.threshold_correlation_matrix` (sigmoid gate) | drop-in replacement |
| **A.1** solve | `thomas_solve` / `newton_solve_rigid` (`scipy.linalg.solve`) | `solve.solve_displacement` (`torch.linalg.solve`) | drop-in replacement |
| spatial windows | `get_spatial_windows` | `windows.get_spatial_windows` (torch port) | port (no grad needed) |
| **Part B** encoder | *(none — does not exist in DREDge)* | `waveform_encoder/` | **net-new** |
| Integration | *(none)* | `pipeline.py` | **net-new** |

The matrix algebra in `solve.py` is identical to the reference
(`neg_hessian_likelihood_term`, `newton_rhs`, `laplacian`); only the linear
*solver* differs (dense `torch.linalg.solve` of the same block-tridiagonal system
the reference solves with a Thomas recursion — same answer, trivially
differentiable).

---

## The four differentiable substitutions (A.1–A.4)

Each is config-toggleable between its **soft** (differentiable) and **hard**
(original DREDge) form, so each step can be ablated and verified against its
limit:

| Step | Module | Soft form | Hard limit recovered when |
|---|---|---|---|
| A.1 solve | `solve.py` | `P = torch.linalg.solve(H, g)`; `H,g` autograd-tracked | (already exact / natively differentiable) |
| A.2 argmax | `xcorr.py` | soft-argmax: `E_softmax(corr/T)[lag]` | `temperature → 0` |
| A.3 threshold | `threshold.py` | `sigmoid(slope·(C − θ_C))·C` | `slope → ∞` |
| A.4 binning | `soft_binning.py` | Gaussian KDE over bins | `bandwidth → 0` |

## 1-D vs 2-D motion (`motion_dims`)

`DredgeConfig.motion_dims` switches the whole pipeline by config:

- `("y",)` — **stock DREDge**: 2-D `(depth, time)` raster, scalar axial motion
  `Δy(t)`. `x` is collapsed.
- `("x","y")` — **full 2-D motion**: 3-D `(x, y, time)` raster, 2-D
  cross-correlation + 2-D soft-argmax, vector motion `(Δx(t), Δy(t))`.

The `x`/`y` least-squares share the **same** Hessian `H` (it only depends on the
weights `U`), so 2-D motion is solved as `solve(H, [g_x, g_y])` — one extra
right-hand side, not a second system. Note: lateral `Δx` is poorly identifiable
on narrow single-shank probes (few columns); it is most useful for multi-shank or
synthetic data. Soft-binning is N-D regardless, so the encoder's `x` localization
always receives motion-loss gradients even when `motion_dims=("y",)`.

---

## Localization caveat (important)

DREDge-AP normally consumes **per-spike localizations** (continuous sub-channel
positions). `infer_motion.py` currently feeds `centroids.npy` — the *neighborhood
channel centroid* (≈ the peak channel), a **channel-resolution proxy, not a real
localization**. So the bundled inference is effectively a Kilosort-style
channel-binned activity drift estimate: it recovers the timing/shape of motion
but is quantized to channel pitch. For true DREDge-AP precision, feed either
amplitude-weighted center-of-mass localizations or the encoder's learned `pos_head`
positions (the latter is the whole point of making soft-binning differentiable in
x/y). Note the motion loss is translation-invariant in absolute position, so a
learned localizer needs an anchor (a COM target / monopole prior), not just the
drift objective.

---

## Layout

```
src/dredge_diff/
  config.py        DredgeConfig + per-substep configs (all the toggles)
  soft_binning.py  A.4  build_raster / soft_assign_1d / hard_histogram
  windows.py       spatial windows (gaussian/rect/triangle, rigid)
  xcorr.py         A.2  normxcorr1d/2d + displacement_from_corr_1d/2d (+ time-tiling)
  threshold.py     A.3  reliability_gate / threshold_correlation_matrix
  solve.py         A.1  temporal_laplacian / neg_hessian_term / solve_displacement
  dredge.py        DiffDredge(nn.Module) assembling A.4→windows→A.2→A.3→A.1
src/waveform_encoder/
  masking.py       block-contiguous nearest-neighbor masking (peak never masked)
  dataset.py       WaveformMaskedDataset + collate_masked + synthetic/extracted builders
  model.py         WaveformMAE (conv encoder, pos MLP, mask token, transformer, decoder)
  loss.py          dredge_loss (masked reconstruction; MSE default, Huber option)
  train.py         standalone self-supervised training loop
src/pipeline.py    SpikeLocalizationMotionPipeline (B → soft-bin → A → P)
src/infer_motion.py    run DREDge on a session, save motion_trace.npy (+ .sbatch)
src/plot_motion.py     overlay the estimate vs the manipulator ground truth
```

---

## Running (inside the Singularity container; GPU jobs via sbatch)

```bash
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVL=/scratch/ap7151/envs/pytorch.ext3

# pretrain the encoder on a real session (GPU)
sbatch src/train_encoder.sbatch dataset1_p1 20 256

# produce a motion trace (GPU)
sbatch src/infer_motion.sbatch dataset1_p1 y hard

# compare against the manipulator ground truth (CPU, light)
singularity exec --overlay ${OVL}:ro $SIF /bin/bash -c \
  "source /ext3/env.sh && cd src && python plot_motion.py"
```

## Two-stage training (supported by design)

1. **Pretrain Part B** standalone with `dredge_loss` on masked channels
   (`waveform_encoder/train.py`).
2. **Fine-tune end-to-end** with a motion-correction loss flowing through Part A
   (`pipeline.SpikeLocalizationMotionPipeline.compute_losses`, which combines the
   auxiliary SSL loss `w_ssl` and the motion loss `w_motion`).
