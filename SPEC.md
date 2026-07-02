# Agent Task: End-to-End Differentiable DREDge with a Learned Upstream Waveform Encoder

## Objective

Implement a single trainable PyTorch pipeline with two coupled parts:

1. **Part A — Differentiable DREDge**: reimplement DREDge's motion-estimation pipeline so every step is backpropagatable, exposing it as a `nn.Module` that can sit inside a larger network and receive gradients from a downstream loss.
2. **Part B — Masked-Transformer Waveform Encoder**: a self-supervised, per-channel waveform encoder/transformer that sits *upstream* of DREDge, denoising/extracting cleaner features from raw AP-band spike waveforms before they are rasterized into DREDge's space-time input.

The point of combining these: DREDge already assumes a continuous space-time activity raster as input. Right now that raster is built by hard-binning raw spike amplitudes/positions, which blocks gradients from flowing upstream. By inserting the waveform encoder before the (now-differentiable) binning step, gradients from any downstream task loss can flow backward through DREDge's linear solve → through the soft-binned raster → through the encoder's reconstructions → into the encoder weights, allowing the feature extractor to be trained (or fine-tuned) jointly with the motion-correction objective rather than only via its own masked-reconstruction loss.

Build this in two clearly separated modules, then wire them together per the Integration section below. Each part should also be independently testable/runnable.

---

## Part A — Make DREDge Fully Differentiable

DREDge is already implemented in Python and leverages PyTorch for its GPU-accelerated routines. The conversion to a fully differentiable pipeline requires replacing a handful of discrete operations with smooth, continuous equivalents. Implement all four of the following:

### A.1 Core optimization step (linear solve)

DREDge's inference problem is a Bayesian inverse problem with normally-distributed observation errors and a quadratic spatiotemporal smoothing prior. Because the objective is entirely quadratic, its Hessian `H` is a highly structured block-tridiagonal matrix, and the latent motion trace `P` is recovered by solving the linear system:

```
H P = g
```

This step is natively differentiable — solving `P = H^{-1} g` via `torch.linalg.solve` (or an implicit-differentiation solve) lets gradients from a downstream loss propagate backward through `P` and into the entries of `H` and `g`. Implement this step using `torch.linalg.solve` (not an explicit inverse), and confirm `H` and `g` are built from autograd-tracked tensors all the way back to their raw inputs.

### A.2 Displacement estimation (the argmax bottleneck)

DREDge finds observed displacements by computing normalized cross-correlations between pairs of time bins and locating the displacement that maximizes correlation.

- **Problem**: the standard `argmax` is a step function — derivative is zero everywhere, so no gradient passes through it.
- **Solution**: replace hard `argmax` with one of:
  - **Soft-Argmax**: take the expected value over a softmax distribution computed from the cross-correlation values, or
  - **Differentiable continuous interpolation**: fit a local paraboloid or spline to the correlation peak and solve analytically for the sub-pixel peak location.

Implement at least the soft-argmax variant (expose temperature as a tunable parameter), and structure the code so the paraboloid/spline variant can be swapped in.

### A.3 Masking and thresholding step

DREDge flags and suppresses unreliable time bins when their peak correlation falls below a threshold `θ_C`, or when local activity `V_bt` is zero. Failing bins get variance set to infinity, zeroing their weight in the Hessian.

- **Problem**: hard threshold indicators are discrete 0/1 switches — zero gradient.
- **Solution**: replace the hard threshold with a smooth gate. Instead of dropping entries abruptly when `C_tt'^(b) < θ_C`, pass the difference `(C_tt'^(b) - θ_C)` through a sharp/high-slope sigmoid to smoothly scale the corresponding Hessian weight between 0 and 1.

Implement this as a configurable sigmoid gate (expose slope/sharpness as a parameter) in place of every hard `if`/boolean-mask threshold currently used to zero out Hessian weights.

### A.4 Input preprocessing and binning

If a neural network sits upstream of DREDge (which it will, per Part B), the input representation must also be differentiable end-to-end:

- **LFP data**: standard filtering + temporal downsampling are linear operations and are already fully differentiable — no changes needed.
- **AP/spike data**: continuous spike positions and amplitudes are normally binned into a discrete 2D space-time image. Standard hard-binning breaks differentiability with respect to continuous spatial coordinates.
  - **Solution**: implement **Soft-Binning** — distribute each spike's contribution across nearby bins via bilinear interpolation or a localized Gaussian kernel density estimate, rather than assigning it to a single hard bin index.

This soft-binning function is the seam where Part B's encoder output enters Part A's pipeline (see Integration).

---

## Part B — Masked-Transformer Waveform Encoder (upstream feature extractor)

A self-supervised transformer that learns clean per-channel waveform representations, generalizing across probe geometries, to be used as the feature extractor feeding the soft-binning step in Part A.

### Data per sample

- `waveforms`: `(N, 90)` — `N` = number of present channels in radius (varies by probe).
- `coords`: `(N, 2)` — `(dx, dy)` in **microns**, relative to the peak channel. Keep in real physical units, *not* normalized by pitch — propagation/attenuation physics depends on actual distance, which is exactly what should transfer across probe geometries.
- `peak_idx`: index of the peak channel (always real — never padded, never masked).

### Collate function

Pads `N → N_max` across the batch and produces two masks:

- `padding_mask`: `(B, N_max)` bool — `True` = slot doesn't exist (geometry padding).
- `content_mask`: `(B, N_max)` bool — `True` = real channel whose waveform is hidden (reconstruction target).

### Masking algorithm (per sample, at data-loading time)

```
n_present = count of real channels (excludes nothing yet)
n_mask    = floor(0.30 * n_present)
candidates = all real channels except peak_idx
seed = random.choice(candidates)
masked_set = grow contiguous block from seed by nearest-spatial-neighbor expansion
             until len(masked_set) == n_mask
             (never include peak_idx)
```

Masking must be **block/contiguous, not scattered** — scattered single-channel masking lets the model cheat via trivial neighbor interpolation.

### Forward pass

```python
# 1. Per-channel waveform encoder (shared across all channels)
#    Conv1d stack, captures local waveform shape (depol/repol phases)
x = waveform_encoder(waveforms)            # (B, N, d_model)

# 2. Relative position embedding (raw micron dx,dy -> d_model)
pos = pos_mlp(coords)                      # (B, N, d_model)

# 3. Swap in mask token where content_mask is True
#    (still gets its real position embedding — model knows "something is here")
x = where(content_mask, mask_token.expand_as(x), x)
x = x + pos

# 4. Transformer encoder, padding-aware
out = transformer_encoder(x, src_key_padding_mask=padding_mask)   # (B, N, d_model)

# 5. Per-channel decoder back to waveform space
recon = decoder_mlp(out)                   # (B, N, 90)

# 6. Loss — ONLY on real + masked positions
target_mask = content_mask & ~padding_mask
loss = dredge_loss(recon[target_mask], waveforms[target_mask])
```

### Module specs (starting point — tune later)

| Component | Spec |
|---|---|
| `waveform_encoder` | `Conv1d(1→32, k=7) → Conv1d(32→64, k=5) → Conv1d(64→128, k=3) → flatten/pool → Linear → d_model` |
| `pos_mlp` | `Linear(2→64) → ReLU → Linear(64→d_model)` |
| `mask_token` | `nn.Parameter(torch.randn(1,1,d_model))`, shared, learned |
| `transformer_encoder` | `nn.TransformerEncoder`, `batch_first=True`, `d_model=128–256`, `nhead=4–8`, `num_layers=4–6`, `dim_feedforward=4×d_model`, `dropout=0.1` |
| `decoder_mlp` | `Linear(d_model→256) → ReLU → Linear(256→90)` |

### Two masks, two jobs — don't conflate

- `padding_mask` → passed as `src_key_padding_mask`; blocks attention to/from non-existent slots entirely.
- `content_mask` → only swaps the embedding to `mask_token` before the position add; the transformer still treats it as a normal valid token (it's a real channel, just hidden). It must never be passed to `src_key_padding_mask`.

---

## Integration: wiring Part B into Part A

1. After Part B's transformer encoder produces `out` (`(B, N, d_model)`) for a batch of spike waveforms, project it back to per-spike amplitude/feature value(s) (reuse or extend `decoder_mlp`) — this is the "cleaned" spike representation.
2. Feed these cleaned per-spike features, together with each spike's continuous `(x, y, t)` position, into the **soft-binning** function from A.4 to build the differentiable space-time activity raster.
3. Pass that raster into the rest of the differentiable DREDge pipeline (A.1–A.3) to produce the motion trace `P`.
4. Confirm with `torch.autograd.gradcheck` (or a manual finite-difference spot check) that gradients computed from a downstream loss on `P` flow all the way back into the waveform encoder's weights, through: linear solve → smooth threshold gates → soft-argmax displacement step → soft-binning → decoder/transformer/encoder.
5. Keep Part B's own masked-reconstruction loss available as an auxiliary/pretraining loss — the design should support (a) pretraining Part B standalone with `dredge_loss` on masked channels, then (b) fine-tuning end-to-end with a motion-correction-derived loss flowing through Part A, possibly combining both losses.

## Deliverables

- `dredge_diff/` — differentiable DREDge module (A.1–A.4), each sub-step independently unit-testable, with the hard-vs-soft variant of each step toggleable via config for ablation.
- `waveform_encoder/` — masked-transformer encoder (Part B), with dataset/collate code implementing the masking algorithm exactly as specified, and a standalone training loop using `dredge_loss` on masked positions.
- `pipeline.py` — integration module wiring B's output into A's soft-binning input, per the Integration section.
- Tests: gradient-flow test (Integration step 4), a unit test per differentiable substitution (soft-argmax vs hard argmax agreement in the zero-temperature limit; sigmoid gate vs hard threshold agreement as slope → ∞; soft-binning vs hard-binning agreement at zero kernel bandwidth), and a masking-algorithm test verifying contiguity and that `peak_idx` is never selected.
- Brief README noting which pieces are drop-in replacements for existing DREDge code vs. net-new (the waveform encoder).
