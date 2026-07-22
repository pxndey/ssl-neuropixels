"""Smoke test for UnifiedDriftLocalizer: verify shapes, grads, drift pin, and that
the dredge loss is non-zero and responds to a forced drift ramp."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from drift_model import UnifiedDriftLocalizer

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

S, N, T = 500, 12, 90
wf = torch.randn(S, N, T, device=device)
xc = torch.randn(S, N, device=device) * 30
zc = torch.randn(S, N, device=device) * 20 + 1000
coords = torch.stack([xc, zc], dim=-1)
mask = torch.ones(S, N, dtype=torch.bool, device=device)
mask[:, 7:] = False
centroid = torch.stack([xc.mean(1), zc.mean(1)], dim=1)
times_sec = torch.sort(torch.rand(S, device=device) * 120.0)[0]

model = UnifiedDriftLocalizer(
    n_channels=N, n_samples=T, total_recording_duration_sec=120.0,
    pos_dim=8, feat_dim=32, hidden=128, num_heads=4, bin_width_sec=1.0,
    max_z=3840.0, temporal_window_bins=30, max_shift_bins=30, beta=15.0,
    sigma=2.0, use_knn=False).to(device)

l_mono, l_dredge, l_smooth, extras = model(wf, coords, mask, centroid, times_sec)
total = l_mono + 1.0 * l_dredge + 1.0 * l_smooth
total.backward()

print(f"device={device}")
print(f"loss_monopole={l_mono.item():.4f}  loss_dredge={l_dredge.item():.6f}  "
      f"loss_smooth={l_smooth.item():.6f}  total={total.item():.4f}")
print(f"ptp_pred {tuple(extras['ptp_pred'].shape)}  "
      f"z_brain_driftcorr {tuple(extras['z_brain_driftcorr'].shape)}")
print(f"drift {tuple(extras['drift'].shape)}  drift[0]={extras['drift'][0].item():.6f} (should be 0)")
print(f"global_drift.grad[0]={model.global_drift.grad[0]} (should be ~0: pinned anchor)")
print(f"global_drift.grad[5]={model.global_drift.grad[5].item():.3e}")
print(f"encoder head grad present: {model.encoder.head[0].weight.grad is not None}")
assert l_dredge.item() > 0, "dredge loss should be non-zero with random init"

print("\n--- forced drift ramp: dredge should decrease when drift compensates ---")
with torch.no_grad():
    ramp = torch.linspace(0, 50, model.num_bins, device=device)
    model.global_drift.copy_(ramp - ramp[0])
l_mono2, l_dredge2, l_smooth2, extras2 = model(wf, coords, mask, centroid, times_sec)
print(f"with ramp drift: loss_dredge={l_dredge2.item():.6f} (was {l_dredge.item():.6f} at D=0)")
sampled = model.get_drift(times_sec)
print(f"ramp drift sampled range [{sampled.min().item():.2f}, {sampled.max().item():.2f}]")

with torch.no_grad():
    model.global_drift.zero_()
model.zero_grad()
print("\nOK")
