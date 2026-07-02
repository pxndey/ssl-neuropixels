"""Does the differentiable DREDge actually work? A controlled synthetic test.

We *know* the ground-truth motion here (we impose it), so this isolates the
estimator from the real-data localization issues.

Three things are checked:

  1. FORWARD ACCURACY -- generate spikes from units at known depths, impose a
     known rigid drift P_true(t), and check DREDge recovers it (both the hard
     argmax and the differentiable soft-argmax variants).

  2. GRADIENTS ARE CORRECT -- torch.autograd.gradcheck on the raster -> P map
     (double precision).

  3. GRADIENTS ARE USEFUL (the whole point of "differentiable") -- in a noisy
     regime (half the units are high-jitter junk), learn a per-unit reliability
     weight purely by back-propagating a motion-matching loss THROUGH DREDge,
     and show the recovered motion improves over flat weighting.

Saves a 3-panel figure and prints a verdict. Run on GPU via eval_diffdredge.sbatch.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dredge_diff import DiffDredge, DredgeConfig
from dredge_diff.config import (
    DisplacementConfig,
    SoftBinConfig,
    ThresholdConfig,
    WindowConfig,
    XcorrConfig,
)


# --------------------------------------------------------------------------- #
def true_motion(T: int) -> np.ndarray:
    t = np.arange(T)
    return (25.0 * np.sin(2 * np.pi * t / 100.0) + 8.0 * np.sin(2 * np.pi * t / 31.0))


def make_units(rng, n, depth_lo, depth_hi, jitter_range, amp_range):
    return dict(
        depth=rng.uniform(depth_lo + 200, depth_hi - 200, n),
        amp=rng.uniform(*amp_range, n).astype(np.float32),
        jitter=rng.uniform(*jitter_range, n),
    )


def emit_spikes(rng, units, P_true, rate, uid_offset=0):
    T = len(P_true)
    depths, times, amps, uids = [], [], [], []
    for u in range(len(units["depth"])):
        for ti in range(T):
            k = rng.poisson(rate)
            if k == 0:
                continue
            depths.append(units["depth"][u] + P_true[ti] + rng.normal(0, units["jitter"][u], k))
            times.append(np.full(k, ti))
            amps.append(units["amp"][u] * np.clip(1 + rng.normal(0, 0.1, k), 0.2, None))
            uids.append(np.full(k, u + uid_offset))
    return (np.concatenate(depths), np.concatenate(times).astype(np.int64),
            np.concatenate(amps).astype(np.float32), np.concatenate(uids).astype(np.int64))


def cfg_for(disp_mode, temp=0.25, bin_um=4.0):
    return DredgeConfig(
        motion_dims=("y",), bin_um=bin_um, bin_s=1.0,
        window=WindowConfig(rigid=True),
        xcorr=XcorrConfig(max_disp_um=60.0, batch_size=512),
        disp=DisplacementConfig(mode=disp_mode, temperature=temp, confidence="expected"),
        thresh=ThresholdConfig(mode="sigmoid", mincorr=0.1, slope=50.0),
        soft_bin=SoftBinConfig(mode="gaussian", bandwidth_um=8.0, normalize=True, trunc=4.0),
    )


def align(P: np.ndarray, Ptrue: np.ndarray):
    """mean-subtract, sign-align; return (aligned P, r, rmse, sign)."""
    Pc = P - P.mean()
    Pt = Ptrue - Ptrue.mean()
    r = np.corrcoef(Pc, Pt)[0, 1]
    s = -1.0 if r < 0 else 1.0
    Pa = s * Pc
    return Pa, abs(r), float(np.sqrt(np.mean((Pa - Pt) ** 2))), s


# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device(args.device)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    T = args.T
    P_true = true_motion(T)
    print(f"imposed drift: peak-to-peak {P_true.ptp():.1f} um over {T} time bins")

    # ---- clean regime: all units low-jitter ------------------------------- #
    good = make_units(rng, args.n_units, 0.0, 3000.0, (2.0, 4.0), (10.0, 35.0))
    d_c, t_c, a_c, u_c = emit_spikes(rng, good, P_true, args.rate)
    print(f"clean: {len(d_c)} spikes from {args.n_units} units")

    def to_dev(d, t, a):
        return (torch.tensor(d, dtype=torch.float32, device=device),
                torch.tensor(t, dtype=torch.long, device=device),
                torch.tensor(a, dtype=torch.float32, device=device))

    dc, tc, ac = to_dev(d_c, t_c, a_c)

    with torch.no_grad():
        P_hard = DiffDredge(cfg_for("hard")).to(device)(
            spike_coords={"y": dc}, spike_features=ac, spike_time_idx=tc, n_time=T).squeeze(-1).cpu().numpy()
        P_soft = DiffDredge(cfg_for("soft")).to(device)(
            spike_coords={"y": dc}, spike_features=ac, spike_time_idx=tc, n_time=T).squeeze(-1).cpu().numpy()
    Ph_a, rh, eh, _ = align(P_hard, P_true)
    Ps_a, rs, es, _ = align(P_soft, P_true)
    print(f"[1] FORWARD ACCURACY (clean):  hard r={rh:.3f} rmse={eh:.2f} um | "
          f"soft r={rs:.3f} rmse={es:.2f} um")

    # ---- gradcheck on raster -> P ----------------------------------------- #
    gc_cfg = cfg_for("soft", bin_um=1.0)
    gc = DiffDredge(gc_cfg).double().to(device)
    raster = (torch.rand(10, 6, dtype=torch.float64, device=device) + 0.1).requires_grad_(True)
    try:
        ok = torch.autograd.gradcheck(lambda r: gc(raster=r), (raster,), eps=1e-6, atol=1e-4, rtol=1e-3)
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"    gradcheck raised: {e}")
    print(f"[2] GRADCHECK (raster -> P, fp64): {'PASS' if ok else 'FAIL'}")

    # ---- noisy regime + learn per-unit reliability weights ---------------- #
    junk = make_units(rng, args.n_units, 0.0, 3000.0, (20.0, 35.0), (10.0, 35.0))
    d_g, t_g, a_g, u_g = emit_spikes(rng, good, P_true, args.rate, uid_offset=0)
    d_j, t_j, a_j, u_j = emit_spikes(rng, junk, P_true, args.rate, uid_offset=args.n_units)
    d_n = np.concatenate([d_g, d_j]); t_n = np.concatenate([t_g, t_j])
    a_n = np.concatenate([a_g, a_j]); u_n = np.concatenate([u_g, u_j])
    n_total_units = 2 * args.n_units
    print(f"noisy: {len(d_n)} spikes ({args.n_units} good + {args.n_units} junk units)")

    dn, tn, an = to_dev(d_n, t_n, a_n)
    un = torch.tensor(u_n, dtype=torch.long, device=device)
    Ptrue_t = torch.tensor(P_true, dtype=torch.float32, device=device)
    Ptrue_c = Ptrue_t - Ptrue_t.mean()

    dredge = DiffDredge(cfg_for("soft")).to(device)
    with torch.no_grad():
        P_before = dredge(spike_coords={"y": dn}, spike_features=torch.ones_like(an),
                          spike_time_idx=tn, n_time=T).squeeze(-1)
    Pb_a, rb, eb, sgn = align(P_before.cpu().numpy(), P_true)

    log_gain = torch.zeros(n_total_units, device=device, requires_grad=True)  # exp(0)=1 -> flat
    opt = torch.optim.Adam([log_gain], lr=args.lr)
    losses = []
    for it in range(args.steps):
        feats = torch.exp(log_gain[un])                     # learned per-unit weight
        P = dredge(spike_coords={"y": dn}, spike_features=feats, spike_time_idx=tn, n_time=T).squeeze(-1)
        Pc = P - P.mean()
        loss = torch.mean((Pc - sgn * Ptrue_c) ** 2)        # sign fixed from the flat-weight run
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        feats = torch.exp(log_gain[un])
        P_after = dredge(spike_coords={"y": dn}, spike_features=feats,
                         spike_time_idx=tn, n_time=T).squeeze(-1)
    Pa_a, ra, ea, _ = align(P_after.cpu().numpy(), P_true)
    gain = torch.exp(log_gain).detach().cpu().numpy()
    g_good = gain[:args.n_units].mean(); g_junk = gain[args.n_units:].mean()
    print(f"[3] LEARN WEIGHTS THROUGH DREDGE (noisy): "
          f"flat r={rb:.3f} rmse={eb:.2f} -> learned r={ra:.3f} rmse={ea:.2f} um")
    print(f"    loss {losses[0]:.2f} -> {losses[-1]:.2f} | learned gain good/junk = "
          f"{g_good:.2f}/{g_junk:.2f} (ratio {g_good/max(g_junk,1e-6):.1f}x)")

    # ---- plot ------------------------------------------------------------- #
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.5))
    tt = np.arange(T)
    Pt_c = P_true - P_true.mean()
    ax[0].plot(tt, Pt_c, "k", lw=2.4, label="true drift")
    ax[0].plot(tt, Ph_a, "C0", lw=1.3, label=f"DREDge hard (r={rh:.2f})")
    ax[0].plot(tt, Ps_a, "C1", lw=1.3, label=f"DREDge soft (r={rs:.2f})")
    ax[0].set_title("[1] forward accuracy on clean synthetic data")
    ax[0].set_xlabel("time bin"); ax[0].set_ylabel("drift (um, mean-sub)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(tt, Pt_c, "k", lw=2.4, label="true drift")
    ax[1].plot(tt, Pb_a, "C3", lw=1.2, alpha=0.8, label=f"flat weights (r={rb:.2f})")
    ax[1].plot(tt, Pa_a, "C2", lw=1.4, label=f"learned weights (r={ra:.2f})")
    ax[1].set_title("[3] learn per-unit weights via backprop through DREDge")
    ax[1].set_xlabel("time bin"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    ax[2].plot(losses, "C4")
    ax[2].set_title(f"motion-matching loss through DREDge\ngain good/junk={g_good:.2f}/{g_junk:.2f}")
    ax[2].set_xlabel("Adam step"); ax[2].set_ylabel("MSE to true drift"); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    out = args.out
    fig.savefig(out, dpi=130)
    print(f"saved -> {out}")

    verdict = (rh > 0.9 and rs > 0.9 and ok and ra >= rb)
    print("\nVERDICT:", "differentiable DREDge looks GOOD" if verdict else "needs a look",
          f"(clean soft r={rs:.2f}, gradcheck={'ok' if ok else 'fail'}, "
          f"noisy flat->learned r {rb:.2f}->{ra:.2f})")


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--T", type=int, default=180)
    p.add_argument("--n-units", type=int, default=50)
    p.add_argument("--rate", type=int, default=8)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="/scratch/ap7151/sln-fixed/plots/diffdredge_eval.png")
    return p


if __name__ == "__main__":
    main(build_argparser().parse_args())
