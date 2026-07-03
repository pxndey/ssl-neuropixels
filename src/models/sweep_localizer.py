"""Ray Tune ASHA hyperparameter sweep for the localizer, one model type per call.

Runs ~8 trials on a truncated (`--max-spikes`) representative session, selecting
on `val_loss`, then writes to <repo>/hpo_runs/<model_type>/:
    best_config.json   - hyperparameters to feed train_localizer's --config-json
    best_result.json   - best val/train/test loss, trial id, session, seed, time
    sweep_analysis.csv  - all-trial results dataframe

Ray's own trial logs/checkpoints go to external scratch (storage_path /
RAY_TMPDIR), never into the repo.
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import torch
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from localizer import NP12_CONFIG, NPULTRA_CONFIG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {"np12": NP12_CONFIG, "npultra": NPULTRA_CONFIG}

REPRESENTATIVE_SESSION = {
    "np12": "runs/dataset1_p1",
    "npultra": "runs/dandi_000957_sub-ZYE-0021_ses-1",
}

COMMON_SPACE = {
    "lr": tune.choice([1e-4, 3e-4, 1e-3, 3e-3]),
    "weight_decay": tune.choice([0.0, 1e-5, 1e-4]),
    "feat_dim": tune.choice([16, 32, 64]),
    "hidden": tune.choice([64, 128, 256]),
    "num_heads": tune.choice([2, 4]),
    "pos_dim": tune.choice([4, 8, 16]),
    "b": tune.choice([0.5, 1.0, 2.0]),
}
SEARCH_SPACES = {
    "np12": dict(COMMON_SPACE),
    "npultra": {**COMMON_SPACE, "knn_k": tune.choice([8, 16, 32])},
}


def trainable(config, base_cfg=None, models_dir=None, session_path=None,
              model_type=None, epochs=None, val_frac=None, test_frac=None,
              seed=None, batch_size=None, max_spikes=None, num_workers=None):
    import sys as _sys
    if models_dir and models_dir not in _sys.path:
        _sys.path.insert(0, models_dir)
    import torch as _torch
    from train_localizer import run_training

    def report_fn(metrics):
        # launched via tune.run -> this is a Tune session; tune.report takes a
        # metrics DICT positionally (not kwargs, and not ray.train.report)
        from ray import tune as _raytune
        _raytune.report(metrics)

    cfg = dict(base_cfg)
    cfg.update(config)
    device = "cuda" if _torch.cuda.is_available() else "cpu"
    run_training(
        cfg=cfg, session_path=session_path, model_type=model_type, epochs=epochs,
        device=device, val_frac=val_frac, test_frac=test_frac, seed=seed,
        batch_size=batch_size, max_spikes=max_spikes, num_workers=num_workers,
        report_fn=report_fn)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-type", choices=["np12", "npultra"], required=True)
    p.add_argument("--session-path", type=str, default=None)
    p.add_argument("--num-samples", type=int, default=8)
    p.add_argument("--max-spikes", type=int, default=100000)
    p.add_argument("--epochs", type=int, default=10, help="ASHA max_t")
    p.add_argument("--grace-period", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpus-per-trial", type=int,
                   default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    model_type = args.model_type
    session_rel = args.session_path or REPRESENTATIVE_SESSION[model_type]
    session_path = str(session_rel if os.path.isabs(session_rel)
                       else REPO_ROOT / session_rel)
    n_gpus = torch.cuda.device_count()

    ray.init(num_cpus=args.cpus_per_trial, num_gpus=max(n_gpus, 1),
             ignore_reinit_error=True, include_dashboard=False)

    scheduler = ASHAScheduler(
        time_attr="training_iteration", metric="val_loss", mode="min",
        max_t=args.epochs, grace_period=args.grace_period, reduction_factor=2)

    trainable_with_params = tune.with_parameters(
        trainable, base_cfg=CONFIGS[model_type], models_dir=MODELS_DIR,
        session_path=session_path, model_type=model_type, epochs=args.epochs,
        val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed,
        batch_size=args.batch_size, max_spikes=args.max_spikes,
        num_workers=args.num_workers)

    storage_path = str(REPO_ROOT / "hpo_scratch")
    print(f"[sweep] model_type={model_type} session={session_path} "
          f"num_samples={args.num_samples} max_spikes={args.max_spikes} "
          f"gpus={n_gpus} storage={storage_path}", flush=True)

    analysis = tune.run(
        trainable_with_params,
        config=SEARCH_SPACES[model_type],
        num_samples=args.num_samples,
        scheduler=scheduler,
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": 1},
        storage_path=storage_path,
        name=f"{model_type}_sweep",
        verbose=1,
    )

    best_config = analysis.get_best_config(metric="val_loss", mode="min", scope="all")
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
    ma = best_trial.metric_analysis

    out_dir = REPO_ROOT / "hpo_runs" / model_type
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2, sort_keys=True)

    best_result = {
        "val_loss": ma.get("val_loss", {}).get("min"),
        "train_loss": ma.get("train_loss", {}).get("last"),
        "test_loss": ma.get("test_loss", {}).get("last"),
        "trial_id": best_trial.trial_id,
        "session": session_path,
        "model_type": model_type,
        "seed": args.seed,
        "max_spikes": args.max_spikes,
        "num_samples": args.num_samples,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_dir / "best_result.json", "w") as f:
        json.dump(best_result, f, indent=2)

    try:
        analysis.results_df.to_csv(out_dir / "sweep_analysis.csv")
    except Exception as e:  # pragma: no cover - informational artifact only
        print(f"[warn] could not write sweep_analysis.csv: {e}", flush=True)

    print(f"[sweep] best_config={json.dumps(best_config, sort_keys=True)}", flush=True)
    print(f"[sweep] best_result={json.dumps(best_result)}", flush=True)
    print(f"[sweep] wrote {out_dir}/best_config.json, best_result.json, sweep_analysis.csv",
          flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
