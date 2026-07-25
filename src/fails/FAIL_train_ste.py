"""Train with Straight-Through Estimator (STE) - standalone."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
from FAIL_drift_model_ste import STEDriftLocalizer
from FAIL_drift_train import (
    run_drift_training, SpikeDataset, split_train_val_test_indices,
    _sample_raster_indices, DRIFT_PRESET, REPO_ROOT
)
import argparse
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description="Train drift model with STE")
    p.add_argument("--model-type", choices=["np12", "npultra"], required=True)
    p.add_argument("--session-path", type=str, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--beta", type=float, default=3.0)
    p.add_argument("--em-schedule", type=str, default="1:1")
    p.add_argument("--checkpoint-path", type=str, default=None)
    p.add_argument("--save-predictions", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    
    args = p.parse_args()
    
    cfg = dict(DRIFT_PRESET[args.model_type])
    cfg["beta"] = args.beta
    cfg["em_schedule"] = args.em_schedule
    
    # Monkey-patch the model class
    import drift_train
    original_class = drift_train.UnifiedDriftLocalizer
    drift_train.UnifiedDriftLocalizer = lambda **kwargs: STEDriftLocalizer(
        ste_beta_forward=1000.0,
        ste_beta_backward=3.0,
        **kwargs
    )
    
    device = "cuda"
    session_path = args.session_path
    
    run_drift_training(
        cfg=cfg,
        session_path=session_path,
        model_type=args.model_type,
        epochs=args.epochs,
        device=device,
        val_frac=0.1,
        test_frac=0.1,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        save_predictions_path=args.save_predictions,
        checkpoint_every=5,
    )

if __name__ == "__main__":
    main()
