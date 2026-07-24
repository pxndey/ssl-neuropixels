"""Train with Straight-Through Estimator (STE)."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FAIL_drift_train import (run_drift_training, build_cfg, main as base_main)
from FAIL_drift_model_ste import STEDriftLocalizer
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ste-beta-forward", type=float, default=1000.0)
    p.add_argument("--ste-beta-backward", type=float, default=3.0)
    
    # Add all base args
    from FAIL_drift_train import main as base_main
    args = p.parse_known_args()[0]
    
    # Override the model creation in drift_train
    import drift_train
    original_model_class = drift_train.UnifiedDriftLocalizer
    drift_train.UnifiedDriftLocalizer = lambda **kwargs: STEDriftLocalizer(
        ste_beta_forward=args.ste_beta_forward,
        ste_beta_backward=args.ste_beta_backward,
        **kwargs
    )
    
    base_main()

if __name__ == "__main__":
    main()
