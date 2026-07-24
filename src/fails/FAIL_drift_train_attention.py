"""Train with Attention-Based Drift."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FAIL_drift_train import (run_drift_training, build_cfg, main as base_main)
from drift_model_attention import AttentionDriftLocalizer
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--attn-heads", type=int, default=4)
    p.add_argument("--attn-temporal-sigma", type=float, default=100.0)
    
    # Add all base args
    from FAIL_drift_train import main as base_main
    args = p.parse_known_args()[0]
    
    # Override the model creation in drift_train
    import drift_train
    original_model_class = drift_train.UnifiedDriftLocalizer
    drift_train.UnifiedDriftLocalizer = lambda **kwargs: AttentionDriftLocalizer(
        attn_heads=args.attn_heads,
        attn_temporal_sigma=args.attn_temporal_sigma,
        **kwargs
    )
    
    base_main()

if __name__ == "__main__":
    main()
