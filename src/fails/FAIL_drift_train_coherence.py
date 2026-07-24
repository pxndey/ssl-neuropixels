"""Train with Direct Temporal Coherence Loss."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from FAIL_drift_train import (run_drift_training, build_cfg, main as base_main)
from drift_model_coherence import CoherenceDriftLocalizer
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--temporal-window-sec", type=float, default=300.0)
    
    # Add all base args
    from FAIL_drift_train import main as base_main
    args = p.parse_known_args()[0]
    
    # Override the model creation in drift_train
    import drift_train
    original_model_class = drift_train.UnifiedDriftLocalizer
    drift_train.UnifiedDriftLocalizer = lambda **kwargs: CoherenceDriftLocalizer(
        temporal_window_sec=args.temporal_window_sec,
        **kwargs
    )
    
    base_main()

if __name__ == "__main__":
    main()
