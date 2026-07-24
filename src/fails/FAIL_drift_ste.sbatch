#!/bin/bash
#SBATCH --job-name=drift_ste
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_ste_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_ste_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general
#SBATCH --mail-type=BEGIN,FAIL,CANCEL,END
#SBATCH --mail-user=ap7151@nyu.edu
set -euo pipefail
REPO=/scratch/ap7151/sln-v2
SESSION_PATH=${REPO}/runs/dataset1_p1
CKPT=${REPO}/checkpoints/dataset1_p1/drift_localizer_ste.pt
mkdir -p ${REPO}/slurm_logs
echo "=== Approach 1: Straight-Through Estimator (STE) ==="
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python -c '
import sys
sys.path.insert(0, \"${REPO}/src/models\")
from drift_model_ste import STEDriftLocalizer
from FAIL_drift_train import run_drift_training, DRIFT_PRESET
import torch

cfg = dict(DRIFT_PRESET[\"np12\"])
cfg[\"beta\"] = 3.0
cfg[\"em_schedule\"] = \"1:1\"

# Monkey patch
import drift_train
FAIL_drift_train.UnifiedDriftLocalizer = lambda **kwargs: STEDriftLocalizer(
    ste_beta_forward=1000.0, ste_beta_backward=3.0, **kwargs
)

run_drift_training(
    cfg=cfg, session_path=\"${SESSION_PATH}\", model_type=\"np12\",
    epochs=50, device=\"cuda\", val_frac=0.1, test_frac=0.1, seed=0,
    checkpoint_path=\"${CKPT}\", save_predictions_path=\"${SESSION_PATH}/inference_drift_ste/predictions.npz\",
    checkpoint_every=5
)
'"
echo "Completed at: $(date)"
