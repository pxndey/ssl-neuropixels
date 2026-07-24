#!/bin/bash
#SBATCH --job-name=drift_em_1to1
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_em_1to1_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_em_1to1_%x_%j.err
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
CKPT=${REPO}/checkpoints/dataset1_p1/drift_localizer_em_1to1.pt
mkdir -p ${REPO}/slurm_logs
echo "=== E-M Drift Training: 1:1 Schedule ==="
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "E-steps: 1, M-steps: 1 per epoch"
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/FAIL_drift_train.py \
        --model-type np12 \
        --session-path ${SESSION_PATH} \
        --epochs 50 \
        --beta 3.0 \
        --em-schedule 1:1 \
        --init-drift-mode smooth_random \
        --init-drift-amplitude 30.0 \
        --lambda-smooth 0.1 \
        --checkpoint-every 5 \
        --checkpoint-path ${CKPT} \
        --save-predictions ${SESSION_PATH}/inference_drift_em_1to1/predictions.npz"
echo "Completed at: $(date)"
