#!/bin/bash
#SBATCH --job-name=drift_coherence
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_coherence_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_coherence_%x_%j.err
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
CKPT=${REPO}/checkpoints/dataset1_p1/drift_localizer_coherence.pt
mkdir -p ${REPO}/slurm_logs
echo "=== Approach 4: Direct Temporal Coherence Loss ==="
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "Directly penalize temporal variance of z_brain"
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/drift_train_coherence.py \
        --model-type np12 \
        --session-path ${SESSION_PATH} \
        --epochs 50 \
        --temporal-window-sec 300.0 \
        --em-schedule 1:1 \
        --init-drift-mode smooth_random \
        --init-drift-amplitude 30.0 \
        --checkpoint-every 5 \
        --checkpoint-path ${CKPT} \
        --save-predictions ${SESSION_PATH}/inference_drift_coherence/predictions.npz"
echo "Completed at: $(date)"
