#!/bin/bash
#SBATCH --job-name=train_encoder
#SBATCH --output=/scratch/ap7151/sln-fixed/slurm_logs/train_encoder_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-fixed/slurm_logs/train_encoder_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

# Self-supervised pretraining of the masked-transformer waveform encoder (Part B)
# on an extracted session's spike neighborhoods.
#
# Usage: sbatch src/train_encoder.sbatch [SESSION_ID] [EPOCHS] [BATCH_SIZE]
#   e.g. sbatch src/train_encoder.sbatch dataset1_p1 20 256

SESSION_ID="${1:-dataset1_p1}"
EPOCHS="${2:-20}"
BATCH_SIZE="${3:-256}"

SESSION_PATH="/scratch/ap7151/sln-fixed/runs/${SESSION_ID}"
SRC=/scratch/ap7151/sln-fixed/src
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVL=/scratch/ap7151/envs/pytorch.ext3

mkdir -p /scratch/ap7151/sln-fixed/slurm_logs

echo "=== Train Waveform Encoder (self-supervised) ==="
echo "Job ID:    ${SLURM_JOB_ID:-?}"
echo "Session:   $SESSION_ID  ($SESSION_PATH)"
echo "Epochs:    $EPOCHS   Batch: $BATCH_SIZE"
echo "Start:     $(date)"
nvidia-smi -L || echo "WARNING: no GPU visible"
echo ""

if [[ ! -f "${SESSION_PATH}/neighborhood_waveforms.npy" ]]; then
    echo "ERROR: ${SESSION_PATH}/neighborhood_waveforms.npy not found" >&2
    exit 1
fi

singularity exec --nv --overlay ${OVL}:ro $SIF \
    /bin/bash -c "source /ext3/env.sh && cd ${SRC} && \
        python -u -m waveform_encoder.train \
            --data ${SESSION_PATH} \
            --epochs ${EPOCHS} \
            --batch-size ${BATCH_SIZE} \
            --device cuda \
            --num-workers 4 \
            --out ${SESSION_PATH}/encoder_ckpt.pt"

echo ""
echo "Completed at: $(date)"
