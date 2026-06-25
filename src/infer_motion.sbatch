#!/bin/bash
#SBATCH --job-name=infer_motion
#SBATCH --output=/scratch/ap7151/sln-fixed/slurm_logs/infer_motion_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-fixed/slurm_logs/infer_motion_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

# Produce a motion trace P(t) with differentiable DREDge for an extracted session.
# Usage: sbatch src/infer_motion.sbatch [SESSION_ID] [MOTION_DIMS] [DISP]
#   e.g. sbatch src/infer_motion.sbatch dataset1_p1 y hard

SESSION_ID="${1:-dataset1_p1}"
MOTION_DIMS="${2:-y}"
DISP="${3:-hard}"

SESSION_PATH="/scratch/ap7151/sln-fixed/runs/${SESSION_ID}"
SRC=/scratch/ap7151/sln-fixed/src
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVL=/scratch/ap7151/envs/pytorch.ext3

mkdir -p /scratch/ap7151/sln-fixed/slurm_logs

echo "=== DREDge motion-trace inference ==="
echo "Job ID:    ${SLURM_JOB_ID:-?}"
echo "Session:   $SESSION_ID  ($SESSION_PATH)"
echo "Dims:      $MOTION_DIMS   Disp: $DISP"
echo "Start:     $(date)"
nvidia-smi -L || echo "WARNING: no GPU visible"
echo ""

if [[ ! -f "${SESSION_PATH}/centroids.npy" ]]; then
    echo "ERROR: ${SESSION_PATH}/centroids.npy not found" >&2
    exit 1
fi

singularity exec --nv --overlay ${OVL}:ro $SIF \
    /bin/bash -c "source /ext3/env.sh && cd ${SRC} && \
        python -u infer_motion.py \
            --session ${SESSION_PATH} \
            --motion-dims ${MOTION_DIMS} \
            --disp ${DISP} \
            --device cuda"

echo ""
echo "Completed at: $(date)"
