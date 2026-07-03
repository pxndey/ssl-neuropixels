#!/bin/bash
#SBATCH --job-name=infer_localizer
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/infer_localizer_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/infer_localizer_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

SESSION_ID="${1:-}"
CHECKPOINT_PATH="${2:-}"

if [[ -z "$SESSION_ID" ]]; then
    echo "Usage: sbatch infer_localizer.sbatch <session_id> [checkpoint_path]"
    echo "Example: sbatch infer_localizer.sbatch dataset1_p1"
    exit 1
fi

REPO=/scratch/ap7151/sln-v2
SESSION_PATH="${REPO}/runs/${SESSION_ID}"

EXTRA=""
if [[ -n "$CHECKPOINT_PATH" ]]; then EXTRA=" --checkpoint-path ${CHECKPOINT_PATH}"; fi

echo "=== Infer Localizer (test-set localizations) ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Session:   $SESSION_ID"
echo "Output:    ${SESSION_PATH}/inference/localizations.npy"
echo "Start:     $(date)"
echo ""

mkdir -p "${REPO}/slurm_logs"

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/infer_localizer.py \
        --session-path ${SESSION_PATH}${EXTRA}"

echo ""
echo "Completed at: $(date)"
