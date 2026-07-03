#!/bin/bash
#SBATCH --job-name=train_localizer
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/train_localizer_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/train_localizer_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

SESSION_ID="${1:-}"
MODEL_TYPE="${2:-}"
CONFIG_JSON="${3:-}"
EPOCHS="${4:-20}"
BATCH_SIZE="${5:-256}"
LR="${6:-}"

if [[ -z "$SESSION_ID" || -z "$MODEL_TYPE" ]]; then
    echo "Usage: sbatch train_localizer.sbatch <session_id> <model_type> [config_json] [epochs] [batch_size] [lr]"
    echo "Example: sbatch train_localizer.sbatch dataset1_p1 np12 /scratch/ap7151/sln-v2/hpo_runs/np12/best_config.json"
    exit 1
fi

REPO=/scratch/ap7151/sln-v2
SESSION_PATH="${REPO}/runs/${SESSION_ID}"

EXTRA=""
if [[ -n "$CONFIG_JSON" ]]; then EXTRA="${EXTRA} --config-json ${CONFIG_JSON}"; fi
if [[ -n "$LR" ]]; then EXTRA="${EXTRA} --lr ${LR}"; fi

echo "=== Train Localizer ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Session:   $SESSION_ID"
echo "Model:     $MODEL_TYPE"
echo "Config:    ${CONFIG_JSON:-<preset defaults>}"
echo "Epochs:    $EPOCHS   Batch: $BATCH_SIZE   LR: ${LR:-<from config/preset>}"
echo "Output:    $SESSION_PATH"
echo "Start:     $(date)"
echo ""

mkdir -p "${REPO}/slurm_logs"

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/train_localizer.py \
        --model-type ${MODEL_TYPE} \
        --session-path ${SESSION_PATH} \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE}${EXTRA}"

echo ""
echo "Completed at: $(date)"
