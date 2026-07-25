#!/bin/bash
#SBATCH --job-name=drift_localizer
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_%x_%j.err
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

SESSION_ID="${1:-}"
MODEL_TYPE="${2:-}"
CONFIG_JSON="${3:-}"
EPOCHS="${4:-20}"
CHUNK_DURATION="${5:-60.0}"
LR="${6:-}"

if [[ -z "$SESSION_ID" || -z "$MODEL_TYPE" ]]; then
    echo "Usage: sbatch drift_train.sbatch <session_id> <model_type> [config_json] [epochs] [chunk_duration_sec] [lr]"
    echo "Example: sbatch drift_train.sbatch dataset1_p1 np12 /scratch/ap7151/sln-v2/hpo_runs/np12_drift/best_config.json 20 60.0"
    exit 1
fi

REPO=/scratch/ap7151/sln-v2
SESSION_PATH="${REPO}/runs/${SESSION_ID}"

EXTRA=""
if [[ -n "$CONFIG_JSON" ]]; then EXTRA="${EXTRA} --config-json ${CONFIG_JSON}"; fi
if [[ -n "$LR" ]]; then EXTRA="${EXTRA} --lr ${LR}"; fi

echo "=== Train Unified Drift + Localizer ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Session:   $SESSION_ID"
echo "Model:     $MODEL_TYPE"
echo "Config:    ${CONFIG_JSON:-<preset defaults>}"
echo "Epochs:    $EPOCHS   Chunk: ${CHUNK_DURATION}s   LR: ${LR:-<from config/preset>}"
echo "Output:    $SESSION_PATH"
echo "Start:     $(date)"
echo ""

mkdir -p "${REPO}/slurm_logs"

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/FAIL_drift_train.py \
        --model-type ${MODEL_TYPE} \
        --session-path ${SESSION_PATH} \
        --epochs ${EPOCHS} \
        --chunk-duration ${CHUNK_DURATION}${EXTRA} \
        --save-predictions ${SESSION_PATH}/inference_drift/predictions.npz"

echo ""
echo "Completed at: $(date)"
