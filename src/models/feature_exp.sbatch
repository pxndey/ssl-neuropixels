#!/bin/bash
#SBATCH --job-name=feat_exp
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/feat_exp_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/feat_exp_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

SESSION_ID="${1:-}"
FEATURE="${2:-}"
CONFIG_JSON="${3:-}"
MODEL_TYPE="${4:-np12}"

if [[ -z "$SESSION_ID" || -z "$FEATURE" ]]; then
    echo "Usage: sbatch feature_exp.sbatch <session_id> <feature> [config_json] [model_type]"
    echo "  feature in: ptp | peak_to_trough | first_half | second_half"
    exit 1
fi

REPO=/scratch/ap7151/sln-v2
SESSION_PATH="${REPO}/runs/${SESSION_ID}"
CKPT="${REPO}/checkpoints/${SESSION_ID}__${FEATURE}/localizer.pt"
INFER_OUT="${SESSION_PATH}/inference_${FEATURE}/localizations.npy"

EXTRA=""
if [[ -n "$CONFIG_JSON" ]]; then EXTRA=" --config-json ${CONFIG_JSON}"; fi

echo "=== Feature Experiment ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Session:  $SESSION_ID   Model: $MODEL_TYPE   Feature: $FEATURE"
echo "Config:   ${CONFIG_JSON:-<preset>}"
echo "Ckpt:     $CKPT"
echo "Infer:    $INFER_OUT"
echo "Start:    $(date)"
echo ""

mkdir -p "${REPO}/slurm_logs"

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && \
        python ${REPO}/src/models/train.py \
            --model-type ${MODEL_TYPE} --session-path ${SESSION_PATH} \
            --recon-feature ${FEATURE} --checkpoint-path ${CKPT}${EXTRA} && \
        python ${REPO}/src/models/infer.py --mode single \
            --session-path ${SESSION_PATH} --checkpoint ${CKPT} --output ${INFER_OUT}"

echo ""
echo "Completed at: $(date)"
