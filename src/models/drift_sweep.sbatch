#!/bin/bash
#SBATCH --job-name=drift_sweep
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_sweep_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_sweep_%x_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general
#SBATCH --mail-type=BEGIN,FAIL,CANCEL,END
#SBATCH --mail-user=ap7151@nyu.edu

set -euo pipefail

MODEL_TYPE="${1:-}"
NUM_SAMPLES="${2:-8}"
SESSION_PATH="${3:-}"
RESUME="${4:-0}"

if [[ -z "$MODEL_TYPE" ]]; then
    echo "Usage: sbatch drift_sweep.sbatch <model_type> [num_samples] [session_path] [resume=0/1]"
    echo "Example: sbatch drift_sweep.sbatch np12 16 runs/dataset1_p2 1"
    exit 1
fi

SESS_ARG=""
if [[ -n "$SESSION_PATH" ]]; then SESS_ARG=" --session-path ${SESSION_PATH}"; fi
RESUME_ARG=""
if [[ "$RESUME" == "1" ]]; then RESUME_ARG=" --resume"; fi

REPO=/scratch/ap7151/sln-v2
export RAY_TMPDIR="${REPO}/.raytmp/${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"
trap 'rm -rf "$RAY_TMPDIR"' EXIT
mkdir -p "${REPO}/slurm_logs"

echo "=== Drift Sweep (Ray Tune ASHA) ==="
echo "Job ID:      $SLURM_JOB_ID"
echo "Model:       ${MODEL_TYPE}_drift"
echo "Num samples: $NUM_SAMPLES"
echo "Resume:      $RESUME"
echo "RAY_TMPDIR:  $RAY_TMPDIR"
echo "Start:       $(date)"
echo ""

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && export RAY_TMPDIR=${RAY_TMPDIR} && python ${REPO}/src/models/drift_train.py --sweep \
        --model-type ${MODEL_TYPE} \
        --num-samples ${NUM_SAMPLES}${SESS_ARG}${RESUME_ARG}"

echo ""
echo "Completed at: $(date)"
echo "Ray temp dir ${RAY_TMPDIR} auto-removed on exit"
