#!/bin/bash
#SBATCH --job-name=drift_joint_p1
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_joint_p1_%A_%a.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_joint_p1_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

REPO=/scratch/ap7151/sln-v2
RUN_SUFFIX="${1:-plan3_v1}"
DRIFT_FIT_SPLIT="${2:-train}"
FINE_KNOTS="${3:-128}"
DRIFT_STEPS_PER_CYCLE="${4:-400}"
FINE_SMOOTHNESS="${5:-0.0002}"
TIME_BIN_SEC="${6:-2.0}"

case "${SLURM_ARRAY_TASK_ID}" in
    0)
        ABLATION=localizer-only
        INITIALIZATION=zero
        ;;
    1)
        ABLATION=frozen-drift
        INITIALIZATION=zero
        ;;
    2)
        ABLATION=alternating
        INITIALIZATION=zero
        ;;
    3)
        ABLATION=full
        INITIALIZATION=zero
        ;;
    4)
        ABLATION=shuffled-times
        INITIALIZATION=zero
        ;;
    5)
        ABLATION=full
        INITIALIZATION=linear
        ;;
    *)
        echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
        ;;
esac

RUN_NAME="${ABLATION}_${INITIALIZATION}_${RUN_SUFFIX}"
OUTPUT_DIR="${REPO}/outputs/drift_joint_p1/${RUN_NAME}"
mkdir -p "${OUTPUT_DIR}"

echo "=== PLAN_3 Alternating Joint Learner ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Ablation: ${ABLATION}"
echo "Initialization: ${INITIALIZATION}"
echo "Drift fit split: ${DRIFT_FIT_SPLIT}"
echo "Fine knots: ${FINE_KNOTS}"
echo "Drift steps per cycle: ${DRIFT_STEPS_PER_CYCLE}"
echo "Fine smoothness: ${FINE_SMOOTHNESS}"
echo "Time bin: ${TIME_BIN_SEC} s"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -lc "source /ext3/env.sh && cd ${REPO} && \
        python src/models/drift_joint_train.py \
            --session-path ${REPO}/runs/dataset1_p1 \
            --localizer-checkpoint ${REPO}/checkpoints/dataset1_p1/localizer.pt \
            --output-dir ${OUTPUT_DIR} \
            --ablation ${ABLATION} \
            --drift-initialization ${INITIALIZATION} \
            --drift-fit-split ${DRIFT_FIT_SPLIT} \
            --fine-knots ${FINE_KNOTS} \
            --drift-steps-per-cycle ${DRIFT_STEPS_PER_CYCLE} \
            --fine-smoothness ${FINE_SMOOTHNESS} \
            --time-bin-sec ${TIME_BIN_SEC}"

echo "Completed at: $(date)"
