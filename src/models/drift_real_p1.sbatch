#!/bin/bash
#SBATCH --job-name=drift_real_p1
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_real_p1_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_real_p1_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

REPO=/scratch/ap7151/sln-v2
RUN_NAME="${1:-dredge_comparison_v1}"
OUTPUT_DIR="${REPO}/outputs/drift_real_p1/${RUN_NAME}"

mkdir -p "${REPO}/slurm_logs" "${OUTPUT_DIR}"

echo "=== Real dataset1_p1 Drift Registration ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Output: ${OUTPUT_DIR}"
echo "Start: $(date)"

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -lc "source /ext3/env.sh && cd ${REPO} && \
        python src/models/drift_real.py \
            --session-path ${REPO}/runs/dataset1_p1 \
            --dredge-motion /scratch/am15577/UnitMatch/Post_Neurips/mp_ladder/results/Steinmetz/dataset1_p1/mp_dredge/motion.npz \
            --output-dir ${OUTPUT_DIR} \
            --initializations zero linear"

echo "Completed at: $(date)"
