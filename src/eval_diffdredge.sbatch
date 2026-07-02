#!/bin/bash
#SBATCH --job-name=eval_diffdredge
#SBATCH --output=/scratch/ap7151/sln-fixed/slurm_logs/eval_diffdredge_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-fixed/slurm_logs/eval_diffdredge_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

# Controlled synthetic evaluation of the differentiable DREDge:
#   (1) forward accuracy vs known imposed drift (hard + soft argmax)
#   (2) gradcheck on raster->P
#   (3) learn per-unit reliability weights by backprop THROUGH DREDge
# Usage: sbatch src/eval_diffdredge.sbatch

SRC=/scratch/ap7151/sln-fixed/src
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVL=/scratch/ap7151/envs/pytorch.ext3

mkdir -p /scratch/ap7151/sln-fixed/slurm_logs /scratch/ap7151/sln-fixed/plots

echo "=== Differentiable DREDge evaluation ==="
echo "Job ID:    ${SLURM_JOB_ID:-?}"
echo "Start:     $(date)"
nvidia-smi -L || echo "WARNING: no GPU visible"
echo ""

singularity exec --nv --overlay ${OVL}:ro $SIF \
    /bin/bash -c "source /ext3/env.sh && cd ${SRC} && python -u eval_diffdredge.py --device cuda"

echo ""
echo "Completed at: $(date)"
