#!/bin/bash
#SBATCH --job-name=crossprobe_eval
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/crossprobe_eval_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/crossprobe_eval_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail
REPO=/scratch/ap7151/sln-v2
echo "=== Cross-probe eval (np12 p1 <-> p2) ==="
echo "Job ID: $SLURM_JOB_ID   Start: $(date)"
mkdir -p "${REPO}/slurm_logs"

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/crossprobe_eval.py"

echo "Completed at: $(date)"
