#!/bin/bash
#SBATCH --job-name=drift_smoke
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_smoke_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_smoke_%x_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --account=torch_pr_60_general
#SBATCH --mail-type=BEGIN,FAIL,CANCEL,END
#SBATCH --mail-user=ap7151@nyu.edu

set -euo pipefail

REPO=/scratch/ap7151/sln-v2
mkdir -p "${REPO}/slurm_logs"

echo "=== Drift Model Smoke Test ==="
echo "Job ID: $SLURM_JOB_ID   Node: $SLURM_JOB_NODELIST"
echo "Start:  $(date)"
echo ""

singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/drift_smoke.py"

echo ""
echo "Completed at: $(date)"
