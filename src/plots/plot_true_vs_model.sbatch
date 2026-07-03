#!/bin/bash
#SBATCH --job-name=plot_true_vs_model
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/plot_true_vs_model_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/plot_true_vs_model_%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail
REPO=/scratch/ap7151/sln-v2
ARGS="$*"   # forwarded to plot_true_vs_model.py (e.g. --inference-name ... --out ... --sessions ...)
echo "=== Plot SLN vs monopolar (y=x) ==="
echo "Job ID: $SLURM_JOB_ID   Args: ${ARGS:-<defaults>}   Start: $(date)"
mkdir -p "${REPO}/slurm_logs"

singularity exec --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/plots/plot_true_vs_model.py ${ARGS}"

echo "Completed at: $(date)"
