#!/bin/bash
#SBATCH --job-name=plot_localizations
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/plot_localizations_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/plot_localizations_%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

REPO=/scratch/ap7151/sln-v2
SESSIONS="${*:-dataset1_p1 dataset1_p2}"   # sessions to (re)plot; default the np12 pair

echo "=== Plot Localizations + Loss ==="
echo "Job ID:   $SLURM_JOB_ID"
echo "Sessions: $SESSIONS"
echo "Start:    $(date)"
echo ""

mkdir -p "${REPO}/slurm_logs"

# CPU-only (matplotlib/numpy); no --nv needed
singularity exec --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && \
        python ${REPO}/src/plots/plot_localizations.py --sessions ${SESSIONS} && \
        python ${REPO}/src/plots/plot_loss.py"

echo ""
echo "Completed at: $(date)"
