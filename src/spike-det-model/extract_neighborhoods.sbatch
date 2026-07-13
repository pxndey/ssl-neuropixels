#!/bin/bash
#SBATCH --job-name=extract_neighborhoods
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/extract_neighborhoods_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/extract_neighborhoods_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

SESSION_ID="${1:-}"
RECORDING_PATH="${2:-}"
RADIUS_UM="${3:-48.0}"

if [[ -z "$SESSION_ID" || -z "$RECORDING_PATH" ]]; then
    echo "Usage: sbatch extract_neighborhoods.sbatch <session_id> <recording_path> [radius_um]"
    echo "Example: sbatch extract_neighborhoods.sbatch dataset1_p1 /scratch/ap7151/RAW_DATA/extra-motion/dataset1_p1"
    exit 1
fi

SESSION_PATH="/scratch/ap7151/sln-v2/runs/${SESSION_ID}"

echo "=== Extract Neighborhoods ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Session:   $SESSION_ID"
echo "Recording: $RECORDING_PATH"
echo "Output:    $SESSION_PATH"
echo "Radius um: $RADIUS_UM"
echo "Start:     $(date)"
echo ""

singularity exec --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python /scratch/ap7151/sln-v2/src/spike-det-model/extract_neighborhoods.py \
        --session-path ${SESSION_PATH} \
        --recording-path ${RECORDING_PATH} \
        --radius-um ${RADIUS_UM}"

echo ""
echo "Completed at: $(date)"
