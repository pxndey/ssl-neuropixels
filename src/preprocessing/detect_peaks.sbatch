#!/bin/bash
#SBATCH --job-name=detect_peaks
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/detect_peaks_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/detect_peaks_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

RECORDING_PATH="${RECORDING_PATH:-}"
SESSION_ID="${SESSION_ID:-}"

if [[ -z "$RECORDING_PATH" || -z "$SESSION_ID" ]]; then
    echo "ERROR: RECORDING_PATH and SESSION_ID must be set" >&2
    exit 1
fi

OUTPUT_PATH="/scratch/ap7151/sln-v2/runs/${SESSION_ID}"

echo "=== Detect Peaks ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Session:   $SESSION_ID"
echo "Recording: $RECORDING_PATH"
echo "Output:    $OUTPUT_PATH"
echo "Start:     $(date)"
echo ""

mkdir -p "$OUTPUT_PATH"
mkdir -p /scratch/ap7151/sln-v2/slurm_logs

singularity exec --overlay /scratch/ap7151/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python /scratch/ap7151/sln-v2/src/preprocessing/detect_peaks.py \
        --recording-path $RECORDING_PATH \
        --output-path $OUTPUT_PATH"

echo ""
echo "Completed at: $(date)"
