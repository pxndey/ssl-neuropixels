#!/bin/bash
# Submit detect_peaks.sbatch for each extra-motion session.
# Usage: bash runs/submit_extra_motion.sh

set -euo pipefail

SBATCH_SCRIPT=/scratch/ap7151/sln-fixed/src/detect_peaks.sbatch
RAW_DATA=/scratch/ap7151/RAW_DATA/extra-motion

sessions=(
    dataset1_p1
    dataset1_p2
    dataset2_p1
    dataset2_p2
    dataset3_p1
    dataset3_p2
)

for session in "${sessions[@]}"; do
    recording_path="${RAW_DATA}/${session}"
    echo "Submitting: $session"
    sbatch \
        --job-name="peaks_${session}" \
        --export="RECORDING_PATH=${recording_path},SESSION_ID=${session}" \
        "$SBATCH_SCRIPT"
done
