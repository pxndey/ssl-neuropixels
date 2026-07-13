#!/bin/bash
#SBATCH --job-name=mono_localize
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/mono_localize_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/mono_localize_%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

SESSION_ID="${1:-}"
if [[ -z "$SESSION_ID" ]]; then
    echo "Usage: sbatch run_localize.sbatch <session_id>"
    echo "  sessions: dataset1_p1 | dataset1_p2 | dandi_000957_sub-ZYE-0021_ses-1"
    exit 1
fi

REPO=/scratch/ap7151/sln-v2
echo "=== Monopolar localization (bandpass+CMR+TPCA+monopolar) on OUR spikes ==="
echo "Job ID:  $SLURM_JOB_ID   Session: $SESSION_ID   CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Start:   $(date)"
echo ""
mkdir -p "${REPO}/slurm_logs"

# parallel (n_jobs>1, mp_context=spawn) -> NO --fakeroot (breaks forked children)
singularity exec --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/localizations/pipeline.py \
        --session-id ${SESSION_ID} --n-jobs ${SLURM_CPUS_PER_TASK}"

echo ""
echo "Completed at: $(date)"
