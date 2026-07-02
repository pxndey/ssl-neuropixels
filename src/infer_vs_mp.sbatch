#!/bin/bash
#SBATCH --job-name=diffdredge_vs_mp
#SBATCH --output=/scratch/ap7151/sln-fixed/slurm_logs/diffdredge_vs_mp_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-fixed/slurm_logs/diffdredge_vs_mp_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

# Run our differentiable DREDge on a session's monopolar localizations, then make
# the 2-line (vs mp_dredge) and 1x3 raster figures.
# Usage: sbatch src/infer_vs_mp.sbatch [SESSION_ID]    e.g. dataset3_p2

SID="${1:-dataset1_p1}"
SRC=/scratch/ap7151/sln-fixed/src
RUN=/scratch/ap7151/sln-fixed/runs/${SID}
LOC=/scratch/ap7151/sln/runs/extra-motion/${SID}/preprocessed/shank_0
REF=/scratch/ap7151/sln/runs/extra-motion/${SID}/mp_dredge/motion.npz
SIF=/share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif
OVL=/scratch/ap7151/envs/pytorch.ext3

mkdir -p /scratch/ap7151/sln-fixed/slurm_logs /scratch/ap7151/sln-fixed/plots "$RUN"

echo "=== diff-DREDge vs mp_dredge :: ${SID} ==="
echo "Job ID: ${SLURM_JOB_ID:-?}  Start: $(date)"
nvidia-smi -L || echo "WARNING: no GPU"
echo ""

if [[ ! -f "${LOC}/localizations.npy" ]]; then
    echo "ERROR: ${LOC}/localizations.npy not found" >&2
    exit 1
fi

singularity exec --nv --overlay ${OVL}:ro $SIF /bin/bash -c "
source /ext3/env.sh && cd ${SRC} && \
python -u infer_motion.py \
    --session ${RUN} \
    --loc-npy ${LOC}/localizations.npy \
    --times-npy ${LOC}/spike_times.npy \
    --depth-col 1 --x-col 0 --amp-col 3 \
    --bin-um 2 --bin-s 1 --max-disp-um 150 --mincorr 0.1 --disp hard \
    --bandwidth-um 3 --spike-chunk 50000 --tag _mp --device cuda && \
python -u plot_mp2.py --session-id ${SID} && \
python -u plot_raster3.py --session-id ${SID}
"

echo ""
echo "Completed at: $(date)"
