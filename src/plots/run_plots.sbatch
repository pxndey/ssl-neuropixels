#!/bin/bash
#SBATCH --job-name=sln_plots
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/plots_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/plots_%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --account=torch_pr_60_general

set -euo pipefail

REPO=/scratch/ap7151/sln-v2
OUT_DIR="${REPO}/plots"

mkdir -p "${REPO}/slurm_logs"
mkdir -p "${OUT_DIR}/raytune"
mkdir -p "${OUT_DIR}/embeddings"

echo "=== SLN-v2 Plots ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start:  $(date)"
echo ""

# ---------------------------------------------------------------------------
# 1) RayTune analysis plots (CPU-only, fast)
singularity exec --overlay /scratch/${USER}/envs/pytorch-plotly.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/plots/plot_raytune.py --out-dir ${OUT_DIR}/raytune"

# ---------------------------------------------------------------------------
# 2) Embedding PCA for existing good checkpoints
#    (edit these paths if you want a different session)
# ---------------------------------------------------------------------------

# NP12 good checkpoint (dataset1_p1, default normalized)
echo "--- Extracting NP12 embeddings ---"
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch-plotly.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/plots/plot_embeddings.py \
        --checkpoint ${REPO}/checkpoints/dataset1_p1/localizer.pt \
        --session-path ${REPO}/runs/dataset1_p1 \
        --out-dir ${OUT_DIR}/embeddings"

# NP Ultra good checkpoint (dandi session)
echo "--- Extracting NP Ultra embeddings ---"
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch-plotly.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/plots/plot_embeddings.py \
        --checkpoint ${REPO}/checkpoints/dandi_000957_sub-ZYE-0021_ses-1/localizer.pt \
        --session-path ${REPO}/runs/dandi_000957_sub-ZYE-0021_ses-1 \
        --out-dir ${OUT_DIR}/embeddings"

# NP Ultra good checkpoint (dandi session)
echo "--- Extracting NP Ultra embeddings ---"
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/plots/plot_embeddings.py \
        --checkpoint ${REPO}/checkpoints/dandi_000957_sub-ZYE-0021_ses-1/localizer.pt \
        --session-path ${REPO}/runs/dandi_000957_sub-ZYE-0021_ses-1 \
        --out-dir ${OUT_DIR}/embeddings"

echo ""
echo "Completed at: $(date)"
echo ""
echo "RayTune plots  -> ${OUT_DIR}/raytune/"
echo "Embedding PCA  -> ${OUT_DIR}/embeddings/"
