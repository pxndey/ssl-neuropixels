#!/bin/bash
#SBATCH --job-name=drift_full_p2
#SBATCH --output=/scratch/ap7151/sln-v2/slurm_logs/drift_full_p2_%x_%j.out
#SBATCH --error=/scratch/ap7151/sln-v2/slurm_logs/drift_full_p2_%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --account=torch_pr_60_general
#SBATCH --mail-type=BEGIN,FAIL,CANCEL,END
#SBATCH --mail-user=ap7151@nyu.edu
set -euo pipefail
REPO=/scratch/ap7151/sln-v2
SESSION_PATH=${REPO}/runs/dataset1_p2
CFG=${REPO}/hpo_runs/np12_drift/best_config.json
CKPT=${REPO}/checkpoints/dataset1_p2/drift_localizer.pt
mkdir -p ${REPO}/slurm_logs
echo "=== Drift Full Train (dataset1_p2, from p1 sweep best config) ==="
echo "Job ID: $SLURM_JOB_ID  Start: $(date)"
echo "Config: ${CFG}"
singularity exec --nv --overlay /scratch/${USER}/envs/pytorch.ext3:ro \
    /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
    /bin/bash -c "source /ext3/env.sh && python ${REPO}/src/models/drift_train.py \
        --model-type np12 \
        --session-path ${SESSION_PATH} \
        --config-json ${CFG} \
        --epochs 50 \
        --checkpoint-every 5 \
        --save-predictions ${SESSION_PATH}/inference_drift/predictions.npz"
echo "Completed at: $(date)"
