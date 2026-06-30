#!/usr/bin/env bash
#SBATCH --job-name=finetune2gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=fine-tune-%j.out
#SBATCH --error=fine-tune-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"

# Load/activate your DrugCLIP environment here if needed.
spack load cuda@12.4.1
source "$REPO_DIR/.venv/bin/activate"

export RUN_NAME=projection_lr1e-4_tau071_lambda-mp1pm0_bs32_uf2_ppl4
export RUN_DIR="$PWD/runs/biosensia_finetune/$RUN_NAME"

SAVE_DIR="$RUN_DIR/checkpoints" \
TMP_SAVE_DIR="$RUN_DIR/tmp" \
TSB_DIR="$RUN_DIR/tensorboard" \
DATA_DIR="$PWD/data/biosensia_finetune" \
INIT_CHECKPOINT="$PWD/external/DrugCLIP/checkpoint_best_original.pt" \
TRAINABLE_PARAMS=projection \
LR=1e-4 \
TEMPERATURE=0.07142857142857142 \
MAX_EPOCH=50 \
BATCH_SIZE=32 \
BATCH_SIZE_VALID=32 \
UPDATE_FREQ=2 \
POSITIVES_PER_LIGAND=4 \
LAMBDA_MOL_TO_POCKET=1.0 \
LAMBDA_POCKET_TO_MOL=0.0 \
METRIC_TOP_K=1,3,5 \
BEST_METRIC=valid_m2p_mrr \
bash train_biosensia_finetune.sh

