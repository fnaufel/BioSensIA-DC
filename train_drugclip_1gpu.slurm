#!/usr/bin/env bash
#SBATCH --job-name=drugclip-1gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=drugclip-%j.out
#SBATCH --error=drugclip-%j.err

set -euo pipefail

REPO_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
DRUGCLIP_DIR="$REPO_DIR/external/DrugCLIP"

cd "$DRUGCLIP_DIR"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"

# Load/activate your DrugCLIP environment here if needed.
spack load cuda@12.4.1
source "$REPO_DIR/.venv/bin/activate"

export NUM_WORKERS=4
export N_GPU=1

bash ./drugclip.sh
