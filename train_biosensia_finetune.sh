#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
DRUGCLIP_DIR="$REPO_DIR/external/DrugCLIP"

DATA_DIR="${DATA_DIR:-$REPO_DIR/data/biosensia_finetune}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/runs/biosensia_finetune/savedir}"
TMP_SAVE_DIR="${TMP_SAVE_DIR:-$REPO_DIR/runs/biosensia_finetune/tmp_save_dir}"
TSB_DIR="${TSB_DIR:-$REPO_DIR/runs/biosensia_finetune/tsb_dir}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-$DRUGCLIP_DIR/checkpoint_best.pt}"

MASTER_PORT="${MASTER_PORT:-10055}"
BATCH_SIZE="${BATCH_SIZE:-12}"
BATCH_SIZE_VALID="${BATCH_SIZE_VALID:-16}"
UPDATE_FREQ="${UPDATE_FREQ:-4}"
MAX_EPOCH="${MAX_EPOCH:-50}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
SEED="${SEED:-1}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_POCKET_ATOMS="${MAX_POCKET_ATOMS:-256}"

TRAINABLE_PARAMS="${TRAINABLE_PARAMS:-projection}"
POSITIVES_PER_LIGAND="${POSITIVES_PER_LIGAND:-2}"
LAMBDA_MOL_TO_POCKET="${LAMBDA_MOL_TO_POCKET:-1.0}"
LAMBDA_POCKET_TO_MOL="${LAMBDA_POCKET_TO_MOL:-0.0}"
TEMPERATURE="${TEMPERATURE:-0.07142857142857142}"
METRIC_TOP_K="${METRIC_TOP_K:-1,3,5}"
BEST_METRIC="${BEST_METRIC:-valid_m2p_mrr}"
PATIENCE="${PATIENCE:-2000}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
fi

if [ -n "${N_GPU:-}" ]; then
  n_gpu="$N_GPU"
else
  visible_gpus="${CUDA_VISIBLE_DEVICES//,/ }"
  n_gpu=0
  for _gpu in $visible_gpus; do
    n_gpu=$((n_gpu + 1))
  done
fi

if [ "$n_gpu" -lt 1 ]; then
  n_gpu=1
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1

if [ -f "$DRUGCLIP_DIR/env-drugclip.sh" ]; then
  source "$DRUGCLIP_DIR/env-drugclip.sh"
fi

mkdir -p "$SAVE_DIR" "$TMP_SAVE_DIR" "$TSB_DIR"

python \
  -m torch.distributed.launch \
  --use-env \
  --nproc_per_node="$n_gpu" \
  --master_port="$MASTER_PORT" \
  "$(which unicore-train)" \
  "$DATA_DIR" \
  --user-dir "$DRUGCLIP_DIR/unimol" \
  --train-subset train \
  --valid-subset valid \
  --num-workers "$NUM_WORKERS" \
  --ddp-backend=c10d \
  --task drugclip \
  --loss biosensia_multi_positive \
  --arch drugclip \
  --max-pocket-atoms "$MAX_POCKET_ATOMS" \
  --optimizer adam \
  --adam-betas "(0.9, 0.999)" \
  --adam-eps 1e-8 \
  --clip-norm 1.0 \
  --lr-scheduler polynomial_decay \
  --lr "$LR" \
  --warmup-ratio "$WARMUP_RATIO" \
  --max-epoch "$MAX_EPOCH" \
  --batch-size "$BATCH_SIZE" \
  --batch-size-valid "$BATCH_SIZE_VALID" \
  --fp16 \
  --fp16-init-scale 4 \
  --fp16-scale-window 256 \
  --update-freq "$UPDATE_FREQ" \
  --seed "$SEED" \
  --tensorboard-logdir "$TSB_DIR" \
  --log-interval 100 \
  --log-format simple \
  --validate-interval 1 \
  --best-checkpoint-metric "$BEST_METRIC" \
  --patience "$PATIENCE" \
  --all-gather-list-size 2048000 \
  --save-dir "$SAVE_DIR" \
  --tmp-save-dir "$TMP_SAVE_DIR" \
  --keep-last-epochs 5 \
  --find-unused-parameters \
  --maximize-best-checkpoint-metric \
  --finetune-from-model "$INIT_CHECKPOINT" \
  --trainable-params "$TRAINABLE_PARAMS" \
  --biosensia-batch-sampler ligand \
  --biosensia-positives-per-ligand "$POSITIVES_PER_LIGAND" \
  --biosensia-lambda-mol-to-pocket "$LAMBDA_MOL_TO_POCKET" \
  --biosensia-lambda-pocket-to-mol "$LAMBDA_POCKET_TO_MOL" \
  --biosensia-temperature "$TEMPERATURE" \
  --biosensia-metric-top-k "$METRIC_TOP_K" \
  --disable-duplicate-mask

