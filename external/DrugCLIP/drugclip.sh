#!/usr/bin/env bash

#####################################################################################
# Usage:                                                                            #
#                                                                                   #
# ./drugclip.sh                  # local default: GPU 0                             #
# GPU_ID=0 ./drugclip.sh         # explicit local single GPU                        #
# GPU_ID=0,1 ./drugclip.sh       # explicit local multiple GPUs                     #
# CUDA_VISIBLE_DEVICES=0,1 ./drugclip.sh  # explicit visible GPUs                   #
# N_GPU=2 CUDA_VISIBLE_DEVICES=0,1 ./drugclip.sh  # override process count          #
# srun ... ./drugclip.sh         # Sagres: respect SLURM's CUDA_VISIBLE_DEVICES     #
#                                                                                   #
# If CUDA_VISIBLE_DEVICES is unset, GPU_ID defaults to 0. By default N_GPU is        #
# inferred from CUDA_VISIBLE_DEVICES; set N_GPU explicitly if your launcher needs    #
# a different --nproc_per_node value.                                               #
#####################################################################################


data_path="data"


save_dir="savedir"

tmp_save_dir="tmp_save_dir"
tsb_dir="tsb_dir"

MASTER_PORT=10055
finetune_mol_model="mol_pre_no_h_220816.pt" # unimol pretrained mol model
finetune_pocket_model="pocket_pre_220816.pt" # unimol pretrained pocket model


batch_size=48
batch_size_valid=64
batch_size_valid=128
epoch=200
dropout=0.0
warmup=0.06
update_freq=1
dist_threshold=8.0
recycling=3
lr=1e-3

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
python \
       -m torch.distributed.launch \
       --use-env \
       --nproc_per_node="$n_gpu" \
       --master_port="$MASTER_PORT" \
       "$(which unicore-train)" \
       "$data_path" \
       --user-dir ./unimol \
       --train-subset train \
       --valid-subset valid \
       --num-workers 8 \
       --ddp-backend=c10d \
       --task drugclip \
       --loss in_batch_softmax \
       --arch drugclip \
       --max-pocket-atoms 256 \
       --optimizer adam \
       --adam-betas "(0.9, 0.999)" \
       --adam-eps 1e-8 \
       --clip-norm 1.0 \
       --lr-scheduler polynomial_decay \
       --lr "$lr" \
       --warmup-ratio "$warmup" \
       --max-epoch "$epoch" \
       --batch-size "$batch_size" \
       --batch-size-valid "$batch_size_valid" \
       --fp16 \
       --fp16-init-scale 4 \
       --fp16-scale-window 256 \
       --update-freq "$update_freq" \
       --seed 1 \
       --tensorboard-logdir "$tsb_dir" \
       --log-interval 100 \
       --log-format simple \
       --validate-interval 1 \
       --best-checkpoint-metric valid_bedroc \
       --patience 2000 \
       --all-gather-list-size 2048000 \
       --save-dir "$save_dir" \
       --tmp-save-dir "$tmp_save_dir" \
       --keep-last-epochs 5 \
       --find-unused-parameters \
       --maximize-best-checkpoint-metric \
       --finetune-pocket-model "$finetune_pocket_model" \
       --finetune-mol-model "$finetune_mol_model"
