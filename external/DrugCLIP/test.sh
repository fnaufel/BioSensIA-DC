
#########################################################################
# Usage:                                                                #
#                                                                       #
# ./test.sh              # local default: GPU 0                         #
# GPU_ID=0 ./test.sh     # explicit local GPU                           #
# srun ... ./test.sh     # Sagres: respect SLURM's CUDA_VISIBLE_DEVICES #
#########################################################################

results_path="./test"  # replace to your results path
batch_size=8
weight_path="checkpoint_best.pt"

TASK="PCBA" # DUDE or PCBA

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
fi

python ./unimol/test.py \
       --user-dir ./unimol $data_path "./data" \
       --valid-subset test \
       --results-path $results_path \
       --num-workers 8 \
       --ddp-backend=c10d \
       --batch-size $batch_size \
       --task drugclip \
       --loss in_batch_softmax \
       --arch drugclip  \
       --fp16 \
       --fp16-init-scale 4 \
       --fp16-scale-window 256  \
       --seed 1 \
       --path $weight_path \
       --log-interval 100 \
       --log-format simple \
       --max-pocket-atoms 511 \
       --test-task $TASK \
