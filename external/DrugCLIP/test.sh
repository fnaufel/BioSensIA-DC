
#########################################################################
# Usage:                                                                #
#                                                                       #
# ./test.sh              # local default: GPU 0                         #
# GPU_ID=0 ./test.sh     # explicit local GPU                           #
# srun ... ./test.sh     # Sagres: respect SLURM's CUDA_VISIBLE_DEVICES #
#########################################################################

results_path="./test"  # replace to your results path
batch_size=8
batch_size_valid=8
weight_path="checkpoint_best.pt"

TASK="PCBA" # DUDE or PCBA

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
fi

python ./unimol/test.py \
       $data_path "./data" \
       --arch drugclip \
       --batch-size $batch_size \
       --batch-size-valid $batch_size_valid \
       --ddp-backend c10d \
       --fp16 \
       --fp16-init-scale 4 \
       --fp16-scale-window 256 \
       --log-format simple \
       --log-interval 100 \
       --loss in_batch_softmax \
       --max-pocket-atoms 511 \
       --num-workers 8 \
       --path $weight_path \
       --results-path $results_path \
       --seed 1 \
       --task drugclip \
       --test-task $TASK \
       --user-dir ./unimol \
       --valid-subset test
