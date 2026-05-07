#####################################################################################
# Usage:                                                                            #
#                                                                                   #
# ./retrieval.sh              # local default: GPU 0                                #
# GPU_ID=0 ./retrieval.sh     # explicit local GPU                                  #
# srun ... ./retrieval.sh     # Sagres: respect SLURM's CUDA_VISIBLE_DEVICES        #
#####################################################################################

results_path="./test"  # replace to your results path
batch_size=8
batch_size_valid=8
top_k=10000
weight_path="checkpoint_best.pt"
MOL_PATH="mols.lmdb" # path to the molecule file
POCKET_PATH="../../data/2ie4.lmdb" # path to the pocket file
EMB_DIR="./data/emb" # path to the cached mol embedding file

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
fi

python ./unimol/retrieval.py \
       $data_path "./data" \
       --arch drugclip \
       --batch-size $batch_size \
       --batch-size-valid $batch_size_valid \
       --ddp-backend c10d \
       --emb-dir $EMB_DIR \
       --fp16 \
       --fp16-init-scale 4 \
       --fp16-scale-window 256 \
       --log-format simple \
       --log-interval 100 \
       --loss in_batch_softmax \
       --max-pocket-atoms 256 \
       --mol-path $MOL_PATH \
       --num-workers 8 \
       --path $weight_path \
       --pocket-path $POCKET_PATH \
       --results-path $results_path \
       --seed 1 \
       --task drugclip \
       --top-k $top_k \
       --user-dir ./unimol \
       --valid-subset test
