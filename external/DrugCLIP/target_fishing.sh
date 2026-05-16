#!/usr/bin/env bash

#####################################################################################
# Usage:                                                                            #
#                                                                                   #
# ./target_fishing.sh              # local default: GPU 0                           #
# GPU_ID=0 ./target_fishing.sh     # explicit local GPU                             #
# srun ... ./target_fishing.sh     # Sagres: respect SLURM's CUDA_VISIBLE_DEVICES   #
#                                                                                   #
# Target fishing is the inverse of retrieval / virtual screening:                   #
#                                                                                   #
#   retrieval:       query pocket(s)   -> rank candidate molecules                  #
#   target fishing:  query molecule(s) -> rank candidate pockets                    #
#                                                                                   #
# The query molecule LMDB is MOL_PATH. The candidate pocket LMDB is POCKET_PATH.    #
# Candidate-pocket embeddings are cached in EMB_DIR so later target-fishing runs    #
# against the same pocket library can reuse them.                                   #
#####################################################################################

results_path="./test"  # kept for Uni-Core argument parity with retrieval.sh
batch_size=8
batch_size_valid=8
top_k=10000
weight_path="checkpoint_best.pt"
DATA_PATH="${DATA_PATH:-./data}" # path containing DrugCLIP dictionaries
MOL_PATH="${MOL_PATH:-../../data/query_mol.lmdb}" # query molecule LMDB
POCKET_PATH="${POCKET_PATH:-../../data/candidate_pockets.lmdb}" # candidate pockets
EMB_DIR="${EMB_DIR:-./data/pocket_emb}" # cached candidate-pocket embeddings

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
fi

python ./unimol/target_fishing.py \
       "$DATA_PATH" \
       --arch drugclip \
       --batch-size "$batch_size" \
       --batch-size-valid "$batch_size_valid" \
       --ddp-backend c10d \
       --emb-dir "$EMB_DIR" \
       --fp16 \
       --fp16-init-scale 4 \
       --fp16-scale-window 256 \
       --log-format simple \
       --log-interval 100 \
       --loss in_batch_softmax \
       --max-pocket-atoms 256 \
       --mol-path "$MOL_PATH" \
       --num-workers 8 \
       --path "$weight_path" \
       --pocket-path "$POCKET_PATH" \
       --results-path "$results_path" \
       --seed 1 \
       --task drugclip \
       --top-k "$top_k" \
       --user-dir ./unimol \
       --valid-subset test
