#!/usr/bin/env bash
set -euo pipefail

#####################################################################################
# Usage:                                                                            #
#                                                                                   #
# ./retrieve_ranked_compounds.sh                 # local default: GPU 0             #
# GPU_ID=0 ./retrieve_ranked_compounds.sh        # explicit local GPU               #
# ENRICH_PUBCHEM=0 ./retrieve_ranked_compounds.sh  # disable PubChem enrichment     #
# ENRICH_CHEMBL=0 ./retrieve_ranked_compounds.sh   # disable ChEMBL enrichment      #
# PUBCHEM_LIMIT=20 ./retrieve_ranked_compounds.sh  # enrich only first 20 PubChem   #
# CHEMBL_LIMIT=20 ./retrieve_ranked_compounds.sh   # enrich only first 20 ChEMBL    #
# srun ... ./retrieve_ranked_compounds.sh        # respect SLURM CUDA visibility    #
#                                                                                   #
# If PUBCHEM_LIMIT or CHEMBL_LIMIT are unset, no corresponding --*-limit argument   #
# is passed, so biosensia_ranked_compounds.py uses its default: no limit.           #
#####################################################################################

TOP_K="${TOP_K:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
BATCH_SIZE_VALID="${BATCH_SIZE_VALID:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
POCKET_PATH="${POCKET_PATH:-data/query-pocket.lmdb}"
OUTPUT_PARQUET="${OUTPUT_PARQUET:-external/DrugCLIP/data/emb/ranked_compounds_enriched.parquet}"

ENRICH_PUBCHEM="${ENRICH_PUBCHEM:-1}"
ENRICH_CHEMBL="${ENRICH_CHEMBL:-1}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
fi

args=(
  "biosensia_ranked_compounds.py"
  --top-k "$TOP_K"
  --batch-size "$BATCH_SIZE"
  --batch-size-valid "$BATCH_SIZE_VALID"
  --num-workers "$NUM_WORKERS"
  --pocket-path "$POCKET_PATH"
  --output-parquet "$OUTPUT_PARQUET"
)

if [ "$ENRICH_PUBCHEM" != "0" ]; then
  args+=(--enrich-pubchem)
  if [ -n "${PUBCHEM_LIMIT:-}" ]; then
    args+=(--pubchem-limit "$PUBCHEM_LIMIT")
  fi
fi

if [ "$ENRICH_CHEMBL" != "0" ]; then
  args+=(--enrich-chembl)
  if [ -n "${CHEMBL_LIMIT:-}" ]; then
    args+=(--chembl-limit "$CHEMBL_LIMIT")
  fi
fi

.venv/bin/python "${args[@]}"
