#!/usr/bin/env bash
#SBATCH --job-name=build_pairs_with_ligand_annotations
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=build_pairs-%j.out
#SBATCH --error=build_pairs-%j.err

set -euo pipefail

source .venv/bin/activate

python - <<PY
import scratch.analyze_pdb_complexes as apc

df = apc.build_complexes_frame(online_ligand=True)
apc.write_frame(df, './scratch/2026-06-07-pairs-with-ligand-annotations.json')
PY

