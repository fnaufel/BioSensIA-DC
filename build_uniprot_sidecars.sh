#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Uso:
  ./build_uniprot_sidecars.sh [CANDIDATE_LMDB] [OUTPUT_DIR] [CACHE_DIR]

Variáveis de ambiente opcionais:
  BATCH_SIZE  Número máximo de IDs PDB por consulta GraphQL (padrão: 100).
  REFRESH     true/1/yes para ignorar o cache normalizado (padrão: false).

Valores posicionais padrão:
  CANDIDATE_LMDB  data/candidate_pockets.lmdb
  OUTPUT_DIR      data
  CACHE_DIR       OUTPUT_DIR/pdb_graphql_cache
EOF
    exit 0
fi

candidate_lmdb="${1:-data/candidate_pockets.lmdb}"
output_dir="${2:-data}"
cache_dir="${3:-${output_dir}/pdb_graphql_cache}"
batch_size="${BATCH_SIZE:-100}"
refresh="${REFRESH:-false}"

printf 'Starting UniProt sidecar build for %s...\n' "${candidate_lmdb}"

uv run python - \
    "${candidate_lmdb}" \
    "${output_dir}" \
    "${cache_dir}" \
    "${batch_size}" \
    "${refresh}" <<'PY'
from pathlib import Path
import sys

from biosensia_uniprot_enrichment import build_uniprot_metadata_sidecars


candidate_lmdb = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
cache_dir = Path(sys.argv[3])
batch_size = int(sys.argv[4])
refresh = sys.argv[5].strip().lower() in {"1", "true", "yes", "sim"}

result = build_uniprot_metadata_sidecars(
    candidate_lmdb,
    output_dir=output_dir,
    cache_dir=cache_dir,
    batch_size=batch_size,
    refresh=refresh,
    show_progress=True,
)

print(f"Candidatos indexados: {result['candidate_rows']}")
print(f"Entradas PDB consultadas: {result['unique_pdb_ids']}")
print(f"SHA-256 lógico da biblioteca: {result['candidate_library_sha256']}")
print(f"Índice de candidatos: {result['candidate_index_path']}")
print(f"Resumo por PDB: {result['pdb_metadata_path']}")
print(f"Relação entidade–UniProt: {result['entity_metadata_path']}")
print(f"Cache GraphQL: {result['cache_dir']}")
PY
