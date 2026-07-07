"""Run target-fishing ranking benchmarks for DrugCLIP/BioSensIA checkpoints."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import lmdb
import polars as pl

from biosensia_finetuning import target_fishing_rank_metrics
from biosensia_target_fishing import (
    DEFAULT_CANDIDATE_POCKETS_LMDB,
    DEFAULT_DRUGCLIP_DIR,
    DEFAULT_QUERY_MOL_LMDB,
    DEFAULT_TARGET_FISHING_TOP_K,
    build_drugclip_target_fishing_args,
    retrieve_pocket_rankings_from_drugclip,
)


def run_target_fishing_benchmark(
    *,
    positives_path: str | Path,
    query_column: str = "query",
    pocket_column: str = "pocket",
    drugclip_dir: str | Path = DEFAULT_DRUGCLIP_DIR,
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    mol_path: str | Path = DEFAULT_QUERY_MOL_LMDB,
    pocket_path: str | Path = DEFAULT_CANDIDATE_POCKETS_LMDB,
    emb_dir: str | Path | None = None,
    top_k: int = DEFAULT_TARGET_FISHING_TOP_K,
    metric_top_k: tuple[int, ...] = (1, 3, 5, 10),
    batch_size: int = 2,
    batch_size_valid: int = 2,
    num_workers: int = 8,
    seed: int = 1,
    fp16: bool = True,
    cpu: bool = False,
) -> dict[str, Any]:
    """Run a BioSensIA target-fishing benchmark for one DrugCLIP checkpoint.

    This is the programmatic entry point behind the CLI in this module. It
    evaluates the inverse of DrugCLIP's usual virtual-screening setup: each
    query molecule is compared with a library of candidate pockets, and the
    benchmark checks whether the known positive pocket identities for that
    molecule appear near the top of the ranked pocket list.

    ``positives_path`` must point to a table in ``.parquet``, ``.csv``,
    ``.jsonl``, or ``.ndjson`` format. The table needs at least two columns:
    one query identifier column and one pocket identifier column. Their names
    are controlled by ``query_column`` and ``pocket_column``. The values in
    ``query_column`` must match the molecule names returned from ``mol_path``;
    the values in ``pocket_column`` must match the candidate pocket names
    returned from ``pocket_path``. Multiple rows may share the same query
    identifier, which represents a multi-positive target-fishing case.

    The model checkpoint, DrugCLIP data directory, query molecule LMDB, pocket
    LMDB, embedding cache directory, batch sizes, worker count, seed, precision,
    and device mode are passed through to the BioSensIA target-fishing helpers
    built around the DrugCLIP task/model APIs. If ``emb_dir`` already contains
    a candidate-pocket embedding cache for the same pocket LMDB basename and
    checkpoint tag, DrugCLIP loads that cache instead of embedding the pockets
    again. Query molecule embeddings are recomputed for the supplied
    ``mol_path``.

    ``top_k`` controls how many ranked pockets are retrieved per query.
    ``metric_top_k`` controls which top-k cutoffs are reported by
    :func:`biosensia_finetuning.target_fishing_rank_metrics`. For interpretable
    metrics, ``top_k`` should be at least ``max(metric_top_k)`` because metrics
    are computed from the retrieved ranked list.

    Returns:
        A JSON-serializable dictionary containing the checkpoint path, query
        molecule LMDB path, candidate pocket LMDB path, positives table path,
        retrieval depth, and target-fishing metrics such as MRR, top-k
        accuracy, and recall@k.
    """

    positives_by_query = read_positive_pairs(
        positives_path,
        query_column=query_column,
        pocket_column=pocket_column,
    )
    args = build_drugclip_target_fishing_args(
        drugclip_dir=drugclip_dir,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        mol_path=mol_path,
        pocket_path=pocket_path,
        emb_dir=emb_dir,
        top_k=top_k,
        batch_size=batch_size,
        batch_size_valid=batch_size_valid,
        num_workers=num_workers,
        seed=seed,
        fp16=fp16,
        cpu=cpu,
    )
    rankings, _scores = retrieve_pocket_rankings_from_drugclip(args)
    metrics = target_fishing_rank_metrics(
        rankings,
        positives_by_query,
        top_k_values=metric_top_k,
    )
    return {
        "checkpoint_path": str(args.path),
        "mol_path": str(args.mol_path),
        "pocket_path": str(args.pocket_path),
        "positives_path": str(positives_path),
        "top_k": top_k,
        "metrics": metrics,
    }


def read_positive_pairs(
    positives_path: str | Path,
    *,
    query_column: str,
    pocket_column: str,
) -> dict[str, set[str]]:
    positives_path = Path(positives_path)
    df = _read_frame(positives_path)
    missing = {query_column, pocket_column}.difference(df.columns)
    if missing:
        raise ValueError(f"Positive-pair table is missing columns: {sorted(missing)}")
    positives: dict[str, set[str]] = defaultdict(set)
    for row in df.select([query_column, pocket_column]).iter_rows(named=True):
        positives[str(row[query_column])].add(str(row[pocket_column]))
    return dict(positives)


def build_positive_pairs_frame_from_lmdb(
    lmdb_path: str | Path,
    *,
    query_field: str = "smi",
    pocket_field: str = "pocket",
    query_column: str = "query",
    pocket_column: str = "pocket",
    unique: bool = True,
) -> pl.DataFrame:
    """Build a positive-pair table from a DrugCLIP-style LMDB split.

    DrugCLIP train/valid LMDB records already encode positive pairs: the ligand
    fields in a record bind to the pocket fields in that same record. This
    helper converts those records into the explicit two-column ground-truth
    table consumed by :func:`run_target_fishing_benchmark`.

    By default, the output columns are ``query`` and ``pocket``, populated from
    the source record's ``smi`` and ``pocket`` fields. Use ``query_field`` and
    ``pocket_field`` when the benchmark should use a different identity, such
    as ``ligand_key`` or ``pocket_geometry_hash``.
    """

    rows = []
    for lmdb_key, record in _iter_pickled_lmdb_records(Path(lmdb_path)):
        missing = [
            field
            for field in (query_field, pocket_field)
            if field not in record or record[field] is None or record[field] == ""
        ]
        if missing:
            raise ValueError(
                f"LMDB record {lmdb_key} in {lmdb_path} is missing "
                f"required field(s): {missing}"
            )
        rows.append(
            {
                query_column: str(record[query_field]),
                pocket_column: str(record[pocket_field]),
            }
        )

    df = pl.DataFrame(
        rows,
        schema={
            query_column: pl.String,
            pocket_column: pl.String,
        },
        orient="row",
    )
    return df.unique(maintain_order=True) if unique else df


def write_positive_pairs_from_lmdb(
    lmdb_path: str | Path,
    output_path: str | Path,
    *,
    query_field: str = "smi",
    pocket_field: str = "pocket",
    query_column: str = "query",
    pocket_column: str = "pocket",
    unique: bool = True,
) -> pl.DataFrame:
    """Write an explicit positive-pair table derived from an LMDB split."""

    output_path = Path(output_path)
    df = build_positive_pairs_frame_from_lmdb(
        lmdb_path,
        query_field=query_field,
        pocket_field=pocket_field,
        query_column=query_column,
        pocket_column=pocket_column,
        unique=unique,
    )
    _write_frame(df, output_path)
    return df


def _read_frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(path)
    raise ValueError("positives file must be .parquet, .csv, .jsonl, or .ndjson")


def _write_frame(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.write_parquet(path)
        return
    if suffix == ".csv":
        df.write_csv(path)
        return
    if suffix in {".jsonl", ".ndjson"}:
        df.write_ndjson(path)
        return
    raise ValueError("positives file must be .parquet, .csv, .jsonl, or .ndjson")


def _iter_pickled_lmdb_records(path: Path):
    env = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with env.begin() as transaction:
            keys = list(transaction.cursor().iternext(values=False))
            keys = _sort_lmdb_keys(keys)
            for key in keys:
                value = transaction.get(key)
                if value is not None:
                    yield key.decode("ascii", errors="replace"), pickle.loads(value)
    finally:
        env.close()


def _sort_lmdb_keys(keys: list[bytes]) -> list[bytes]:
    try:
        return sorted(keys, key=lambda key: int(key.decode("ascii")))
    except ValueError:
        return sorted(keys)


def _parse_top_k(value: str) -> tuple[int, ...]:
    top_k = tuple(int(item) for item in value.split(",") if item.strip())
    if not top_k or any(item <= 0 for item in top_k):
        raise argparse.ArgumentTypeError("top-k values must be positive integers")
    return top_k


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark target-fishing rankings for one checkpoint."
    )
    parser.add_argument("--positives-path", type=Path, required=True)
    parser.add_argument("--query-column", default="query")
    parser.add_argument("--pocket-column", default="pocket")
    parser.add_argument("--drugclip-dir", type=Path, default=DEFAULT_DRUGCLIP_DIR)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--mol-path", type=Path, default=DEFAULT_QUERY_MOL_LMDB)
    parser.add_argument("--pocket-path", type=Path, default=DEFAULT_CANDIDATE_POCKETS_LMDB)
    parser.add_argument("--emb-dir", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TARGET_FISHING_TOP_K)
    parser.add_argument("--metric-top-k", type=_parse_top_k, default=(1, 3, 5, 10))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-size-valid", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_target_fishing_benchmark(
        positives_path=args.positives_path,
        query_column=args.query_column,
        pocket_column=args.pocket_column,
        drugclip_dir=args.drugclip_dir,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        mol_path=args.mol_path,
        pocket_path=args.pocket_path,
        emb_dir=args.emb_dir,
        top_k=args.top_k,
        metric_top_k=args.metric_top_k,
        batch_size=args.batch_size,
        batch_size_valid=args.batch_size_valid,
        num_workers=args.num_workers,
        seed=args.seed,
        fp16=not args.no_fp16,
        cpu=args.cpu,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
