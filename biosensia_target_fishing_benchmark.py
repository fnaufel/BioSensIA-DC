"""Run target-fishing ranking benchmarks for DrugCLIP/BioSensIA checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

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


def _read_frame(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pl.read_ndjson(path)
    raise ValueError("positives file must be .parquet, .csv, .jsonl, or .ndjson")


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
