"""Utilities for BioSensIA-DC DrugCLIP fine-tuning data and metrics."""

from __future__ import annotations

import argparse
import pickle
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import polars as pl
from rdkit import Chem

from lmdb_helpers import write_lmdb_records


DEFAULT_DRUGCLIP_DATA_DIR = Path("external/DrugCLIP/data")
DEFAULT_FINETUNE_DATA_DIR = Path("data/biosensia_finetune")
DEFAULT_PAIR_TABLE = DEFAULT_FINETUNE_DATA_DIR / "training_data_pairs.parquet"
DEFAULT_SPLITS = ("train", "valid")


def build_biosensia_finetuning_data(
    *,
    source_data_dir: str | Path = DEFAULT_DRUGCLIP_DATA_DIR,
    output_data_dir: str | Path = DEFAULT_FINETUNE_DATA_DIR,
    pair_table_path: str | Path | None = DEFAULT_PAIR_TABLE,
    splits: Sequence[str] = DEFAULT_SPLITS,
    ligand_policy: str = "inchikey_or_smiles",
    pocket_policy: str = "metadata_pocket",
    overwrite: bool = True,
    map_size: int = 1 << 40,
) -> dict[str, Any]:
    """Create a DrugCLIP data directory annotated for BioSensIA fine-tuning.

    The output directory contains copies of ``dict_mol.txt`` and ``dict_pkt.txt``
    plus LMDB files with the same records as the source splits. Each record gets
    two extra fields:

    ``ligand_key``
        Reproducible ligand identity used to group multiple positive pockets.

    ``pocket_key``
        Reproducible pocket identity used by the positive-pair mask.
    """

    source_data_dir = Path(source_data_dir)
    output_data_dir = Path(output_data_dir)
    output_data_dir.mkdir(parents=True, exist_ok=True)

    for dictionary_name in ("dict_mol.txt", "dict_pkt.txt"):
        source_dictionary = source_data_dir / dictionary_name
        if not source_dictionary.exists():
            raise FileNotFoundError(f"DrugCLIP dictionary not found: {source_dictionary}")
        shutil.copy2(source_dictionary, output_data_dir / dictionary_name)

    metadata = _load_pair_metadata(pair_table_path)
    split_summaries = {}
    for split in splits:
        source_lmdb = source_data_dir / f"{split}.lmdb"
        output_lmdb = output_data_dir / f"{split}.lmdb"
        split_summaries[split] = annotate_lmdb_records(
            source_lmdb,
            output_lmdb,
            split=split,
            pair_metadata=metadata,
            ligand_policy=ligand_policy,
            pocket_policy=pocket_policy,
            overwrite=overwrite,
            map_size=map_size,
        )

    return {
        "source_data_dir": str(source_data_dir),
        "output_data_dir": str(output_data_dir),
        "pair_table_path": str(pair_table_path) if pair_table_path else None,
        "ligand_policy": ligand_policy,
        "pocket_policy": pocket_policy,
        "splits": split_summaries,
    }


def annotate_lmdb_records(
    source_lmdb: str | Path,
    output_lmdb: str | Path,
    *,
    split: str,
    pair_metadata: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    ligand_policy: str = "inchikey_or_smiles",
    pocket_policy: str = "metadata_pocket",
    overwrite: bool = True,
    map_size: int = 1 << 40,
) -> dict[str, Any]:
    """Copy one DrugCLIP LMDB split and add BioSensIA identity keys."""

    source_lmdb = Path(source_lmdb)
    if not source_lmdb.exists():
        raise FileNotFoundError(f"LMDB split not found: {source_lmdb}")

    records = []
    ligand_keys = set()
    pocket_keys = set()
    metadata_hits = 0
    for lmdb_key, record in _iter_lmdb_records(source_lmdb):
        metadata_row = (
            pair_metadata.get((split, lmdb_key))
            if pair_metadata is not None
            else None
        )
        if metadata_row is not None:
            metadata_hits += 1
        annotated = dict(record)
        ligand_key = choose_ligand_key(
            record,
            metadata_row,
            policy=ligand_policy,
        )
        pocket_key = choose_pocket_key(
            record,
            metadata_row,
            policy=pocket_policy,
        )
        annotated["ligand_key"] = ligand_key
        annotated["pocket_key"] = pocket_key
        annotated["biosensia_ligand_policy"] = ligand_policy
        annotated["biosensia_pocket_policy"] = pocket_policy
        records.append(annotated)
        ligand_keys.add(ligand_key)
        pocket_keys.add(pocket_key)

    write_lmdb_records(records, output_lmdb, overwrite=overwrite, map_size=map_size)
    return {
        "source_lmdb": str(source_lmdb),
        "output_lmdb": str(output_lmdb),
        "records": len(records),
        "metadata_hits": metadata_hits,
        "ligands": len(ligand_keys),
        "pockets": len(pocket_keys),
    }


def choose_ligand_key(
    record: Mapping[str, Any],
    metadata_row: Mapping[str, Any] | None,
    *,
    policy: str,
) -> str:
    """Return the ligand grouping key for one record."""

    metadata_row = metadata_row or {}
    if policy == "inchikey":
        return _first_nonempty(
            metadata_row.get("ligand_inchikey"),
            _mol_to_inchikey(record.get("mol")),
            _smiles_to_inchikey(record.get("smi")),
            required_name="ligand InChIKey",
        )
    if policy == "canonical_smiles":
        return _first_nonempty(
            metadata_row.get("ligand_smiles"),
            _mol_to_smiles(record.get("mol")),
            _smiles_to_canonical_smiles(record.get("smi")),
            record.get("smi"),
            required_name="ligand SMILES",
        )
    if policy == "inchikey_or_smiles":
        return _first_nonempty(
            metadata_row.get("ligand_inchikey"),
            _mol_to_inchikey(record.get("mol")),
            _smiles_to_inchikey(record.get("smi")),
            metadata_row.get("ligand_smiles"),
            _mol_to_smiles(record.get("mol")),
            _smiles_to_canonical_smiles(record.get("smi")),
            record.get("smi"),
            required_name="ligand identity",
        )
    if policy == "raw_smi":
        return _first_nonempty(record.get("smi"), required_name="raw SMILES")
    raise ValueError(
        "ligand_policy must be one of: inchikey, canonical_smiles, "
        "inchikey_or_smiles, raw_smi"
    )


def choose_pocket_key(
    record: Mapping[str, Any],
    metadata_row: Mapping[str, Any] | None,
    *,
    policy: str,
) -> str:
    """Return the pocket grouping key for one record."""

    metadata_row = metadata_row or {}
    if policy == "metadata_pocket":
        return _first_nonempty(
            metadata_row.get("pocket"),
            record.get("pocket"),
            required_name="pocket identity",
        )
    if policy == "geometry_hash":
        return _first_nonempty(
            metadata_row.get("pocket_geometry_hash"),
            record.get("pocket"),
            required_name="pocket geometry hash",
        )
    if policy == "raw_pocket":
        return _first_nonempty(record.get("pocket"), required_name="raw pocket")
    raise ValueError(
        "pocket_policy must be one of: metadata_pocket, geometry_hash, raw_pocket"
    )


def target_fishing_rank_metrics(
    rankings: Mapping[str, Sequence[str]],
    positives_by_query: Mapping[str, Iterable[str]],
    *,
    top_k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Compute target-fishing ranking metrics from named pocket rankings."""

    query_count = 0
    reciprocal_rank_sum = 0.0
    hit_counts = {int(top_k): 0 for top_k in top_k_values}
    recall_sums = {int(top_k): 0.0 for top_k in top_k_values}

    for query_key, ranking in rankings.items():
        positives = set(positives_by_query.get(query_key, ()))
        if not positives:
            continue
        query_count += 1
        ranking = list(ranking)
        first_rank = None
        for rank, pocket_key in enumerate(ranking, start=1):
            if pocket_key in positives:
                first_rank = rank
                break
        if first_rank is not None:
            reciprocal_rank_sum += 1.0 / first_rank
        for top_k in hit_counts:
            top_hits = positives.intersection(ranking[:top_k])
            hit_counts[top_k] += bool(top_hits)
            recall_sums[top_k] += len(top_hits) / len(positives)

    if query_count == 0:
        raise ValueError("No benchmark queries have known positives")

    metrics = {"queries": float(query_count), "mrr": reciprocal_rank_sum / query_count}
    for top_k in sorted(hit_counts):
        metrics[f"top{top_k}_accuracy"] = hit_counts[top_k] / query_count
        metrics[f"recall_at_{top_k}"] = recall_sums[top_k] / query_count
    return metrics


def _load_pair_metadata(
    pair_table_path: str | Path | None,
) -> dict[tuple[str, str], dict[str, Any]] | None:
    if pair_table_path is None:
        return None
    pair_table_path = Path(pair_table_path)
    if not pair_table_path.exists():
        return None
    df = pl.read_parquet(pair_table_path)
    required = {"split", "lmdb_key"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Pair table is missing required columns: {sorted(missing)}")
    metadata = {}
    for row in df.iter_rows(named=True):
        metadata[(str(row["split"]), str(row["lmdb_key"]))] = row
    return metadata


def _iter_lmdb_records(path: Path):
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
                    yield key.decode("ascii"), pickle.loads(value)
    finally:
        env.close()


def _sort_lmdb_keys(keys: list[bytes]) -> list[bytes]:
    try:
        return sorted(keys, key=lambda key: int(key.decode("ascii")))
    except ValueError:
        return sorted(keys)


def _first_nonempty(*values: Any, required_name: str) -> str:
    for value in values:
        if value is not None and value != "":
            if isinstance(value, float) and np.isnan(value):
                continue
            return str(value)
    raise ValueError(f"Could not determine {required_name}")


def _mol_to_smiles(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _mol_to_inchikey(mol: Chem.Mol | None) -> str | None:
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def _smiles_to_canonical_smiles(smiles: Any) -> str | None:
    if smiles is None or smiles == "":
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return _mol_to_smiles(mol)


def _smiles_to_inchikey(smiles: Any) -> str | None:
    if smiles is None or smiles == "":
        return None
    mol = Chem.MolFromSmiles(str(smiles))
    return _mol_to_inchikey(mol)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BioSensIA-annotated DrugCLIP fine-tuning LMDBs."
    )
    parser.add_argument(
        "--source-data-dir",
        type=Path,
        default=DEFAULT_DRUGCLIP_DATA_DIR,
        help=f"DrugCLIP data directory. Default: {DEFAULT_DRUGCLIP_DATA_DIR}",
    )
    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=DEFAULT_FINETUNE_DATA_DIR,
        help=f"Annotated output data directory. Default: {DEFAULT_FINETUNE_DATA_DIR}",
    )
    parser.add_argument(
        "--pair-table",
        type=Path,
        default=DEFAULT_PAIR_TABLE,
        help=f"Optional metadata parquet. Default: {DEFAULT_PAIR_TABLE}",
    )
    parser.add_argument(
        "--splits",
        default="train,valid",
        help="comma-separated LMDB split names to annotate",
    )
    parser.add_argument(
        "--ligand-policy",
        default="inchikey_or_smiles",
        choices=["inchikey", "canonical_smiles", "inchikey_or_smiles", "raw_smi"],
    )
    parser.add_argument(
        "--pocket-policy",
        default="metadata_pocket",
        choices=["metadata_pocket", "geometry_hash", "raw_pocket"],
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="fail if an output LMDB already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    summary = build_biosensia_finetuning_data(
        source_data_dir=args.source_data_dir,
        output_data_dir=args.output_data_dir,
        pair_table_path=args.pair_table,
        splits=splits,
        ligand_policy=args.ligand_policy,
        pocket_policy=args.pocket_policy,
        overwrite=not args.no_overwrite,
    )
    print(summary)


if __name__ == "__main__":
    main()
