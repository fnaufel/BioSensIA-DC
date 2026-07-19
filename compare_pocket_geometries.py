#!/usr/bin/env python3
"""Compare fine-tuning LMDB pockets with their ``combine_set`` geometries.

The comparison is deliberately performed in the deposited coordinate frame; it
does not rotate or translate either structure.  A second, heavy-atom-only
comparison distinguishes genuine coordinate changes from hydrogen removal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import polars as pl
from tqdm.auto import tqdm


DEFAULT_LMDBS = (
    Path("data/biosensia_finetune/train.lmdb"),
    Path("data/biosensia_finetune/valid.lmdb"),
)
DEFAULT_COMBINE_SET = Path("external/DrugCLIP/data/pdb/combine_set")
DEFAULT_SIDECAR = Path("data/biosensia_finetune/training_data_pairs.parquet")
DEFAULT_OUTPUT = Path("data/pocket_geometry_comparison.parquet")
DEFAULT_DUPLICATES_OUTPUT = Path("data/pocket_geometry_duplicates.parquet")


def iter_lmdb(
    path: Path, *, show_progress: bool = True
) -> Iterator[tuple[str, Mapping[str, Any]]]:
    env = lmdb.open(
        str(path), subdir=False, readonly=True, lock=False, readahead=False,
        meminit=False, max_readers=256,
    )
    try:
        with env.begin() as transaction:
            records = tqdm(
                transaction.cursor(),
                total=transaction.stat()["entries"],
                desc=f"Comparing {path.name}",
                unit="record",
                disable=not show_progress,
            )
            for key, value in records:
                yield key.decode("ascii", errors="replace"), pickle.loads(value)
    finally:
        env.close()


def normalize_atoms_and_coordinates(
    atoms: Any, coordinates: Any, *, source: str
) -> tuple[list[str], np.ndarray]:
    atom_list = [str(atom).strip() for atom in atoms]
    array = np.asarray(coordinates, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{source}: coordinates have shape {array.shape}, expected (n, 3)")
    if len(atom_list) != len(array):
        raise ValueError(
            f"{source}: {len(atom_list)} atom labels but {len(array)} coordinates"
        )
    return atom_list, array


def is_hydrogen(atom_name: str) -> bool:
    """Recognize PDB-style hydrogen atom names (H, 1HG2, HD1, etc.)."""
    name = atom_name.strip().upper().lstrip("0123456789")
    return name.startswith("H")


def without_hydrogens(
    atoms: Sequence[str], coordinates: np.ndarray
) -> tuple[list[str], np.ndarray]:
    keep = np.fromiter((not is_hydrogen(atom) for atom in atoms), dtype=bool)
    return [atom for atom, selected in zip(atoms, keep) if selected], coordinates[keep]


def geometry_hash(atoms: Sequence[str], coordinates: np.ndarray) -> str:
    digest = hashlib.sha256()
    for atom in atoms:
        digest.update(atom.encode("utf-8"))
        digest.update(b"\0")
    digest.update(np.asarray(coordinates, dtype=np.float32).tobytes())
    return digest.hexdigest()


def comparison_metrics(
    query_atoms: Sequence[str],
    query_coordinates: np.ndarray,
    reference_atoms: Sequence[str],
    reference_coordinates: np.ndarray,
    *,
    atol: float,
) -> dict[str, Any]:
    atoms_equal = list(query_atoms) == list(reference_atoms)
    coordinate_shape_equal = query_coordinates.shape == reference_coordinates.shape
    comparable = atoms_equal and coordinate_shape_equal
    if comparable:
        delta = query_coordinates - reference_coordinates
        max_abs = float(np.max(np.abs(delta))) if delta.size else 0.0
        rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))) if len(delta) else 0.0
        coordinates_equal = bool(np.allclose(
            query_coordinates, reference_coordinates, rtol=0.0, atol=atol
        ))
    else:
        max_abs = None
        rmsd = None
        coordinates_equal = False
    return {
        "atoms_equal": atoms_equal,
        "coordinate_shape_equal": coordinate_shape_equal,
        "coordinates_equal": coordinates_equal,
        "geometry_equal": comparable and coordinates_equal,
        "max_abs_coordinate_error": max_abs,
        "rmsd": rmsd,
    }


def read_pocket_pdb(path: Path) -> tuple[list[str], np.ndarray]:
    atoms: list[str] = []
    coordinates: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[:6].strip() != "ATOM":
                continue
            atoms.append(line[12:16].strip())
            coordinates.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if not atoms:
        raise ValueError(f"{path}: no ATOM records")
    return atoms, np.asarray(coordinates, dtype=np.float64)


def load_reference(combine_set: Path, pdb_id: str) -> tuple[list[str], np.ndarray, Path]:
    directory = combine_set / pdb_id
    data_pkl = directory / "data.pkl"
    if data_pkl.exists():
        with data_pkl.open("rb") as handle:
            record = pickle.load(handle)
        atoms, coordinates = normalize_atoms_and_coordinates(
            record["pocket_atoms"], record["pocket_coordinates"], source=str(data_pkl)
        )
        return atoms, coordinates, data_pkl

    candidates = [directory / f"{pdb_id}_pocket.pdb", *sorted(directory.glob("*pocket*.pdb"))]
    for candidate in dict.fromkeys(candidates):
        if candidate.exists():
            atoms, coordinates = read_pocket_pdb(candidate)
            return atoms, coordinates, candidate
    raise FileNotFoundError(f"no data.pkl or pocket PDB under {directory}")


def load_sidecar(path: Path | None) -> dict[tuple[str, str], str]:
    if path is None or not path.exists():
        return {}
    table = pl.read_parquet(path)
    required = {"split", "lmdb_key"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} lacks required columns: {sorted(required - set(table.columns))}")
    identifier_column = next(
        (column for column in ("pdb_id", "raw_pocket") if column in table.columns), None
    )
    if identifier_column is None:
        raise ValueError(f"{path} has neither pdb_id nor raw_pocket")
    result = {}
    for split, key, identifier in table.select("split", "lmdb_key", identifier_column).iter_rows():
        if identifier is not None and str(identifier).strip():
            result[(str(split), str(key))] = str(identifier).strip().lower()
    return result


def compare(
    lmdb_paths: Sequence[Path],
    combine_set: Path,
    *,
    sidecar_path: Path | None = DEFAULT_SIDECAR,
    atol: float = 1e-4,
    show_progress: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    sidecar = load_sidecar(sidecar_path)
    reference_cache: dict[str, tuple[list[str], np.ndarray, Path] | Exception] = {}
    rows: list[dict[str, Any]] = []
    occurrences: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    for lmdb_path in lmdb_paths:
        split = lmdb_path.stem
        for key, record in iter_lmdb(lmdb_path, show_progress=show_progress):
            raw_id = record.get("pocket")
            pdb_id = sidecar.get((split, key), str(raw_id).strip().lower() if raw_id else "")
            base = {
                "pdb_id": pdb_id,
                "split": split,
                "lmdb_path": str(lmdb_path),
                "lmdb_key": key,
                "raw_pocket": str(raw_id) if raw_id is not None else None,
            }
            try:
                atoms, coords = normalize_atoms_and_coordinates(
                    record["pocket_atoms"], record["pocket_coordinates"],
                    source=f"{lmdb_path}:{key}",
                )
                digest = geometry_hash(atoms, coords)
                occurrences[pdb_id].append((split, key, digest))
                if pdb_id not in reference_cache:
                    try:
                        reference_cache[pdb_id] = load_reference(combine_set, pdb_id)
                    except Exception as error:  # retain error so duplicate IDs are cheap
                        reference_cache[pdb_id] = error
                reference = reference_cache[pdb_id]
                if isinstance(reference, Exception):
                    raise reference
                ref_atoms, ref_coords, ref_path = reference
                full = comparison_metrics(atoms, coords, ref_atoms, ref_coords, atol=atol)
                heavy_atoms, heavy_coords = without_hydrogens(atoms, coords)
                ref_heavy_atoms, ref_heavy_coords = without_hydrogens(ref_atoms, ref_coords)
                heavy = comparison_metrics(
                    heavy_atoms, heavy_coords, ref_heavy_atoms, ref_heavy_coords, atol=atol
                )
                status = "full_match" if full["geometry_equal"] else (
                    "heavy_atom_match" if heavy["geometry_equal"] else "mismatch"
                )
                rows.append({
                    **base, "status": status, "error": None,
                    "reference_path": str(ref_path), "lmdb_geometry_hash": digest,
                    "lmdb_atom_count": len(atoms), "reference_atom_count": len(ref_atoms),
                    "lmdb_heavy_atom_count": len(heavy_atoms),
                    "reference_heavy_atom_count": len(ref_heavy_atoms),
                    **{f"full_{name}": value for name, value in full.items()},
                    **{f"heavy_{name}": value for name, value in heavy.items()},
                })
            except Exception as error:
                rows.append({**base, "status": "error", "error": f"{type(error).__name__}: {error}"})

    duplicate_rows = []
    for pdb_id, items in sorted(occurrences.items()):
        if len(items) < 2:
            continue
        hashes = Counter(digest for _, _, digest in items)
        duplicate_rows.append({
            "pdb_id": pdb_id,
            "occurrence_count": len(items),
            "unique_lmdb_geometry_count": len(hashes),
            "has_different_lmdb_geometries": len(hashes) > 1,
            "geometry_hash_counts_json": json.dumps(hashes, sort_keys=True),
            "records_json": json.dumps(
                [{"split": split, "lmdb_key": key, "geometry_hash": digest}
                 for split, key, digest in items],
                sort_keys=True,
            ),
        })
    return pl.DataFrame(rows, infer_schema_length=None), pl.DataFrame(
        duplicate_rows,
        schema={
            "pdb_id": pl.String, "occurrence_count": pl.Int64,
            "unique_lmdb_geometry_count": pl.Int64,
            "has_different_lmdb_geometries": pl.Boolean,
            "geometry_hash_counts_json": pl.String, "records_json": pl.String,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lmdb", type=Path, nargs="+", default=list(DEFAULT_LMDBS))
    parser.add_argument("--combine-set", type=Path, default=DEFAULT_COMBINE_SET)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--no-sidecar", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--duplicates-output", type=Path, default=DEFAULT_DUPLICATES_OUTPUT)
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable LMDB progress bars."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.atol < 0:
        raise SystemExit("--atol must be non-negative")
    comparisons, duplicates = compare(
        args.lmdb, args.combine_set,
        sidecar_path=None if args.no_sidecar else args.sidecar,
        atol=args.atol,
        show_progress=not args.no_progress,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.duplicates_output.parent.mkdir(parents=True, exist_ok=True)
    comparisons.write_parquet(args.output)
    duplicates.write_parquet(args.duplicates_output)
    status_counts = comparisons.group_by("status").len().sort("status").to_dicts()
    print(json.dumps({
        "records": comparisons.height,
        "status_counts": status_counts,
        "duplicate_pdb_ids": duplicates.height,
        "duplicate_pdb_ids_with_different_geometries": (
            duplicates.filter(pl.col("has_different_lmdb_geometries")).height
        ),
        "comparison_output": str(args.output),
        "duplicates_output": str(args.duplicates_output),
    }, indent=2))


if __name__ == "__main__":
    main()
