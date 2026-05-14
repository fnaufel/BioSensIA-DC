"""Utilities for preparing DrugCLIP target-fishing inputs."""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any

import lmdb

from biosensia_retrieval import (
    DEFAULT_COMBINE_SET_DIR,
    DEFAULT_POCKET_RADIUS_ANGSTROM,
    _build_local_pocket_record,
    _normalize_pocket_record,
)


DEFAULT_CANDIDATE_POCKETS_LMDB = Path("data/candidate_pockets.lmdb")

_PDB_ID_RE = re.compile(r"^[0-9][0-9A-Za-z]{3}$")


def build_candidate_pockets_lmdb(
    output_path: str | Path = DEFAULT_CANDIDATE_POCKETS_LMDB,
    *,
    combine_set_dir: str | Path = DEFAULT_COMBINE_SET_DIR,
    radius: float = DEFAULT_POCKET_RADIUS_ANGSTROM,
    prefer_data_pkl: bool = True,
    include_pocket_hetatm: bool = False,
    overwrite: bool = True,
    skip_invalid: bool = False,
    map_size: int = 1 << 40,
    commit_interval: int = 1000,
) -> dict[str, Any]:
    """Build the candidate-pocket LMDB used by target fishing.

    The output records intentionally match the schema consumed by
    ``DrugCLIPTask.load_pockets_dataset``:

    ``{"pocket": str, "pocket_atoms": list[str], "pocket_coordinates": ndarray}``

    Parameters
    ----------
    output_path:
        Destination LMDB file. Defaults to ``data/candidate_pockets.lmdb``.
    combine_set_dir:
        DrugCLIP ``combine_set`` directory containing one subdirectory per PDB ID.
    radius:
        Distance cutoff, in Angstrom, used when a pocket must be generated from
        protein and ligand files instead of loaded from ``data.pkl``.
    prefer_data_pkl:
        Use bundled ``data.pkl`` files first when available.
    include_pocket_hetatm:
        Include HETATM rows if falling back to a local ``*_pocket.pdb`` file.
    overwrite:
        Replace an existing output LMDB.
    skip_invalid:
        If true, skip pockets that cannot be read or normalized and report them
        in the returned summary. If false, raise on the first invalid pocket.
    map_size:
        LMDB map size.
    commit_interval:
        Number of records to write per LMDB transaction.

    Returns
    -------
    dict[str, Any]
        Build summary with output path, candidate directory count, written
        record count, and skipped entries.
    """

    if commit_interval <= 0:
        raise ValueError("commit_interval must be greater than 0")

    combine_set_dir = Path(combine_set_dir)
    output_path = Path(output_path)

    bundle_dirs = list(_iter_candidate_bundle_dirs(combine_set_dir))
    if not bundle_dirs:
        raise FileNotFoundError(
            f"No PDB-like bundle directories found in {combine_set_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"LMDB already exists: {output_path}")

    tmp_output_path = output_path.with_name(f".{output_path.name}.tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    skipped: list[dict[str, str]] = []
    written = 0
    completed_write = False
    env = lmdb.open(
        str(tmp_output_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=map_size,
    )
    transaction = env.begin(write=True)
    try:
        for pdb_id, bundle_dir in bundle_dirs:
            try:
                record, _source = _build_local_pocket_record(
                    pdb_id,
                    bundle_dir,
                    radius=radius,
                    prefer_data_pkl=prefer_data_pkl,
                    include_pocket_hetatm=include_pocket_hetatm,
                )
                if record is None:
                    raise FileNotFoundError(
                        f"No usable pocket data found in {bundle_dir}"
                    )
                record = _normalize_pocket_record(record, fallback_name=pdb_id)
            except Exception as exc:
                if not skip_invalid:
                    raise RuntimeError(
                        f"Failed to build candidate pocket record for {pdb_id} "
                        f"from {bundle_dir}"
                    ) from exc
                skipped.append(
                    {
                        "accession": pdb_id,
                        "source": str(bundle_dir),
                        "error": str(exc),
                    }
                )
                continue

            transaction.put(
                str(written).encode("ascii"),
                pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
            )
            written += 1

            if written % commit_interval == 0:
                transaction.commit()
                transaction = env.begin(write=True)

        transaction.commit()
        transaction = None
        completed_write = True
    except Exception:
        if transaction is not None:
            transaction.abort()
        raise
    finally:
        env.close()
        if not completed_write:
            tmp_output_path.unlink(missing_ok=True)

    if written == 0:
        tmp_output_path.unlink(missing_ok=True)
        raise ValueError("No candidate pocket records were written")

    os.replace(tmp_output_path, output_path)

    return {
        "output_path": str(output_path),
        "combine_set_dir": str(combine_set_dir),
        "candidate_dirs": len(bundle_dirs),
        "pockets": written,
        "skipped": len(skipped),
        "skipped_entries": skipped,
    }


def _iter_candidate_bundle_dirs(combine_set_dir: Path) -> list[tuple[str, Path]]:
    if not combine_set_dir.exists():
        raise FileNotFoundError(f"combine_set directory not found: {combine_set_dir}")
    if not combine_set_dir.is_dir():
        raise NotADirectoryError(f"combine_set path is not a directory: {combine_set_dir}")

    return [
        (path.name.lower(), path)
        for path in sorted(combine_set_dir.iterdir())
        if path.is_dir() and _PDB_ID_RE.fullmatch(path.name)
    ]


__all__ = ["DEFAULT_CANDIDATE_POCKETS_LMDB", "build_candidate_pockets_lmdb"]
