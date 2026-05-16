"""Utilities for preparing DrugCLIP target-fishing inputs."""

from __future__ import annotations

import os
import pickle
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import polars as pl
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from biosensia_retrieval import (
    DEFAULT_COMBINE_SET_DIR,
    DEFAULT_POCKET_RADIUS_ANGSTROM,
    _first_existing,
    _normalize_pocket_record,
    _record_from_data_pkl,
    _record_from_pocket_pdb,
    _record_from_protein_and_ligand,
    _write_lmdb,
)


DEFAULT_CANDIDATE_POCKETS_LMDB = Path("data/candidate_pockets.lmdb")
DEFAULT_DRUGCLIP_MOLS_LMDB = Path("external/DrugCLIP/mols.lmdb")
DEFAULT_DRUGCLIP_MOLS_INDEX_LMDB = Path("data/mols_index.lmdb")
DEFAULT_MOL_DOWNLOAD_DIR = Path("data/molecules")
PUBCHEM_SDF_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
    "{namespace}/{identifier}/SDF?record_type={record_type}"
)
PUBCHEM_USER_AGENT = "BioSensIA-DC/0.1 (molecule LMDB preparation)"

_PDB_ID_RE = re.compile(r"^[0-9][0-9A-Za-z]{3}$")
_MOL_SLUG_RE = re.compile(r"[^0-9A-Za-z_.-]+")
_MOL_INDEX_LOOKUP_DB = b"lookup"
_MOL_INDEX_META_DB = b"meta"
_MOL_INDEX_SCHEMA_VERSION = "1"


def create_mol_lmdb(
    molecules: Iterable[str],
    output_path: str | Path,
    *,
    source_lmdb_path: str | Path = DEFAULT_DRUGCLIP_MOLS_LMDB,
    mol_index_path: str | Path | None = DEFAULT_DRUGCLIP_MOLS_INDEX_LMDB,
    work_dir: str | Path | None = None,
    download_missing: bool = True,
    overwrite: bool = True,
    map_size: int = 1 << 40,
    timeout_seconds: float = 60,
    random_seed: int = 1,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Create a DrugCLIP molecule LMDB for one or more molecules.

    The output records intentionally match the schema consumed by
    ``DrugCLIPTask.load_retrieval_mols_dataset``:

    ``{"atoms": list[str], "coordinates": list[ndarray], "smi": str}``

    Parameters
    ----------
    molecules:
        Molecule identifiers. SMILES strings are supported directly. DrugCLIP
        IDs are matched against the local source LMDB. Missing molecules are
        resolved through PubChem when ``download_missing`` is true; use
        ``cid:2244`` or ``2244`` for a PubChem CID, or a PubChem compound name.
    output_path:
        Destination LMDB file path.
    source_lmdb_path:
        Existing DrugCLIP molecule LMDB to search first. Defaults to
        ``external/DrugCLIP/mols.lmdb``.
    mol_index_path:
        Optional LMDB index built by ``build_mol_lmdb_index``. If this exists
        and matches ``source_lmdb_path``, it is used before a sequential scan.
        Set to ``None`` to force sequential search.
    work_dir:
        Directory used for downloaded molecule SDF files. Defaults to
        ``data/molecules``.
    download_missing:
        Download missing molecules from PubChem. If PubChem cannot resolve a
        SMILES string, a 3D conformer is generated locally from the SMILES.
    overwrite:
        Replace an existing output LMDB file.
    map_size:
        LMDB map size.
    timeout_seconds:
        Network timeout for PubChem downloads.
    random_seed:
        RDKit conformer generation seed for downloaded 2D records or local
        SMILES fallback.
    show_progress:
        Show progress while scanning the source LMDB and reporting fallback
        download/generation steps.

    Returns
    -------
    list[dict[str, Any]]
        Build summaries for each requested molecule.
    """

    if isinstance(molecules, str):
        molecule_queries = [_normalize_molecule_query(molecules)]
    else:
        molecule_queries = [
            _normalize_molecule_query(molecule) for molecule in molecules
        ]
    if not molecule_queries:
        raise ValueError("molecules must contain at least one molecule identifier")

    source_lmdb_path = Path(source_lmdb_path)
    mol_index_path = Path(mol_index_path) if mol_index_path is not None else None
    work_dir = Path(work_dir) if work_dir is not None else DEFAULT_MOL_DOWNLOAD_DIR

    local_records: dict[int, tuple[dict[str, Any], str]] = {}
    if mol_index_path is not None:
        local_records = _find_molecule_records_in_index(
            molecule_queries,
            source_lmdb_path,
            mol_index_path,
            show_progress=show_progress,
        )

    missing_query_indices = [
        query_index
        for query_index in range(len(molecule_queries))
        if query_index not in local_records
    ]
    if missing_query_indices:
        sequential_records = _find_molecule_records_in_lmdb(
            [molecule_queries[query_index] for query_index in missing_query_indices],
            source_lmdb_path,
            show_progress=show_progress,
        )
        for local_query_index, record in sequential_records.items():
            local_records[missing_query_indices[local_query_index]] = record

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for molecule_index, query in enumerate(molecule_queries):
        found = local_records.get(molecule_index)
        if found is None:
            if not download_missing:
                raise FileNotFoundError(
                    f"No usable local molecule data found for {query['query']!r} "
                    f"in {source_lmdb_path}"
                )
            if show_progress:
                tqdm.write(
                    f"Did not find {query['query']!r} in {source_lmdb_path}; "
                    "trying PubChem/download fallback."
                )
            record, source = _download_molecule_record(
                query["query"],
                work_dir=work_dir / _safe_molecule_slug(query["query"]),
                timeout_seconds=timeout_seconds,
                random_seed=random_seed,
                show_progress=show_progress,
            )
        else:
            record, source = found
            if show_progress:
                tqdm.write(f"Found {query['query']!r} in {source}.")

        record = _normalize_molecule_record(record, fallback_smiles=query["query"])
        records.append(record)
        summaries.append(
            {
                "molecule": query["query"],
                "smiles": record["smi"],
                "source": source,
                "molecule_index": molecule_index,
                "molecule_atoms": len(record["atoms"]),
                "conformers": len(record["coordinates"]),
                "output_path": str(output_path),
            }
        )

    if show_progress:
        tqdm.write(f"Writing {len(records)} molecule record(s) to {output_path}.")
    _write_lmdb(records, output_path, overwrite=overwrite, map_size=map_size)
    return summaries


def build_mol_lmdb_index(
    index_path: str | Path = DEFAULT_DRUGCLIP_MOLS_INDEX_LMDB,
    *,
    source_lmdb_path: str | Path = DEFAULT_DRUGCLIP_MOLS_LMDB,
    overwrite: bool = True,
    map_size: int = 1 << 40,
    commit_interval: int = 10000,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Build an LMDB lookup index for a DrugCLIP molecule LMDB.

    The index maps canonical SMILES and lower-cased molecule IDs to numeric
    keys in ``source_lmdb_path``. It is stored separately from the source LMDB
    so the DrugCLIP-compatible numeric-key molecule file remains unchanged.
    """

    if commit_interval <= 0:
        raise ValueError("commit_interval must be greater than 0")

    source_lmdb_path = Path(source_lmdb_path)
    index_path = Path(index_path)
    if not source_lmdb_path.exists():
        raise FileNotFoundError(f"Source molecule LMDB not found: {source_lmdb_path}")
    if index_path.exists() and not overwrite:
        raise FileExistsError(f"Molecule index LMDB already exists: {index_path}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_index_path = index_path.with_name(f".{index_path.name}.tmp")
    if tmp_index_path.exists():
        tmp_index_path.unlink()

    source_env = lmdb.open(
        str(source_lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    index_env = lmdb.open(
        str(tmp_index_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=map_size,
        max_dbs=2,
    )
    lookup_db = index_env.open_db(_MOL_INDEX_LOOKUP_DB, dupsort=True)
    meta_db = index_env.open_db(_MOL_INDEX_META_DB)

    indexed_records = 0
    lookup_values = 0
    completed_write = False
    transaction = index_env.begin(write=True)
    progress = None
    pending_progress = 0
    try:
        with source_env.begin() as source_transaction:
            source_entries = source_env.stat()["entries"]
            if show_progress:
                progress = tqdm(
                    total=source_entries,
                    desc=f"Indexing {source_lmdb_path.name}",
                    unit="record",
                )

            for source_key, value in source_transaction.cursor():
                record = pickle.loads(value)
                for lookup_key in _molecule_record_index_keys(record):
                    transaction.put(lookup_key, source_key, db=lookup_db)
                    lookup_values += 1
                indexed_records += 1

                if progress is not None:
                    pending_progress += 1
                    if pending_progress >= 1000:
                        progress.update(pending_progress)
                        pending_progress = 0

                if indexed_records % commit_interval == 0:
                    transaction.commit()
                    transaction = index_env.begin(write=True)

            _write_molecule_index_metadata(
                transaction,
                meta_db,
                source_lmdb_path=source_lmdb_path,
                source_entries=source_entries,
                indexed_records=indexed_records,
                lookup_values=lookup_values,
            )
            transaction.commit()
            transaction = None
            completed_write = True
    except Exception:
        if transaction is not None:
            transaction.abort()
        raise
    finally:
        if progress is not None:
            if pending_progress:
                progress.update(pending_progress)
            progress.close()
        source_env.close()
        index_env.close()
        if not completed_write:
            tmp_index_path.unlink(missing_ok=True)

    os.replace(tmp_index_path, index_path)
    summary = {
        "index_path": str(index_path),
        "source_lmdb_path": str(source_lmdb_path),
        "source_entries": source_entries,
        "indexed_records": indexed_records,
        "lookup_values": lookup_values,
    }
    if show_progress:
        tqdm.write(
            f"Built molecule index {index_path} with {lookup_values} lookup "
            f"value(s) for {indexed_records} record(s)."
        )
    return summary


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
        for pdb_id, bundle_dir in tqdm(
            bundle_dirs,
            desc="Building candidate pockets LMDB",
            unit="pocket",
        ):
            try:
                record, _source = _build_candidate_pocket_record(
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


def build_candidate_pockets_frame(
    lmdb_path: str | Path = DEFAULT_CANDIDATE_POCKETS_LMDB,
) -> pl.DataFrame:
    """Read a candidate-pocket LMDB into a compact Polars dataframe."""

    lmdb_path = Path(lmdb_path)
    rows: list[dict[str, Any]] = []
    env = lmdb.open(
        str(lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with env.begin() as transaction:
            for index in range(env.stat()["entries"]):
                value = transaction.get(str(index).encode("ascii"))
                if value is None:
                    raise KeyError(f"Missing numeric LMDB key {index} in {lmdb_path}")
                record = pickle.loads(value)
                rows.append(
                    {
                        "pocket": str(record["pocket"]),
                        "pocket_atoms": len(record["pocket_atoms"]),
                    }
                )
    finally:
        env.close()

    return pl.DataFrame(
        rows,
        schema={
            "pocket": pl.String,
            "pocket_atoms": pl.Int64,
        },
    )


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


def _build_candidate_pocket_record(
    pdb_id: str,
    bundle_dir: Path,
    *,
    radius: float,
    prefer_data_pkl: bool,
    include_pocket_hetatm: bool,
) -> tuple[dict[str, Any], str]:
    errors: list[str] = []

    data_pkl = bundle_dir / "data.pkl"
    if prefer_data_pkl and data_pkl.exists():
        result = _try_candidate_record(
            "data.pkl",
            str(data_pkl),
            lambda: _record_from_data_pkl(data_pkl, pocket_name=pdb_id),
            pdb_id,
            errors,
        )
        if result is not None:
            return result

    protein_path = _first_existing(
        [
            bundle_dir / f"{pdb_id}_protein.pdb",
            bundle_dir / "receptor.pdb",
            bundle_dir / f"{pdb_id}.pdb",
        ],
        fallback_globs=[
            (bundle_dir, "*_protein.pdb"),
            (bundle_dir, "receptor*.pdb"),
        ],
    )
    ligand_path = _first_existing(
        [
            bundle_dir / f"{pdb_id}_ligand.mol2",
            bundle_dir / "crystal_ligand.mol2",
            bundle_dir / f"{pdb_id}_ligand.sdf",
            bundle_dir / "crystal_ligand.sdf",
        ],
        fallback_globs=[
            (bundle_dir, "*_ligand.mol2"),
            (bundle_dir, "*ligand*.mol2"),
            (bundle_dir, "*_ligand.sdf"),
            (bundle_dir, "*ligand*.sdf"),
        ],
    )
    if protein_path is not None and ligand_path is not None:
        result = _try_candidate_record(
            "protein+ligand",
            f"{protein_path} + {ligand_path}",
            lambda: _record_from_protein_and_ligand(
                protein_path,
                ligand_path,
                pocket_name=pdb_id,
                radius=radius,
            ),
            pdb_id,
            errors,
        )
        if result is not None:
            return result

    pocket_pdb = _first_existing(
        [
            bundle_dir / f"{pdb_id}_pocket.pdb",
            bundle_dir / f"{pdb_id}_pocket6A.pdb",
            bundle_dir / "pocket.pdb",
        ],
        fallback_globs=[
            (bundle_dir, "*_pocket.pdb"),
            (bundle_dir, "*pocket*.pdb"),
        ],
    )
    if pocket_pdb is not None:
        result = _try_candidate_record(
            "pocket PDB",
            str(pocket_pdb),
            lambda: _record_from_pocket_pdb(
                pocket_pdb,
                pocket_name=pdb_id,
                include_hetatm=include_pocket_hetatm,
            ),
            pdb_id,
            errors,
        )
        if result is not None:
            return result

    if errors:
        raise ValueError("; ".join(errors))
    raise FileNotFoundError(f"No usable pocket data found in {bundle_dir}")


def _try_candidate_record(
    source_kind: str,
    source: str,
    build_record: Callable[[], dict[str, Any]],
    pdb_id: str,
    errors: list[str],
) -> tuple[dict[str, Any], str] | None:
    try:
        record = build_record()
        return _normalize_pocket_record(record, fallback_name=pdb_id), source
    except Exception as exc:
        errors.append(f"{source_kind} ({source}): {exc}")
        return None


def _normalize_molecule_query(molecule: str) -> dict[str, Any]:
    query = str(molecule).strip()
    if not query:
        raise ValueError("molecule identifiers must be non-empty strings")

    variants = {query, query.lower()}
    canonical_smiles = _canonical_smiles(query)
    if canonical_smiles is not None:
        variants.add(canonical_smiles)

    return {
        "query": query,
        "variants": variants,
        "canonical_smiles": canonical_smiles,
    }


def _find_molecule_records_in_index(
    molecule_queries: list[dict[str, Any]],
    source_lmdb_path: Path,
    index_path: Path,
    *,
    show_progress: bool,
) -> dict[int, tuple[dict[str, Any], str]]:
    if not index_path.exists():
        if show_progress:
            tqdm.write(f"Molecule index not found: {index_path}")
        return {}
    if not source_lmdb_path.exists():
        if show_progress:
            tqdm.write(f"Source molecule LMDB not found: {source_lmdb_path}")
        return {}
    if show_progress:
        tqdm.write(
            f"Searching molecule index {index_path} "
            f"for {len(molecule_queries)} molecule query/queries."
        )

    source_env = lmdb.open(
        str(source_lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        try:
            index_env = lmdb.open(
                str(index_path),
                subdir=False,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=256,
                max_dbs=2,
            )
        except lmdb.Error as exc:
            if show_progress:
                tqdm.write(f"Could not open molecule index {index_path}: {exc}")
            return {}

        try:
            try:
                lookup_db = index_env.open_db(_MOL_INDEX_LOOKUP_DB)
                meta_db = index_env.open_db(_MOL_INDEX_META_DB)
            except lmdb.Error as exc:
                if show_progress:
                    tqdm.write(f"Molecule index {index_path} is not usable: {exc}")
                return {}

            with index_env.begin(db=meta_db) as meta_transaction:
                if not _molecule_index_matches_source(
                    meta_transaction,
                    source_lmdb_path=source_lmdb_path,
                    source_entries=source_env.stat()["entries"],
                ):
                    if show_progress:
                        tqdm.write(
                            f"Molecule index {index_path} does not match "
                            f"{source_lmdb_path}; falling back to sequential search."
                        )
                    return {}

            found: dict[int, tuple[dict[str, Any], str]] = {}
            with (
                index_env.begin(db=lookup_db) as index_transaction,
                source_env.begin() as source_transaction,
            ):
                cursor = index_transaction.cursor()
                for query_index, query in enumerate(molecule_queries):
                    if show_progress:
                        tqdm.write(
                            f"Searching molecule index {index_path} "
                            f"for {query['query']!r}."
                        )
                    source_keys: list[bytes] = []
                    for lookup_key in _molecule_query_index_keys(query):
                        if not cursor.set_key(lookup_key):
                            continue
                        source_keys.extend(cursor.iternext_dup())
                    for source_key in source_keys:
                        value = source_transaction.get(source_key)
                        if value is None:
                            continue
                        record = pickle.loads(value)
                        found[query_index] = (
                            _normalize_molecule_record(
                                record,
                                fallback_smiles=query["query"],
                            ),
                            (
                                f"{source_lmdb_path}:"
                                f"{source_key.decode('ascii', errors='replace')}"
                            ),
                        )
                        break
            if show_progress and found:
                tqdm.write(
                    f"Found {len(found)} molecule record(s) via {index_path}."
                )
            return found
        finally:
            index_env.close()
    finally:
        source_env.close()


def _find_molecule_records_in_lmdb(
    molecule_queries: list[dict[str, Any]],
    source_lmdb_path: Path,
    *,
    show_progress: bool,
) -> dict[int, tuple[dict[str, Any], str]]:
    if not source_lmdb_path.exists():
        if show_progress:
            tqdm.write(f"Source molecule LMDB not found: {source_lmdb_path}")
        return {}

    lookup: dict[str, list[int]] = {}
    for query_index, query in enumerate(molecule_queries):
        for variant in query["variants"]:
            lookup.setdefault(variant, []).append(query_index)

    found: dict[int, tuple[dict[str, Any], str]] = {}
    env = lmdb.open(
        str(source_lmdb_path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    progress = None
    pending_progress = 0
    try:
        with env.begin() as transaction:
            if show_progress:
                progress = tqdm(
                    total=env.stat()["entries"],
                    desc=f"Searching {source_lmdb_path.name}",
                    unit="record",
                )
            for key, value in transaction.cursor():
                if progress is not None:
                    pending_progress += 1
                    if pending_progress >= 1000:
                        progress.update(pending_progress)
                        pending_progress = 0

                if len(found) == len(molecule_queries):
                    break

                record = pickle.loads(value)
                matches = _matched_molecule_query_indices(record, lookup)
                if not matches:
                    continue

                source = f"{source_lmdb_path}:{key.decode('ascii', errors='replace')}"
                for query_index in matches:
                    if query_index in found:
                        continue
                    found[query_index] = (
                        _normalize_molecule_record(
                            record,
                            fallback_smiles=molecule_queries[query_index]["query"],
                        ),
                        source,
                    )
                if len(found) == len(molecule_queries):
                    break
    finally:
        if progress is not None:
            if pending_progress:
                progress.update(pending_progress)
            progress.close()
        env.close()

    return found


def _write_molecule_index_metadata(
    transaction: lmdb.Transaction,
    meta_db: lmdb._Database,
    *,
    source_lmdb_path: Path,
    source_entries: int,
    indexed_records: int,
    lookup_values: int,
) -> None:
    metadata = {
        "schema_version": _MOL_INDEX_SCHEMA_VERSION,
        "source_lmdb_path": str(source_lmdb_path.resolve()),
        "source_entries": int(source_entries),
        "indexed_records": int(indexed_records),
        "lookup_values": int(lookup_values),
    }
    for key, value in metadata.items():
        transaction.put(
            key.encode("ascii"),
            pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL),
            db=meta_db,
        )


def _molecule_index_matches_source(
    transaction: lmdb.Transaction,
    *,
    source_lmdb_path: Path,
    source_entries: int,
) -> bool:
    metadata = _read_molecule_index_metadata(transaction)
    return (
        metadata.get("schema_version") == _MOL_INDEX_SCHEMA_VERSION
        and metadata.get("source_lmdb_path") == str(source_lmdb_path.resolve())
        and metadata.get("source_entries") == int(source_entries)
    )


def _read_molecule_index_metadata(transaction: lmdb.Transaction) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in transaction.cursor():
        metadata[key.decode("ascii")] = pickle.loads(value)
    return metadata


def _molecule_record_index_keys(record: dict[str, Any]) -> set[bytes]:
    keys: set[bytes] = set()
    smi = record.get("smi")
    if smi:
        smi = str(smi).strip()
        if smi:
            keys.add(_molecule_index_lookup_key("smi", smi))
            canonical_smiles = _canonical_smiles(smi)
            if canonical_smiles is not None:
                keys.add(_molecule_index_lookup_key("smi", canonical_smiles))

    for key in ("IDs", "id", "name"):
        for identifier in _iter_string_values(record.get(key)):
            keys.add(_molecule_index_lookup_key("id", identifier.lower()))
    return keys


def _molecule_query_index_keys(query: dict[str, Any]) -> set[bytes]:
    keys: set[bytes] = set()
    for variant in query["variants"]:
        keys.add(_molecule_index_lookup_key("smi", variant))
        keys.add(_molecule_index_lookup_key("id", variant.lower()))
    return keys


def _molecule_index_lookup_key(kind: str, value: str) -> bytes:
    digest = sha256(str(value).encode("utf-8")).hexdigest()
    return f"{kind}:{digest}".encode("ascii")


def _matched_molecule_query_indices(
    record: dict[str, Any],
    lookup: dict[str, list[int]],
) -> list[int]:
    matched: list[int] = []
    for identifier in _molecule_record_identifiers(record):
        matched.extend(lookup.get(identifier, []))
    return matched


def _molecule_record_identifiers(record: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    if record.get("smi"):
        identifiers.add(str(record["smi"]).strip())

    for key in ("IDs", "id", "name"):
        value = record.get(key)
        for item in _iter_string_values(value):
            identifiers.add(item)
            identifiers.add(item.lower())
    return identifiers


def _iter_string_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    item = str(value).strip()
    return [item] if item else []


def _download_molecule_record(
    molecule: str,
    *,
    work_dir: Path,
    timeout_seconds: float,
    random_seed: int,
    show_progress: bool,
) -> tuple[dict[str, Any], str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    for namespace, identifier in _pubchem_identifier_candidates(molecule):
        for record_type, force_embed in (("3d", False), ("2d", True)):
            target_path = work_dir / f"pubchem_{namespace}_{record_type}.sdf"
            try:
                if show_progress:
                    if target_path.exists():
                        tqdm.write(f"Using cached PubChem SDF: {target_path}")
                    else:
                        tqdm.write(
                            f"Downloading PubChem {record_type.upper()} SDF "
                            f"for {molecule!r}."
                        )
                _ensure_pubchem_sdf(
                    namespace,
                    identifier,
                    target_path,
                    record_type=record_type,
                    timeout_seconds=timeout_seconds,
                )
                record = _record_from_molecule_sdf(
                    target_path,
                    fallback_smiles=molecule,
                    force_embed=force_embed,
                    random_seed=random_seed,
                )
                record.setdefault("subset", "pubchem")
                if namespace == "cid":
                    record.setdefault("IDs", f"CID:{identifier}")
                return record, str(target_path)
            except Exception as exc:
                errors.append(f"PubChem {namespace}/{record_type}: {exc}")

    canonical_smiles = _canonical_smiles(molecule)
    if canonical_smiles is not None:
        if show_progress:
            tqdm.write(f"Generating RDKit conformer for {canonical_smiles!r}.")
        record = _record_from_smiles(
            canonical_smiles,
            random_seed=random_seed,
        )
        record.setdefault("subset", "generated_from_smiles")
        return record, f"generated from SMILES:{canonical_smiles}"

    raise FileNotFoundError(
        f"Could not download molecule data for {molecule!r}. "
        + "; ".join(errors)
    )


def _pubchem_identifier_candidates(molecule: str) -> list[tuple[str, str]]:
    stripped = molecule.strip()
    lower = stripped.lower()
    if lower.startswith("cid:"):
        cid = stripped.split(":", 1)[1].strip()
        return [("cid", cid)] if cid else []
    if stripped.isdigit():
        return [("cid", stripped)]

    quoted = urllib.parse.quote(stripped, safe="")
    if _canonical_smiles(stripped) is not None:
        return [("smiles", quoted)]
    return [("name", quoted)]


def _ensure_pubchem_sdf(
    namespace: str,
    identifier: str,
    target_path: Path,
    *,
    record_type: str,
    timeout_seconds: float,
) -> None:
    if target_path.exists():
        return

    url = PUBCHEM_SDF_URL.format(
        namespace=namespace,
        identifier=identifier,
        record_type=record_type,
    )
    request = urllib.request.Request(url, headers={"User-Agent": PUBCHEM_USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            with target_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
    except urllib.error.HTTPError as exc:
        raise FileNotFoundError(f"{url} returned HTTP {exc.code}") from exc


def _record_from_molecule_sdf(
    path: Path,
    *,
    fallback_smiles: str,
    force_embed: bool,
    random_seed: int,
) -> dict[str, Any]:
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
    mol = next((item for item in supplier if item is not None), None)
    if mol is None:
        raise ValueError(f"No molecule found in SDF {path}")
    return _record_from_rdkit_mol(
        mol,
        fallback_smiles=fallback_smiles,
        force_embed=force_embed,
        random_seed=random_seed,
    )


def _record_from_smiles(smiles: str, *, random_seed: int) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles!r}")
    return _record_from_rdkit_mol(
        mol,
        fallback_smiles=smiles,
        force_embed=True,
        random_seed=random_seed,
    )


def _record_from_rdkit_mol(
    mol: Chem.Mol,
    *,
    fallback_smiles: str,
    force_embed: bool,
    random_seed: int,
) -> dict[str, Any]:
    working_mol = Chem.Mol(mol)
    if force_embed or working_mol.GetNumConformers() == 0:
        working_mol = _embed_molecule_3d(working_mol, random_seed=random_seed)

    working_mol = Chem.RemoveHs(working_mol)
    if working_mol.GetNumConformers() == 0:
        raise ValueError("Molecule has no conformer")

    conformer = working_mol.GetConformer(0)
    atoms = [atom.GetSymbol() for atom in working_mol.GetAtoms()]
    coordinates = np.asarray(conformer.GetPositions(), dtype=np.float32)
    smiles = _canonical_smiles(Chem.MolToSmiles(working_mol)) or fallback_smiles
    return {
        "atoms": atoms,
        "coordinates": [coordinates],
        "smi": smiles,
    }


def _embed_molecule_3d(mol: Chem.Mol, *, random_seed: int) -> Chem.Mol:
    working_mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(random_seed)
    status = AllChem.EmbedMolecule(working_mol, params)
    if status != 0:
        status = AllChem.EmbedMolecule(working_mol, useRandomCoords=True)
    if status != 0:
        raise ValueError(f"Could not generate conformer for {Chem.MolToSmiles(mol)}")

    try:
        AllChem.MMFFOptimizeMolecule(working_mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(working_mol)
    return working_mol


def _normalize_molecule_record(
    record: dict[str, Any],
    *,
    fallback_smiles: str,
) -> dict[str, Any]:
    atoms = [str(atom).strip() for atom in record["atoms"]]
    coordinates = _normalize_molecule_coordinates(record["coordinates"], len(atoms))
    if not atoms:
        raise ValueError("atoms must contain at least one atom")

    normalized = dict(record)
    normalized["atoms"] = atoms
    normalized["coordinates"] = coordinates
    normalized["smi"] = str(record.get("smi") or fallback_smiles)
    return normalized


def _normalize_molecule_coordinates(
    coordinates: Any,
    expected_atoms: int,
) -> list[np.ndarray]:
    if isinstance(coordinates, np.ndarray):
        if coordinates.ndim == 2:
            conformers = [coordinates]
        elif coordinates.ndim == 3:
            conformers = [coordinates[index] for index in range(coordinates.shape[0])]
        else:
            raise ValueError(
                f"coordinates must have 2 or 3 dimensions, got {coordinates.shape}"
            )
    else:
        conformers = list(coordinates)

    if not conformers:
        raise ValueError("coordinates must contain at least one conformer")

    normalized: list[np.ndarray] = []
    for conformer in conformers:
        array = np.asarray(conformer, dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(
                f"each conformer must have shape (n_atoms, 3), got {array.shape}"
            )
        if array.shape[0] != expected_atoms:
            raise ValueError(
                "atoms and conformer coordinates must have the same length "
                f"({expected_atoms} != {array.shape[0]})"
            )
        normalized.append(array)
    return normalized


def _canonical_smiles(molecule: str) -> str | None:
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(str(molecule))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def _safe_molecule_slug(molecule: str) -> str:
    slug = _MOL_SLUG_RE.sub("_", molecule.strip())[:80].strip("._")
    return slug or "molecule"


__all__ = [
    "DEFAULT_CANDIDATE_POCKETS_LMDB",
    "DEFAULT_DRUGCLIP_MOLS_INDEX_LMDB",
    "DEFAULT_DRUGCLIP_MOLS_LMDB",
    "build_mol_lmdb_index",
    "build_candidate_pockets_frame",
    "build_candidate_pockets_lmdb",
    "create_mol_lmdb",
]
