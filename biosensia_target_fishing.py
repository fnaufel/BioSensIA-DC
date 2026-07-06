"""Utilities for BioSensIA target fishing with DrugCLIP encoders."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import polars as pl
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem
from tqdm.auto import tqdm

from lmdb_helpers import write_lmdb_records


LOGGER = logging.getLogger(__name__)


DEFAULT_CANDIDATE_POCKETS_LMDB = Path("data/candidate_pockets.lmdb")
DEFAULT_QUERY_MOL_LMDB = Path("data/query_mol.lmdb")
DEFAULT_DRUGCLIP_DIR = Path("external/DrugCLIP")
DEFAULT_DRUGCLIP_DATA_DIR = DEFAULT_DRUGCLIP_DIR / "data"
DEFAULT_DRUGCLIP_CHECKPOINT = DEFAULT_DRUGCLIP_DIR / "checkpoint_best.pt"
DEFAULT_DRUGCLIP_MOLS_LMDB = Path("external/DrugCLIP/mols.lmdb")
DEFAULT_DRUGCLIP_MOLS_INDEX_LMDB = Path("data/mols_index.lmdb")
DEFAULT_MOL_DOWNLOAD_DIR = Path("data/molecules")
DEFAULT_RANKED_POCKETS_PATH = Path(
    "external/DrugCLIP/data/pocket_emb/ranked_pockets.txt"
)
DEFAULT_TARGET_FISHING_TOP_K = 1000
RCSB_PDB_STRUCTURE_URL_PREFIX = "https://www.rcsb.org/structure/"
PUBCHEM_SDF_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
    "{namespace}/{identifier}/SDF?record_type={record_type}"
)
PUBCHEM_USER_AGENT = "BioSensIA-DC/0.1 (molecule LMDB preparation)"

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
        with a compatible schema version and the same source entry count, index
        misses go directly to the download/generation fallback. If the index is
        missing or unusable, the source LMDB is searched sequentially. Set to
        ``None`` to force sequential search.
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
    should_scan_source_lmdb = True
    if mol_index_path is not None:
        local_records, should_scan_source_lmdb = _find_molecule_records_in_index(
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
    if missing_query_indices and should_scan_source_lmdb:
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
    write_lmdb_records(records, output_path, overwrite=overwrite, map_size=map_size)
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
    source_data_dir: str | Path = DEFAULT_DRUGCLIP_DATA_DIR,
    splits: Sequence[str] = ("train", "valid"),
    overwrite: bool = True,
    skip_invalid: bool = False,
    map_size: int = 1 << 40,
    commit_interval: int = 1000,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Build the target-fishing candidate-pocket LMDB from DrugCLIP splits.

    The candidate pockets are copied from the DrugCLIP training/validation
    LMDB records instead of being re-extracted from ``combine_set`` structure
    bundles. This preserves the exact pocket atom lists and coordinates seen by
    DrugCLIP during training and validation, which avoids geometry mismatches
    caused by rebuilding pockets from PDB files.

    Each valid input row becomes one output candidate. Duplicates are kept on
    purpose: ``train.lmdb`` contains repeated raw PDB IDs, and some repeated IDs
    have distinct pocket geometries. The output record keeps the raw ``pocket``
    value so existing positive-pair tables that use raw PDB IDs continue to
    match benchmark rankings. Duplicate pocket names in ranked outputs are
    therefore expected.

    Output records contain the fields consumed by
    ``DrugCLIPTask.load_pockets_dataset`` plus provenance/debug fields:

    ``pocket``
        Raw pocket identifier copied from the source LMDB, usually a PDB ID.
    ``pocket_atoms``
        Pocket atom names/types copied from the source record.
    ``pocket_coordinates``
        Pocket coordinates copied from the source record and normalized to a
        ``float32`` array with shape ``(n_atoms, 3)``.
    ``source_split``
        Source split name, for example ``"train"`` or ``"valid"``.
    ``source_lmdb_key``
        Numeric LMDB key of the source row, stored as a string.
    ``pocket_geometry_hash``
        SHA-256 hash over the normalized atom list and coordinates. This is not
        used for ranking, but it makes duplicate and geometry-specific analyses
        reproducible.

    If this function overwrites an existing candidate file, regenerate or remove
    the corresponding pocket embedding cache before benchmarking. DrugCLIP's
    pocket cache is keyed by the LMDB basename and checkpoint tag, not by the
    LMDB file contents.

    Parameters
    ----------
    output_path:
        Destination LMDB file. Defaults to ``data/candidate_pockets.lmdb``.
    source_data_dir:
        DrugCLIP data directory containing ``{split}.lmdb`` files. Defaults to
        ``external/DrugCLIP/data``.
    splits:
        Source LMDB split names to concatenate. The default ``("train",
        "valid")`` creates candidates from both DrugCLIP fine-tuning splits.
    overwrite:
        Replace an existing output LMDB.
    skip_invalid:
        If true, skip source records whose pocket fields cannot be read or
        normalized and report them in the returned summary. If false, raise on
        the first invalid record.
    map_size:
        LMDB map size.
    commit_interval:
        Number of records to write per LMDB transaction.
    show_progress:
        Display progress bars while scanning the source LMDB files.

    Returns
    -------
    dict[str, Any]
        Build summary with output path, source split counts, written pocket
        count, and skipped entries.
    """

    if commit_interval <= 0:
        raise ValueError("commit_interval must be greater than 0")

    output_path = Path(output_path)
    source_data_dir = Path(source_data_dir)
    splits = tuple(str(split).strip() for split in splits if str(split).strip())
    if not splits:
        raise ValueError("splits must contain at least one split name")

    source_lmdbs = {split: source_data_dir / f"{split}.lmdb" for split in splits}
    for split, source_lmdb in source_lmdbs.items():
        if not source_lmdb.exists():
            raise FileNotFoundError(
                f"Source LMDB for split {split!r} not found: {source_lmdb}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"LMDB already exists: {output_path}")

    tmp_output_path = output_path.with_name(f".{output_path.name}.tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()

    skipped: list[dict[str, str]] = []
    split_summaries: dict[str, dict[str, Any]] = {}
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
        for split, source_lmdb in source_lmdbs.items():
            source_records = _lmdb_entry_count(source_lmdb)
            split_written = 0
            split_skipped = 0
            record_iter = tqdm(
                _iter_pickled_lmdb_records(source_lmdb),
                total=source_records,
                desc=f"Copying {split}.lmdb pockets",
                unit="record",
                disable=not show_progress,
            )
            for source_key, source_record in record_iter:
                try:
                    record = _candidate_pocket_record_from_lmdb_record(
                        source_record,
                        split=split,
                        source_key=source_key,
                    )
                except Exception as exc:
                    if not skip_invalid:
                        raise RuntimeError(
                            "Failed to build candidate pocket record from "
                            f"{source_lmdb}:{source_key}"
                        ) from exc
                    skipped.append(
                        {
                            "split": split,
                            "source_lmdb": str(source_lmdb),
                            "lmdb_key": source_key,
                            "error": str(exc),
                        }
                    )
                    split_skipped += 1
                    continue

                transaction.put(
                    str(written).encode("ascii"),
                    pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL),
                )
                written += 1
                split_written += 1

                if written % commit_interval == 0:
                    transaction.commit()
                    transaction = env.begin(write=True)

            split_summaries[split] = {
                "source_lmdb": str(source_lmdb),
                "source_records": source_records,
                "pockets": split_written,
                "skipped": split_skipped,
            }

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
        "source_data_dir": str(source_data_dir),
        "splits": split_summaries,
        "pockets": written,
        "skipped": len(skipped),
        "skipped_entries": skipped,
    }


def _lmdb_entry_count(path: Path) -> int:
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
        return env.stat()["entries"]
    finally:
        env.close()


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


def _candidate_pocket_record_from_lmdb_record(
    record: dict[str, Any],
    *,
    split: str,
    source_key: str,
) -> dict[str, Any]:
    pocket = str(record.get("pocket") or "").strip()
    if not pocket:
        raise ValueError("source record is missing a non-empty pocket field")

    if "pocket_atoms" not in record:
        raise ValueError("source record is missing pocket_atoms")
    if "pocket_coordinates" not in record:
        raise ValueError("source record is missing pocket_coordinates")

    atoms = [str(atom).strip() for atom in record["pocket_atoms"]]
    coordinates = np.asarray(record["pocket_coordinates"], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            "pocket_coordinates must have shape (n_atoms, 3), "
            f"got {coordinates.shape}"
        )
    if len(atoms) != len(coordinates):
        raise ValueError(
            "pocket_atoms and pocket_coordinates must have the same length "
            f"({len(atoms)} != {len(coordinates)})"
        )
    if len(atoms) == 0:
        raise ValueError("source record has no pocket atoms")

    return {
        "pocket": pocket,
        "pocket_atoms": atoms,
        "pocket_coordinates": coordinates,
        "source_split": split,
        "source_lmdb_key": source_key,
        "pocket_geometry_hash": _pocket_geometry_hash(atoms, coordinates),
    }


def _pocket_geometry_hash(atoms: Sequence[str], coordinates: np.ndarray) -> str:
    digest = sha256()
    digest.update("\0".join(atoms).encode("utf-8"))
    digest.update(np.asarray(coordinates, dtype=np.float32).tobytes())
    return digest.hexdigest()


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


def build_ranked_pockets_frame(
    ranked_pockets_path: str | Path = DEFAULT_RANKED_POCKETS_PATH,
) -> pl.DataFrame:
    """Read BioSensIA target-fishing scores and add predictable RCSB PDB links."""

    ranked_pockets_path = Path(ranked_pockets_path)
    if not ranked_pockets_path.exists():
        raise FileNotFoundError(f"ranked pockets file not found: {ranked_pockets_path}")

    df = pl.read_csv(
        ranked_pockets_path,
        separator="\t",
        has_header=False,
        schema={
            "pocket": pl.String,
            "drugclip_score": pl.Float64,
        },
    )

    return df.with_columns(
        pl.concat_str(
            [
                pl.lit(RCSB_PDB_STRUCTURE_URL_PREFIX),
                pl.col("pocket").str.to_uppercase(),
            ]
        ).alias("pdb_url")
    )


def build_drugclip_target_fishing_args(
    *,
    drugclip_dir: str | Path = DEFAULT_DRUGCLIP_DIR,
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    mol_path: str | Path = DEFAULT_QUERY_MOL_LMDB,
    pocket_path: str | Path = DEFAULT_CANDIDATE_POCKETS_LMDB,
    emb_dir: str | Path | None = None,
    results_path: str | Path | None = None,
    top_k: int = DEFAULT_TARGET_FISHING_TOP_K,
    batch_size: int = 2,
    batch_size_valid: int = 2,
    num_workers: int = 8,
    seed: int = 1,
    fp16: bool = True,
    cpu: bool = False,
):
    """Create the Uni-Core args object used by BioSensIA target fishing.

    The registered Uni-Core task remains DrugCLIP's ``drugclip`` task. This
    helper only builds an argument namespace from repository-root paths so the
    target-fishing entry point can live in BioSensIA-DC instead of inside
    ``external/DrugCLIP``.
    """

    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    from unicore import options, utils

    drugclip_dir = Path(drugclip_dir).resolve()
    user_dir = drugclip_dir / "unimol"
    data_dir = Path(data_dir).resolve() if data_dir else drugclip_dir / "data"
    checkpoint_path = (
        Path(checkpoint_path).resolve()
        if checkpoint_path
        else drugclip_dir / "checkpoint_best.pt"
    )
    mol_path = Path(mol_path).resolve()
    pocket_path = Path(pocket_path).resolve()
    emb_dir = (
        Path(emb_dir).resolve()
        if emb_dir
        else drugclip_dir / "data" / "pocket_emb"
    )
    results_path = (
        Path(results_path).resolve() if results_path else drugclip_dir / "test"
    )

    # Import DrugCLIP's custom task/model/loss registrations before parsing
    # ``--task drugclip`` and ``--arch drugclip``.
    utils.import_user_module(argparse.Namespace(user_dir=str(user_dir)))

    parser = options.get_validation_parser()
    parser.add_argument(
        "--mol-path",
        type=str,
        default="",
        help="path for query molecule data",
    )
    parser.add_argument(
        "--pocket-path",
        type=str,
        default="",
        help="path for candidate pocket data",
    )
    parser.add_argument(
        "--emb-dir",
        type=str,
        default="",
        help="path for saved candidate-pocket embedding data",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TARGET_FISHING_TOP_K,
        help="number of top-ranked pockets to write",
    )
    options.add_model_args(parser)

    input_args = [
        str(data_dir),
        "--arch",
        "drugclip",
        "--batch-size",
        str(batch_size),
        "--batch-size-valid",
        str(batch_size_valid),
        "--ddp-backend",
        "c10d",
        "--emb-dir",
        str(emb_dir),
        "--fp16-init-scale",
        "4",
        "--fp16-scale-window",
        "256",
        "--log-format",
        "simple",
        "--log-interval",
        "100",
        "--loss",
        "in_batch_softmax",
        "--max-pocket-atoms",
        "256",
        "--mol-path",
        str(mol_path),
        "--num-workers",
        str(num_workers),
        "--path",
        str(checkpoint_path),
        "--pocket-path",
        str(pocket_path),
        "--results-path",
        str(results_path),
        "--seed",
        str(seed),
        "--task",
        "drugclip",
        "--top-k",
        str(top_k),
        "--user-dir",
        str(user_dir),
        "--valid-subset",
        "test",
    ]
    if fp16:
        input_args.append("--fp16")
    if cpu:
        input_args.append("--cpu")

    return options.parse_args_and_arch(parser, input_args=input_args)


def retrieve_pockets_from_drugclip(args) -> tuple[list[str], np.ndarray]:
    """Load DrugCLIP and return ``task.retrieve_pockets`` names and scores."""

    task, model = load_drugclip_model_for_target_fishing(args)
    return task.retrieve_pockets(
        model,
        args.mol_path,
        args.pocket_path,
        args.emb_dir,
        args.top_k,
    )


def retrieve_pocket_rankings_from_drugclip(
    args,
) -> tuple[dict[str, list[str]], dict[str, np.ndarray]]:
    """Load DrugCLIP and return one pocket ranking per query molecule."""

    task, model = load_drugclip_model_for_target_fishing(args)
    return task.rank_pockets_by_query(
        model,
        args.mol_path,
        args.pocket_path,
        args.emb_dir,
        args.top_k,
    )


def load_drugclip_model_for_target_fishing(args):
    """Load a DrugCLIP checkpoint and return ``(task, model)`` for inference."""

    import torch
    from unicore import checkpoint_utils, tasks

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if use_cuda:
        torch.cuda.set_device(args.device_id)
    else:
        LOGGER.warning(
            "CUDA is not available or --cpu was set. BioSensIA target fishing "
            "may fail because DrugCLIP's encoder path moves batches to CUDA."
        )

    LOGGER.info("loading model(s) from %s", args.path)
    state = checkpoint_utils.load_checkpoint_to_cpu(args.path)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    model.load_state_dict(state["model"], strict=False)

    if use_fp16:
        model.half()
    if use_cuda:
        model.cuda()

    model.eval()
    return task, model


def write_ranked_pockets(
    names: Iterable[str],
    scores: Iterable[float],
    output_path: str | Path = DEFAULT_RANKED_POCKETS_PATH,
) -> Path:
    """Write target-fishing pocket scores as DrugCLIP-compatible TSV."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for name, score in zip(names, scores):
            handle.write(f"{name}\t{score}\n")
    return output_path


def target_fishing_main(args, *, output_path: str | Path | None = None) -> Path:
    """Run target fishing from a parsed Uni-Core argument namespace."""

    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than 0")

    names, scores = retrieve_pockets_from_drugclip(args)
    ranked_pockets_path = (
        Path(output_path)
        if output_path is not None
        else Path(args.emb_dir) / "ranked_pockets.txt"
    )
    ranked_pockets_path = write_ranked_pockets(names, scores, ranked_pockets_path)
    LOGGER.info("wrote ranked pockets to %s", ranked_pockets_path)
    return ranked_pockets_path


def run_target_fishing(
    *,
    drugclip_dir: str | Path = DEFAULT_DRUGCLIP_DIR,
    data_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    mol_path: str | Path = DEFAULT_QUERY_MOL_LMDB,
    pocket_path: str | Path = DEFAULT_CANDIDATE_POCKETS_LMDB,
    emb_dir: str | Path | None = None,
    results_path: str | Path | None = None,
    output_path: str | Path | None = None,
    top_k: int = DEFAULT_TARGET_FISHING_TOP_K,
    batch_size: int = 2,
    batch_size_valid: int = 2,
    num_workers: int = 8,
    seed: int = 1,
    fp16: bool = True,
    cpu: bool = False,
) -> Path:
    """Build Uni-Core args, run target fishing, and return the TSV path."""

    args = build_drugclip_target_fishing_args(
        drugclip_dir=drugclip_dir,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        mol_path=mol_path,
        pocket_path=pocket_path,
        emb_dir=emb_dir,
        results_path=results_path,
        top_k=top_k,
        batch_size=batch_size,
        batch_size_valid=batch_size_valid,
        num_workers=num_workers,
        seed=seed,
        fp16=fp16,
        cpu=cpu,
    )
    return target_fishing_main(args, output_path=output_path)


def _build_target_fishing_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run BioSensIA target fishing with DrugCLIP encoders from the "
            "BioSensIA-DC directory."
        ),
    )
    parser.add_argument(
        "--drugclip-dir",
        type=Path,
        default=DEFAULT_DRUGCLIP_DIR,
        help="DrugCLIP checkout containing unimol/, data/, and checkpoint files.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="directory containing DrugCLIP dictionaries; defaults to DRUGCLIP_DIR/data.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="DrugCLIP checkpoint; defaults to DRUGCLIP_DIR/checkpoint_best.pt.",
    )
    parser.add_argument(
        "--mol-path",
        type=Path,
        default=DEFAULT_QUERY_MOL_LMDB,
        help="query molecule LMDB.",
    )
    parser.add_argument(
        "--pocket-path",
        type=Path,
        default=DEFAULT_CANDIDATE_POCKETS_LMDB,
        help="candidate pocket LMDB.",
    )
    parser.add_argument(
        "--emb-dir",
        type=Path,
        default=None,
        help="candidate-pocket embedding cache directory; defaults to DRUGCLIP_DIR/data/pocket_emb.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Uni-Core results directory; defaults to DRUGCLIP_DIR/test.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="ranked pockets TSV; defaults to EMB_DIR/ranked_pockets.txt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TARGET_FISHING_TOP_K,
        help="number of top-ranked pockets to write.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--batch-size-valid", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable or disable fp16 inference.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help=(
            "request CPU execution; the DrugCLIP encoder path used for target "
            "fishing is normally GPU-only."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOGLEVEL", "INFO"),
        help="Python logging level.",
    )
    return parser


def cli_main(argv: list[str] | None = None) -> Path:
    """CLI entry point for ``python -m biosensia_target_fishing``."""

    parser = _build_target_fishing_cli_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=str(args.log_level).upper(),
    )
    return run_target_fishing(
        drugclip_dir=args.drugclip_dir,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
        mol_path=args.mol_path,
        pocket_path=args.pocket_path,
        emb_dir=args.emb_dir,
        results_path=args.results_path,
        output_path=args.output_path,
        top_k=args.top_k,
        batch_size=args.batch_size,
        batch_size_valid=args.batch_size_valid,
        num_workers=args.num_workers,
        seed=args.seed,
        fp16=args.fp16,
        cpu=args.cpu,
    )


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
) -> tuple[dict[int, tuple[dict[str, Any], str]], bool]:
    if not index_path.exists():
        if show_progress:
            tqdm.write(f"Molecule index not found: {index_path}")
        return {}, True
    if not source_lmdb_path.exists():
        if show_progress:
            tqdm.write(f"Source molecule LMDB not found: {source_lmdb_path}")
        return {}, False
    if show_progress:
        tqdm.write(
            f"Searching molecule index {index_path} "
            f"for {len(molecule_queries)} molecule query/queries."
        )

    try:
        source_env = lmdb.open(
            str(source_lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=256,
        )
    except lmdb.Error as exc:
        if show_progress:
            tqdm.write(f"Could not open source molecule LMDB {source_lmdb_path}: {exc}")
        return {}, False

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
            return {}, True

        try:
            try:
                lookup_db = index_env.open_db(_MOL_INDEX_LOOKUP_DB)
                meta_db = index_env.open_db(_MOL_INDEX_META_DB)
            except lmdb.Error as exc:
                if show_progress:
                    tqdm.write(f"Molecule index {index_path} is not usable: {exc}")
                return {}, True

            with index_env.begin(db=meta_db) as meta_transaction:
                if not _molecule_index_matches_source(
                    meta_transaction,
                    source_entries=source_env.stat()["entries"],
                ):
                    if show_progress:
                        tqdm.write(
                            f"Molecule index {index_path} is not compatible with "
                            f"{source_lmdb_path}; falling back to sequential search."
                        )
                    return {}, True

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
            elif show_progress:
                tqdm.write(f"No molecule records found via {index_path}.")
            return found, False
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
    source_entries: int,
) -> bool:
    metadata = _read_molecule_index_metadata(transaction)
    return (
        metadata.get("schema_version") == _MOL_INDEX_SCHEMA_VERSION
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
    "DEFAULT_DRUGCLIP_CHECKPOINT",
    "DEFAULT_DRUGCLIP_DATA_DIR",
    "DEFAULT_DRUGCLIP_DIR",
    "DEFAULT_DRUGCLIP_MOLS_INDEX_LMDB",
    "DEFAULT_DRUGCLIP_MOLS_LMDB",
    "DEFAULT_QUERY_MOL_LMDB",
    "DEFAULT_RANKED_POCKETS_PATH",
    "DEFAULT_TARGET_FISHING_TOP_K",
    "RCSB_PDB_STRUCTURE_URL_PREFIX",
    "build_drugclip_target_fishing_args",
    "build_candidate_pockets_frame",
    "build_candidate_pockets_lmdb",
    "build_mol_lmdb_index",
    "build_ranked_pockets_frame",
    "cli_main",
    "create_mol_lmdb",
    "load_drugclip_model_for_target_fishing",
    "retrieve_pocket_rankings_from_drugclip",
    "retrieve_pockets_from_drugclip",
    "run_target_fishing",
    "target_fishing_main",
    "write_ranked_pockets",
]


if __name__ == "__main__":
    cli_main()
