"""Build a pocket-ligand identity table from DrugCLIP train/valid LMDB files.

The default input is the DrugCLIP training data under ``external/DrugCLIP/data``:

    import scratch.analyze_training_data as atd
    df = atd.build_training_frame()

or from the command line:

    uv run python scratch/analyze_training_data.py

Rows are LMDB records, not deduplicated biological pairs. This is intentional:
``train.lmdb`` contains many repeated raw ``(pocket, smi)`` identifiers with
different pocket geometries, so ``split`` and ``lmdb_key`` are part of the row
identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lmdb
import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm


DEFAULT_LMDB_PATHS = {
    "train": Path("external/DrugCLIP/data/train.lmdb"),
    "valid": Path("external/DrugCLIP/data/valid.lmdb"),
}
DEFAULT_COMBINE_SET_DIR = Path("external/DrugCLIP/data/pdb/combine_set")
DEFAULT_OUTPUT_PATH = Path("data/biosensia_finetune/training_data_pairs.parquet")
DEFAULT_CACHE_PATH = Path("scratch/training_data_metadata_cache.json")

PD_BE_SIFTS_UNIPROT_URL = (
    "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
)
PUBCHEM_CIDS_BY_INCHIKEY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
    "{inchikey}/cids/JSON"
)
CHEMBL_MOLECULE_BY_INCHIKEY_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
USER_AGENT = "BioSensIA-DC/0.1 (DrugCLIP training data pair analysis)"

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}
STANDARD_AMINO_ACID_CODES = set(AA3_TO_AA1) | {
    "ASH",
    "CYM",
    "CYX",
    "GLH",
    "HID",
    "HIE",
    "HIP",
}

PAIR_COLUMNS = [
    "pocket",
    "ligand",
    "pdb_id",
    "pocket_id_source",
    "pocket_uniprot_accessions",
    "pocket_chain_ids",
    "pocket_parse_status",
    "ligand_inchikey",
    "ligand_smiles",
    "ligand_ccd_id",
    "ligand_ccd_ids",
    "ligand_type",
    "ligand_id_source",
    "ligand_parse_status",
    "pubchem_cid",
    "chembl_id",
    "protein_file",
    "pocket_file",
    "ligand_sdf_file",
    "ligand_mol2_file",
]

ROW_COLUMNS = [
    "split",
    "lmdb_path",
    "lmdb_key",
    *PAIR_COLUMNS,
    "is_positive",
    "label_source",
    "raw_pocket",
    "raw_smi",
    "raw_record_fields",
    "pocket_geometry_hash",
    "ligand_conformer_count",
    "ligand_atom_count",
    "pocket_atom_count",
    "residue_count",
]


class MetadataFetchError(RuntimeError):
    """Raised when an online metadata lookup fails."""


def build_training_frame(
    lmdb_paths: Mapping[str, str | Path] | None = None,
    *,
    combine_set_dir: str | Path = DEFAULT_COMBINE_SET_DIR,
    online_pocket: bool = True,
    online_ligand: bool = False,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    limit_per_split: int | None = None,
    timeout_seconds: float = 30,
    sleep_seconds: float = 0,
    strict_online: bool = False,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Return a Polars frame of positive train/valid LMDB records.

    Parameters
    ----------
    lmdb_paths:
        Mapping from split name to LMDB path. By default, includes the local
        DrugCLIP ``train.lmdb`` and ``valid.lmdb`` files.
    combine_set_dir:
        Optional local PDB complex tree used to recover chain, UniProt, CCD, and
        source-file metadata by joining on the LMDB ``pocket`` PDB id.
    online_pocket:
        Fetch missing UniProt annotations from PDBe SIFTS. Defaults to true, as
        in ``analyze_pdb_complexes.py``.
    online_ligand:
        Fetch PubChem and ChEMBL ligand metadata by InChIKey. Defaults to false,
        as in ``analyze_pdb_complexes.py``.
    cache_path:
        JSON cache for enabled online metadata lookups. Set to ``None`` to
        disable caching.
    limit_per_split:
        Optional number of LMDB records to process from each split.
    timeout_seconds:
        Per-request timeout for online metadata lookups.
    sleep_seconds:
        Optional delay after each online request.
    strict_online:
        Raise on online lookup failures instead of using fallbacks/nulls.
    show_progress:
        Display a progress bar.
    """

    if lmdb_paths is None:
        lmdb_paths = DEFAULT_LMDB_PATHS
    if limit_per_split is not None and limit_per_split < 0:
        raise ValueError("limit_per_split must be greater than or equal to 0")

    combine_set_dir = Path(combine_set_dir)
    cache = _load_cache(cache_path)
    pocket_cache: dict[str, dict[str, Any]] = {}
    ligand_cache: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []

    RDLogger.DisableLog("rdApp.*")

    for split, lmdb_path in lmdb_paths.items():
        lmdb_path = Path(lmdb_path)
        if not lmdb_path.exists():
            raise FileNotFoundError(f"LMDB file not found for split {split!r}: {lmdb_path}")

        iterator = _iter_lmdb_records(lmdb_path, limit=limit_per_split)
        if show_progress:
            iterator = tqdm(
                iterator,
                total=_lmdb_record_count(lmdb_path, limit=limit_per_split),
                desc=f"Analyzing {split}.lmdb",
                unit="record",
            )

        for lmdb_key, record in iterator:
            rows.append(
                analyze_training_record(
                    record,
                    split=split,
                    lmdb_path=lmdb_path,
                    lmdb_key=lmdb_key,
                    combine_set_dir=combine_set_dir,
                    online_pocket=online_pocket,
                    online_ligand=online_ligand,
                    cache=cache,
                    pocket_cache=pocket_cache,
                    ligand_cache=ligand_cache,
                    timeout_seconds=timeout_seconds,
                    sleep_seconds=sleep_seconds,
                    strict_online=strict_online,
                )
            )

    _save_cache(cache_path, cache)
    return pl.DataFrame(rows, schema=ROW_COLUMNS, orient="row")


def analyze_training_record(
    record: dict[str, Any],
    *,
    split: str,
    lmdb_path: str | Path,
    lmdb_key: str,
    combine_set_dir: Path,
    online_pocket: bool,
    online_ligand: bool,
    cache: dict[str, Any],
    pocket_cache: dict[str, dict[str, Any]],
    ligand_cache: dict[tuple[str, str], dict[str, Any]],
    timeout_seconds: float,
    sleep_seconds: float,
    strict_online: bool,
) -> dict[str, Any]:
    """Analyze one raw LMDB record and return one output row."""

    raw_pocket = str(record.get("pocket") or "").lower()
    if not raw_pocket:
        raise ValueError(f"record {lmdb_key} in {split} has no pocket field")
    raw_smi = str(record.get("smi") or "")
    pocket_geometry_hash = _pocket_geometry_hash(record)

    if raw_pocket not in pocket_cache:
        pocket_cache[raw_pocket] = extract_pocket_identity(
            raw_pocket,
            combine_set_dir=combine_set_dir,
            online=online_pocket,
            cache=cache,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            strict_online=strict_online,
        )
    pocket_info = pocket_cache[raw_pocket]

    ligand_cache_key = (raw_pocket, raw_smi)
    if ligand_cache_key not in ligand_cache:
        files = _complex_files(combine_set_dir / raw_pocket)
        ligand_cache[ligand_cache_key] = extract_ligand_identity(
            raw_pocket,
            mol=record.get("mol"),
            raw_smi=raw_smi,
            ligand_mol2=files["ligand_mol2_file"],
            online=online_ligand,
            cache=cache,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
            strict_online=strict_online,
        )
    ligand_info = ligand_cache[ligand_cache_key]

    return {
        "split": split,
        "lmdb_path": str(lmdb_path),
        "lmdb_key": lmdb_key,
        "pocket": pocket_info["pocket"],
        "ligand": ligand_info["ligand"],
        "pdb_id": raw_pocket,
        "pocket_id_source": pocket_info["pocket_id_source"],
        "pocket_uniprot_accessions": pocket_info["uniprot_accessions"],
        "pocket_chain_ids": pocket_info["chain_ids"],
        "pocket_parse_status": pocket_info["parse_status"],
        "ligand_inchikey": ligand_info["inchikey"],
        "ligand_smiles": ligand_info["smiles"],
        "ligand_ccd_id": ligand_info["ccd_id"],
        "ligand_ccd_ids": ligand_info["ccd_ids"],
        "ligand_type": ligand_info["ligand_type"],
        "ligand_id_source": ligand_info["ligand_id_source"],
        "ligand_parse_status": ligand_info["parse_status"],
        "pubchem_cid": ligand_info["pubchem_cid"],
        "chembl_id": ligand_info["chembl_id"],
        "protein_file": _string_path(pocket_info["protein_file"]),
        "pocket_file": _string_path(pocket_info["pocket_file"]),
        "ligand_sdf_file": _string_path(pocket_info["ligand_sdf_file"]),
        "ligand_mol2_file": _string_path(pocket_info["ligand_mol2_file"]),
        "is_positive": True,
        "label_source": "missing_label_defaults_to_positive",
        "raw_pocket": raw_pocket,
        "raw_smi": raw_smi,
        "raw_record_fields": sorted(record.keys()),
        "pocket_geometry_hash": pocket_geometry_hash,
        "ligand_conformer_count": _safe_len(record.get("coordinates")),
        "ligand_atom_count": _safe_len(record.get("atoms")),
        "pocket_atom_count": _safe_len(record.get("pocket_atoms")),
        "residue_count": _safe_len(record.get("residue")),
    }


def extract_pocket_identity(
    pdb_id: str,
    *,
    combine_set_dir: str | Path,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
    strict_online: bool,
) -> dict[str, Any]:
    """Return the chosen pocket identifier and annotations for a PDB id."""

    pdb_id = pdb_id.lower()
    complex_dir = Path(combine_set_dir) / pdb_id
    files = _complex_files(complex_dir)
    local_files_found = any(path is not None for path in files.values())

    status_parts = []
    if local_files_found:
        status_parts.append("combine_set:found")
    else:
        status_parts.append("combine_set:missing")

    chain_ids = sorted(
        _parse_chain_ids(files["pocket_file"])
        or _parse_chain_ids(files["protein_file"])
    )
    if chain_ids:
        status_parts.append("chains:local")
    else:
        status_parts.append("chains:missing")

    try:
        mappings = _get_pdb_uniprot_mappings(
            pdb_id,
            online=online,
            cache=cache,
            timeout_seconds=timeout_seconds,
            sleep_seconds=sleep_seconds,
        )
    except MetadataFetchError as exc:
        if strict_online:
            raise
        mappings = []
        status_parts.append(f"sifts_error:{exc}")

    if mappings:
        accessions = _accessions_for_chains(mappings, chain_ids)
        if accessions:
            if chain_ids:
                status_parts.append("ok:sifts")
            else:
                status_parts.append("ok:sifts_no_chain_filter")
            return {
                "pocket": "+".join(accessions),
                "pocket_id_source": "uniprot_sifts",
                "uniprot_accessions": accessions,
                "chain_ids": chain_ids,
                "parse_status": ";".join(status_parts),
                **files,
            }
        status_parts.append("sifts_no_chain_match")
    else:
        status_parts.append("sifts_disabled" if not online else "sifts_missing")

    fallback = _protein_sequence_hash(files["protein_file"], chain_ids)
    if fallback is not None:
        status_parts.append("fallback:protein_sequence_hash")
        return {
            "pocket": fallback,
            "pocket_id_source": "seqsha1",
            "uniprot_accessions": [],
            "chain_ids": chain_ids,
            "parse_status": ";".join(status_parts),
            **files,
        }

    status_parts.append("fallback:pdb_id")
    return {
        "pocket": f"pdb:{pdb_id}",
        "pocket_id_source": "pdb_id",
        "uniprot_accessions": [],
        "chain_ids": chain_ids,
        "parse_status": ";".join(status_parts),
        **files,
    }


def extract_ligand_identity(
    pdb_id: str,
    *,
    mol: Chem.Mol | None,
    raw_smi: str,
    ligand_mol2: Path | None,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
    strict_online: bool,
) -> dict[str, Any]:
    """Return the chosen ligand identifier and annotations for an LMDB record."""

    status_parts = []
    mol_smiles = _mol_to_smiles(mol) if mol is not None else None
    mol_inchikey = _mol_to_inchikey(mol) if mol is not None else None
    if mol is None:
        status_parts.append("mol:missing")
    elif mol_smiles or mol_inchikey:
        status_parts.append("ok:mol")
    else:
        status_parts.append("mol:unusable")

    raw_mol = _mol_from_smiles(raw_smi)
    raw_smiles = _mol_to_smiles(raw_mol) if raw_mol is not None else None
    raw_inchikey = _mol_to_inchikey(raw_mol) if raw_mol is not None else None
    if raw_mol is not None:
        status_parts.append("ok:raw_smi")
    elif raw_smi:
        status_parts.append("raw_smi:unparseable")
    else:
        status_parts.append("raw_smi:missing")

    if mol_smiles and raw_smiles:
        if mol_smiles == raw_smiles:
            status_parts.append("mol_raw_smi:match")
        else:
            status_parts.append("mol_raw_smi:mismatch")

    inchikey = mol_inchikey or raw_inchikey
    smiles = mol_smiles or raw_smiles or raw_smi or None
    if mol_inchikey:
        ligand_id_source = "inchikey:mol"
    elif raw_inchikey:
        ligand_id_source = "inchikey:raw_smi"
    elif smiles:
        ligand_id_source = "canonical_smiles_hash"
    else:
        ligand_id_source = "pdb_id"

    ccd_ids = _parse_mol2_ccd_ids(ligand_mol2)
    ccd_id = "+".join(ccd_ids) if ccd_ids else None
    ligand = _choose_ligand_id(
        pdb_id=pdb_id,
        inchikey=inchikey,
        smiles=smiles,
        ccd_id=ccd_id,
        id_source=ligand_id_source,
    )
    ligand_type = _classify_ligand_type(ccd_ids, mol or raw_mol)

    pubchem_cid = None
    chembl_id = None
    if inchikey is not None:
        try:
            pubchem_cid = _get_pubchem_cid(
                inchikey,
                online=online,
                cache=cache,
                timeout_seconds=timeout_seconds,
                sleep_seconds=sleep_seconds,
            )
            chembl_id = _get_chembl_id(
                inchikey,
                online=online,
                cache=cache,
                timeout_seconds=timeout_seconds,
                sleep_seconds=sleep_seconds,
            )
        except MetadataFetchError:
            if strict_online:
                raise
            status_parts.append("online_ligand:error")

    return {
        "ligand": ligand,
        "inchikey": inchikey,
        "smiles": smiles,
        "ccd_id": ccd_id,
        "ccd_ids": ccd_ids,
        "ligand_type": ligand_type,
        "ligand_id_source": ligand_id_source,
        "parse_status": ";".join(status_parts),
        "pubchem_cid": pubchem_cid,
        "chembl_id": chembl_id,
    }


def write_frame(df: pl.DataFrame, output_path: str | Path) -> None:
    """Write a Polars frame to Parquet, CSV, or JSON based on suffix."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".parquet":
        df.write_parquet(output_path)
    elif suffix == ".csv":
        df.write_csv(output_path)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        df.write_ndjson(output_path)
    else:
        raise ValueError(
            f"Unsupported output suffix {suffix!r}; use .parquet, .csv, or .jsonl"
        )


def _iter_lmdb_records(
    lmdb_path: Path,
    *,
    limit: int | None,
):
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
        with env.begin() as txn:
            keys = list(txn.cursor().iternext(values=False))
            keys = _sort_lmdb_keys(keys)
            if limit is not None:
                keys = keys[:limit]
            for key in keys:
                value = txn.get(key)
                if value is None:
                    continue
                yield key.decode("ascii"), pickle.loads(value)
    finally:
        env.close()


def _lmdb_record_count(lmdb_path: Path, *, limit: int | None) -> int:
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
        with env.begin() as txn:
            count = txn.stat()["entries"]
    finally:
        env.close()
    return min(count, limit) if limit is not None else count


def _sort_lmdb_keys(keys: list[bytes]) -> list[bytes]:
    try:
        return sorted(keys, key=lambda key: int(key.decode("ascii")))
    except ValueError:
        return sorted(keys)


def _complex_files(complex_dir: Path) -> dict[str, Path | None]:
    pdb_id = complex_dir.name.lower()
    return {
        "protein_file": _first_existing(
            complex_dir / f"{pdb_id}_protein.pdb",
            complex_dir / f"{pdb_id.upper()}_protein.pdb",
        ),
        "pocket_file": _first_existing(
            complex_dir / f"{pdb_id}_pocket.pdb",
            complex_dir / f"{pdb_id}_pocket6A.pdb",
            complex_dir / f"{pdb_id.upper()}_pocket.pdb",
            complex_dir / f"{pdb_id.upper()}_pocket6A.pdb",
        ),
        "ligand_sdf_file": _first_existing(
            complex_dir / f"{pdb_id}_ligand.sdf",
            complex_dir / f"{pdb_id.upper()}_ligand.sdf",
        ),
        "ligand_mol2_file": _first_existing(
            complex_dir / f"{pdb_id}_ligand.mol2",
            complex_dir / f"{pdb_id.upper()}_ligand.mol2",
        ),
    }


def _first_existing(*paths: Path) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def _parse_chain_ids(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()

    chain_ids = set()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM  ", "HETATM")) and len(line) > 21:
                chain_id = line[21].strip()
                if chain_id:
                    chain_ids.add(chain_id)
    return chain_ids


def _get_pdb_uniprot_mappings(
    pdb_id: str,
    *,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
) -> list[dict[str, str]]:
    pdb_id = pdb_id.lower()
    if not online:
        return []

    pdb_cache = cache.setdefault("pdb_uniprot_mappings", {})
    if pdb_id in pdb_cache:
        return pdb_cache[pdb_id] or []

    url = PD_BE_SIFTS_UNIPROT_URL.format(pdb_id=pdb_id)
    data = _get_json(url, timeout_seconds=timeout_seconds)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    entry = data.get(pdb_id) or data.get(pdb_id.upper()) or {}
    uniprot = entry.get("UniProt") or {}
    mappings = []
    for accession, payload in uniprot.items():
        for mapping in payload.get("mappings", []):
            chain_id = str(mapping.get("chain_id") or "").strip()
            struct_asym_id = str(mapping.get("struct_asym_id") or "").strip()
            entity_id = str(mapping.get("entity_id") or "").strip()
            mappings.append(
                {
                    "accession": accession,
                    "chain_id": chain_id,
                    "struct_asym_id": struct_asym_id,
                    "entity_id": entity_id,
                }
            )

    pdb_cache[pdb_id] = mappings
    return mappings


def _accessions_for_chains(
    mappings: list[dict[str, str]],
    chain_ids: list[str],
) -> list[str]:
    if not chain_ids:
        return sorted({mapping["accession"] for mapping in mappings})

    chain_id_set = set(chain_ids)
    accessions = {
        mapping["accession"]
        for mapping in mappings
        if mapping.get("chain_id") in chain_id_set
        or mapping.get("struct_asym_id") in chain_id_set
    }
    if not accessions:
        all_accessions = {mapping["accession"] for mapping in mappings}
        if len(all_accessions) == 1:
            accessions = all_accessions
    return sorted(accessions)


def _protein_sequence_hash(
    protein_pdb: Path | None,
    chain_ids: list[str],
) -> str | None:
    sequences = _parse_seqres_sequences(protein_pdb)
    if not sequences:
        sequences = _parse_atom_sequences(protein_pdb)
    if not sequences:
        return None

    if chain_ids:
        selected = {
            chain_id: sequences[chain_id]
            for chain_id in chain_ids
            if chain_id in sequences and sequences[chain_id]
        }
        if selected:
            sequences = selected

    sequence_blob = "|".join(
        f"{chain_id}:{sequence}" for chain_id, sequence in sorted(sequences.items())
    )
    digest = hashlib.sha1(sequence_blob.encode("ascii")).hexdigest()
    return f"seqsha1:{digest}"


def _parse_seqres_sequences(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    residues_by_chain: dict[str, list[str]] = defaultdict(list)
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("SEQRES"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            chain_id = parts[2]
            residues_by_chain[chain_id].extend(parts[4:])

    return {
        chain_id: "".join(AA3_TO_AA1.get(residue, "X") for residue in residues)
        for chain_id, residues in residues_by_chain.items()
    }


def _parse_atom_sequences(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}

    residues_by_chain: dict[str, list[str]] = defaultdict(list)
    seen_residues = set()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or len(line) < 27:
                continue
            chain_id = line[21].strip() or "_"
            residue_key = (chain_id, line[22:27])
            if residue_key in seen_residues:
                continue
            seen_residues.add(residue_key)
            residue_name = line[17:20].strip()
            residues_by_chain[chain_id].append(AA3_TO_AA1.get(residue_name, "X"))

    return {
        chain_id: "".join(residues)
        for chain_id, residues in residues_by_chain.items()
    }


def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


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
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    return inchikey or None


def _parse_mol2_ccd_ids(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []

    ccd_ids = set()
    in_atom_block = False
    with path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line.startswith("@<TRIPOS>") and in_atom_block:
                break
            if not in_atom_block:
                continue

            parts = line.split()
            if len(parts) >= 8:
                ccd_id = parts[7].strip()
                if ccd_id and ccd_id != "<0>":
                    ccd_ids.add(ccd_id)
    return sorted(ccd_ids)


def _classify_ligand_type(
    ccd_ids: list[str],
    mol: Chem.Mol | None,
) -> str:
    if ccd_ids:
        aa_like = [ccd_id for ccd_id in ccd_ids if ccd_id in STANDARD_AMINO_ACID_CODES]
        if len(aa_like) == len(ccd_ids) and len(ccd_ids) > 1:
            return "peptide"
        if aa_like:
            return "peptide_or_composite"
        if len(ccd_ids) == 1:
            return "small_molecule"
        return "composite"

    if mol is not None:
        return "small_molecule_or_unknown"
    return "unknown"


def _choose_ligand_id(
    *,
    pdb_id: str,
    inchikey: str | None,
    smiles: str | None,
    ccd_id: str | None,
    id_source: str,
) -> str:
    if inchikey and id_source.startswith("inchikey"):
        return inchikey
    if smiles:
        digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
        return f"smilesha1:{digest}"
    if ccd_id:
        return f"ccd:{ccd_id}"
    return f"pdb_ligand:{pdb_id}"


def _get_pubchem_cid(
    inchikey: str,
    *,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
) -> int | None:
    if not online:
        return None

    pubchem_cache = cache.setdefault("pubchem_cid_by_inchikey", {})
    if inchikey in pubchem_cache:
        return pubchem_cache[inchikey]

    url = PUBCHEM_CIDS_BY_INCHIKEY_URL.format(
        inchikey=urllib.parse.quote(inchikey, safe="")
    )
    data = _get_json(url, timeout_seconds=timeout_seconds, allow_not_found=True)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    cids = (data or {}).get("IdentifierList", {}).get("CID", [])
    cid = int(cids[0]) if cids else None
    pubchem_cache[inchikey] = cid
    return cid


def _get_chembl_id(
    inchikey: str,
    *,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
) -> str | None:
    if not online:
        return None

    chembl_cache = cache.setdefault("chembl_id_by_inchikey", {})
    if inchikey in chembl_cache:
        return chembl_cache[inchikey]

    query = urllib.parse.urlencode(
        {
            "molecule_structures__standard_inchi_key": inchikey,
            "limit": 1,
        }
    )
    data = _get_json(
        f"{CHEMBL_MOLECULE_BY_INCHIKEY_URL}?{query}",
        timeout_seconds=timeout_seconds,
        allow_not_found=True,
    )
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)

    molecules = (data or {}).get("molecules", [])
    chembl_id = molecules[0].get("molecule_chembl_id") if molecules else None
    chembl_cache[inchikey] = chembl_id
    return chembl_id


def _get_json(
    url: str,
    *,
    timeout_seconds: float,
    allow_not_found: bool = False,
) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if allow_not_found and exc.code == 404:
            return None
        raise MetadataFetchError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise MetadataFetchError(f"{type(exc).__name__} for {url}") from exc


def _load_cache(cache_path: str | Path | None) -> dict[str, Any]:
    if cache_path is None:
        return {}

    cache_path = Path(cache_path)
    if not cache_path.exists():
        return {}
    with cache_path.open() as handle:
        return json.load(handle)


def _save_cache(cache_path: str | Path | None, cache: dict[str, Any]) -> None:
    if cache_path is None:
        return

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _pocket_geometry_hash(record: dict[str, Any]) -> str | None:
    atoms = record.get("pocket_atoms")
    coordinates = record.get("pocket_coordinates")
    if atoms is None or coordinates is None:
        return None

    try:
        coord_array = np.asarray(coordinates, dtype=np.float32)
    except (TypeError, ValueError):
        return None

    digest = hashlib.sha1()
    digest.update("\0".join(map(str, atoms)).encode("utf-8"))
    digest.update(str(coord_array.shape).encode("ascii"))
    digest.update(np.ascontiguousarray(coord_array).tobytes())
    residue = record.get("residue")
    if residue is not None:
        digest.update("\0".join(map(str, residue)).encode("utf-8"))
    return f"pocketgeomsha1:{digest.hexdigest()}"


def _safe_len(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _string_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Polars-compatible positive-pair table from DrugCLIP "
            "train.lmdb and valid.lmdb."
        )
    )
    parser.add_argument(
        "--train-lmdb",
        type=Path,
        default=DEFAULT_LMDB_PATHS["train"],
        help=f"Training LMDB path. Default: {DEFAULT_LMDB_PATHS['train']}",
    )
    parser.add_argument(
        "--valid-lmdb",
        type=Path,
        default=DEFAULT_LMDB_PATHS["valid"],
        help=f"Validation LMDB path. Default: {DEFAULT_LMDB_PATHS['valid']}",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip train.lmdb.",
    )
    parser.add_argument(
        "--no-valid",
        action="store_true",
        help="Skip valid.lmdb.",
    )
    parser.add_argument(
        "--combine-set-dir",
        type=Path,
        default=DEFAULT_COMBINE_SET_DIR,
        help=f"Local complex directory tree. Default: {DEFAULT_COMBINE_SET_DIR}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output .parquet, .csv, or .jsonl file. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Metadata cache path. Default: {DEFAULT_CACHE_PATH}",
    )
    parser.add_argument(
        "--no-online-pocket",
        action="store_true",
        help=(
            "Do not fetch UniProt metadata for pocket IDs; use sequence-hash "
            "or PDB-id fallbacks instead."
        ),
    )
    parser.add_argument(
        "--online-ligand",
        action="store_true",
        help=(
            "Fetch PubChem and ChEMBL ligand metadata. When off, those IDs stay "
            "null."
        ),
    )
    parser.add_argument(
        "--strict-online",
        action="store_true",
        help="Raise on metadata lookup failures instead of using fallbacks/nulls.",
    )
    parser.add_argument(
        "--limit-per-split",
        type=int,
        default=None,
        help="Process only the first N records from each selected split.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=30,
        help="Per-request timeout for online metadata lookups.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0,
        help="Delay after each online request, useful for rate limiting.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Hide progress bars.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lmdb_paths = {}
    if not args.no_train:
        lmdb_paths["train"] = args.train_lmdb
    if not args.no_valid:
        lmdb_paths["valid"] = args.valid_lmdb
    if not lmdb_paths:
        raise ValueError("At least one split must be selected.")

    df = build_training_frame(
        lmdb_paths,
        combine_set_dir=args.combine_set_dir,
        online_pocket=not args.no_online_pocket,
        online_ligand=args.online_ligand,
        cache_path=args.cache_path,
        limit_per_split=args.limit_per_split,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        strict_online=args.strict_online,
        show_progress=not args.no_progress,
    )
    write_frame(df, args.output)
    print(f"Wrote {df.height} rows and {df.width} columns to {args.output}")
    print(df.select(["split", "lmdb_key", "pocket", "ligand", "pdb_id"]).head())


if __name__ == "__main__":
    main()
