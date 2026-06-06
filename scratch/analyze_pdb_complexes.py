"""Build a pocket-ligand identity table from DrugCLIP PDB complexes.

The primary identifiers follow the decisions made for pair analysis:

* ``pocket`` is a sorted UniProt accession set from PDBe SIFTS mappings.
  If no mapping is available, the script falls back to a protein sequence hash.
* ``ligand`` is the RDKit standard InChIKey parsed from the local ligand SDF
  or MOL2 file. Canonical SMILES and external IDs are stored as annotations.

The script can be imported and used directly:

    import scratch.analyze_pdb_complexes as apc
    df = apc.build_complexes_frame()

or run from the command line:

    uv run python scratch/analyze_pdb_complexes.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import polars as pl
from rdkit import Chem, RDLogger
from tqdm.auto import tqdm


DEFAULT_COMBINE_SET_DIR = Path("external/DrugCLIP/data/pdb/combine_set")
DEFAULT_OUTPUT_PATH = Path("scratch/pdb_complex_pairs.parquet")
DEFAULT_CACHE_PATH = Path("scratch/pdb_complexes_metadata_cache.json")

PD_BE_SIFTS_UNIPROT_URL = (
    "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id}"
)
PUBCHEM_CIDS_BY_INCHIKEY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/"
    "{inchikey}/cids/JSON"
)
CHEMBL_MOLECULE_BY_INCHIKEY_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
USER_AGENT = "BioSensIA-DC/0.1 (PDB complex pair analysis)"

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

ROW_COLUMNS = [
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


class MetadataFetchError(RuntimeError):
    """Raised when an online metadata lookup fails."""


def build_complexes_frame(
    combine_set_dir: str | Path = DEFAULT_COMBINE_SET_DIR,
    *,
    online: bool = True,
    cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    limit: int | None = None,
    timeout_seconds: float = 30,
    sleep_seconds: float = 0,
    strict_online: bool = False,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Return a Polars frame of positive pocket-ligand pairs.

    Parameters
    ----------
    combine_set_dir:
        Directory containing one subdirectory per PDB complex.
    online:
        Fetch UniProt/PubChem/ChEMBL annotations from PDBe, PubChem, and ChEMBL.
        If false, only cached online metadata is used; missing pocket IDs fall
        back to sequence hashes, and missing ligand database IDs remain null.
    cache_path:
        JSON cache for online metadata. Set to ``None`` to disable caching.
    limit:
        Optional number of complexes to process, useful for smoke tests.
    timeout_seconds:
        Per-request timeout for online metadata lookups.
    sleep_seconds:
        Optional delay after each online request.
    strict_online:
        Raise on online lookup failures instead of falling back to nulls/hashes.
    show_progress:
        Display a progress bar.
    """

    combine_set_dir = Path(combine_set_dir)
    if not combine_set_dir.exists():
        raise FileNotFoundError(f"combine_set directory not found: {combine_set_dir}")

    complex_dirs = sorted(path for path in combine_set_dir.iterdir() if path.is_dir())
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be greater than or equal to 0")
        complex_dirs = complex_dirs[:limit]

    cache = _load_cache(cache_path)
    rows = []
    iterator: Iterable[Path] = complex_dirs
    if show_progress:
        iterator = tqdm(complex_dirs, desc="Analyzing PDB complexes", unit="complex")

    RDLogger.DisableLog("rdApp.*")
    for complex_dir in iterator:
        rows.append(
            analyze_complex(
                complex_dir,
                online=online,
                cache=cache,
                timeout_seconds=timeout_seconds,
                sleep_seconds=sleep_seconds,
                strict_online=strict_online,
            )
        )

    _save_cache(cache_path, cache)
    return pl.DataFrame(rows, schema=ROW_COLUMNS, orient="row")


def analyze_complex(
    complex_dir: str | Path,
    *,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
    strict_online: bool,
) -> dict[str, Any]:
    """Analyze one complex directory and return one output row."""

    complex_dir = Path(complex_dir)
    pdb_id = complex_dir.name.lower()
    files = _complex_files(complex_dir)

    pocket_info = extract_pocket_identity(
        pdb_id,
        protein_pdb=files["protein_pdb"],
        pocket_pdb=files["pocket_pdb"],
        online=online,
        cache=cache,
        timeout_seconds=timeout_seconds,
        sleep_seconds=sleep_seconds,
        strict_online=strict_online,
    )
    ligand_info = extract_ligand_identity(
        pdb_id,
        ligand_sdf=files["ligand_sdf"],
        ligand_mol2=files["ligand_mol2"],
        online=online,
        cache=cache,
        timeout_seconds=timeout_seconds,
        sleep_seconds=sleep_seconds,
        strict_online=strict_online,
    )

    return {
        "pocket": pocket_info["pocket"],
        "ligand": ligand_info["ligand"],
        "pdb_id": pdb_id,
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
        "protein_file": _string_path(files["protein_pdb"]),
        "pocket_file": _string_path(files["pocket_pdb"]),
        "ligand_sdf_file": _string_path(files["ligand_sdf"]),
        "ligand_mol2_file": _string_path(files["ligand_mol2"]),
    }


def extract_pocket_identity(
    pdb_id: str,
    *,
    protein_pdb: Path | None,
    pocket_pdb: Path | None,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
    strict_online: bool,
) -> dict[str, Any]:
    """Return the chosen pocket identifier and its annotations."""

    chain_ids = sorted(_parse_chain_ids(pocket_pdb) or _parse_chain_ids(protein_pdb))
    accessions: list[str] = []
    status_parts = []

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
            return {
                "pocket": "+".join(accessions),
                "pocket_id_source": "uniprot_sifts",
                "uniprot_accessions": accessions,
                "chain_ids": chain_ids,
                "parse_status": "ok:sifts",
            }
        status_parts.append("sifts_no_chain_match")
    else:
        status_parts.append("sifts_missing")

    fallback = _protein_sequence_hash(protein_pdb, chain_ids)
    if fallback is not None:
        status_parts.append("fallback:protein_sequence_hash")
        return {
            "pocket": fallback,
            "pocket_id_source": "seqsha1",
            "uniprot_accessions": [],
            "chain_ids": chain_ids,
            "parse_status": ";".join(status_parts),
        }

    status_parts.append("fallback:pdb_id")
    return {
        "pocket": f"pdb:{pdb_id}",
        "pocket_id_source": "pdb_id",
        "uniprot_accessions": [],
        "chain_ids": chain_ids,
        "parse_status": ";".join(status_parts),
    }


def extract_ligand_identity(
    pdb_id: str,
    *,
    ligand_sdf: Path | None,
    ligand_mol2: Path | None,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
    strict_online: bool,
) -> dict[str, Any]:
    """Return the chosen ligand identifier and its annotations."""

    ccd_ids = _parse_mol2_ccd_ids(ligand_mol2)
    ccd_id = "+".join(ccd_ids) if ccd_ids else None
    mol, parse_status = _load_ligand_mol(ligand_sdf, ligand_mol2)
    smiles = None
    inchikey = None

    if mol is not None:
        smiles = _mol_to_smiles(mol)
        inchikey = _mol_to_inchikey(mol)

    ligand, ligand_id_source = _choose_ligand_id(
        pdb_id=pdb_id,
        inchikey=inchikey,
        smiles=smiles,
        ccd_id=ccd_id,
    )
    ligand_type = _classify_ligand_type(ccd_ids, mol)

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

    return {
        "ligand": ligand,
        "inchikey": inchikey,
        "smiles": smiles,
        "ccd_id": ccd_id,
        "ccd_ids": ccd_ids,
        "ligand_type": ligand_type,
        "ligand_id_source": ligand_id_source,
        "parse_status": parse_status,
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


def _complex_files(complex_dir: Path) -> dict[str, Path | None]:
    pdb_id = complex_dir.name.lower()
    return {
        "protein_pdb": _first_existing(
            complex_dir / f"{pdb_id}_protein.pdb",
            complex_dir / f"{pdb_id.upper()}_protein.pdb",
        ),
        "pocket_pdb": _first_existing(
            complex_dir / f"{pdb_id}_pocket.pdb",
            complex_dir / f"{pdb_id}_pocket6A.pdb",
            complex_dir / f"{pdb_id.upper()}_pocket.pdb",
            complex_dir / f"{pdb_id.upper()}_pocket6A.pdb",
        ),
        "ligand_sdf": _first_existing(
            complex_dir / f"{pdb_id}_ligand.sdf",
            complex_dir / f"{pdb_id.upper()}_ligand.sdf",
        ),
        "ligand_mol2": _first_existing(
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
    pdb_cache = cache.setdefault("pdb_uniprot_mappings", {})
    if pdb_id in pdb_cache:
        return pdb_cache[pdb_id] or []
    if not online:
        return []

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


def _load_ligand_mol(
    ligand_sdf: Path | None,
    ligand_mol2: Path | None,
) -> tuple[Chem.Mol | None, str]:
    if ligand_sdf is not None:
        mol = _load_sdf_mol(ligand_sdf, sanitize=True)
        if mol is not None:
            return mol, "ok:sdf"
        mol = _load_sdf_mol(ligand_sdf, sanitize=False)
        if mol is not None:
            return mol, "ok:sdf_unsanitized"

    if ligand_mol2 is not None:
        mol = _load_mol2_mol(ligand_mol2, sanitize=True)
        if mol is not None:
            return mol, "ok:mol2"
        mol = _load_mol2_mol(ligand_mol2, sanitize=False)
        if mol is not None:
            return mol, "ok:mol2_unsanitized"

    return None, "error:no_rdkit_molecule"


def _load_sdf_mol(path: Path, *, sanitize: bool) -> Chem.Mol | None:
    try:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=sanitize)
        return next((mol for mol in supplier if mol is not None), None)
    except Exception:
        return None


def _load_mol2_mol(path: Path, *, sanitize: bool) -> Chem.Mol | None:
    try:
        return Chem.MolFromMol2File(
            str(path),
            sanitize=sanitize,
            removeHs=False,
            cleanupSubstructures=True,
        )
    except Exception:
        return None


def _mol_to_smiles(mol: Chem.Mol) -> str | None:
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _mol_to_inchikey(mol: Chem.Mol) -> str | None:
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
) -> tuple[str, str]:
    if inchikey:
        return inchikey, "inchikey"
    if smiles:
        digest = hashlib.sha1(smiles.encode("utf-8")).hexdigest()
        return f"smilesha1:{digest}", "canonical_smiles_hash"
    if ccd_id:
        return f"ccd:{ccd_id}", "ccd_id"
    return f"pdb_ligand:{pdb_id}", "pdb_id"


def _get_pubchem_cid(
    inchikey: str,
    *,
    online: bool,
    cache: dict[str, Any],
    timeout_seconds: float,
    sleep_seconds: float,
) -> int | None:
    pubchem_cache = cache.setdefault("pubchem_cid_by_inchikey", {})
    if inchikey in pubchem_cache:
        return pubchem_cache[inchikey]
    if not online:
        return None

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
    chembl_cache = cache.setdefault("chembl_id_by_inchikey", {})
    if inchikey in chembl_cache:
        return chembl_cache[inchikey]
    if not online:
        return None

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


def _string_path(path: Path | None) -> str | None:
    return str(path) if path is not None else None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Polars-compatible pocket-ligand identity table from "
            "external/DrugCLIP/data/pdb/combine_set."
        )
    )
    parser.add_argument(
        "--combine-set-dir",
        type=Path,
        default=DEFAULT_COMBINE_SET_DIR,
        help=f"Input complex directory tree. Default: {DEFAULT_COMBINE_SET_DIR}",
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
        "--no-online",
        action="store_true",
        help=(
            "Do not fetch missing UniProt/PubChem/ChEMBL metadata. Cached values "
            "are still used."
        ),
    )
    parser.add_argument(
        "--strict-online",
        action="store_true",
        help="Raise on metadata lookup failures instead of using fallbacks/nulls.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N complexes.",
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
        help="Hide the progress bar.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    df = build_complexes_frame(
        args.combine_set_dir,
        online=not args.no_online,
        cache_path=args.cache_path,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=args.sleep_seconds,
        strict_online=args.strict_online,
        show_progress=not args.no_progress,
    )
    write_frame(df, args.output)
    print(f"Wrote {df.height} rows and {df.width} columns to {args.output}")
    print(df.select(["pocket", "ligand", "pdb_id"]).head())


if __name__ == "__main__":
    main()
