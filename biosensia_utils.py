"""Utilities for preparing DrugCLIP retrieval inputs."""

from __future__ import annotations

import pickle
import re
import shutil
import urllib.request
from pathlib import Path
from typing import Any, Iterable

import lmdb
import numpy as np
import pandas as pd
from biopandas.mol2 import PandasMol2
from biopandas.pdb import PandasPdb
from rdkit import Chem


DEFAULT_COMBINE_SET_DIR = Path("external/DrugCLIP/data/pdb/combine_set")
DEFAULT_POCKET_RADIUS_ANGSTROM = 6.0
RCSB_PDB_DOWNLOAD_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

_PDB_ID_RE = re.compile(r"^[0-9][0-9A-Za-z]{3}$")

_EXCLUDED_HETATM_RESIDUES = {
    "HOH",
    "WAT",
    "DOD",
    "H2O",
    "NA",
    "K",
    "CL",
    "CA",
    "MG",
    "MN",
    "ZN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "SO4",
    "PO4",
    "GOL",
    "EDO",
}


def create_pocket_lmdb(
    accessions: Iterable[str],
    output_path: str | Path,
    *,
    combine_set_dir: str | Path = DEFAULT_COMBINE_SET_DIR,
    work_dir: str | Path | None = None,
    radius: float = DEFAULT_POCKET_RADIUS_ANGSTROM,
    prefer_data_pkl: bool = True,
    download_missing: bool = True,
    reference_ligand: str | None = None,
    include_pocket_hetatm: bool = False,
    overwrite: bool = True,
    map_size: int = 1 << 40,
) -> list[dict[str, Any]]:
    """Create a DrugCLIP pocket LMDB for one or more PDB IDs.

    Parameters
    ----------
    accessions:
        PDB IDs such as ``"2ie4"``. A single string is accepted. Protein names
        or UniProt accessions are intentionally not resolved automatically
        because a protein can map to many structures and binding sites.
    output_path:
        Destination LMDB file path. DrugCLIP retrieval expects this file to
        contain records with ``pocket``, ``pocket_atoms``, and
        ``pocket_coordinates``.
    combine_set_dir:
        Local DrugCLIP structure bundle directory. The function first checks
        ``{combine_set_dir}/{pdb_id}``.
    work_dir:
        Directory used for downloaded PDB files. Defaults to ``combine_set_dir``.
    radius:
        Distance cutoff, in Angstrom, used to define a pocket from a
        protein-ligand complex.
    prefer_data_pkl:
        Use local ``data.pkl`` first when available. This is the most faithful
        path for bundled DrugCLIP data.
    download_missing:
        Download a PDB file from RCSB when no usable local bundle exists.
    reference_ligand:
        Optional PDB ligand residue name, such as ``"OKA"``. This is only used
        when extracting a pocket from HETATM records in a downloaded PDB file.
    include_pocket_hetatm:
        Include HETATM rows when falling back to a local ``*_pocket.pdb`` file.
        By default only protein ATOM rows are used.
    overwrite:
        Replace an existing output LMDB file.
    map_size:
        LMDB map size.

    Returns
    -------
    list[dict[str, Any]]
        Build summaries for each accession.
    """

    if isinstance(accessions, str):
        pdb_ids = [_normalize_pdb_id(accessions)]
    else:
        pdb_ids = [_normalize_pdb_id(accession) for accession in accessions]
    if not pdb_ids:
        raise ValueError("accessions must contain at least one PDB ID")

    combine_set_dir = Path(combine_set_dir)
    work_dir = Path(work_dir) if work_dir is not None else combine_set_dir

    records: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for pocket_index, pdb_id in enumerate(pdb_ids):
        record, source = _build_local_pocket_record(
            pdb_id,
            combine_set_dir / pdb_id,
            radius=radius,
            prefer_data_pkl=prefer_data_pkl,
            include_pocket_hetatm=include_pocket_hetatm,
        )

        if record is None:
            if not download_missing:
                raise FileNotFoundError(
                    f"No usable local pocket data found for PDB ID {pdb_id!r}"
                )
            pdb_path = _ensure_downloaded_pdb(pdb_id, work_dir / pdb_id)
            record = _record_from_protein_pdb_and_hetatm_ligand(
                pdb_path,
                pocket_name=pdb_id,
                radius=radius,
                reference_ligand=reference_ligand,
            )
            source = str(pdb_path)

        record = _normalize_pocket_record(record, fallback_name=pdb_id)
        records.append(record)
        summaries.append(
            {
                "accession": pdb_id,
                "pocket": record["pocket"],
                "source": source,
                "pocket_index": pocket_index,
                "pocket_atoms": len(record["pocket_atoms"]),
                "output_path": str(output_path),
            }
        )

    _write_lmdb(records, output_path, overwrite=overwrite, map_size=map_size)
    return summaries


def read_lmdb_records(path: str | Path) -> list[dict[str, Any]]:
    """Read pickled DrugCLIP LMDB records.

    The same helper works for DrugCLIP pocket and molecule LMDB files because
    both store pickled dictionaries under numeric keys.
    """

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
            if _has_numeric_lmdb_keys(keys):
                keys = sorted(keys, key=lambda key: int(key.decode("ascii")))
            return [
                pickle.loads(value)
                for key in keys
                if (value := transaction.get(key)) is not None
            ]
    finally:
        env.close()


def _build_local_pocket_record(
    pdb_id: str,
    bundle_dir: Path,
    *,
    radius: float,
    prefer_data_pkl: bool,
    include_pocket_hetatm: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    if not bundle_dir.exists():
        return None, None

    data_pkl = bundle_dir / "data.pkl"
    if prefer_data_pkl and data_pkl.exists():
        return _record_from_data_pkl(data_pkl, pocket_name=pdb_id), str(data_pkl)

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
        record = _record_from_protein_and_ligand(
            protein_path,
            ligand_path,
            pocket_name=pdb_id,
            radius=radius,
        )
        return record, f"{protein_path} + {ligand_path}"

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
        return (
            _record_from_pocket_pdb(
                pocket_pdb,
                pocket_name=pdb_id,
                include_hetatm=include_pocket_hetatm,
            ),
            str(pocket_pdb),
        )

    return None, None


def _record_from_data_pkl(path: Path, *, pocket_name: str) -> dict[str, Any]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    return {
        "pocket": str(data.get("pocket") or pocket_name),
        "pocket_atoms": data["pocket_atoms"],
        "pocket_coordinates": data["pocket_coordinates"],
    }


def _record_from_protein_and_ligand(
    protein_path: Path,
    ligand_path: Path,
    *,
    pocket_name: str,
    radius: float,
) -> dict[str, Any]:
    protein = _read_protein_atom_table(protein_path)
    ligand_coordinates = _read_ligand_coordinates(ligand_path)
    return _select_pocket_record(
        protein,
        ligand_coordinates,
        pocket_name=pocket_name,
        radius=radius,
    )


def _record_from_protein_pdb_and_hetatm_ligand(
    pdb_path: Path,
    *,
    pocket_name: str,
    radius: float,
    reference_ligand: str | None,
) -> dict[str, Any]:
    protein = _read_protein_atom_table(pdb_path)
    ligand_coordinates = _read_ligand_coordinates_from_pdb_hetatm(
        pdb_path,
        reference_ligand=reference_ligand,
    )
    return _select_pocket_record(
        protein,
        ligand_coordinates,
        pocket_name=pocket_name,
        radius=radius,
    )


def _record_from_pocket_pdb(
    pocket_path: Path,
    *,
    pocket_name: str,
    include_hetatm: bool,
) -> dict[str, Any]:
    pdb = PandasPdb().read_pdb(str(pocket_path))
    frames = [pdb.df["ATOM"]]
    if include_hetatm:
        frames.append(pdb.df["HETATM"])
    table = _concat_nonempty_frames(frames)
    if table.empty:
        raise ValueError(f"No pocket atoms found in {pocket_path}")
    return {
        "pocket": pocket_name,
        "pocket_atoms": _clean_string_list(table["atom_name"]),
        "pocket_coordinates": _coords_from_table(table),
    }


def _read_protein_atom_table(path: Path) -> dict[str, np.ndarray | list[str]]:
    pdb = PandasPdb().read_pdb(str(path))
    atom_table = pdb.df["ATOM"]
    if atom_table.empty:
        raise ValueError(f"No protein ATOM records found in {path}")
    return {
        "atom_names": _clean_string_list(atom_table["atom_name"]),
        "coordinates": _coords_from_table(atom_table),
        "residue_ids": _residue_ids(atom_table),
    }


def _read_ligand_coordinates(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".mol2":
        mol2 = PandasMol2().read_mol2(str(path))
        return mol2.df[["x", "y", "z"]].to_numpy(dtype=np.float32)
    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        mol = next((item for item in supplier if item is not None), None)
        if mol is None or mol.GetNumConformers() == 0:
            raise ValueError(f"No conformer found in ligand SDF {path}")
        return np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float32)
    raise ValueError(f"Unsupported ligand file format: {path}")


def _read_ligand_coordinates_from_pdb_hetatm(
    pdb_path: Path,
    *,
    reference_ligand: str | None,
) -> np.ndarray:
    pdb = PandasPdb().read_pdb(str(pdb_path))
    hetatm = pdb.df["HETATM"].copy()
    if hetatm.empty:
        raise ValueError(f"No HETATM ligand records found in {pdb_path}")

    hetatm["residue_name"] = hetatm["residue_name"].astype(str).str.strip()
    if reference_ligand is not None:
        ligand_rows = hetatm[
            hetatm["residue_name"].str.upper() == reference_ligand.upper()
        ]
        if ligand_rows.empty:
            raise ValueError(
                f"Ligand {reference_ligand!r} was not found in {pdb_path}"
            )
        return _coords_from_table(ligand_rows)

    candidates = hetatm[
        ~hetatm["residue_name"].str.upper().isin(_EXCLUDED_HETATM_RESIDUES)
    ]
    if candidates.empty:
        raise ValueError(f"No non-solvent HETATM ligand candidate found in {pdb_path}")

    group_keys = ["chain_id", "residue_name", "residue_number", "insertion"]
    groups = candidates.groupby(group_keys, dropna=False)
    _, ligand_rows = max(groups, key=lambda item: len(item[1]))
    return _coords_from_table(ligand_rows)


def _select_pocket_record(
    protein: dict[str, np.ndarray | list[str]],
    ligand_coordinates: np.ndarray,
    *,
    pocket_name: str,
    radius: float,
) -> dict[str, Any]:
    protein_coordinates = np.asarray(protein["coordinates"], dtype=np.float32)
    residue_ids = np.asarray(protein["residue_ids"], dtype=str)
    atom_names = np.asarray(protein["atom_names"], dtype=object)

    if ligand_coordinates.size == 0:
        raise ValueError("Ligand coordinate array is empty")

    near_atom_mask = np.zeros(len(protein_coordinates), dtype=bool)
    radius_squared = radius * radius
    for start in range(0, len(ligand_coordinates), 128):
        chunk = ligand_coordinates[start : start + 128]
        distances_squared = np.sum(
            (protein_coordinates[:, None, :] - chunk[None, :, :]) ** 2,
            axis=2,
        )
        near_atom_mask |= np.any(distances_squared < radius_squared, axis=1)

    pocket_residue_ids = set(residue_ids[near_atom_mask])
    if not pocket_residue_ids:
        raise ValueError(f"No pocket residues found within {radius:g} A of ligand")

    pocket_mask = np.array([item in pocket_residue_ids for item in residue_ids])
    return {
        "pocket": pocket_name,
        "pocket_atoms": atom_names[pocket_mask].tolist(),
        "pocket_coordinates": protein_coordinates[pocket_mask],
    }


def _write_lmdb(
    records: list[dict[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool,
    map_size: int,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"LMDB already exists: {output_path}")
        output_path.unlink()

    env = lmdb.open(
        str(output_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=map_size,
    )
    try:
        with env.begin(write=True) as transaction:
            for index, record in enumerate(records):
                transaction.put(str(index).encode("ascii"), pickle.dumps(record))
    finally:
        env.close()


def _ensure_downloaded_pdb(pdb_id: str, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{pdb_id}.pdb"
    if target_path.exists():
        return target_path

    url = RCSB_PDB_DOWNLOAD_URL.format(pdb_id=pdb_id.upper())
    with urllib.request.urlopen(url, timeout=60) as response:
        with target_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    return target_path


def _normalize_pdb_id(accession: str) -> str:
    pdb_id = str(accession).strip().lower()
    if not _PDB_ID_RE.fullmatch(pdb_id):
        raise ValueError(
            f"{accession!r} is not a 4-character PDB ID. "
            "Choose a structure first, for example '2ie4' for PP2A."
        )
    return pdb_id


def _normalize_pocket_record(
    record: dict[str, Any],
    *,
    fallback_name: str,
) -> dict[str, Any]:
    atoms = [str(atom).strip() for atom in record["pocket_atoms"]]
    coordinates = np.asarray(record["pocket_coordinates"], dtype=np.float32)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError(
            f"pocket_coordinates must have shape (n_atoms, 3), got {coordinates.shape}"
        )
    if len(atoms) != len(coordinates):
        raise ValueError(
            "pocket_atoms and pocket_coordinates must have the same length "
            f"({len(atoms)} != {len(coordinates)})"
        )
    return {
        "pocket": str(record.get("pocket") or fallback_name),
        "pocket_atoms": atoms,
        "pocket_coordinates": coordinates,
    }


def _coords_from_table(table: Any) -> np.ndarray:
    return table[["x_coord", "y_coord", "z_coord"]].to_numpy(dtype=np.float32)


def _residue_ids(table: Any) -> list[str]:
    chain_id = table["chain_id"].astype(str).str.strip()
    residue_number = table["residue_number"].astype(str).str.strip()
    insertion = table["insertion"].astype(str).str.strip()
    return (chain_id + ":" + residue_number + ":" + insertion).tolist()


def _clean_string_list(values: Any) -> list[str]:
    return [str(value).strip() for value in values.tolist()]


def _concat_nonempty_frames(frames: list[Any]) -> Any:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return frames[0]
    if len(nonempty) == 1:
        return nonempty[0]
    return pd.concat(nonempty, ignore_index=True)


def _first_existing(
    candidates: list[Path],
    *,
    fallback_globs: list[tuple[Path, str]],
) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    for directory, pattern in fallback_globs:
        matches = sorted(directory.glob(pattern))
        if matches:
            return matches[0]
    return None


def _has_numeric_lmdb_keys(keys: list[bytes]) -> bool:
    if not keys:
        return False
    try:
        return all(key.decode("ascii").isdigit() for key in keys)
    except UnicodeDecodeError:
        return False


__all__ = ["create_pocket_lmdb", "read_lmdb_records"]
