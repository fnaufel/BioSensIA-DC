import pickle
from pathlib import Path

import lmdb
import numpy as np

from compare_pocket_geometries import compare


def _write_lmdb(path: Path, records: list[dict]) -> None:
    env = lmdb.open(str(path), subdir=False, map_size=1 << 20)
    with env.begin(write=True) as transaction:
        for index, record in enumerate(records):
            transaction.put(str(index).encode(), pickle.dumps(record))
    env.close()


def _write_reference(root: Path, pdb_id: str, atoms: list[str], coords: np.ndarray) -> None:
    directory = root / pdb_id
    directory.mkdir(parents=True)
    with (directory / "data.pkl").open("wb") as handle:
        pickle.dump({"pocket_atoms": atoms, "pocket_coordinates": coords}, handle)


def test_reports_heavy_atom_match_and_distinct_duplicate_geometries(tmp_path: Path) -> None:
    combine_set = tmp_path / "combine_set"
    reference_coords = np.array([[0, 0, 0], [9, 9, 9], [1, 0, 0]], dtype=float)
    _write_reference(combine_set, "1abc", ["N", "H", "CA"], reference_coords)
    lmdb_path = tmp_path / "train.lmdb"
    _write_lmdb(lmdb_path, [
        {"pocket": "1ABC", "pocket_atoms": ["N", "CA"],
         "pocket_coordinates": reference_coords[[0, 2]]},
        {"pocket": "1abc", "pocket_atoms": ["N", "CA"],
         "pocket_coordinates": np.array([[0, 0, 0], [2, 0, 0]], dtype=float)},
    ])

    rows, duplicates = compare([lmdb_path], combine_set, sidecar_path=None)

    assert rows["status"].to_list() == ["heavy_atom_match", "mismatch"]
    assert duplicates.row(0, named=True)["occurrence_count"] == 2
    assert duplicates.row(0, named=True)["unique_lmdb_geometry_count"] == 2
    assert duplicates.row(0, named=True)["has_different_lmdb_geometries"] is True


def test_reports_full_match(tmp_path: Path) -> None:
    combine_set = tmp_path / "combine_set"
    coords = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    _write_reference(combine_set, "2xyz", ["C"], coords)
    lmdb_path = tmp_path / "valid.lmdb"
    _write_lmdb(lmdb_path, [
        {"pocket": "2xyz", "pocket_atoms": ["C"], "pocket_coordinates": coords}
    ])

    rows, duplicates = compare([lmdb_path], combine_set, sidecar_path=None)

    assert rows.row(0, named=True)["status"] == "full_match"
    assert duplicates.is_empty()
