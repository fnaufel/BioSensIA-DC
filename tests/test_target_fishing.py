import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from biosensia_target_fishing import (
    build_candidate_pockets_frame,
    build_candidate_pockets_lmdb,
    build_mol_lmdb_index,
    create_mol_lmdb,
)
from lmdb_helpers import read_lmdb_records, write_lmdb_records


REPO_ROOT = Path(__file__).resolve().parents[1]
COMBINE_SET_DIR = REPO_ROOT / "external/DrugCLIP/data/pdb/combine_set"
TWO_IE_FOUR_DIR = COMBINE_SET_DIR / "2ie4"
TWO_R_ONE_W_DIR = COMBINE_SET_DIR / "2r1w"


def write_test_lmdb(path: Path, records: list[dict]) -> None:
    write_lmdb_records(records, path, overwrite=True, map_size=1 << 20)


def test_build_candidate_pockets_lmdb_writes_encoder_schema(tmp_path):
    combine_set_dir = tmp_path / "combine_set"
    bundle_dir = combine_set_dir / "2ie4"
    bundle_dir.mkdir(parents=True)
    shutil.copy(TWO_IE_FOUR_DIR / "data.pkl", bundle_dir / "data.pkl")
    (combine_set_dir / "readme").mkdir()
    (combine_set_dir / "index").mkdir()

    output_path = tmp_path / "candidate_pockets.lmdb"
    summary = build_candidate_pockets_lmdb(
        output_path,
        combine_set_dir=combine_set_dir,
    )

    assert summary == {
        "output_path": str(output_path),
        "combine_set_dir": str(combine_set_dir),
        "candidate_dirs": 1,
        "pockets": 1,
        "skipped": 0,
        "skipped_entries": [],
    }

    records = read_lmdb_records(output_path)
    assert len(records) == 1
    assert set(records[0]) == {"pocket", "pocket_atoms", "pocket_coordinates"}
    assert records[0]["pocket"] == "2ie4"
    assert len(records[0]["pocket_atoms"]) == 546
    assert np.asarray(records[0]["pocket_coordinates"]).shape == (546, 3)
    assert np.asarray(records[0]["pocket_coordinates"]).dtype == np.float32


def test_build_candidate_pockets_frame_reads_pocket_atom_counts(tmp_path):
    combine_set_dir = tmp_path / "combine_set"
    bundle_dir = combine_set_dir / "2ie4"
    bundle_dir.mkdir(parents=True)
    shutil.copy(TWO_IE_FOUR_DIR / "data.pkl", bundle_dir / "data.pkl")
    output_path = tmp_path / "candidate_pockets.lmdb"
    build_candidate_pockets_lmdb(output_path, combine_set_dir=combine_set_dir)

    df = build_candidate_pockets_frame(output_path)

    assert df.columns == ["pocket", "pocket_atoms"]
    assert df.to_dicts() == [{"pocket": "2ie4", "pocket_atoms": 546}]


def test_build_candidate_pockets_lmdb_raises_for_invalid_bundle_by_default(tmp_path):
    combine_set_dir = tmp_path / "combine_set"
    (combine_set_dir / "9zzz").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="9zzz"):
        build_candidate_pockets_lmdb(
            tmp_path / "candidate_pockets.lmdb",
            combine_set_dir=combine_set_dir,
        )


def test_build_candidate_pockets_lmdb_falls_back_from_empty_data_pkl_to_pocket_pdb(
    tmp_path,
):
    combine_set_dir = tmp_path / "combine_set"
    bundle_dir = combine_set_dir / "2r1w"
    bundle_dir.mkdir(parents=True)
    with (bundle_dir / "data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "pocket": "2r1w",
                "pocket_atoms": [],
                "pocket_coordinates": [],
            },
            handle,
        )
    shutil.copy(TWO_R_ONE_W_DIR / "2r1w_pocket.pdb", bundle_dir / "2r1w_pocket.pdb")

    output_path = tmp_path / "candidate_pockets.lmdb"
    summary = build_candidate_pockets_lmdb(
        output_path,
        combine_set_dir=combine_set_dir,
    )

    assert summary["candidate_dirs"] == 1
    assert summary["pockets"] == 1
    assert summary["skipped"] == 0

    record = read_lmdb_records(output_path)[0]
    assert set(record) == {"pocket", "pocket_atoms", "pocket_coordinates"}
    assert record["pocket"] == "2r1w"
    assert len(record["pocket_atoms"]) == 216
    assert np.asarray(record["pocket_coordinates"]).shape == (216, 3)


def test_build_candidate_pockets_lmdb_can_skip_invalid_bundles(tmp_path):
    combine_set_dir = tmp_path / "combine_set"
    bundle_dir = combine_set_dir / "2ie4"
    bundle_dir.mkdir(parents=True)
    shutil.copy(TWO_IE_FOUR_DIR / "data.pkl", bundle_dir / "data.pkl")
    (combine_set_dir / "9zzz").mkdir()

    output_path = tmp_path / "candidate_pockets.lmdb"
    summary = build_candidate_pockets_lmdb(
        output_path,
        combine_set_dir=combine_set_dir,
        skip_invalid=True,
    )

    assert summary["candidate_dirs"] == 2
    assert summary["pockets"] == 1
    assert summary["skipped"] == 1
    assert summary["skipped_entries"][0]["accession"] == "9zzz"
    assert len(read_lmdb_records(output_path)) == 1


def test_create_mol_lmdb_copies_matching_records_from_source_lmdb(tmp_path):
    source_path = tmp_path / "source_mols.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "atoms": ["C", "C", "O"],
                "coordinates": [np.zeros((3, 3), dtype=np.float64)],
                "smi": "CCO",
                "IDs": "ethanol-record",
                "subset": "test-subset",
            }
        ],
    )
    output_path = tmp_path / "mols.lmdb"

    summaries = create_mol_lmdb(
        "CCO",
        output_path,
        source_lmdb_path=source_path,
        download_missing=False,
        show_progress=False,
    )

    assert summaries == [
        {
            "molecule": "CCO",
            "smiles": "CCO",
            "source": f"{source_path}:0",
            "molecule_index": 0,
            "molecule_atoms": 3,
            "conformers": 1,
            "output_path": str(output_path),
        }
    ]
    record = read_lmdb_records(output_path)[0]
    assert record["atoms"] == ["C", "C", "O"]
    assert record["smi"] == "CCO"
    assert record["IDs"] == "ethanol-record"
    assert np.asarray(record["coordinates"][0]).dtype == np.float32


def test_create_mol_lmdb_matches_source_by_canonical_smiles(tmp_path):
    source_path = tmp_path / "source_mols.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "atoms": ["C", "C", "O"],
                "coordinates": [np.ones((3, 3), dtype=np.float32)],
                "smi": "CCO",
            }
        ],
    )

    output_path = tmp_path / "mols.lmdb"
    summaries = create_mol_lmdb(
        ["OCC"],
        output_path,
        source_lmdb_path=source_path,
        download_missing=False,
        show_progress=False,
    )

    assert summaries[0]["source"] == f"{source_path}:0"
    assert read_lmdb_records(output_path)[0]["smi"] == "CCO"


def test_create_mol_lmdb_matches_source_by_drugclip_id(tmp_path):
    source_path = tmp_path / "source_mols.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "atoms": ["N"],
                "coordinates": [np.array([[1.0, 2.0, 3.0]], dtype=np.float32)],
                "smi": "N",
                "IDs": "F0007-0960",
            }
        ],
    )

    output_path = tmp_path / "mols.lmdb"
    summaries = create_mol_lmdb(
        ["f0007-0960"],
        output_path,
        source_lmdb_path=source_path,
        download_missing=False,
        show_progress=False,
    )

    assert summaries[0]["smiles"] == "N"
    assert summaries[0]["source"] == f"{source_path}:0"
    assert read_lmdb_records(output_path)[0]["IDs"] == "F0007-0960"


def test_build_mol_lmdb_index_supports_fast_smiles_and_id_lookup(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "source_mols.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "atoms": ["C", "C", "O"],
                "coordinates": [np.ones((3, 3), dtype=np.float32)],
                "smi": "CCO",
                "IDs": "ethanol-record",
            },
            {
                "atoms": ["N"],
                "coordinates": [np.zeros((1, 3), dtype=np.float32)],
                "smi": "N",
                "IDs": "nitrogen-record",
            },
        ],
    )
    index_path = tmp_path / "mols_index.lmdb"

    summary = build_mol_lmdb_index(
        index_path,
        source_lmdb_path=source_path,
        show_progress=False,
    )

    assert summary == {
        "index_path": str(index_path),
        "source_lmdb_path": str(source_path),
        "source_entries": 2,
        "indexed_records": 2,
        "lookup_values": 4,
    }

    def fail_sequential_scan(*args, **kwargs):
        raise AssertionError("sequential scan should not be used")

    monkeypatch.setattr(
        "biosensia_target_fishing._find_molecule_records_in_lmdb",
        fail_sequential_scan,
    )

    smiles_output_path = tmp_path / "mols_from_smiles.lmdb"
    smiles_summaries = create_mol_lmdb(
        ["OCC"],
        smiles_output_path,
        source_lmdb_path=source_path,
        mol_index_path=index_path,
        download_missing=False,
        show_progress=False,
    )
    assert smiles_summaries[0]["source"] == f"{source_path}:0"
    assert read_lmdb_records(smiles_output_path)[0]["smi"] == "CCO"

    id_output_path = tmp_path / "mols_from_id.lmdb"
    id_summaries = create_mol_lmdb(
        ["nitrogen-record"],
        id_output_path,
        source_lmdb_path=source_path,
        mol_index_path=index_path,
        download_missing=False,
        show_progress=False,
    )
    assert id_summaries[0]["source"] == f"{source_path}:1"
    assert read_lmdb_records(id_output_path)[0]["smi"] == "N"


def test_create_mol_lmdb_reuses_index_for_different_source_path_same_entry_count(
    monkeypatch,
    tmp_path,
):
    records = [
        {
            "atoms": ["C", "C", "O"],
            "coordinates": [np.ones((3, 3), dtype=np.float32)],
            "smi": "CCO",
            "IDs": "ethanol-record",
        }
    ]
    indexed_source_path = tmp_path / "sagres_mols.lmdb"
    local_source_path = tmp_path / "local_mols.lmdb"
    write_test_lmdb(indexed_source_path, records)
    write_test_lmdb(local_source_path, records)
    index_path = tmp_path / "mols_index.lmdb"
    build_mol_lmdb_index(
        index_path,
        source_lmdb_path=indexed_source_path,
        show_progress=False,
    )

    def fail_sequential_scan(*args, **kwargs):
        raise AssertionError("sequential scan should not be used")

    monkeypatch.setattr(
        "biosensia_target_fishing._find_molecule_records_in_lmdb",
        fail_sequential_scan,
    )

    output_path = tmp_path / "mols_from_reused_index.lmdb"
    summaries = create_mol_lmdb(
        ["OCC"],
        output_path,
        source_lmdb_path=local_source_path,
        mol_index_path=index_path,
        download_missing=False,
        show_progress=False,
    )

    assert summaries[0]["source"] == f"{local_source_path}:0"
    assert read_lmdb_records(output_path)[0]["smi"] == "CCO"


def test_create_mol_lmdb_downloads_after_usable_index_miss_without_full_scan(
    monkeypatch,
    tmp_path,
):
    source_path = tmp_path / "source_mols.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "atoms": ["C"],
                "coordinates": [np.zeros((1, 3), dtype=np.float32)],
                "smi": "C",
            }
        ],
    )
    index_path = tmp_path / "mols_index.lmdb"
    build_mol_lmdb_index(
        index_path,
        source_lmdb_path=source_path,
        show_progress=False,
    )

    def fail_sequential_scan(*args, **kwargs):
        raise AssertionError("sequential scan should not be used after index miss")

    monkeypatch.setattr(
        "biosensia_target_fishing._find_molecule_records_in_lmdb",
        fail_sequential_scan,
    )

    calls = {}

    def fake_download_molecule_record(
        molecule: str,
        *,
        work_dir: Path,
        timeout_seconds: float,
        random_seed: int,
        show_progress: bool,
    ) -> tuple[dict, str]:
        calls["download"] = (
            molecule,
            work_dir,
            timeout_seconds,
            random_seed,
            show_progress,
        )
        return (
            {
                "atoms": ["O"],
                "coordinates": [np.zeros((1, 3), dtype=np.float32)],
                "smi": "O",
            },
            str(work_dir / "generated.sdf"),
        )

    monkeypatch.setattr(
        "biosensia_target_fishing._download_molecule_record",
        fake_download_molecule_record,
    )

    output_path = tmp_path / "query_mols.lmdb"
    summaries = create_mol_lmdb(
        ["O"],
        output_path,
        source_lmdb_path=source_path,
        mol_index_path=index_path,
        work_dir=tmp_path / "downloads",
        timeout_seconds=10,
        random_seed=42,
        show_progress=False,
    )

    assert calls["download"] == (
        "O",
        tmp_path / "downloads/O",
        10,
        42,
        False,
    )
    assert summaries[0]["source"] == str(tmp_path / "downloads/O/generated.sdf")
    assert read_lmdb_records(output_path)[0]["smi"] == "O"


def test_build_mol_lmdb_index_respects_overwrite_false(tmp_path):
    source_path = tmp_path / "source_mols.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "atoms": ["C"],
                "coordinates": [np.zeros((1, 3), dtype=np.float32)],
                "smi": "C",
            }
        ],
    )
    index_path = tmp_path / "mols_index.lmdb"
    build_mol_lmdb_index(
        index_path,
        source_lmdb_path=source_path,
        show_progress=False,
    )

    with pytest.raises(FileExistsError):
        build_mol_lmdb_index(
            index_path,
            source_lmdb_path=source_path,
            overwrite=False,
            show_progress=False,
        )


def test_create_mol_lmdb_missing_local_data_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not-found"):
        create_mol_lmdb(
            ["not-found"],
            tmp_path / "mols.lmdb",
            source_lmdb_path=tmp_path / "missing_source.lmdb",
            download_missing=False,
            show_progress=False,
        )


def test_create_mol_lmdb_download_path_is_offline_mockable(monkeypatch, tmp_path):
    calls = {}

    def fake_download_molecule_record(
        molecule: str,
        *,
        work_dir: Path,
        timeout_seconds: float,
        random_seed: int,
        show_progress: bool,
    ) -> tuple[dict, str]:
        calls["download"] = (
            molecule,
            work_dir,
            timeout_seconds,
            random_seed,
            show_progress,
        )
        return (
            {
                "atoms": ["C", "O"],
                "coordinates": [np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])],
                "smi": "CO",
                "IDs": "CID:887",
            },
            str(work_dir / "pubchem_cid_3d.sdf"),
        )

    monkeypatch.setattr(
        "biosensia_target_fishing._download_molecule_record",
        fake_download_molecule_record,
    )

    output_path = tmp_path / "downloaded_mols.lmdb"
    summaries = create_mol_lmdb(
        ["cid:887"],
        output_path,
        source_lmdb_path=tmp_path / "missing_source.lmdb",
        work_dir=tmp_path / "downloads",
        timeout_seconds=10,
        random_seed=42,
        show_progress=False,
    )

    assert calls["download"] == (
        "cid:887",
        tmp_path / "downloads/cid_887",
        10,
        42,
        False,
    )
    assert summaries[0]["source"] == str(
        tmp_path / "downloads/cid_887/pubchem_cid_3d.sdf"
    )
    assert summaries[0]["molecule_atoms"] == 2
    record = read_lmdb_records(output_path)[0]
    assert record["smi"] == "CO"
    assert record["IDs"] == "CID:887"
    assert np.asarray(record["coordinates"][0]).shape == (2, 3)


def test_create_mol_lmdb_respects_overwrite_false(tmp_path):
    source_path = tmp_path / "source_mols.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "atoms": ["C"],
                "coordinates": [np.zeros((1, 3), dtype=np.float32)],
                "smi": "C",
            }
        ],
    )
    output_path = tmp_path / "mols.lmdb"
    create_mol_lmdb(
        ["C"],
        output_path,
        source_lmdb_path=source_path,
        download_missing=False,
        show_progress=False,
    )

    with pytest.raises(FileExistsError):
        create_mol_lmdb(
            ["C"],
            output_path,
            source_lmdb_path=source_path,
            download_missing=False,
            overwrite=False,
            show_progress=False,
        )
