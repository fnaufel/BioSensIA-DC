import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from biosensia_retrieval import read_lmdb_records
from biosensia_target_fishing import (
    build_candidate_pockets_frame,
    build_candidate_pockets_lmdb,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMBINE_SET_DIR = REPO_ROOT / "external/DrugCLIP/data/pdb/combine_set"
TWO_IE_FOUR_DIR = COMBINE_SET_DIR / "2ie4"
TWO_R_ONE_W_DIR = COMBINE_SET_DIR / "2r1w"


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
