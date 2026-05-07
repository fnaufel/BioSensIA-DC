import pickle
import shutil
from pathlib import Path

import lmdb
import numpy as np
import pytest

from biosensia_utils import create_pocket_lmdb, read_lmdb_records


REPO_ROOT = Path(__file__).resolve().parents[1]
COMBINE_SET_DIR = REPO_ROOT / "external/DrugCLIP/data/pdb/combine_set"
TWO_IE_FOUR_DIR = COMBINE_SET_DIR / "2ie4"


def test_read_lmdb_records_accepts_molecule_style_lmdb(tmp_path):
    output_path = tmp_path / "mols.lmdb"
    env = lmdb.open(
        str(output_path),
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        map_size=1 << 20,
    )
    try:
        with env.begin(write=True) as transaction:
            transaction.put(b"10", pickle.dumps({"smi": "CCC"}))
            transaction.put(b"2", pickle.dumps({"smi": "CC"}))
            transaction.put(b"0", pickle.dumps({"smi": "C"}))
    finally:
        env.close()

    records = read_lmdb_records(output_path)

    assert [record["smi"] for record in records] == ["C", "CC", "CCC"]


def test_create_pocket_lmdb_accepts_single_pdb_id_string(tmp_path):
    output_path = tmp_path / "pocket.lmdb"

    summaries = create_pocket_lmdb(
        "2ie4",
        output_path,
        combine_set_dir=COMBINE_SET_DIR,
        download_missing=False,
    )

    assert summaries == [
        {
            "accession": "2ie4",
            "pocket": "2ie4",
            "source": str(TWO_IE_FOUR_DIR / "data.pkl"),
            "pocket_index": 0,
            "pocket_atoms": 546,
            "output_path": str(output_path),
        }
    ]
    records = read_lmdb_records(output_path)
    assert len(records) == 1
    assert set(records[0]) == {"pocket", "pocket_atoms", "pocket_coordinates"}
    assert records[0]["pocket"] == "2ie4"
    assert len(records[0]["pocket_atoms"]) == 546
    assert np.asarray(records[0]["pocket_coordinates"]).shape == (546, 3)


def test_create_pocket_lmdb_accepts_pdb_id_list(tmp_path):
    output_path = tmp_path / "pockets.lmdb"

    summaries = create_pocket_lmdb(
        ["2IE4"],
        output_path,
        combine_set_dir=COMBINE_SET_DIR,
        download_missing=False,
    )

    assert summaries[0]["accession"] == "2ie4"
    assert len(read_lmdb_records(output_path)) == 1


@pytest.mark.parametrize("accessions", [["PP2A"], ["P67775"], []])
def test_create_pocket_lmdb_rejects_non_pdb_inputs(accessions, tmp_path):
    with pytest.raises(ValueError):
        create_pocket_lmdb(
            accessions,
            tmp_path / "pocket.lmdb",
            combine_set_dir=COMBINE_SET_DIR,
            download_missing=False,
        )


def test_create_pocket_lmdb_uses_protein_and_ligand_fallback(tmp_path):
    output_path = tmp_path / "generated_pocket.lmdb"

    summaries = create_pocket_lmdb(
        ["2ie4"],
        output_path,
        combine_set_dir=COMBINE_SET_DIR,
        download_missing=False,
        prefer_data_pkl=False,
    )

    assert "2ie4_protein.pdb" in summaries[0]["source"]
    assert "2ie4_ligand.mol2" in summaries[0]["source"]
    assert summaries[0]["pocket_atoms"] == 546

    record = read_lmdb_records(output_path)[0]
    assert record["pocket"] == "2ie4"
    assert len(record["pocket_atoms"]) == 546
    assert np.asarray(record["pocket_coordinates"]).shape == (546, 3)


def test_create_pocket_lmdb_uses_pocket_pdb_fallback(tmp_path):
    combine_set_dir = tmp_path / "combine_set"
    bundle_dir = combine_set_dir / "2ie4"
    bundle_dir.mkdir(parents=True)
    shutil.copy(TWO_IE_FOUR_DIR / "2ie4_pocket.pdb", bundle_dir / "2ie4_pocket.pdb")

    output_path = tmp_path / "pocket_from_pdb.lmdb"
    summaries = create_pocket_lmdb(
        ["2ie4"],
        output_path,
        combine_set_dir=combine_set_dir,
        download_missing=False,
    )

    assert summaries[0]["source"] == str(bundle_dir / "2ie4_pocket.pdb")
    assert summaries[0]["pocket_atoms"] == 436
    assert len(read_lmdb_records(output_path)[0]["pocket_atoms"]) == 436

    output_path_with_hetatm = tmp_path / "pocket_from_pdb_with_hetatm.lmdb"
    summaries_with_hetatm = create_pocket_lmdb(
        ["2ie4"],
        output_path_with_hetatm,
        combine_set_dir=combine_set_dir,
        download_missing=False,
        include_pocket_hetatm=True,
    )

    assert summaries_with_hetatm[0]["pocket_atoms"] == 442
    assert len(read_lmdb_records(output_path_with_hetatm)[0]["pocket_atoms"]) == 442


def test_create_pocket_lmdb_missing_local_data_without_download_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        create_pocket_lmdb(
            ["9zzz"],
            tmp_path / "missing.lmdb",
            combine_set_dir=tmp_path / "empty_combine_set",
            download_missing=False,
        )


def test_create_pocket_lmdb_download_path_is_offline_mockable(monkeypatch, tmp_path):
    calls = {}

    def fake_download(pdb_id: str, target_dir: Path) -> Path:
        calls["download"] = (pdb_id, target_dir)
        return target_dir / f"{pdb_id}.pdb"

    def fake_record_from_downloaded_pdb(
        pdb_path: Path,
        *,
        pocket_name: str,
        radius: float,
        reference_ligand: str | None,
    ) -> dict:
        calls["record"] = (pdb_path, pocket_name, radius, reference_ligand)
        return {
            "pocket": pocket_name,
            "pocket_atoms": ["N"],
            "pocket_coordinates": np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        }

    monkeypatch.setattr("biosensia_utils._ensure_downloaded_pdb", fake_download)
    monkeypatch.setattr(
        "biosensia_utils._record_from_protein_pdb_and_hetatm_ligand",
        fake_record_from_downloaded_pdb,
    )

    output_path = tmp_path / "downloaded.lmdb"
    summaries = create_pocket_lmdb(
        ["9zzz"],
        output_path,
        combine_set_dir=tmp_path / "empty_combine_set",
        work_dir=tmp_path / "downloads",
        reference_ligand="OKA",
    )

    assert calls["download"] == ("9zzz", tmp_path / "downloads/9zzz")
    assert calls["record"] == (
        tmp_path / "downloads/9zzz/9zzz.pdb",
        "9zzz",
        6.0,
        "OKA",
    )
    assert summaries[0]["source"] == str(tmp_path / "downloads/9zzz/9zzz.pdb")
    assert summaries[0]["pocket_atoms"] == 1
    record = read_lmdb_records(output_path)[0]
    assert record["pocket"] == "9zzz"
    assert record["pocket_atoms"] == ["N"]
    assert np.asarray(record["pocket_coordinates"]).shape == (1, 3)


def test_create_pocket_lmdb_respects_overwrite_false(tmp_path):
    output_path = tmp_path / "pocket.lmdb"
    create_pocket_lmdb(
        ["2ie4"],
        output_path,
        combine_set_dir=COMBINE_SET_DIR,
        download_missing=False,
    )

    with pytest.raises(FileExistsError):
        create_pocket_lmdb(
            ["2ie4"],
            output_path,
            combine_set_dir=COMBINE_SET_DIR,
            download_missing=False,
            overwrite=False,
        )
