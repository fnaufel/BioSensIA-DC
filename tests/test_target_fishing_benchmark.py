import pytest

from biosensia_target_fishing_benchmark import (
    build_positive_pairs_frame_from_lmdb,
    read_positive_pairs,
    write_positive_pairs_from_lmdb,
)
from lmdb_helpers import write_lmdb_records


def write_test_lmdb(path, records):
    write_lmdb_records(records, path, overwrite=True, map_size=1 << 20)


def test_write_positive_pairs_from_lmdb_writes_unique_query_pocket_table(tmp_path):
    lmdb_path = tmp_path / "valid.lmdb"
    write_test_lmdb(
        lmdb_path,
        [
            {"smi": "CCO", "pocket": "1abc"},
            {"smi": "CCO", "pocket": "1abc"},
            {"smi": "N", "pocket": "2def"},
        ],
    )
    output_path = tmp_path / "valid_positives.parquet"

    df = write_positive_pairs_from_lmdb(lmdb_path, output_path)

    assert df.to_dicts() == [
        {"query": "CCO", "pocket": "1abc"},
        {"query": "N", "pocket": "2def"},
    ]
    assert read_positive_pairs(
        output_path,
        query_column="query",
        pocket_column="pocket",
    ) == {
        "CCO": {"1abc"},
        "N": {"2def"},
    }


def test_build_positive_pairs_frame_from_lmdb_can_keep_duplicate_rows(tmp_path):
    lmdb_path = tmp_path / "valid.lmdb"
    write_test_lmdb(
        lmdb_path,
        [
            {"ligand_key": "lig-a", "pocket_geometry_hash": "geom-1"},
            {"ligand_key": "lig-a", "pocket_geometry_hash": "geom-1"},
        ],
    )

    df = build_positive_pairs_frame_from_lmdb(
        lmdb_path,
        query_field="ligand_key",
        pocket_field="pocket_geometry_hash",
        query_column="ligand",
        pocket_column="pocket_geometry",
        unique=False,
    )

    assert df.to_dicts() == [
        {"ligand": "lig-a", "pocket_geometry": "geom-1"},
        {"ligand": "lig-a", "pocket_geometry": "geom-1"},
    ]


def test_build_positive_pairs_frame_from_lmdb_requires_identity_fields(tmp_path):
    lmdb_path = tmp_path / "valid.lmdb"
    write_test_lmdb(lmdb_path, [{"smi": "CCO"}])

    with pytest.raises(ValueError, match="pocket"):
        build_positive_pairs_frame_from_lmdb(lmdb_path)
