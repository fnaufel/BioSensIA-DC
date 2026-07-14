import json
import urllib.error
from pathlib import Path

import polars as pl
import pytest

import biosensia_uniprot_enrichment as enrichment
from biosensia_uniprot_enrichment import (
    RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY,
    RCSBGraphQLError,
    aggregate_pocket_scores_by_protein,
    build_entity_uniprot_frame,
    build_pdb_metadata_frame,
    build_pdb_uniprot_cache,
    build_row_level_metadata,
    build_uniprot_metadata_sidecars,
    collect_lmdb_rows,
    determine_entry_mapping_status,
    fetch_rcsb_graphql_batch,
    is_pdb_id,
    normalize_rcsb_entry,
    write_enriched_lmdb,
)
from lmdb_helpers import read_lmdb_records, write_lmdb_records


def protein_entity(
    entity_id: str,
    accessions: list[str] | None,
    *,
    description: str | None = None,
    label_chains: list[str] | None = None,
    author_chains: list[str] | None = None,
    polymer_type: str = "Protein",
    organism: str = "Homo sapiens",
    taxonomy_id: int = 9606,
) -> dict:
    references = []
    for index, accession in enumerate(accessions or []):
        references.append(
            {
                "database_name": "UniProt",
                "database_accession": accession,
                "database_isoform": f"{accession}-1" if index == 0 else None,
                "provenance_source": "SIFTS",
                "entity_sequence_coverage": 0.9,
                "reference_sequence_coverage": 0.8,
            }
        )
    references.append(
        {
            "database_name": "GenBank",
            "database_accession": f"GB-{entity_id}",
        }
    )
    return {
        "rcsb_id": f"1ABC_{entity_id}",
        "entity_poly": {
            "type": "polypeptide(L)",
            "rcsb_entity_polymer_type": polymer_type,
        },
        "rcsb_polymer_entity": {
            "pdbx_description": description or f"Protein {entity_id}",
        },
        "rcsb_polymer_entity_container_identifiers": {
            "entity_id": entity_id,
            "asym_ids": label_chains or [entity_id],
            "auth_asym_ids": author_chains or [entity_id],
            "reference_sequence_identifiers": references,
        },
        "rcsb_entity_source_organism": [
            {
                "ncbi_scientific_name": organism,
                "ncbi_taxonomy_id": taxonomy_id,
            }
        ],
    }


def entry(pdb_id: str, entities: list[dict]) -> dict:
    copied_entities = []
    for entity_json in entities:
        copied = dict(entity_json)
        copied["rcsb_id"] = f"{pdb_id.upper()}_{copied['rcsb_polymer_entity_container_identifiers']['entity_id']}"
        copied_entities.append(copied)
    return {"rcsb_id": pdb_id.upper(), "polymer_entities": copied_entities}


def normalized_entry(pdb_id: str, entities: list[dict]) -> dict:
    return normalize_rcsb_entry(
        entry(pdb_id, entities),
        retrieved_at="2026-07-14T12:00:00+00:00",
    )


def write_test_lmdb(path: Path, records: list[dict]) -> None:
    write_lmdb_records(records, path, overwrite=True, map_size=1 << 20)


def test_graphql_query_uses_current_documented_rcsb_fields():
    assert "rcsb_polymer_entity {" in RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY
    assert "pdbx_description" in RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY
    assert "rcsb_entity_polymer_type" in RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY
    assert "database_isoform" in RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY
    assert "provenance_source" in RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY
    assert "entity_poly {\n        type\n        pdbx_description" not in RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY


@pytest.mark.parametrize(
    ("protein_entities", "mapped_entities", "accessions", "expected"),
    [
        (0, 0, 0, "no_protein_entity"),
        (2, 0, 0, "no_uniprot_mapping"),
        (2, 1, 1, "partial_uniprot_mapping"),
        (1, 1, 1, "ok_single_uniprot"),
        (2, 2, 1, "ok_multiple_entities_same_uniprot"),
        (1, 1, 2, "ambiguous_multiple_uniprot"),
        (2, 2, 2, "ambiguous_multiple_uniprot"),
    ],
)
def test_determine_entry_mapping_status(
    protein_entities,
    mapped_entities,
    accessions,
    expected,
):
    assert determine_entry_mapping_status(
        protein_entity_count=protein_entities,
        mapped_protein_entity_count=mapped_entities,
        unique_uniprot_accession_count=accessions,
    ) == expected


def test_determine_entry_mapping_status_rejects_impossible_counts():
    with pytest.raises(ValueError, match="cannot exceed"):
        determine_entry_mapping_status(
            protein_entity_count=1,
            mapped_protein_entity_count=2,
            unique_uniprot_accession_count=1,
        )


@pytest.mark.parametrize(
    ("entities", "expected_status", "expected_accessions"),
    [
        ([], "no_protein_entity", []),
        ([protein_entity("1", None)], "no_uniprot_mapping", []),
        (
            [protein_entity("1", ["P11111"]), protein_entity("2", None)],
            "partial_uniprot_mapping",
            ["P11111"],
        ),
        (
            [protein_entity("1", ["P11111"])],
            "ok_single_uniprot",
            ["P11111"],
        ),
        (
            [
                protein_entity("1", ["P11111"]),
                protein_entity("2", ["P11111"]),
            ],
            "ok_multiple_entities_same_uniprot",
            ["P11111"],
        ),
        (
            [
                protein_entity("1", ["P11111"]),
                protein_entity("2", ["Q22222"]),
            ],
            "ambiguous_multiple_uniprot",
            ["P11111", "Q22222"],
        ),
    ],
)
def test_normalize_rcsb_entry_assigns_complete_statuses(
    entities,
    expected_status,
    expected_accessions,
):
    metadata = normalized_entry("1abc", entities)

    assert metadata["entry_mapping_status"] == expected_status
    assert metadata["all_uniprot_accessions"] == expected_accessions
    assert metadata["protein_entity_count"] == len(entities)


def test_normalize_rcsb_entry_preserves_entity_relationships_and_provenance():
    metadata = normalized_entry(
        "1abc",
        [
            protein_entity(
                "2",
                ["P67775"],
                description="PP2A catalytic subunit",
                label_chains=["B"],
                author_chains=["C"],
            )
        ],
    )

    entity_metadata = metadata["protein_entities"][0]
    mapping = entity_metadata["uniprot_mappings"][0]
    assert entity_metadata["rcsb_polymer_entity_id"] == "2"
    assert entity_metadata["label_asym_ids"] == ["B"]
    assert entity_metadata["auth_asym_ids"] == ["C"]
    assert entity_metadata["organism_names"] == ["Homo sapiens"]
    assert entity_metadata["organism_taxonomy_ids"] == ["9606"]
    assert mapping == {
        "uniprot_accession": "P67775",
        "uniprot_isoform": "P67775-1",
        "provenance_source": "SIFTS",
        "entity_sequence_coverage": 0.9,
        "reference_sequence_coverage": 0.8,
    }


def test_normalize_rcsb_entry_ignores_nonprotein_polymer_entities():
    dna = protein_entity("3", ["NOT-USED"], polymer_type="DNA")
    dna["entity_poly"]["type"] = "polydeoxyribonucleotide"

    metadata = normalized_entry("1abc", [dna])

    assert metadata["entry_mapping_status"] == "no_protein_entity"
    assert metadata["protein_entities"] == []


def test_normalize_rcsb_entry_retains_partial_graphql_warnings():
    metadata = normalize_rcsb_entry(
        entry("1abc", [protein_entity("1", ["P11111"])]),
        retrieved_at="2026-07-14T12:00:00+00:00",
        graphql_errors=["A sibling entry failed"],
    )

    assert metadata["entry_mapping_status"] == "ok_single_uniprot"
    assert metadata["graphql_status"] == "partial_error"
    assert metadata["metadata_warnings"] == ["A sibling entry failed"]


def test_fetch_rcsb_graphql_batch_posts_uppercase_ids_and_accepts_partial_data():
    calls = []

    def transport(endpoint, payload, timeout):
        calls.append((endpoint, payload, timeout))
        return {
            "data": {"entries": [entry("1abc", [])]},
            "errors": [{"message": "Other entry failed", "path": ["entries", 1]}],
        }

    response = fetch_rcsb_graphql_batch(
        ["1abc"],
        timeout_seconds=12,
        transport=transport,
    )

    assert response["data"]["entries"][0]["rcsb_id"] == "1ABC"
    assert calls[0][1]["variables"] == {"ids": ["1ABC"]}
    assert calls[0][1]["query"] == RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY
    assert calls[0][2] == 12


def test_fetch_rcsb_graphql_batch_rejects_errors_without_usable_data():
    def transport(_endpoint, _payload, _timeout):
        return {"errors": [{"message": "Schema failure"}]}

    with pytest.raises(RCSBGraphQLError, match="Schema failure"):
        fetch_rcsb_graphql_batch(["1abc"], transport=transport)


def test_default_graphql_transport_retries_network_errors(monkeypatch):
    attempts = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"data": {"entries": []}}).encode("utf-8")

    def fake_urlopen(_request, timeout):
        attempts.append(timeout)
        if len(attempts) == 1:
            raise urllib.error.URLError("temporary failure")
        return FakeResponse()

    sleeps = []
    monkeypatch.setattr(enrichment.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(enrichment.time, "sleep", sleeps.append)

    response = fetch_rcsb_graphql_batch(
        ["1abc"],
        max_retries=1,
        initial_backoff_seconds=0.25,
    )

    assert response == {"data": {"entries": []}}
    assert attempts == [60.0, 60.0]
    assert sleeps == [0.25]


def test_build_pdb_uniprot_cache_reconciles_missing_ids_and_reuses_cache(tmp_path):
    calls = []

    def transport(_endpoint, payload, _timeout):
        calls.append(payload["variables"]["ids"])
        return {
            "data": {
                "entries": [
                    entry("1abc", [protein_entity("1", ["P11111"])]),
                    entry("2def", [protein_entity("1", ["Q22222"])]),
                ]
            }
        }

    cache_dir = tmp_path / "cache"
    metadata = build_pdb_uniprot_cache(
        ["1abc", "2def", "3ghi"],
        cache_dir,
        transport=transport,
    )

    assert calls == [["1ABC", "2DEF", "3GHI"]]
    assert metadata["1ABC"]["entry_mapping_status"] == "ok_single_uniprot"
    assert metadata["3GHI"]["entry_mapping_status"] == "pdb_not_found"
    assert (cache_dir / "1ABC.json").exists()
    assert (cache_dir / "3GHI.json").exists()
    assert len(list((cache_dir / "_raw").glob("*.json"))) == 1

    def unexpected_transport(*_args):
        raise AssertionError("valid cache should prevent a second request")

    cached = build_pdb_uniprot_cache(
        ["3ghi", "1abc", "2def"],
        cache_dir,
        transport=unexpected_transport,
    )
    assert cached == metadata


def test_build_pdb_uniprot_cache_splits_oversized_batches(tmp_path):
    calls = []

    def transport(_endpoint, payload, _timeout):
        pdb_ids = payload["variables"]["ids"]
        calls.append(pdb_ids)
        if len(pdb_ids) > 1:
            raise RCSBGraphQLError("too large", status_code=413)
        return {"data": {"entries": [entry(pdb_ids[0], [])]}}

    metadata = build_pdb_uniprot_cache(
        ["1abc", "2def"],
        tmp_path / "cache",
        transport=transport,
    )

    assert calls == [["1ABC", "2DEF"], ["1ABC"], ["2DEF"]]
    assert set(metadata) == {"1ABC", "2DEF"}


def test_build_pdb_uniprot_cache_retries_cached_graphql_errors(tmp_path):
    cache_dir = tmp_path / "cache"

    def failing_transport(_endpoint, _payload, _timeout):
        raise RCSBGraphQLError("temporary")

    failed = build_pdb_uniprot_cache(
        ["1abc"],
        cache_dir,
        transport=failing_transport,
    )
    assert failed["1ABC"]["entry_mapping_status"] == "graphql_error"

    calls = []

    def recovered_transport(_endpoint, payload, _timeout):
        calls.append(payload["variables"]["ids"])
        return {
            "data": {
                "entries": [entry("1abc", [protein_entity("1", ["P11111"])])]
            }
        }

    recovered = build_pdb_uniprot_cache(
        ["1abc"],
        cache_dir,
        transport=recovered_transport,
    )
    assert calls == [["1ABC"]]
    assert recovered["1ABC"]["entry_mapping_status"] == "ok_single_uniprot"


def test_collect_lmdb_rows_preserves_raw_pocket_and_canonicalizes_join_key(tmp_path):
    lmdb_path = tmp_path / "train.lmdb"
    write_test_lmdb(
        lmdb_path,
        [
            {"pocket": "1aBc", "pocket_geometry_hash": "hash-1"},
            {"pocket": "AlphaFold-model"},
        ],
    )

    rows = collect_lmdb_rows({"train": lmdb_path})

    assert rows == [
        {
            "source_split": "train",
            "source_lmdb_key": "0",
            "raw_pocket_id": "1aBc",
            "pdb_id": "1ABC",
            "pocket_geometry_hash": "hash-1",
        },
        {
            "source_split": "train",
            "source_lmdb_key": "1",
            "raw_pocket_id": "AlphaFold-model",
            "pdb_id": None,
            "pocket_geometry_hash": None,
        },
    ]
    assert is_pdb_id("1aBc")
    assert not is_pdb_id("P67775")


def test_frames_preserve_normalized_relations_and_row_selection_rules():
    metadata = {
        "1ABC": normalized_entry(
            "1abc",
            [protein_entity("1", ["P11111"], label_chains=["A"])],
        ),
        "2DEF": normalized_entry(
            "2def",
            [protein_entity("1", ["Q22222"]), protein_entity("2", None)],
        ),
    }
    pdb_frame = build_pdb_metadata_frame(metadata)
    entity_frame = build_entity_uniprot_frame(metadata)
    row_frame = build_row_level_metadata(
        [
            {
                "source_split": "train",
                "source_lmdb_key": "0",
                "raw_pocket_id": "1abc",
                "pdb_id": "1ABC",
                "pocket_geometry_hash": None,
            },
            {
                "source_split": "train",
                "source_lmdb_key": "1",
                "raw_pocket_id": "2def",
                "pdb_id": "2DEF",
                "pocket_geometry_hash": None,
            },
            {
                "source_split": "train",
                "source_lmdb_key": "2",
                "raw_pocket_id": "not-pdb",
                "pdb_id": None,
                "pocket_geometry_hash": None,
            },
        ],
        metadata,
    )

    assert pdb_frame.schema == enrichment.PDB_METADATA_SCHEMA
    assert entity_frame.schema == enrichment.ENTITY_METADATA_SCHEMA
    assert entity_frame.filter(pl.col("pdb_id") == "2DEF").height == 2
    assert (
        entity_frame.filter(
            (pl.col("pdb_id") == "2DEF")
            & (pl.col("rcsb_polymer_entity_id") == "2")
        )["uniprot_accession"][0]
        is None
    )
    rows = row_frame.to_dicts()
    assert rows[0]["selected_uniprot_accession"] == "P11111"
    assert rows[1]["protein_mapping_status"] == "partial_uniprot_mapping"
    assert rows[1]["selected_uniprot_accession"] is None
    assert rows[2]["protein_mapping_status"] == "non_pdb_pocket_id"


def test_write_enriched_lmdb_keeps_original_pocket_identifier(tmp_path):
    source_path = tmp_path / "candidate.lmdb"
    write_test_lmdb(
        source_path,
        [
            {
                "pocket": "1aBc",
                "pocket_atoms": ["C"],
                "pocket_coordinates": [[0.0, 0.0, 0.0]],
                "source_split": "train",
                "source_lmdb_key": "7",
                "custom_field": "kept",
            }
        ],
    )
    metadata = normalized_entry("1abc", [protein_entity("1", ["P11111"])])
    row_frame = build_row_level_metadata(
        [
            {
                "source_split": "train",
                "source_lmdb_key": "7",
                "raw_pocket_id": "1aBc",
                "pdb_id": "1ABC",
                "pocket_geometry_hash": None,
            }
        ],
        {"1ABC": metadata},
    )

    output_path = tmp_path / "candidate.enriched.lmdb"
    summary = write_enriched_lmdb(source_path, row_frame, output_path)
    record = read_lmdb_records(output_path)[0]

    assert summary["enriched_records"] == 1
    assert record["pocket"] == "1aBc"
    assert record["custom_field"] == "kept"
    assert record["pdb_id"] == "1ABC"
    assert record["entry_uniprot_accessions"] == ["P11111"]
    assert record["selected_uniprot_accession"] == "P11111"
    assert record["protein_mapping_status"] == "ok_single_uniprot"


def test_write_enriched_lmdb_supports_raw_split_identity(tmp_path):
    source_path = tmp_path / "train.lmdb"
    write_test_lmdb(source_path, [{"pocket": "1abc", "value": 42}])
    metadata = normalized_entry("1abc", [protein_entity("1", ["P11111"])])
    row_frame = build_row_level_metadata(
        [
            {
                "source_split": "train",
                "source_lmdb_key": "0",
                "raw_pocket_id": "1abc",
                "pdb_id": "1ABC",
                "pocket_geometry_hash": None,
            }
        ],
        {"1ABC": metadata},
    )

    output_path = tmp_path / "train.enriched.lmdb"
    write_enriched_lmdb(
        source_path,
        row_frame,
        output_path,
        source_split="train",
    )

    record = read_lmdb_records(output_path)[0]
    assert record["pocket"] == "1abc"
    assert record["value"] == 42
    assert record["entry_uniprot_accessions"] == ["P11111"]


def test_aggregate_pocket_scores_by_protein_reports_support_and_ambiguity():
    metadata = {
        "1ABC": normalized_entry(
            "1abc",
            [protein_entity("1", ["P11111"], description="Protein P")],
        ),
        "2DEF": normalized_entry(
            "2def",
            [
                protein_entity("1", ["P11111"], description="Protein P"),
                protein_entity("2", ["Q22222"], description="Protein Q"),
            ],
        ),
        "3GHI": normalized_entry(
            "3ghi",
            [protein_entity("1", ["P11111"], description="Protein P")],
        ),
    }
    entity_frame = build_entity_uniprot_frame(metadata)
    ranked = pl.DataFrame(
        [
            {
                "query": "aspirin",
                "pocket": "1abc",
                "drugclip_score": 0.8,
                "source_split": "train",
                "source_lmdb_key": "0",
                "pocket_geometry_hash": "g1",
            },
            {
                "query": "aspirin",
                "pocket": "2def",
                "drugclip_score": 0.9,
                "source_split": "train",
                "source_lmdb_key": "1",
                "pocket_geometry_hash": "g2",
            },
            {
                "query": "aspirin",
                "pocket": "3ghi",
                "drugclip_score": 0.7,
                "source_split": "train",
                "source_lmdb_key": "2",
                "pocket_geometry_hash": "g3",
            },
        ]
    )

    strict = aggregate_pocket_scores_by_protein(
        ranked,
        entity_frame,
        ambiguity_mode="strict",
    )
    exploratory = aggregate_pocket_scores_by_protein(
        ranked,
        entity_frame,
        ambiguity_mode="exploratory",
    )

    strict_row = strict.to_dicts()[0]
    assert strict_row["uniprot_accession"] == "P11111"
    assert strict_row["protein_score"] == 0.8
    assert strict_row["best_pocket"] == "1abc"
    assert strict_row["best_source_lmdb_key"] == "0"
    assert strict_row["support_count"] == 2
    assert strict_row["unique_pdb_count"] == 2
    assert strict_row["supporting_pdb_ids"] == ["1ABC", "3GHI"]
    assert strict_row["has_ambiguous_support"] is False

    exploratory_rows = exploratory.to_dicts()
    assert [row["uniprot_accession"] for row in exploratory_rows] == [
        "P11111",
        "Q22222",
    ]
    assert exploratory_rows[0]["protein_score"] == 0.9
    assert exploratory_rows[0]["best_pocket"] == "2def"
    assert exploratory_rows[0]["support_count"] == 3
    assert exploratory_rows[0]["unique_pdb_count"] == 3
    assert exploratory_rows[0]["has_ambiguous_support"] is True
    assert exploratory_rows[1]["protein_score"] == 0.9
    assert exploratory_rows[1]["has_ambiguous_support"] is True
    supporting_hits = json.loads(exploratory_rows[0]["supporting_hits_json"])
    assert supporting_hits[1]["source_split"] == "train"
    assert supporting_hits[1]["pocket_geometry_hash"] == "g2"


def test_end_to_end_sidecars_and_enriched_candidate_lmdb_smoke(tmp_path):
    source_dir = tmp_path / "source"
    train_path = source_dir / "train.lmdb"
    write_test_lmdb(
        train_path,
        [
            {"pocket": "1abc", "payload": "a"},
            {"pocket": "2def", "payload": "b"},
            {"pocket": "not-pdb", "payload": "c"},
        ],
    )

    def transport(_endpoint, payload, _timeout):
        requested = payload["variables"]["ids"]
        entries = []
        if "1ABC" in requested:
            entries.append(entry("1abc", [protein_entity("1", ["P11111"])]))
        if "2DEF" in requested:
            entries.append(
                entry(
                    "2def",
                    [protein_entity("1", ["Q22222"]), protein_entity("2", None)],
                )
            )
        return {"data": {"entries": entries}}

    result = build_uniprot_metadata_sidecars(
        {"train": train_path},
        output_dir=tmp_path / "metadata",
        transport=transport,
    )

    assert result["lmdb_rows"] == 3
    assert result["unique_pdb_ids"] == 2
    assert Path(result["pdb_metadata_path"]).exists()
    assert Path(result["entity_metadata_path"]).exists()
    assert Path(result["row_metadata_path"]).exists()
    assert result["row_metadata"]["protein_mapping_status"].to_list() == [
        "ok_single_uniprot",
        "partial_uniprot_mapping",
        "non_pdb_pocket_id",
    ]

    enriched_path = tmp_path / "train.enriched.lmdb"
    write_enriched_lmdb(
        train_path,
        result["row_metadata"],
        enriched_path,
        source_split="train",
    )
    records = read_lmdb_records(enriched_path)
    assert [record["pocket"] for record in records] == [
        "1abc",
        "2def",
        "not-pdb",
    ]
    assert records[0]["entry_uniprot_accessions"] == ["P11111"]
    assert records[1]["selected_uniprot_accession"] is None
    assert records[2]["protein_mapping_status"] == "non_pdb_pocket_id"
