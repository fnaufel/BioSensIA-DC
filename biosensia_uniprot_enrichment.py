"""RCSB-backed UniProt metadata enrichment for BioSensIA pocket libraries.

The functions in this module deliberately perform *entry-level* annotation.
Given a PDB-like value stored in ``record["pocket"]``, they discover the
protein polymer entities present in that PDB entry and the UniProt references
reported for those entities.  They do not claim that every returned protein
contains, contacts, or biologically owns the pocket represented by the LMDB
record.

The implementation never rewrites the candidate LMDB.  A lightweight candidate
index links exact LMDB keys to canonical PDB IDs, while normalized PDB and
protein-entity metadata live in Parquet sidecars.  Per-PDB JSON cache files keep
API responses auditable and allow normalization to evolve without duplicating
entry-level metadata in every LMDB record.
"""

from __future__ import annotations

import json
import pickle
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import lmdb
import polars as pl
from tqdm.auto import tqdm


RCSB_GRAPHQL_ENDPOINT = "https://data.rcsb.org/graphql"
RCSB_MAPPING_SOURCE = "RCSB PDB Data API GraphQL"
RCSB_MAPPING_METHOD = "rcsb_graphql_entry_level"
RCSB_USER_AGENT = "BioSensIA-DC/0.1 (PDB-to-UniProt metadata enrichment)"
DEFAULT_GRAPHQL_BATCH_SIZE = 100
DEFAULT_GRAPHQL_TIMEOUT_SECONDS = 60.0
DEFAULT_CACHE_DIR = Path("data/pdb_graphql_cache")
DEFAULT_PDB_METADATA_PATH = Path("data/pdb_uniprot_metadata.parquet")
DEFAULT_ENTITY_METADATA_PATH = Path("data/pdb_entity_uniprot_metadata.parquet")
DEFAULT_CANDIDATE_INDEX_PATH = Path("data/candidate_pocket_index.parquet")
CACHE_SCHEMA_VERSION = "1"

PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")

RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY = """
query EntryProteinMappings($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    polymer_entities {
      rcsb_id
      entity_poly {
        type
        rcsb_entity_polymer_type
      }
      rcsb_polymer_entity {
        pdbx_description
      }
      rcsb_polymer_entity_container_identifiers {
        entity_id
        asym_ids
        auth_asym_ids
        reference_sequence_identifiers {
          database_name
          database_accession
          database_isoform
          provenance_source
          entity_sequence_coverage
          reference_sequence_coverage
        }
      }
      rcsb_entity_source_organism {
        ncbi_scientific_name
        ncbi_taxonomy_id
      }
    }
  }
}
""".strip()

RCSB_QUERY_SHA256 = sha256(
    RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY.encode("utf-8")
).hexdigest()

UNAMBIGUOUS_MAPPING_STATUSES = frozenset(
    {"ok_single_uniprot", "ok_multiple_entities_same_uniprot"}
)

PDB_METADATA_SCHEMA = {
    "pdb_id": pl.String,
    "all_uniprot_accessions": pl.List(pl.String),
    "all_uniprot_isoforms": pl.List(pl.String),
    "selected_uniprot_accession": pl.String,
    "all_protein_names": pl.List(pl.String),
    "all_organisms": pl.List(pl.String),
    "all_organism_taxonomy_ids": pl.List(pl.String),
    "rcsb_polymer_entity_ids": pl.List(pl.String),
    "protein_entity_count": pl.Int64,
    "mapped_protein_entity_count": pl.Int64,
    "entry_mapping_status": pl.String,
    "graphql_status": pl.String,
    "retrieved_at": pl.String,
    "mapping_method": pl.String,
    "mapping_source": pl.String,
    "query_sha256": pl.String,
    "metadata_warnings": pl.List(pl.String),
}

ENTITY_METADATA_SCHEMA = {
    "pdb_id": pl.String,
    "rcsb_polymer_entity_id": pl.String,
    "rcsb_id": pl.String,
    "polymer_type": pl.String,
    "entity_description": pl.String,
    "label_asym_ids": pl.List(pl.String),
    "auth_asym_ids": pl.List(pl.String),
    "organism_names": pl.List(pl.String),
    "organism_taxonomy_ids": pl.List(pl.String),
    "uniprot_accession": pl.String,
    "uniprot_isoform": pl.String,
    "mapping_provenance": pl.String,
    "entity_sequence_coverage": pl.Float64,
    "reference_sequence_coverage": pl.Float64,
    "entry_mapping_status": pl.String,
    "retrieved_at": pl.String,
}

CANDIDATE_INDEX_SCHEMA = {
    "candidate_library_sha256": pl.String,
    "candidate_library_entries": pl.Int64,
    "candidate_lmdb_key": pl.String,
    "source_split": pl.String,
    "source_lmdb_key": pl.String,
    "raw_pocket_id": pl.String,
    "pdb_id": pl.String,
    "pocket_geometry_hash": pl.String,
}

PROTEIN_RANKING_SCHEMA = {
    "query": pl.String,
    "protein_rank": pl.Int64,
    "uniprot_accession": pl.String,
    "protein_score": pl.Float64,
    "protein_names": pl.List(pl.String),
    "organisms": pl.List(pl.String),
    "organism_taxonomy_ids": pl.List(pl.String),
    "best_pocket": pl.String,
    "best_pdb_id": pl.String,
    "best_candidate_library_sha256": pl.String,
    "best_candidate_lmdb_key": pl.String,
    "best_source_split": pl.String,
    "best_source_lmdb_key": pl.String,
    "best_pocket_geometry_hash": pl.String,
    "best_support_mapping_status": pl.String,
    "support_count": pl.Int64,
    "unique_pdb_count": pl.Int64,
    "supporting_pdb_ids": pl.List(pl.String),
    "supporting_pocket_ids": pl.List(pl.String),
    "supporting_mapping_statuses": pl.List(pl.String),
    "has_ambiguous_support": pl.Boolean,
    "aggregation_method": pl.String,
    "ambiguity_mode": pl.String,
    "supporting_hits_json": pl.String,
}


class RCSBGraphQLError(RuntimeError):
    """Represent a transport, HTTP, or unusable GraphQL response failure.

    Parameters
    ----------
    message:
        Human-readable failure description suitable for cache warnings.
    status_code:
        Optional HTTP status code.  Batch orchestration uses this to split
        requests rejected as too large without treating schema errors as
        retryable size failures.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


GraphQLTransport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp with second precision."""

    return datetime.now(UTC).replace(microsecond=0).isoformat()


def is_pdb_id(value: object) -> bool:
    """Return whether ``value`` is a classic four-character PDB identifier.

    This validation intentionally reflects the identifiers present in the
    DrugCLIP/BioSensIA datasets.  It does not classify AlphaFold or other
    computed-model identifiers as PDB IDs.
    """

    return bool(PDB_ID_RE.fullmatch(str(value).strip())) if value is not None else False


def canonicalize_pdb_id(value: object) -> str | None:
    """Return an uppercase PDB ID or ``None`` for a non-PDB-like value."""

    if not is_pdb_id(value):
        return None
    return str(value).strip().upper()


def iter_lmdb_records(lmdb_path: str | Path) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield source key and unpickled record from a DrugCLIP-style LMDB.

    Numeric ASCII keys are yielded numerically rather than lexicographically,
    preserving the order expected by DrugCLIP datasets.  Non-numeric keys are
    sorted bytewise.  The environment is opened read-only without locking so
    this function can inspect large immutable datasets without copying them.

    Parameters
    ----------
    lmdb_path:
        Path to a single-file, pickle-backed LMDB.

    Yields
    ------
    tuple[str, dict[str, Any]]
        Decoded LMDB key and deserialized dictionary record.
    """

    path = Path(lmdb_path)
    if not path.exists():
        raise FileNotFoundError(f"LMDB not found: {path}")

    environment = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    try:
        with environment.begin() as transaction:
            keys = list(transaction.cursor().iternext(values=False))
            keys = _sort_lmdb_keys(keys)
            for key in keys:
                value = transaction.get(key)
                if value is None:
                    continue
                record = pickle.loads(value)
                if not isinstance(record, dict):
                    raise TypeError(
                        f"LMDB record {key!r} in {path} is not a dictionary"
                    )
                yield key.decode("ascii", errors="replace"), record
    finally:
        environment.close()


def collect_lmdb_rows(
    lmdb_paths: Mapping[str, str | Path],
) -> list[dict[str, Any]]:
    """Collect row identities and PDB-like pocket IDs from LMDB splits.

    This compatibility utility inventories original split LMDBs before a
    candidate library is assembled.  ``raw_pocket_id`` always preserves the
    source value; ``pdb_id`` is an uppercase canonical join key or ``None``.
    Existing ``pocket_geometry_hash`` values are retained when present.

    Parameters
    ----------
    lmdb_paths:
        Mapping from logical split name, such as ``"train"``, to LMDB path.

    Returns
    -------
    list[dict[str, Any]]
        One row per LMDB record, including split, key, raw pocket value,
        canonical PDB ID, and optional geometry hash.
    """

    rows: list[dict[str, Any]] = []
    for source_split, lmdb_path in lmdb_paths.items():
        normalized_split = str(source_split).strip()
        if not normalized_split:
            raise ValueError("LMDB split names must be non-empty")
        for source_key, record in iter_lmdb_records(lmdb_path):
            raw_value = record.get("pocket")
            raw_pocket_id = "" if raw_value is None else str(raw_value)
            rows.append(
                {
                    "source_split": normalized_split,
                    "source_lmdb_key": source_key,
                    "raw_pocket_id": raw_pocket_id,
                    "pdb_id": canonicalize_pdb_id(raw_pocket_id),
                    "pocket_geometry_hash": _optional_string(
                        record.get("pocket_geometry_hash")
                    ),
                }
            )
    return rows


def collect_unique_pdb_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Return canonical PDB IDs present in lightweight LMDB row metadata."""

    return {
        str(row["pdb_id"]).upper()
        for row in rows
        if row.get("pdb_id") is not None
    }


def build_candidate_pocket_index_frame(
    candidate_lmdb_path: str | Path,
    *,
    show_progress: bool = True,
) -> pl.DataFrame:
    """Build an exact, lightweight index for one candidate-pocket LMDB.

    The LMDB key is the primary row identity.  ``source_split`` and
    ``source_lmdb_key`` preserve provenance when the candidate library was
    created by :func:`biosensia_target_fishing.build_candidate_pockets_lmdb`.
    ``pdb_id`` is the foreign key into the PDB and entity metadata sidecars.
    The original ``pocket`` value is copied to ``raw_pocket_id`` without
    normalization.

    A SHA-256 digest is calculated from the ordered logical LMDB key/value
    stream, not from LMDB page bytes.  This makes the library identity stable
    across equivalent LMDB files with different map sizes or page layouts.
    The digest and entry count are repeated in the Parquet table; Parquet
    dictionary encoding keeps this repetition inexpensive and lets any
    extracted row identify the exact candidate library it belongs to.

    Parameters
    ----------
    candidate_lmdb_path:
        DrugCLIP-compatible candidate-pocket LMDB.  Records must be pickled
        dictionaries containing a non-empty ``pocket`` field.  Keys must be
        ASCII text so they can be represented losslessly in Parquet.
    show_progress:
        Display a progress bar while reading and hashing candidate records.

    Returns
    -------
    polars.DataFrame
        One row per LMDB entry with ``candidate_lmdb_key`` as the exact primary
        key and ``pdb_id`` as the metadata foreign key.
    """

    path = Path(candidate_lmdb_path)
    if not path.exists():
        raise FileNotFoundError(f"Candidate pocket LMDB not found: {path}")

    environment = lmdb.open(
        str(path),
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=256,
    )
    rows: list[dict[str, Any]] = []
    digest = sha256()
    try:
        with environment.begin() as transaction:
            keys = _sort_lmdb_keys(
                list(transaction.cursor().iternext(values=False))
            )
            for key in tqdm(
                keys,
                desc="Indexing candidate pockets",
                unit="record",
                disable=not show_progress,
            ):
                value = transaction.get(key)
                if value is None:
                    continue
                digest.update(len(key).to_bytes(8, "big"))
                digest.update(key)
                digest.update(len(value).to_bytes(8, "big"))
                digest.update(value)

                record = pickle.loads(value)
                if not isinstance(record, dict):
                    raise TypeError(
                        f"Candidate LMDB record {key!r} in {path} is not a dictionary"
                    )
                raw_value = record.get("pocket")
                raw_pocket_id = "" if raw_value is None else str(raw_value)
                if not raw_pocket_id.strip():
                    raise ValueError(
                        f"Candidate LMDB record {key!r} in {path} has no pocket"
                    )
                try:
                    candidate_lmdb_key = key.decode("ascii")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"Candidate LMDB key {key!r} in {path} is not ASCII"
                    ) from exc
                rows.append(
                    {
                        "candidate_lmdb_key": candidate_lmdb_key,
                        "source_split": _optional_string(
                            record.get("source_split")
                        ),
                        "source_lmdb_key": _optional_string(
                            record.get("source_lmdb_key")
                        ),
                        "raw_pocket_id": raw_pocket_id,
                        "pdb_id": canonicalize_pdb_id(raw_pocket_id),
                        "pocket_geometry_hash": _optional_string(
                            record.get("pocket_geometry_hash")
                        ),
                    }
                )
    finally:
        environment.close()

    library_sha256 = digest.hexdigest()
    entry_count = len(rows)
    indexed_rows = [
        {
            "candidate_library_sha256": library_sha256,
            "candidate_library_entries": entry_count,
            **row,
        }
        for row in rows
    ]
    return pl.DataFrame(
        indexed_rows,
        schema=CANDIDATE_INDEX_SCHEMA,
        orient="row",
    )


def fetch_rcsb_graphql_batch(
    pdb_ids: Sequence[str],
    *,
    endpoint: str = RCSB_GRAPHQL_ENDPOINT,
    timeout_seconds: float = DEFAULT_GRAPHQL_TIMEOUT_SECONDS,
    transport: GraphQLTransport | None = None,
    max_retries: int = 4,
    initial_backoff_seconds: float = 1.0,
) -> dict[str, Any]:
    """Fetch entry-level protein mappings for one batch of PDB IDs.

    The function uses a JSON POST body and accepts GraphQL responses that
    contain both ``data`` and ``errors`` so callers can retain partial data.
    A response with errors but no usable ``data.entries`` raises
    :class:`RCSBGraphQLError`.  HTTP 429 and 5xx responses, connection errors,
    and timeouts are retried by the default transport with exponential
    backoff.  Tests and offline workflows can inject a deterministic
    ``transport`` callable.

    Parameters
    ----------
    pdb_ids:
        One or more classic PDB IDs.  IDs are canonicalized to uppercase.
    endpoint:
        RCSB GraphQL endpoint.
    timeout_seconds:
        Per-request network timeout.
    transport:
        Optional callable receiving ``(endpoint, payload, timeout_seconds)``
        and returning the decoded JSON object.
    max_retries:
        Maximum retry count used by the default HTTP transport.
    initial_backoff_seconds:
        Initial exponential-backoff delay.

    Returns
    -------
    dict[str, Any]
        Decoded GraphQL response containing ``data`` and optionally ``errors``.
    """

    canonical_ids = _validate_pdb_ids(pdb_ids)
    if not canonical_ids:
        raise ValueError("pdb_ids must contain at least one PDB ID")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0")

    payload = {
        "query": RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY,
        "variables": {"ids": canonical_ids},
    }
    if transport is None:
        response = _post_json_with_retries(
            endpoint,
            payload,
            timeout_seconds,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
        )
    else:
        response = transport(endpoint, payload, timeout_seconds)

    if not isinstance(response, Mapping):
        raise RCSBGraphQLError("GraphQL response is not a JSON object")
    decoded = dict(response)
    entries = (decoded.get("data") or {}).get("entries")
    if entries is None:
        errors = _graphql_error_messages(decoded.get("errors"))
        detail = "; ".join(errors) if errors else "response has no data.entries"
        raise RCSBGraphQLError(f"Unusable GraphQL response: {detail}")
    if not isinstance(entries, list):
        raise RCSBGraphQLError("GraphQL data.entries is not a list")
    return decoded


def normalize_rcsb_entry(
    entry_json: Mapping[str, Any],
    *,
    retrieved_at: str | None = None,
    graphql_errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Normalize one RCSB GraphQL entry into entry and entity metadata.

    Only protein polymer entities are retained.  Protein classification uses
    RCSB's normalized ``entity_poly.rcsb_entity_polymer_type`` field and falls
    back to the raw PDBx polymer type for defensive compatibility.  UniProt
    mappings preserve accession, isoform, provenance, and sequence coverage.

    The entry status is based on mapping completeness:

    * ``no_protein_entity``: no protein polymer entity was returned;
    * ``no_uniprot_mapping``: protein entities exist but none map to UniProt;
    * ``partial_uniprot_mapping``: only some protein entities map to UniProt;
    * ``ok_single_uniprot``: one mapped entity and one unique accession;
    * ``ok_multiple_entities_same_uniprot``: every protein entity maps and all
      map to the same accession;
    * ``ambiguous_multiple_uniprot``: complete mapping yields several unique
      accessions, including multi-accession fusion/chimeric entities.

    Parameters
    ----------
    entry_json:
        One object from GraphQL ``data.entries``.
    retrieved_at:
        Retrieval timestamp.  Defaults to the current UTC time.
    graphql_errors:
        Batch-level GraphQL error messages retained as warnings.  Usable entry
        data remains normalized and receives ``graphql_status="partial_error"``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable normalized entry metadata with nested protein
        entities and the original GraphQL entry for auditability.
    """

    pdb_id = canonicalize_pdb_id(entry_json.get("rcsb_id"))
    if pdb_id is None:
        raise ValueError("RCSB entry is missing a valid four-character rcsb_id")

    protein_entities = []
    for entity_json in entry_json.get("polymer_entities") or []:
        if not isinstance(entity_json, Mapping) or not _is_protein_entity(entity_json):
            continue
        protein_entities.append(_normalize_protein_entity(pdb_id, entity_json))

    accessions = _ordered_unique(
        mapping["uniprot_accession"]
        for entity in protein_entities
        for mapping in entity["uniprot_mappings"]
    )
    isoforms = _ordered_unique(
        mapping["uniprot_isoform"]
        for entity in protein_entities
        for mapping in entity["uniprot_mappings"]
        if mapping["uniprot_isoform"] is not None
    )
    mapped_entity_count = sum(
        bool(entity["uniprot_accessions"]) for entity in protein_entities
    )
    mapping_status = determine_entry_mapping_status(
        protein_entity_count=len(protein_entities),
        mapped_protein_entity_count=mapped_entity_count,
        unique_uniprot_accession_count=len(accessions),
    )
    warnings = _ordered_unique(str(error) for error in graphql_errors if str(error))
    if mapping_status == "partial_uniprot_mapping":
        warnings.append(
            f"Only {mapped_entity_count} of {len(protein_entities)} protein "
            "entities have UniProt mappings."
        )

    return {
        "pdb_id": pdb_id,
        "retrieved_at": retrieved_at or utc_now_iso(),
        "mapping_source": RCSB_MAPPING_SOURCE,
        "mapping_method": RCSB_MAPPING_METHOD,
        "query_sha256": RCSB_QUERY_SHA256,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "entry_mapping_status": mapping_status,
        "graphql_status": "partial_error" if warnings and graphql_errors else "ok",
        "protein_entity_count": len(protein_entities),
        "mapped_protein_entity_count": mapped_entity_count,
        "protein_entities": protein_entities,
        "all_uniprot_accessions": accessions,
        "all_uniprot_isoforms": isoforms,
        "all_protein_names": _ordered_unique(
            entity["entity_description"]
            for entity in protein_entities
            if entity["entity_description"] is not None
        ),
        "all_organisms": _ordered_unique(
            organism
            for entity in protein_entities
            for organism in entity["organism_names"]
        ),
        "all_organism_taxonomy_ids": _ordered_unique(
            taxonomy_id
            for entity in protein_entities
            for taxonomy_id in entity["organism_taxonomy_ids"]
        ),
        "metadata_warnings": warnings,
        "raw_graphql_entry": dict(entry_json),
    }


def determine_entry_mapping_status(
    *,
    protein_entity_count: int,
    mapped_protein_entity_count: int,
    unique_uniprot_accession_count: int,
) -> str:
    """Classify entry-level UniProt mapping completeness and ambiguity.

    The decision order intentionally checks entity completeness before unique
    accession cardinality.  This prevents an entry with one mapped protein and
    several unmapped proteins from being mislabeled ``ok_single_uniprot`` and
    keeps ``ok_multiple_entities_same_uniprot`` reachable.
    """

    if min(
        protein_entity_count,
        mapped_protein_entity_count,
        unique_uniprot_accession_count,
    ) < 0:
        raise ValueError("mapping counts must be non-negative")
    if mapped_protein_entity_count > protein_entity_count:
        raise ValueError("mapped protein entities cannot exceed protein entities")
    if protein_entity_count == 0:
        return "no_protein_entity"
    if mapped_protein_entity_count == 0 or unique_uniprot_accession_count == 0:
        return "no_uniprot_mapping"
    if mapped_protein_entity_count < protein_entity_count:
        return "partial_uniprot_mapping"
    if unique_uniprot_accession_count == 1:
        if protein_entity_count > 1:
            return "ok_multiple_entities_same_uniprot"
        return "ok_single_uniprot"
    return "ambiguous_multiple_uniprot"


def build_pdb_uniprot_cache(
    pdb_ids: Iterable[str],
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    batch_size: int = DEFAULT_GRAPHQL_BATCH_SIZE,
    refresh: bool = False,
    retry_error_cache: bool = True,
    endpoint: str = RCSB_GRAPHQL_ENDPOINT,
    timeout_seconds: float = DEFAULT_GRAPHQL_TIMEOUT_SECONDS,
    transport: GraphQLTransport | None = None,
    show_progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Build or reuse normalized per-PDB metadata cache files.

    Successful and negative lookup results are written as ``<PDB>.json``.
    Raw batch responses are written separately under ``_raw/`` with the exact
    query, requested IDs, timestamp, and decoded response.  This separation
    preserves reproducibility without making downstream code depend on the
    external response shape.

    Cached files are reused only when their cache schema and GraphQL query hash
    match this module.  ``graphql_error`` cache entries remain available for
    audit but are retried by default, avoiding permanent caching of transient
    failures.  HTTP 413/414 batch failures are recursively split until a
    request succeeds or a single PDB ID fails.

    Parameters
    ----------
    pdb_ids:
        Iterable of PDB IDs.  Duplicates are removed after canonicalization.
    cache_dir:
        Directory for normalized per-PDB cache files and ``_raw`` responses.
    batch_size:
        Maximum requested entries per GraphQL call.
    refresh:
        Ignore reusable cache entries and fetch every requested ID.
    retry_error_cache:
        Retry cached ``graphql_error`` entries when ``refresh`` is false.
    endpoint, timeout_seconds, transport:
        Passed to :func:`fetch_rcsb_graphql_batch`.
    show_progress:
        Display progress while fetching batches not satisfied by the cache.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping from canonical PDB ID to normalized metadata.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    canonical_ids = sorted(set(_validate_pdb_ids(list(pdb_ids))))
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "_raw").mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, Any]] = {}
    pending = []
    for pdb_id in canonical_ids:
        cached = None if refresh else _read_reusable_cache(cache_path / f"{pdb_id}.json")
        if cached is not None and not (
            retry_error_cache and cached.get("entry_mapping_status") == "graphql_error"
        ):
            metadata[pdb_id] = cached
        else:
            pending.append(pdb_id)

    batches = [
        pending[start : start + batch_size]
        for start in range(0, len(pending), batch_size)
    ]
    for batch in tqdm(
        batches,
        desc="Fetching PDB metadata",
        unit="batch",
        disable=not show_progress,
    ):
        _fetch_cache_batch(
            batch,
            cache_path=cache_path,
            metadata=metadata,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    return metadata


def build_pdb_metadata_frame(
    pdb_metadata: Mapping[str, Mapping[str, Any]],
) -> pl.DataFrame:
    """Build one-row-per-PDB summary metadata with an explicit schema."""

    rows = []
    for pdb_id in sorted(pdb_metadata):
        metadata = pdb_metadata[pdb_id]
        accessions = list(metadata.get("all_uniprot_accessions", []))
        selected_accession = (
            accessions[0]
            if metadata.get("entry_mapping_status")
            in UNAMBIGUOUS_MAPPING_STATUSES
            and len(accessions) == 1
            else None
        )
        rows.append(
            {
                "pdb_id": pdb_id,
                "all_uniprot_accessions": accessions,
                "all_uniprot_isoforms": metadata.get("all_uniprot_isoforms", []),
                "selected_uniprot_accession": selected_accession,
                "all_protein_names": metadata.get("all_protein_names", []),
                "all_organisms": metadata.get("all_organisms", []),
                "all_organism_taxonomy_ids": metadata.get(
                    "all_organism_taxonomy_ids", []
                ),
                "rcsb_polymer_entity_ids": [
                    entity["rcsb_polymer_entity_id"]
                    for entity in metadata.get("protein_entities", [])
                ],
                "protein_entity_count": metadata.get("protein_entity_count", 0),
                "mapped_protein_entity_count": metadata.get(
                    "mapped_protein_entity_count", 0
                ),
                "entry_mapping_status": metadata.get("entry_mapping_status"),
                "graphql_status": metadata.get("graphql_status"),
                "retrieved_at": metadata.get("retrieved_at"),
                "mapping_method": metadata.get(
                    "mapping_method", RCSB_MAPPING_METHOD
                ),
                "mapping_source": metadata.get("mapping_source", RCSB_MAPPING_SOURCE),
                "query_sha256": metadata.get("query_sha256", RCSB_QUERY_SHA256),
                "metadata_warnings": metadata.get("metadata_warnings", []),
            }
        )
    return pl.DataFrame(rows, schema=PDB_METADATA_SCHEMA, orient="row")


def build_entity_uniprot_frame(
    pdb_metadata: Mapping[str, Mapping[str, Any]],
) -> pl.DataFrame:
    """Build a normalized protein-entity-to-UniProt relation table.

    Each mapped accession receives one row per protein entity.  Protein
    entities without a UniProt reference are retained with a null accession,
    which is essential for auditing ``partial_uniprot_mapping`` and
    ``no_uniprot_mapping`` entries.  Names, organisms, chains, provenance, and
    coverage remain associated with the entity/accession that supplied them.
    """

    rows = []
    for pdb_id in sorted(pdb_metadata):
        metadata = pdb_metadata[pdb_id]
        for entity in metadata.get("protein_entities", []):
            mappings = entity.get("uniprot_mappings") or [None]
            for mapping in mappings:
                rows.append(
                    {
                        "pdb_id": pdb_id,
                        "rcsb_polymer_entity_id": entity.get(
                            "rcsb_polymer_entity_id"
                        ),
                        "rcsb_id": entity.get("rcsb_id"),
                        "polymer_type": entity.get("polymer_type"),
                        "entity_description": entity.get("entity_description"),
                        "label_asym_ids": entity.get("label_asym_ids", []),
                        "auth_asym_ids": entity.get("auth_asym_ids", []),
                        "organism_names": entity.get("organism_names", []),
                        "organism_taxonomy_ids": entity.get(
                            "organism_taxonomy_ids", []
                        ),
                        "uniprot_accession": (
                            mapping.get("uniprot_accession") if mapping else None
                        ),
                        "uniprot_isoform": (
                            mapping.get("uniprot_isoform") if mapping else None
                        ),
                        "mapping_provenance": (
                            mapping.get("provenance_source") if mapping else None
                        ),
                        "entity_sequence_coverage": (
                            mapping.get("entity_sequence_coverage") if mapping else None
                        ),
                        "reference_sequence_coverage": (
                            mapping.get("reference_sequence_coverage")
                            if mapping
                            else None
                        ),
                        "entry_mapping_status": metadata.get("entry_mapping_status"),
                        "retrieved_at": metadata.get("retrieved_at"),
                    }
                )
    return pl.DataFrame(rows, schema=ENTITY_METADATA_SCHEMA, orient="row")


def build_uniprot_metadata_sidecars(
    candidate_lmdb_path: str | Path,
    *,
    output_dir: str | Path = "data",
    cache_dir: str | Path | None = None,
    batch_size: int = DEFAULT_GRAPHQL_BATCH_SIZE,
    refresh: bool = False,
    endpoint: str = RCSB_GRAPHQL_ENDPOINT,
    timeout_seconds: float = DEFAULT_GRAPHQL_TIMEOUT_SECONDS,
    transport: GraphQLTransport | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    """Build three Parquet sidecars without modifying the candidate LMDB.

    The output consists of a one-row-per-PDB summary, a normalized
    protein-entity-to-UniProt relation, and a lightweight exact index of the
    candidate LMDB.  Entry-level metadata is never copied into LMDB records or
    repeated in the candidate index.  The index joins to both metadata tables
    through ``pdb_id`` and joins back to the binary library through
    ``candidate_lmdb_key`` plus ``candidate_library_sha256``.

    Parameters
    ----------
    candidate_lmdb_path:
        Candidate-pocket library used by BioSensIA target fishing.  Its records
        are read only and remain byte-for-byte unchanged.
    output_dir:
        Directory receiving ``pdb_uniprot_metadata.parquet``,
        ``pdb_entity_uniprot_metadata.parquet``, and
        ``candidate_pocket_index.parquet``.
    cache_dir:
        Directory for normalized per-PDB JSON and raw GraphQL responses.
        Defaults to ``<output_dir>/pdb_graphql_cache``.
    batch_size:
        Maximum number of PDB IDs per GraphQL request.
    refresh:
        Ignore compatible normalized cache entries when true.
    endpoint, timeout_seconds, transport:
        Network configuration passed to build_pdb_uniprot_cache.
        ``transport`` is injectable for deterministic offline tests.
    show_progress:
        Print a startup message and display progress bars for LMDB indexing
        and uncached GraphQL batches.

    Returns
    -------
    dict[str, Any]
        The three dataframes, output paths, cache path, candidate row count,
        candidate library digest, and unique PDB count.
    """

    if show_progress:
        print(
            f"Building UniProt metadata sidecars from {candidate_lmdb_path}...",
            flush=True,
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_cache_dir = (
        Path(cache_dir)
        if cache_dir is not None
        else output_path / "pdb_graphql_cache"
    )
    candidate_index = build_candidate_pocket_index_frame(
        candidate_lmdb_path,
        show_progress=show_progress,
    )
    pdb_ids = {
        pdb_id
        for pdb_id in candidate_index["pdb_id"].to_list()
        if pdb_id is not None
    }
    metadata = build_pdb_uniprot_cache(
        pdb_ids,
        resolved_cache_dir,
        batch_size=batch_size,
        refresh=refresh,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        transport=transport,
        show_progress=show_progress,
    )
    pdb_frame = build_pdb_metadata_frame(metadata)
    entity_frame = build_entity_uniprot_frame(metadata)

    pdb_output = output_path / DEFAULT_PDB_METADATA_PATH.name
    entity_output = output_path / DEFAULT_ENTITY_METADATA_PATH.name
    candidate_index_output = output_path / DEFAULT_CANDIDATE_INDEX_PATH.name
    pdb_frame.write_parquet(pdb_output)
    entity_frame.write_parquet(entity_output)
    candidate_index.write_parquet(candidate_index_output)
    candidate_library_sha256 = (
        candidate_index["candidate_library_sha256"][0]
        if candidate_index.height
        else sha256().hexdigest()
    )
    return {
        "pdb_metadata": pdb_frame,
        "entity_metadata": entity_frame,
        "candidate_index": candidate_index,
        "pdb_metadata_path": str(pdb_output),
        "entity_metadata_path": str(entity_output),
        "candidate_index_path": str(candidate_index_output),
        "cache_dir": str(resolved_cache_dir),
        "candidate_lmdb_path": str(candidate_lmdb_path),
        "candidate_rows": candidate_index.height,
        "candidate_library_sha256": candidate_library_sha256,
        "unique_pdb_ids": len(pdb_ids),
    }


def aggregate_pocket_scores_by_protein(
    ranked_pockets: pl.DataFrame,
    entity_metadata: pl.DataFrame,
    *,
    candidate_index: pl.DataFrame | None = None,
    ambiguity_mode: str = "strict",
    query_column: str | None = "query",
    pocket_column: str = "pocket",
    score_column: str = "drugclip_score",
    candidate_key_column: str = "candidate_lmdb_key",
) -> pl.DataFrame:
    """Aggregate ranked pocket hits into detailed UniProt protein rankings.

    Scores are aggregated by maximum pocket score.  In ``strict`` mode, only
    entries classified ``ok_single_uniprot`` or
    ``ok_multiple_entities_same_uniprot`` contribute.  In ``exploratory``
    mode, every reported accession receives the pocket score and ambiguous or
    partial evidence is flagged.

    The result reports the best supporting pocket and available candidate and
    source-row provenance, support count, unique PDB count, all supporting PDB
    and pocket identifiers, mapping statuses, ambiguity, associated
    names/organisms, and a JSON representation of every supporting hit.
    ``support_count`` counts ranked candidate rows while ``unique_pdb_count``
    exposes representation imbalance from repeated structures or augmented
    pocket geometries.

    Parameters
    ----------
    ranked_pockets:
        Pocket ranking frame.  Required columns are ``pocket_column`` and
        ``score_column``.  Optional ``source_split``, ``source_lmdb_key``, and
        ``pocket_geometry_hash`` values are propagated into support details.
        If ``query_column`` is absent or ``None``, all rows belong to a single
        synthetic query named ``"__single_query__"``.
    entity_metadata:
        Normalized frame from :func:`build_entity_uniprot_frame`.
    candidate_index:
        Optional exact index from :func:`build_candidate_pocket_index_frame`.
        When supplied, every ranked row must carry ``candidate_key_column``.
        Candidate-library identity, canonical PDB ID, source provenance, and
        geometry hash are then read from the index rather than trusted from
        duplicated ranking columns.
    ambiguity_mode:
        ``"strict"`` or ``"exploratory"``.
    query_column, pocket_column, score_column, candidate_key_column:
        Input column names.

    Returns
    -------
    polars.DataFrame
        Protein rankings ordered by query, descending score, and accession.
    """

    if ambiguity_mode not in {"strict", "exploratory"}:
        raise ValueError("ambiguity_mode must be 'strict' or 'exploratory'")
    missing_columns = {pocket_column, score_column}.difference(ranked_pockets.columns)
    if missing_columns:
        raise ValueError(f"Ranked pocket table is missing columns: {sorted(missing_columns)}")

    entity_lookup = _build_entity_ranking_lookup(entity_metadata)
    candidate_lookup = _build_candidate_index_lookup(
        ranked_pockets,
        candidate_index,
        candidate_key_column=candidate_key_column,
    )
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for hit_index, hit in enumerate(ranked_pockets.iter_rows(named=True)):
        raw_pocket = str(hit[pocket_column])
        indexed_candidate = None
        candidate_lmdb_key = _optional_string(hit.get(candidate_key_column))
        if candidate_lookup is not None:
            indexed_candidate = candidate_lookup[candidate_lmdb_key]
            if indexed_candidate["raw_pocket_id"] != raw_pocket:
                raise ValueError(
                    "Ranked pocket does not match candidate index for key "
                    f"{candidate_lmdb_key!r}: {raw_pocket!r} != "
                    f"{indexed_candidate['raw_pocket_id']!r}"
                )
        provenance = indexed_candidate if indexed_candidate is not None else hit
        pdb_id = canonicalize_pdb_id(
            provenance.get("pdb_id") or raw_pocket
        )
        if pdb_id is None or pdb_id not in entity_lookup:
            continue
        query = (
            str(hit[query_column])
            if query_column and query_column in hit and hit[query_column] is not None
            else "__single_query__"
        )
        score = float(hit[score_column])
        for accession, relation in entity_lookup[pdb_id].items():
            status = relation["entry_mapping_status"]
            if ambiguity_mode == "strict" and status not in UNAMBIGUOUS_MAPPING_STATUSES:
                continue
            support = {
                "hit_index": hit_index,
                "pocket": raw_pocket,
                "pdb_id": pdb_id,
                "score": score,
                "candidate_library_sha256": _optional_string(
                    provenance.get("candidate_library_sha256")
                ),
                "candidate_lmdb_key": candidate_lmdb_key,
                "source_split": _optional_string(
                    provenance.get("source_split")
                ),
                "source_lmdb_key": _optional_string(
                    provenance.get("source_lmdb_key")
                ),
                "pocket_geometry_hash": _optional_string(
                    provenance.get("pocket_geometry_hash")
                ),
                "mapping_status": status,
                "rcsb_polymer_entity_ids": relation["rcsb_polymer_entity_ids"],
            }
            key = (query, accession)
            group = grouped.setdefault(
                key,
                {
                    "query": query,
                    "uniprot_accession": accession,
                    "protein_names": set(),
                    "organisms": set(),
                    "organism_taxonomy_ids": set(),
                    "supports": [],
                },
            )
            group["protein_names"].update(relation["protein_names"])
            group["organisms"].update(relation["organisms"])
            group["organism_taxonomy_ids"].update(
                relation["organism_taxonomy_ids"]
            )
            group["supports"].append(support)

    rows = []
    for group in grouped.values():
        supports = group["supports"]
        best = max(supports, key=lambda support: support["score"])
        statuses = _ordered_unique(support["mapping_status"] for support in supports)
        rows.append(
            {
                "query": group["query"],
                "protein_rank": 0,
                "uniprot_accession": group["uniprot_accession"],
                "protein_score": best["score"],
                "protein_names": sorted(group["protein_names"]),
                "organisms": sorted(group["organisms"]),
                "organism_taxonomy_ids": sorted(group["organism_taxonomy_ids"]),
                "best_pocket": best["pocket"],
                "best_pdb_id": best["pdb_id"],
                "best_candidate_library_sha256": best[
                    "candidate_library_sha256"
                ],
                "best_candidate_lmdb_key": best["candidate_lmdb_key"],
                "best_source_split": best["source_split"],
                "best_source_lmdb_key": best["source_lmdb_key"],
                "best_pocket_geometry_hash": best["pocket_geometry_hash"],
                "best_support_mapping_status": best["mapping_status"],
                "support_count": len(supports),
                "unique_pdb_count": len({support["pdb_id"] for support in supports}),
                "supporting_pdb_ids": sorted(
                    {support["pdb_id"] for support in supports}
                ),
                "supporting_pocket_ids": sorted(
                    {support["pocket"] for support in supports}
                ),
                "supporting_mapping_statuses": statuses,
                "has_ambiguous_support": any(
                    status not in UNAMBIGUOUS_MAPPING_STATUSES for status in statuses
                ),
                "aggregation_method": "max",
                "ambiguity_mode": ambiguity_mode,
                "supporting_hits_json": json.dumps(
                    supports, sort_keys=True, separators=(",", ":")
                ),
            }
        )

    rows.sort(
        key=lambda row: (
            row["query"],
            -row["protein_score"],
            row["uniprot_accession"],
        )
    )
    current_query = None
    current_rank = 0
    for row in rows:
        if row["query"] != current_query:
            current_query = row["query"]
            current_rank = 1
        else:
            current_rank += 1
        row["protein_rank"] = current_rank
    return pl.DataFrame(rows, schema=PROTEIN_RANKING_SCHEMA, orient="row")


def _normalize_protein_entity(
    pdb_id: str,
    entity_json: Mapping[str, Any],
) -> dict[str, Any]:
    identifiers = entity_json.get("rcsb_polymer_entity_container_identifiers") or {}
    entity_poly = entity_json.get("entity_poly") or {}
    entity_summary = entity_json.get("rcsb_polymer_entity") or {}
    entity_id = _optional_string(identifiers.get("entity_id"))
    if entity_id is None:
        rcsb_id = _optional_string(entity_json.get("rcsb_id")) or ""
        entity_id = rcsb_id.rsplit("_", maxsplit=1)[-1] or rcsb_id

    mappings = []
    for reference in identifiers.get("reference_sequence_identifiers") or []:
        if not isinstance(reference, Mapping):
            continue
        database_name = str(reference.get("database_name") or "").upper()
        accession = _optional_string(reference.get("database_accession"))
        if database_name != "UNIPROT" or accession is None:
            continue
        mappings.append(
            {
                "uniprot_accession": accession,
                "uniprot_isoform": _optional_string(reference.get("database_isoform")),
                "provenance_source": _optional_string(
                    reference.get("provenance_source")
                ),
                "entity_sequence_coverage": _optional_float(
                    reference.get("entity_sequence_coverage")
                ),
                "reference_sequence_coverage": _optional_float(
                    reference.get("reference_sequence_coverage")
                ),
            }
        )
    mappings = _deduplicate_mappings(mappings)

    organisms = entity_json.get("rcsb_entity_source_organism") or []
    return {
        "pdb_id": pdb_id,
        "rcsb_polymer_entity_id": entity_id,
        "rcsb_id": _optional_string(entity_json.get("rcsb_id")),
        "polymer_type": _optional_string(
            entity_poly.get("rcsb_entity_polymer_type") or entity_poly.get("type")
        ),
        "entity_description": _optional_string(entity_summary.get("pdbx_description")),
        "label_asym_ids": _string_list(identifiers.get("asym_ids")),
        "auth_asym_ids": _string_list(identifiers.get("auth_asym_ids")),
        "organism_names": _ordered_unique(
            _optional_string(organism.get("ncbi_scientific_name"))
            for organism in organisms
            if isinstance(organism, Mapping)
            and _optional_string(organism.get("ncbi_scientific_name")) is not None
        ),
        "organism_taxonomy_ids": _ordered_unique(
            str(organism["ncbi_taxonomy_id"])
            for organism in organisms
            if isinstance(organism, Mapping)
            and organism.get("ncbi_taxonomy_id") is not None
        ),
        "uniprot_mappings": mappings,
        "uniprot_accessions": _ordered_unique(
            mapping["uniprot_accession"] for mapping in mappings
        ),
    }


def _is_protein_entity(entity_json: Mapping[str, Any]) -> bool:
    entity_poly = entity_json.get("entity_poly") or {}
    normalized_type = str(entity_poly.get("rcsb_entity_polymer_type") or "").lower()
    raw_type = str(entity_poly.get("type") or "").lower()
    return normalized_type == "protein" or raw_type.startswith("polypeptide")


def _failure_metadata(
    *,
    pdb_id: str | None,
    status: str,
    warning: str,
    retrieved_at: str | None = None,
) -> dict[str, Any]:
    return {
        "pdb_id": pdb_id,
        "retrieved_at": retrieved_at or utc_now_iso(),
        "mapping_source": RCSB_MAPPING_SOURCE,
        "mapping_method": RCSB_MAPPING_METHOD,
        "query_sha256": RCSB_QUERY_SHA256,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "entry_mapping_status": status,
        "graphql_status": "error" if status == "graphql_error" else "ok",
        "protein_entity_count": 0,
        "mapped_protein_entity_count": 0,
        "protein_entities": [],
        "all_uniprot_accessions": [],
        "all_uniprot_isoforms": [],
        "all_protein_names": [],
        "all_organisms": [],
        "all_organism_taxonomy_ids": [],
        "metadata_warnings": [warning],
        "raw_graphql_entry": None,
    }


def _fetch_cache_batch(
    pdb_ids: list[str],
    *,
    cache_path: Path,
    metadata: dict[str, dict[str, Any]],
    endpoint: str,
    timeout_seconds: float,
    transport: GraphQLTransport | None,
) -> None:
    if not pdb_ids:
        return
    retrieved_at = utc_now_iso()
    try:
        payload = fetch_rcsb_graphql_batch(
            pdb_ids,
            endpoint=endpoint,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
    except RCSBGraphQLError as exc:
        if exc.status_code in {413, 414} and len(pdb_ids) > 1:
            midpoint = len(pdb_ids) // 2
            for smaller_batch in (pdb_ids[:midpoint], pdb_ids[midpoint:]):
                _fetch_cache_batch(
                    smaller_batch,
                    cache_path=cache_path,
                    metadata=metadata,
                    endpoint=endpoint,
                    timeout_seconds=timeout_seconds,
                    transport=transport,
                )
            return
        for pdb_id in pdb_ids:
            normalized = _failure_metadata(
                pdb_id=pdb_id,
                status="graphql_error",
                warning=str(exc),
                retrieved_at=retrieved_at,
            )
            _write_json(cache_path / f"{pdb_id}.json", normalized)
            metadata[pdb_id] = normalized
        return

    _write_raw_batch_cache(
        cache_path / "_raw",
        pdb_ids,
        payload,
        retrieved_at,
        endpoint=endpoint,
    )
    errors = _graphql_error_messages(payload.get("errors"))
    returned_entries = {
        str(entry.get("rcsb_id") or "").upper(): entry
        for entry in payload["data"]["entries"]
        if isinstance(entry, Mapping) and entry.get("rcsb_id")
    }
    for pdb_id in pdb_ids:
        entry = returned_entries.get(pdb_id)
        if entry is None:
            if errors:
                normalized = _failure_metadata(
                    pdb_id=pdb_id,
                    status="graphql_error",
                    warning="; ".join(errors),
                    retrieved_at=retrieved_at,
                )
            else:
                normalized = _failure_metadata(
                    pdb_id=pdb_id,
                    status="pdb_not_found",
                    warning="RCSB returned no entry for the requested PDB ID.",
                    retrieved_at=retrieved_at,
                )
        else:
            normalized = normalize_rcsb_entry(
                entry,
                retrieved_at=retrieved_at,
                graphql_errors=errors,
            )
        _write_json(cache_path / f"{pdb_id}.json", normalized)
        metadata[pdb_id] = normalized


def _post_json_with_retries(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
    *,
    max_retries: int,
    initial_backoff_seconds: float,
) -> Mapping[str, Any]:
    if max_retries < 0:
        raise ValueError("max_retries must be non-negative")
    if initial_backoff_seconds < 0:
        raise ValueError("initial_backoff_seconds must be non-negative")

    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": RCSB_USER_AGENT,
        },
        method="POST",
    )
    delay = initial_backoff_seconds
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                try:
                    return json.loads(response.read().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RCSBGraphQLError(
                        "RCSB GraphQL returned an invalid JSON response"
                    ) from exc
        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt == max_retries:
                detail = _http_error_detail(exc)
                raise RCSBGraphQLError(
                    f"RCSB GraphQL HTTP {exc.code}: {detail}",
                    status_code=exc.code,
                ) from exc
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            time.sleep(retry_after if retry_after is not None else delay)
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_retries:
                raise RCSBGraphQLError(
                    f"RCSB GraphQL network failure: {exc}"
                ) from exc
            time.sleep(delay)
        delay *= 2
    raise AssertionError("unreachable retry loop")


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        body = error.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    return body[:500] or str(error.reason)


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def _read_reusable_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        cached.get("cache_schema_version") != CACHE_SCHEMA_VERSION
        or cached.get("query_sha256") != RCSB_QUERY_SHA256
    ):
        return None
    return cached


def _write_raw_batch_cache(
    raw_cache_dir: Path,
    pdb_ids: Sequence[str],
    response: Mapping[str, Any],
    retrieved_at: str,
    *,
    endpoint: str,
) -> None:
    request_digest = sha256(
        (retrieved_at + "\0" + "\0".join(pdb_ids)).encode("utf-8")
    ).hexdigest()[:16]
    filename = f"{retrieved_at.replace(':', '').replace('+', '_')}_{request_digest}.json"
    _write_json(
        raw_cache_dir / filename,
        {
            "retrieved_at": retrieved_at,
            "endpoint": endpoint,
            "query": RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY,
            "query_sha256": RCSB_QUERY_SHA256,
            "requested_pdb_ids": list(pdb_ids),
            "response": dict(response),
        },
    )


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _build_entity_ranking_lookup(
    entity_metadata: pl.DataFrame,
) -> dict[str, dict[str, dict[str, Any]]]:
    required = {
        "pdb_id",
        "rcsb_polymer_entity_id",
        "entity_description",
        "organism_names",
        "organism_taxonomy_ids",
        "uniprot_accession",
        "entry_mapping_status",
    }
    missing = required.difference(entity_metadata.columns)
    if missing:
        raise ValueError(f"Entity metadata is missing columns: {sorted(missing)}")

    lookup: dict[str, dict[str, dict[str, Any]]] = {}
    for row in entity_metadata.iter_rows(named=True):
        accession = _optional_string(row.get("uniprot_accession"))
        pdb_id = canonicalize_pdb_id(row.get("pdb_id"))
        if accession is None or pdb_id is None:
            continue
        relation = lookup.setdefault(pdb_id, {}).setdefault(
            accession,
            {
                "protein_names": set(),
                "organisms": set(),
                "organism_taxonomy_ids": set(),
                "rcsb_polymer_entity_ids": set(),
                "entry_mapping_status": row["entry_mapping_status"],
            },
        )
        if row.get("entity_description"):
            relation["protein_names"].add(str(row["entity_description"]))
        relation["organisms"].update(row.get("organism_names") or [])
        relation["organism_taxonomy_ids"].update(
            row.get("organism_taxonomy_ids") or []
        )
        if row.get("rcsb_polymer_entity_id"):
            relation["rcsb_polymer_entity_ids"].add(
                str(row["rcsb_polymer_entity_id"])
            )

    for accessions in lookup.values():
        for relation in accessions.values():
            for key in (
                "protein_names",
                "organisms",
                "organism_taxonomy_ids",
                "rcsb_polymer_entity_ids",
            ):
                relation[key] = sorted(relation[key])
    return lookup


def _build_candidate_index_lookup(
    ranked_pockets: pl.DataFrame,
    candidate_index: pl.DataFrame | None,
    *,
    candidate_key_column: str,
) -> dict[str, dict[str, Any]] | None:
    """Validate and index an optional candidate-library sidecar.

    The validation prevents an exact LMDB key from being interpreted against
    the wrong library, rejects duplicate keys, and requires every ranked row
    to resolve before protein aggregation begins.
    """

    if candidate_index is None:
        return None
    if candidate_key_column not in ranked_pockets.columns:
        raise ValueError(
            "Ranked pocket table must contain "
            f"{candidate_key_column!r} when candidate_index is supplied"
        )
    missing = set(CANDIDATE_INDEX_SCHEMA).difference(candidate_index.columns)
    if missing:
        raise ValueError(f"Candidate index is missing columns: {sorted(missing)}")

    lookup: dict[str, dict[str, Any]] = {}
    library_digests: set[str] = set()
    declared_entry_counts: set[int] = set()
    for row in candidate_index.iter_rows(named=True):
        key = _optional_string(row.get("candidate_lmdb_key"))
        if key is None:
            raise ValueError("Candidate index contains a null or empty LMDB key")
        if key in lookup:
            raise ValueError(f"Candidate index contains duplicate LMDB key {key!r}")
        digest = _optional_string(row.get("candidate_library_sha256"))
        if digest is None:
            raise ValueError("Candidate index contains an empty library digest")
        library_digests.add(digest)
        declared_entry_counts.add(int(row["candidate_library_entries"]))
        lookup[key] = row

    if len(library_digests) > 1:
        raise ValueError("Candidate index combines more than one candidate library")
    if len(declared_entry_counts) > 1:
        raise ValueError("Candidate index has inconsistent library entry counts")
    if declared_entry_counts and declared_entry_counts != {candidate_index.height}:
        raise ValueError(
            "Candidate index entry count does not match the number of index rows"
        )

    ranked_keys = []
    for value in ranked_pockets[candidate_key_column].to_list():
        key = _optional_string(value)
        if key is None:
            raise ValueError("Ranked pocket table contains a null candidate LMDB key")
        ranked_keys.append(key)
    unresolved = sorted(set(ranked_keys).difference(lookup))
    if unresolved:
        raise ValueError(
            "Ranked pocket table contains keys absent from candidate index: "
            f"{unresolved}"
        )
    return lookup


def _validate_pdb_ids(pdb_ids: Sequence[str]) -> list[str]:
    canonical = []
    for pdb_id in pdb_ids:
        normalized = canonicalize_pdb_id(pdb_id)
        if normalized is None:
            raise ValueError(f"Not a classic PDB ID: {pdb_id!r}")
        canonical.append(normalized)
    return _ordered_unique(canonical)


def _graphql_error_messages(errors: object) -> list[str]:
    if not isinstance(errors, list):
        return []
    messages = []
    for error in errors:
        if isinstance(error, Mapping):
            message = str(error.get("message") or error)
            path = error.get("path")
            if path:
                message = f"{message} (path={path})"
        else:
            message = str(error)
        messages.append(message)
    return _ordered_unique(messages)


def _deduplicate_mappings(mappings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduplicated = []
    seen = set()
    for mapping in mappings:
        identity = tuple(sorted(mapping.items()))
        if identity not in seen:
            seen.add(identity)
            deduplicated.append(mapping)
    return deduplicated


def _ordered_unique(values: Iterable[Any]) -> list[Any]:
    output = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def _string_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _sort_lmdb_keys(keys: list[bytes]) -> list[bytes]:
    try:
        return sorted(keys, key=lambda key: int(key.decode("ascii")))
    except (UnicodeDecodeError, ValueError):
        return sorted(keys)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CANDIDATE_INDEX_SCHEMA",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CANDIDATE_INDEX_PATH",
    "DEFAULT_ENTITY_METADATA_PATH",
    "DEFAULT_GRAPHQL_BATCH_SIZE",
    "DEFAULT_PDB_METADATA_PATH",
    "ENTITY_METADATA_SCHEMA",
    "PDB_ID_RE",
    "PDB_METADATA_SCHEMA",
    "PROTEIN_RANKING_SCHEMA",
    "RCSB_ENTRY_PROTEIN_MAPPINGS_QUERY",
    "RCSB_GRAPHQL_ENDPOINT",
    "RCSBGraphQLError",
    "UNAMBIGUOUS_MAPPING_STATUSES",
    "aggregate_pocket_scores_by_protein",
    "build_candidate_pocket_index_frame",
    "build_entity_uniprot_frame",
    "build_pdb_metadata_frame",
    "build_pdb_uniprot_cache",
    "build_uniprot_metadata_sidecars",
    "canonicalize_pdb_id",
    "collect_lmdb_rows",
    "collect_unique_pdb_ids",
    "determine_entry_mapping_status",
    "fetch_rcsb_graphql_batch",
    "is_pdb_id",
    "iter_lmdb_records",
    "normalize_rcsb_entry",
    "utc_now_iso",
]
