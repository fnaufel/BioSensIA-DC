"""Helpers for reading and writing pickle-backed LMDB record files."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import lmdb


def read_lmdb_records(
    path: str | Path,
    head_n: int | None = None,
) -> list[dict[str, Any]]:
    """Read pickled DrugCLIP LMDB records.

    The same helper works for DrugCLIP pocket and molecule LMDB files because
    both store pickled dictionaries under numeric keys.
    When ``head_n`` is set, dense numeric-key LMDBs are read by direct key
    lookup so only the requested prefix is deserialized.
    """

    if head_n is not None:
        if head_n < 0:
            raise ValueError("head_n must be greater than or equal to 0")
        if head_n == 0:
            return []

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
            if head_n is not None:
                return _read_lmdb_record_head(
                    transaction,
                    head_n,
                    entry_count=env.stat()["entries"],
                )

            keys = list(transaction.cursor().iternext(values=False))
            if _has_numeric_lmdb_keys(keys):
                keys = sorted(keys, key=lambda key: int(key.decode("ascii")))
            return [
                loads_lmdb_record(value)
                for key in keys
                if (value := transaction.get(key)) is not None
            ]
    finally:
        env.close()


def write_lmdb_records(
    records: Iterable[dict[str, Any]],
    output_path: str | Path,
    *,
    overwrite: bool,
    map_size: int,
) -> None:
    """Write pickled records to an LMDB with numeric ASCII keys."""

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


def _has_numeric_lmdb_keys(keys: list[bytes]) -> bool:
    if not keys:
        return False
    try:
        return all(key.decode("ascii").isdigit() for key in keys)
    except UnicodeDecodeError:
        return False


def _read_lmdb_record_head(
    transaction: lmdb.Transaction,
    head_n: int,
    *,
    entry_count: int,
) -> list[dict[str, Any]]:
    records = _read_dense_numeric_lmdb_record_head(
        transaction,
        head_n,
        entry_count=entry_count,
    )
    if records is not None:
        return records

    keys = list(transaction.cursor().iternext(values=False))
    if _has_numeric_lmdb_keys(keys):
        keys = sorted(keys, key=lambda key: int(key.decode("ascii")))

    records = []
    for key in keys:
        value = transaction.get(key)
        if value is not None:
            records.append(loads_lmdb_record(value))
        if len(records) >= head_n:
            break
    return records


def _read_dense_numeric_lmdb_record_head(
    transaction: lmdb.Transaction,
    head_n: int,
    *,
    entry_count: int,
) -> list[dict[str, Any]] | None:
    if entry_count == 0:
        return []

    records = []
    for index in range(entry_count):
        if len(records) >= head_n:
            break
        value = transaction.get(str(index).encode("ascii"))
        if value is None:
            return None
        records.append(loads_lmdb_record(value))
    return records


def loads_lmdb_record(value: bytes) -> dict[str, Any]:
    """Deserialize one pickled LMDB record value."""

    return pickle.loads(value)


def _loads_lmdb_record(value: bytes) -> dict[str, Any]:
    return loads_lmdb_record(value)


__all__ = ["loads_lmdb_record", "read_lmdb_records", "write_lmdb_records"]
