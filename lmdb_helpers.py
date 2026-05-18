"""Helpers for reading and writing pickle-backed LMDB record files."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable

import lmdb


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


def loads_lmdb_record(value: bytes) -> dict[str, Any]:
    """Deserialize one pickled LMDB record value."""

    return pickle.loads(value)


def _loads_lmdb_record(value: bytes) -> dict[str, Any]:
    return loads_lmdb_record(value)


__all__ = ["loads_lmdb_record", "read_lmdb_records", "write_lmdb_records"]
