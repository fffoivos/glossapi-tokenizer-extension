#!/usr/bin/env python3
"""Small, dependency-light helpers for the full-corpus finalization stages."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator


READ_CHUNK_BYTES = 8 * 1024 * 1024


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required manifest is missing: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(READ_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any], *, immutable: bool = True) -> None:
    if immutable and path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_output_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    return destination.with_name(f".{destination.name}.partial")


def discover_parquet(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Parquet root is missing: {root}")
    files = sorted(path for path in root.rglob("*.parquet") if not path.name.startswith("."))
    if not files:
        raise ValueError(f"no Parquet files found below {root}")
    return files


def iter_parquet_batches(
    files: Iterable[Path],
    *,
    columns: list[str] | None = None,
    batch_size: int = 2048,
) -> Iterator[tuple[Path, Any]]:
    import pyarrow.parquet as pq

    for path in files:
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        if columns is not None:
            missing = sorted(set(columns) - available)
            if missing:
                raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns, use_threads=False):
            yield path, batch


def parquet_file_receipt(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    import pyarrow.parquet as pq

    metadata = pq.ParquetFile(path).metadata
    return {
        "path": str(path.relative_to(relative_to) if relative_to is not None else path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
    }


def sql_string(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def sql_path_list(paths: Iterable[Path]) -> str:
    values = ",".join(sql_string(path.resolve()) for path in paths)
    if not values:
        raise ValueError("SQL path list cannot be empty")
    return f"[{values}]"


def configure_duckdb(connection: Any, *, temporary_directory: Path, memory_limit: str, threads: int) -> None:
    temporary_directory.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET temp_directory={sql_string(temporary_directory.resolve())}")
    connection.execute(f"SET memory_limit={sql_string(memory_limit)}")
    connection.execute(f"SET threads={max(1, int(threads))}")
    connection.execute("SET preserve_insertion_order=false")


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
