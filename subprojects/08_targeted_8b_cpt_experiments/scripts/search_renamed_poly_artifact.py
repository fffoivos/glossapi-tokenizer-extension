#!/usr/bin/env python3
"""Find a renamed copy of the exact historical ``poly_train`` Parquet.

Discovery is read-only. The scan first uses byte bounds and the Parquet
``PAR1`` header/trailer signature rather than a filename extension, then uses
Parquet footer metadata and reads text/source columns only for 14,929-row
candidates. This includes content-addressed cache blobs with no ``.parquet``
suffix. It never materializes, reconstructs, renames, or rewrites corpus data.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


EXPECTED_ROWS = 14_929
EXPECTED_TEXT_CHARS = 409_101_812
EXPECTED_UTF8_BYTES = 802_061_905
EXPECTED_SOURCES = {
    "1000_prwta_xronia_ellhnikhs": 771,
    "Ekklisiastika_Keimena": 543,
    "Wikisource_Greek_texts": 2_738,
    "klasikh_arx_ell_grammateia": 540,
    "scholarios_graeca_patristic": 10_337,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_parquet_signature(path: Path) -> bool:
    """Return whether *path* has the standard Parquet header and trailer."""

    try:
        with path.open("rb") as stream:
            if stream.read(4) != b"PAR1":
                return False
            stream.seek(-4, os.SEEK_END)
            return stream.read(4) == b"PAR1"
    except (OSError, ValueError):
        return False


def parquet_paths(root: Path, minimum_bytes: int, maximum_bytes: int) -> list[Path]:
    """Return signature-identified Parquet paths without walking symlink dirs."""

    command = [
        "find",
        str(root),
        "-xdev",
        "(",
        "(",
        "-type",
        "f",
        "-size",
        f"+{minimum_bytes - 1}c",
        "-size",
        f"-{maximum_bytes + 1}c",
        ")",
        "-o",
        "-type",
        "l",
        ")",
        "-print0",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    paths: dict[tuple[int, int], Path] = {}
    for raw in result.stdout.split(b"\0"):
        if not raw:
            continue
        discovered = Path(os.fsdecode(raw))
        try:
            resolved = discovered.resolve(strict=True)
            stat = resolved.stat()
        except (FileNotFoundError, OSError, RuntimeError):
            continue
        if not resolved.is_file() or not minimum_bytes <= stat.st_size <= maximum_bytes:
            continue
        if has_parquet_signature(resolved):
            paths.setdefault((stat.st_dev, stat.st_ino), resolved)
    return sorted(paths.values(), key=str)


def inspect_candidate(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path)}
    try:
        stat = path.stat()
        result.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        parquet = pq.ParquetFile(path)
        rows = int(parquet.metadata.num_rows)
        result["rows"] = rows
        result["schema_columns"] = list(parquet.schema_arrow.names)
        if rows != EXPECTED_ROWS:
            result["classification"] = "row_count_mismatch"
            return result
        if not {"source_dataset", "text"}.issubset(parquet.schema_arrow.names):
            result["classification"] = "schema_mismatch"
            return result

        observed_rows = 0
        text_chars = 0
        utf8_bytes = 0
        sources: collections.Counter[str] = collections.Counter()
        for batch in parquet.iter_batches(
            columns=["source_dataset", "text"], batch_size=1_024, use_threads=True
        ):
            values = batch.to_pydict()
            for source, text_value in zip(
                values["source_dataset"], values["text"], strict=True
            ):
                text = "" if text_value is None else str(text_value)
                observed_rows += 1
                text_chars += len(text)
                utf8_bytes += len(text.encode("utf-8"))
                sources[str(source)] += 1
        result.update(
            observed_rows=observed_rows,
            text_chars=text_chars,
            utf8_bytes=utf8_bytes,
            source_counts=dict(sorted(sources.items())),
        )
        exact = (
            observed_rows == EXPECTED_ROWS
            and text_chars == EXPECTED_TEXT_CHARS
            and utf8_bytes == EXPECTED_UTF8_BYTES
            and dict(sources) == EXPECTED_SOURCES
        )
        result["classification"] = (
            "exact_frozen_split_match" if exact else "content_mismatch"
        )
        if exact:
            result["sha256"] = sha256_file(path)
        return result
    except Exception as exc:  # preserve evidence for unreadable candidates
        result["classification"] = "unreadable_parquet"
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-bytes", type=int, default=10_000_000)
    parser.add_argument("--maximum-bytes", type=int, default=2_000_000_000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {args.output}")
    if not 0 < args.minimum_bytes <= args.maximum_bytes:
        raise ValueError("invalid candidate byte bounds")

    unique: dict[str, Path] = {}
    roots: list[dict[str, Any]] = []
    for root in args.root:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(resolved)
        candidates = parquet_paths(resolved, args.minimum_bytes, args.maximum_bytes)
        roots.append({
            "path": str(resolved),
            "signature_identified_parquet_files": len(candidates),
        })
        for candidate in candidates:
            unique.setdefault(str(candidate), candidate)

    inspected = [inspect_candidate(path) for path in unique.values()]
    row_matches = [row for row in inspected if row.get("rows") == EXPECTED_ROWS]
    exact = [
        row for row in row_matches
        if row.get("classification") == "exact_frozen_split_match"
    ]
    unreadable = [
        row for row in inspected if row["classification"] == "unreadable_parquet"
    ]
    payload = {
        "schema_version": "targeted_8b_renamed_poly_artifact_search_v2",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "contract": {
            "discovery_only": True,
            "reconstruction_performed": False,
            "substitution_performed": False,
            "corpus_files_written": 0,
            "artifact_authority": "cscs",
        },
        "expected": {
            "rows": EXPECTED_ROWS,
            "text_chars": EXPECTED_TEXT_CHARS,
            "utf8_bytes": EXPECTED_UTF8_BYTES,
            "source_counts": EXPECTED_SOURCES,
        },
        "candidate_byte_bounds": [args.minimum_bytes, args.maximum_bytes],
        "discovery_rule": "PAR1_header_and_trailer_regardless_of_filename",
        "searched_roots": roots,
        "plausible_unique_parquet_files": len(inspected),
        "row_count_candidate_count": len(row_matches),
        "unreadable_candidate_count": len(unreadable),
        "exact_match_count": len(exact),
        "row_count_candidates": row_matches,
        "unreadable_candidates": unreadable,
        "exact_matches": exact,
    }
    write_atomic(args.output, payload)
    print(json.dumps({
        "ok": True,
        "plausible": len(inspected),
        "row_matches": len(row_matches),
        "exact_matches": len(exact),
        "output": str(args.output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
