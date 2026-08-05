#!/usr/bin/env python3
"""Freeze exact text hashes for every document selected by the full-8B run."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np


CATALOG_DTYPE = np.dtype(
    [
        ("pool", "u1"),
        ("task_index", "<u4"),
        ("document_index", "<u4"),
        ("tokens", "<u4"),
        ("identity", "V16"),
        ("order", "V16"),
    ],
    align=False,
)
CONTENT_DTYPE = np.dtype(
    [("content", "V32"), ("pool", "u1"), ("row", "<u8"), ("identity", "V16")],
    align=False,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(path)
    return value


def file_receipt(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if rows is not None:
        value["rows"] = rows
    return value


def extract_selected_hashes(
    pool: str,
    catalog_path: Path,
    tasks: dict[int, dict[str, Any]],
    output: BinaryIO,
) -> tuple[int, int, list[dict[str, Any]]]:
    source_catalog = np.memmap(catalog_path, mode="r", dtype=CATALOG_DTYPE)
    catalog = np.array(source_catalog, copy=True)
    del source_catalog
    # Selection catalogs retain the seeded schedule order. Ledger reconstruction
    # needs source-local order, so derive it explicitly rather than assuming it.
    catalog.sort(order=("task_index", "document_index"), kind="stable")
    if catalog.size > 1 and np.any(
        (catalog["task_index"][1:] == catalog["task_index"][:-1])
        & (catalog["document_index"][1:] <= catalog["document_index"][:-1])
    ):
        raise ValueError(f"duplicate or unordered source-local catalog row: {pool}")
    written = 0
    tokens = 0
    bindings: list[dict[str, Any]] = []
    start = 0
    while start < catalog.size:
        task_index = int(catalog[start]["task_index"])
        end = start + 1
        while end < catalog.size and int(catalog[end]["task_index"]) == task_index:
            end += 1
        rows = catalog[start:end]
        indexes = rows["document_index"].astype(np.uint64, copy=False)
        if indexes.size > 1 and np.any(indexes[1:] <= indexes[:-1]):
            raise ValueError(f"source-local document order drift: {pool}/{task_index}")
        task = tasks[task_index]
        if task["pool"] != pool:
            raise ValueError("catalog/task pool drift")
        manifest_path = Path(task["source_manifest"]["path"])
        manifest = read_json(manifest_path)
        ledger = Path(manifest["outputs"]["retained_ledger"]["path"])
        cursor = 0
        with ledger.open("r", encoding="utf-8") as handle:
            for document_index, line in enumerate(handle):
                if cursor == indexes.size:
                    break
                if document_index != int(indexes[cursor]):
                    continue
                row = json.loads(line)
                digest = bytes.fromhex(str(row["text_sha256"]))
                if len(digest) != 32 or int(row["tokens"]) != int(rows[cursor]["tokens"]):
                    raise ValueError(f"selected ledger/catalog drift: {pool}/{task_index}/{document_index}")
                identity = hashlib.sha256(
                    str(row["doc_id"]).encode("utf-8") + b"\0" + digest
                ).digest()[:16]
                if identity != bytes(rows[cursor]["identity"]):
                    raise ValueError("selected document identity drift")
                output.write(digest)
                written += 1
                tokens += int(row["tokens"])
                cursor += 1
        if cursor != indexes.size:
            raise ValueError(f"selected ledger rows are missing: {pool}/{task_index}")
        bindings.append(
            {
                "task_index": task_index,
                "selected_documents": int(indexes.size),
                "source_manifest": task["source_manifest"],
                "retained_ledger": file_receipt(ledger, rows=int(manifest["outputs"]["retained_ledger"]["rows"])),
            }
        )
        start = end
    return written, tokens, bindings


def write_unique_sorted(arrays: list[np.ndarray], path: Path) -> tuple[int, int]:
    combined = np.concatenate(arrays)
    rows = int(combined.size)
    combined.sort(kind="quicksort")
    unique_rows = 0
    with path.open("wb") as output:
        chunk_rows = 1_000_000
        previous: bytes | None = None
        for start in range(0, rows, chunk_rows):
            chunk = combined[start : start + chunk_rows]
            keep = np.ones(chunk.size, dtype=bool)
            if chunk.size:
                keep[1:] = chunk[1:] != chunk[:-1]
                if previous is not None and bytes(chunk[0]) == previous:
                    keep[0] = False
                previous = bytes(chunk[-1])
            selected = chunk[keep]
            output.write(selected.tobytes(order="C"))
            unique_rows += int(selected.size)
        output.flush()
        os.fsync(output.fileno())
    return rows, unique_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-receipt", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    pool = read_json(args.pool_receipt)
    if pool.get("schema_version") != "apertus_schedule_pool_corpus_v1" or pool.get("status") != "completed":
        raise ValueError("pool corpus is not complete")
    tasks = {int(row["task_index"]): row for row in pool["tasks"]}
    selected_arrays: list[np.ndarray] = []
    pool_outputs: dict[str, Any] = {}
    for pool_name in ("foreign_replay", "old_greek_replay"):
        catalog_path = args.stage_root / "inventory/catalog" / f"{pool_name}.source_local_selected.catalog45"
        raw_path = args.output_dir / f"{pool_name}.selected.sha32"
        with raw_path.open("wb") as output:
            documents, tokens, bindings = extract_selected_hashes(pool_name, catalog_path, tasks, output)
            output.flush()
            os.fsync(output.fileno())
        selected = np.memmap(raw_path, mode="r", dtype="V32")
        if int(selected.size) != documents:
            raise RuntimeError("selected content row accounting drift")
        selected_arrays.append(selected)
        pool_outputs[pool_name] = {
            "documents": documents,
            "tokens": tokens,
            "catalog": file_receipt(catalog_path, rows=documents),
            "content_hashes": file_receipt(raw_path, rows=documents),
            "source_tasks": bindings,
        }
    modern_path = args.stage_root / "inventory/raw/modern.content57"
    modern = np.memmap(modern_path, mode="r", dtype=CONTENT_DTYPE)
    modern_hashes = modern["content"]
    combined_path = args.output_dir / "selected_training_content.sorted.unique.sha32"
    input_rows, unique_rows = write_unique_sorted([modern_hashes, *selected_arrays], combined_path)
    payload = {
        "schema_version": "apertus_full_8b_selected_training_content_v1",
        "status": "frozen",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pool_corpus_receipt": file_receipt(args.pool_receipt),
        "modern_content": file_receipt(modern_path, rows=int(modern.size)),
        "selected_non_modern": pool_outputs,
        "combined": {
            **file_receipt(combined_path, rows=unique_rows),
            "record_bytes": 32,
            "input_rows": input_rows,
            "duplicate_rows_collapsed": input_rows - unique_rows,
            "sort": "ascending_sha256_utf8_text",
        },
    }
    receipt = args.output_dir / "selected_training_content_receipt.json"
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "input_rows": input_rows, "unique_rows": unique_rows}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
