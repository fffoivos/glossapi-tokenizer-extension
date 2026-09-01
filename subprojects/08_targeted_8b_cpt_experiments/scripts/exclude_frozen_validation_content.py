#!/usr/bin/env python3
"""Exclude exact frozen-validation content without deduplicating training rows."""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from contract_utils import file_binding, read_json, require, sha256_file, write_json_atomic


_HELDOUT_HASHES: set[str] = set()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path}:{number}: expected object")
            yield value


def freeze_heldout_hashes(manifest_path: Path) -> tuple[set[str], list[dict[str, Any]]]:
    manifest = read_json(manifest_path)
    require(manifest.get("schema_version") == "apertus_full_8b_validation_manifest_v1", "validation schema drift")
    require(manifest.get("status") == "frozen", "validation manifest is not frozen")
    hashes: set[str] = set()
    panels: list[dict[str, Any]] = []
    for panel in manifest.get("panels", []):
        raw = panel.get("raw_jsonl")
        require(isinstance(raw, dict), f"validation panel lacks raw_jsonl: {panel.get('name')}")
        path = Path(str(raw["path"]))
        require(path.is_file(), f"validation JSONL missing: {path}")
        require(path.stat().st_size == int(raw["bytes"]), f"validation bytes drift: {path}")
        require(sha256_file(path) == raw["sha256"], f"validation hash drift: {path}")
        rows = 0
        panel_hashes: set[str] = set()
        for row in _iter_jsonl(path):
            text = row.get("text")
            require(isinstance(text, str), f"validation text missing: {path}")
            panel_hashes.add(hashlib.sha256(text.encode("utf-8")).hexdigest())
            rows += 1
        require(rows == int(raw["rows"]), f"validation rows drift: {path}")
        hashes.update(panel_hashes)
        panels.append(
            {
                "name": str(panel["name"]),
                "raw_jsonl": file_binding(path),
                "documents": rows,
                "unique_exact_text_hashes": len(panel_hashes),
            }
        )
    require(len(panels) == 13, f"expected 13 validation panels, found {len(panels)}")
    return hashes, panels


def ledger_schema() -> pa.Schema:
    return pa.schema(
        [
            ("source_dataset", pa.string()),
            ("source_doc_id", pa.string()),
            ("input_text_sha256", pa.string()),
            ("action", pa.string()),
            ("reason", pa.string()),
        ]
    )


def worker_init(hashes: set[str]) -> None:
    global _HELDOUT_HASHES
    _HELDOUT_HASHES = hashes


def _temp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def process_file(task: tuple[str, str, str, str, str]) -> dict[str, Any]:
    input_name, relative, output_root, excluded_root, ledger_root = task
    input_path = Path(input_name)
    kept_path = Path(output_root) / relative
    excluded_path = Path(excluded_root) / relative
    ledger_path = Path(ledger_root) / relative
    parquet = pq.ParquetFile(input_path)
    required = {"source_dataset", "source_doc_id", "text"}
    require(required.issubset(parquet.schema_arrow.names), f"identity/text columns missing: {input_path}")
    for path in (kept_path, excluded_path, ledger_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    kept_temp, excluded_temp, ledger_temp = map(_temp, (kept_path, excluded_path, ledger_path))
    kept_writer = pq.ParquetWriter(kept_temp, parquet.schema_arrow, compression="zstd")
    excluded_writer = pq.ParquetWriter(excluded_temp, parquet.schema_arrow, compression="zstd")
    ledger_writer = pq.ParquetWriter(ledger_temp, ledger_schema(), compression="zstd")
    counts: collections.Counter[str] = collections.Counter()
    try:
        for batch in parquet.iter_batches(batch_size=1024, use_threads=False):
            kept: list[dict[str, Any]] = []
            excluded: list[dict[str, Any]] = []
            ledger: list[dict[str, Any]] = []
            for row in batch.to_pylist():
                text_hash = hashlib.sha256(str(row.get("text") or "").encode("utf-8")).hexdigest()
                action = "exclude" if text_hash in _HELDOUT_HASHES else "keep"
                ledger.append(
                    {
                        "source_dataset": str(row.get("source_dataset") or ""),
                        "source_doc_id": str(row.get("source_doc_id") or ""),
                        "input_text_sha256": text_hash,
                        "action": action,
                        "reason": "frozen_validation_exact_content" if action == "exclude" else "not_frozen_validation_content",
                    }
                )
                counts["input"] += 1
                if action == "exclude":
                    excluded.append(row)
                    counts["excluded"] += 1
                else:
                    kept.append(row)
                    counts["kept"] += 1
            if kept:
                kept_writer.write_table(pa.Table.from_pylist(kept, schema=parquet.schema_arrow))
            if excluded:
                excluded_writer.write_table(pa.Table.from_pylist(excluded, schema=parquet.schema_arrow))
            ledger_writer.write_table(pa.Table.from_pylist(ledger, schema=ledger_schema()))
    except BaseException:
        kept_writer.close()
        excluded_writer.close()
        ledger_writer.close()
        for path in (kept_temp, excluded_temp, ledger_temp):
            path.unlink(missing_ok=True)
        raise
    kept_writer.close()
    excluded_writer.close()
    ledger_writer.close()
    for source, destination in ((kept_temp, kept_path), (excluded_temp, excluded_path), (ledger_temp, ledger_path)):
        os.replace(source, destination)
    return {
        "relative_path": relative,
        "counts": dict(counts),
        "kept": file_binding(kept_path),
        "excluded": file_binding(excluded_path),
        "ledger": file_binding(ledger_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    require(not args.output_root.exists(), f"immutable output root exists: {args.output_root}")
    require(args.workers >= 1, "--workers must be positive")
    heldout_hashes, panels = freeze_heldout_hashes(args.validation_manifest)
    kept_root = args.output_root / "data"
    excluded_root = args.output_root / "excluded"
    ledger_root = args.output_root / "ledger"
    files = sorted(args.input.rglob("*.parquet"))
    require(files, f"no input Parquet files below {args.input}")
    tasks = [
        (str(path), path.relative_to(args.input).as_posix(), str(kept_root), str(excluded_root), str(ledger_root))
        for path in files
    ]
    created_root = False
    try:
        for root in (kept_root, excluded_root, ledger_root):
            root.mkdir(parents=True)
            created_root = True
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=mp.get_context("fork"),
            initializer=worker_init,
            initargs=(heldout_hashes,),
        ) as executor:
            receipts = list(executor.map(process_file, tasks, chunksize=1))
        totals: collections.Counter[str] = collections.Counter()
        for receipt in receipts:
            totals.update(receipt["counts"])
        require(totals["input"] == totals["kept"] + totals["excluded"], "row accounting mismatch")
        payload = {
            "schema_version": "targeted_8b_validation_content_exclusion_v1",
            "status": "completed",
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "input": str(args.input.resolve()),
            "output": str(kept_root.resolve()),
            "excluded": str(excluded_root.resolve()),
            "ledger": str(ledger_root.resolve()),
            "validation_manifest": file_binding(args.validation_manifest),
            "validation_panels": panels,
            "unique_frozen_validation_text_hashes": len(heldout_hashes),
            "counts": dict(totals),
            "files": sorted(receipts, key=lambda row: row["relative_path"]),
            "invariants": {
                "exact_utf8_content_match_only": True,
                "training_rows_compared_independently": True,
                "row_order_and_multiplicity_preserved_within_each_partition": True,
                "global_deduplication_performed": False,
                "near_deduplication_performed": False,
            },
        }
        write_json_atomic(args.output_root / "exclusion_manifest.json", payload)
    except BaseException:
        if created_root:
            shutil.rmtree(args.output_root, ignore_errors=True)
        raise
    print(json.dumps({"ok": True, "counts": dict(totals)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
