#!/usr/bin/env python3
"""Verify curriculum v2 train/validation outputs and held-out id exclusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import orjson
import pyarrow.parquet as pq


TRAIN_BINS = ("hplt_only", "glossapi_only", "replay_only")
TOKS = ("base", "ext")
NEW_VALS = ("hplt", "openarchives", "greek_phd")
FORGET_VALS = ("english", "de", "ru", "zh", "code", "old_greek")


def read_ids(path: Path) -> set[str]:
    table = pq.read_table(path, columns=["doc_id"])
    return set(table.column("doc_id").to_pylist())


def require_file(path: Path) -> dict[str, object]:
    ok = path.is_file() and path.stat().st_size > 0
    return {"path": str(path), "ok": ok, "bytes": path.stat().st_size if path.exists() else 0}


def scan_jsonl(path: Path, new_ids: set[str], forget_ids: set[str]) -> dict[str, object]:
    rows = 0
    missing_doc_id = 0
    new_hits: list[str] = []
    forget_hits: list[str] = []
    with path.open("rb") as fh:
        for line in fh:
            if not line.strip():
                continue
            rows += 1
            doc_id = orjson.loads(line).get("doc_id")
            if doc_id is None:
                missing_doc_id += 1
                continue
            if doc_id in new_ids and len(new_hits) < 20:
                new_hits.append(doc_id)
            if doc_id in forget_ids and len(forget_hits) < 20:
                forget_hits.append(doc_id)
            if rows % 1_000_000 == 0:
                print(f"[scan] {path.name}: {rows:,} rows", flush=True)
    return {
        "path": str(path),
        "rows": rows,
        "missing_doc_id": missing_doc_id,
        "new_holdout_overlap_sample": new_hits,
        "forget_holdout_overlap_sample": forget_hits,
        "new_holdout_overlap_count_capped": len(new_hits),
        "forget_holdout_overlap_count_capped": len(forget_hits),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    stage = args.stage
    megout = stage / "megatron"
    output = args.output or stage / "verify_curriculum_outputs.json"

    expected: list[dict[str, object]] = []
    for name in TRAIN_BINS:
        for tok in TOKS:
            for ext in ("bin", "idx"):
                expected.append(require_file(megout / f"{name}_{tok}_text_document.{ext}"))
    for name in NEW_VALS:
        for tok in TOKS:
            for ext in ("bin", "idx"):
                expected.append(require_file(megout / f"val_{name}_{tok}_text_document.{ext}"))
    for name in FORGET_VALS:
        for tok in TOKS:
            for ext in ("bin", "idx"):
                expected.append(require_file(megout / f"val_forget_{name}_{tok}_text_document.{ext}"))

    missing = [item for item in expected if not item["ok"]]
    if missing:
        report = {"ok": False, "missing": missing}
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise SystemExit(f"missing expected files: {len(missing)}")

    new_ids = read_ids(stage / "val_holdout_ids.parquet")
    forget_ids = read_ids(stage / "forget_holdout_ids.parquet")

    scans = [
        scan_jsonl(stage / f"{name}_final.jsonl", new_ids, forget_ids)
        for name in TRAIN_BINS
    ]
    overlap_failures = [
        scan
        for scan in scans
        if scan["new_holdout_overlap_sample"] or scan["forget_holdout_overlap_sample"]
    ]
    missing_doc_id = [scan for scan in scans if scan["missing_doc_id"]]

    report = {
        "ok": not overlap_failures and not missing_doc_id,
        "expected_files": expected,
        "new_holdout_ids": len(new_ids),
        "forget_holdout_ids": len(forget_ids),
        "scans": scans,
        "overlap_failures": overlap_failures,
        "missing_doc_id_failures": missing_doc_id,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": report["ok"], "output": str(output)}, sort_keys=True), flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
