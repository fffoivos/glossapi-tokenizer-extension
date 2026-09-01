#!/usr/bin/env python3
"""Fail-closed finalizer for one split GreekMMLU plus document evaluation."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file_binding(row: dict, *, label: str) -> Path:
    path = Path(row["path"]).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(row["bytes"])
        or sha256_file(path) != row["sha256"]
    ):
        raise ValueError(f"{label} file binding failed: {path}")
    return path


def atomic_write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--iteration-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    parser.add_argument("--greekmmlu-receipt", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--per-document-root", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "iteration_root", "attempt_root", "greekmmlu_receipt",
        "validation_manifest", "per_document_root",
    ):
        setattr(args, name, getattr(args, name).resolve())

    if args.attempt_root.parent != args.iteration_root:
        raise ValueError("attempt root is not a child of iteration root")
    greek = read_json(args.greekmmlu_receipt)
    if (
        greek.get("schema_version") != "exact_checkpoint_native_greekmmlu_receipt_v1"
        or greek.get("status") != "completed"
        or int(greek.get("checkpoint", {}).get("iteration", -1)) != args.iteration
    ):
        raise ValueError("GreekMMLU receipt is not the exact completed checkpoint")

    manifest = read_json(args.validation_manifest)
    if (
        manifest.get("schema_version") != "apertus_full_8b_validation_manifest_v1"
        or manifest.get("status") != "frozen"
        or len(manifest.get("panels", [])) != 13
    ):
        raise ValueError("validation manifest is not the frozen 13-panel contract")

    receipts = []
    for panel in manifest["panels"]:
        name = str(panel["name"])
        receipt_path = args.per_document_root / f"{name}.receipt.json"
        receipt = read_json(receipt_path)
        if (
            receipt.get("schema_version") != "apertus_per_document_validation_v1"
            or receipt.get("status") != "completed"
        ):
            raise ValueError(f"incomplete per-document receipt: {name}")
        input_path = require_file_binding(receipt["input"], label=f"{name} input")
        expected_input = Path(panel["raw_jsonl"]["path"]).resolve()
        if input_path != expected_input or receipt["input"]["sha256"] != panel["raw_jsonl"]["sha256"]:
            raise ValueError(f"per-document input drift: {name}")
        require_file_binding(receipt["output"], label=f"{name} output")
        if int(receipt.get("aggregate", {}).get("documents", 0)) <= 0:
            raise ValueError(f"empty per-document receipt: {name}")
        receipts.append({
            "panel": name,
            "path": str(receipt_path),
            "sha256": sha256_file(receipt_path),
        })

    canonical = {
        "schema_version": "apertus_full_8b_authoritative_checkpoint_evaluation_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "iteration": args.iteration,
        "attempt": args.attempt,
        "attempt_root": str(args.attempt_root),
        "greekmmlu_receipt": str(args.greekmmlu_receipt),
        "greekmmlu_receipt_sha256": sha256_file(args.greekmmlu_receipt),
        "per_document_root": str(args.per_document_root),
        "per_document_receipts": receipts,
        "validation_manifest": {
            "path": str(args.validation_manifest),
            "sha256": sha256_file(args.validation_manifest),
        },
        "resource_policy": "split_one_node_debug_jobs_with_90_total_node_minute_cap",
    }
    atomic_write_new(args.iteration_root / "authoritative_attempt.json", canonical)
    print(json.dumps({"ok": True, "iteration": args.iteration, "panels": len(receipts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
