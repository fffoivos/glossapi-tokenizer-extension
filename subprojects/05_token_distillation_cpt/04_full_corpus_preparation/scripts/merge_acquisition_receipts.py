#!/usr/bin/env python3
"""Merge verified Hugging Face and MDC acquisitions into one downstream receipt."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_receipt(
    *,
    sources_path: Path,
    hf_path: Path,
    mdc_path: Path,
    destination_root: Path,
) -> dict[str, Any]:
    config = load_object(sources_path)
    hf = load_object(hf_path)
    mdc = load_object(mdc_path)
    errors: list[str] = []
    config_hash = sha256_file(sources_path)
    expected = {
        "nanochat_base",
        *(
            str(row["source_id"])
            for row in config.get("sources", [])
            if isinstance(row, dict) and row.get("source_id")
        ),
    }
    if hf.get("schema_version") != "full_cpt_acquisition_receipt_v1" or hf.get("status") != "passed":
        errors.append("Hugging Face acquisition receipt is not passed")
    if mdc.get("schema_version") != "full_cpt_mdc_acquisition_receipt_v1" or mdc.get("status") != "passed":
        errors.append("MDC acquisition receipt is not passed")
    for label, receipt in (("HF", hf), ("MDC", mdc)):
        if receipt.get("sources_config_sha256") != config_hash:
            errors.append(f"{label} acquisition receipt is not bound to current sources.json")
    if hf.get("code_commit") != mdc.get("code_commit"):
        errors.append("HF and MDC acquisition code commits differ")

    rows: dict[str, dict[str, Any]] = {}
    for label, receipt in (("HF", hf), ("MDC", mdc)):
        for row in receipt.get("sources", []):
            source_id = str(row.get("source_id", ""))
            if not source_id:
                errors.append(f"{label} receipt contains a source without source_id")
            elif source_id in rows:
                errors.append(f"source {source_id!r} appears in multiple acquisition receipts")
            else:
                rows[source_id] = dict(row)
    missing = expected - set(rows)
    if missing:
        errors.append(f"combined acquisition is missing configured sources: {sorted(missing)}")
    external_ids = {
        str(row["source_id"])
        for row in config.get("sources", [])
        if isinstance(row, dict)
        and row.get("acquisition_kind") == "mozilla_data_collective"
    }
    if external_ids - {
        str(row.get("source_id")) for row in mdc.get("sources", [])
    }:
        errors.append("one or more MDC routes were not supplied by the MDC receipt")
    destination_root = destination_root.resolve()
    for source_id, row in rows.items():
        for file in row.get("files", []):
            path = Path(str(file.get("local_path", ""))).resolve()
            try:
                path.relative_to(destination_root)
            except ValueError:
                errors.append(f"{source_id}: acquired file is outside destination root: {path}")
                continue
            if not path.is_file() or path.stat().st_size != int(file.get("size", -1)):
                errors.append(f"{source_id}: acquired file is missing or size-drifted: {path}")
    if errors:
        raise ValueError("acquisition receipt merge failed:\n- " + "\n- ".join(errors))
    return {
        "schema_version": "full_cpt_acquisition_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": hf["code_commit"],
        "destination": str(destination_root),
        "sources_config": str(sources_path.resolve()),
        "sources_config_sha256": config_hash,
        "component_receipts": [
            {"kind": "huggingface", "path": str(hf_path.resolve()), "sha256": sha256_file(hf_path)},
            {"kind": "mozilla_data_collective", "path": str(mdc_path.resolve()), "sha256": sha256_file(mdc_path)},
        ],
        "content_verification": "HF LFS/blob verification plus MDC archive and extracted-file SHA-256 verification",
        "sources": [rows[source_id] for source_id in sorted(rows)],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--hf-receipt", type=Path, required=True)
    parser.add_argument("--mdc-receipt", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        sources_path=args.sources,
        hf_path=args.hf_receipt,
        mdc_path=args.mdc_receipt,
        destination_root=args.destination_root,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps({"ok": True, "sources": len(receipt["sources"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
