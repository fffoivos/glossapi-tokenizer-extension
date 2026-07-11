#!/usr/bin/env python3
"""Verify downloaded Parquet schemas against the tracked source registry.

This is a staging gate, not a normalizer. Non-Parquet sources are recorded as
adapter-required and must be validated by their source-specific normalizer.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
from pathlib import Path


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        os.unlink(temporary)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def registry_entries(config: dict) -> dict[str, dict]:
    return {
        "nanochat_base": config["base"],
        **{source["source_id"]: source for source in config["sources"]},
    }


def verify_source(source: dict, locked: dict, destination: Path) -> tuple[dict, list[str]]:
    source_id = locked["source_id"]
    local_dir = destination / source_id / locked["revision"]
    errors: list[str] = []
    parquet_files = [
        local_dir / row["path"]
        for row in locked["selected_files"]
        if row["path"].lower().endswith(".parquet")
    ]
    if not parquet_files:
        return (
            {
                "source_id": source_id,
                "status": "adapter_required_non_parquet",
                "selected_files": len(locked["selected_files"]),
                "note": "validate fields and rows in the source-specific streaming adapter",
            },
            errors,
        )

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised on Clariden
        raise RuntimeError("install pyarrow in the Phase-04 runtime") from exc

    candidate_text = set(source.get("text_columns", [])) | set(
        source.get("alternate_text_columns", [])
    )
    required_text = set(source.get("required_text_columns", []))
    candidate_ids = set(source.get("id_columns", []))
    schemas: list[dict[str, object]] = []
    total_rows = 0
    for path in parquet_files:
        if not path.is_file():
            errors.append(f"{source_id}: missing staged file {path}")
            continue
        parquet = pq.ParquetFile(path)
        columns = set(parquet.schema_arrow.names)
        present_text = sorted(candidate_text & columns)
        missing_required_text = sorted(required_text - columns)
        present_ids = sorted(candidate_ids & columns)
        if missing_required_text:
            errors.append(
                f"{source_id}:{path.name}: missing required text columns "
                f"{missing_required_text}"
            )
        if candidate_text and not present_text:
            errors.append(
                f"{source_id}:{path.name}: none of the candidate text columns are present: "
                f"{sorted(candidate_text)}"
            )
        if candidate_ids and not present_ids:
            errors.append(
                f"{source_id}:{path.name}: none of the candidate id columns are present: {sorted(candidate_ids)}"
            )
        rows = parquet.metadata.num_rows
        total_rows += rows
        schemas.append(
            {
                "path": str(path),
                "rows": rows,
                "row_groups": parquet.num_row_groups,
                "columns": sorted(columns),
                "present_id_columns": present_ids,
                "text_columns": present_text,
            }
        )
    return (
        {
            "source_id": source_id,
            "status": "ok" if not errors else "error",
            "files": schemas,
            "rows": total_rows,
        },
        errors,
    )


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_json(args.sources)
    lock = load_json(args.lock)
    registry = registry_entries(config)
    reports: list[dict] = []
    errors: list[str] = []
    config_sha256 = sha256_file(args.sources)
    if lock.get("sources_config_sha256") != config_sha256:
        errors.append(
            "source lock/config mismatch: resolve a new lock from this exact sources.json before staging"
        )
    skipped_artifacts: list[str] = []
    for locked in lock["sources"]:
        source_id = locked["source_id"]
        source = registry.get(source_id)
        if source is None:
            skipped_artifacts.append(source_id)
            continue
        report, source_errors = verify_source(source, locked, args.destination)
        reports.append(report)
        errors.extend(source_errors)

    result = {
        "schema_version": "full_cpt_staged_schema_audit_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "ok": not errors,
        "sources_config": str(args.sources.resolve()),
        "sources_config_sha256": config_sha256,
        "source_lock": str(args.lock.resolve()),
        "source_lock_sha256": sha256_file(args.lock),
        "destination": str(args.destination.resolve()),
        "sources": reports,
        "skipped_artifacts": skipped_artifacts,
        "errors": errors,
    }
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable schema audit: {args.output}")
    write_json_atomic(args.output, result)
    print(json.dumps({"ok": not errors, "sources": len(reports), "errors": len(errors), "output": str(args.output)}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
