#!/usr/bin/env python3
"""Copy the 18 candidates plus Nanochat from Iopsstor to Eiger-visible Capstor.

This is an xfer-only bridge.  It streams and hashes each file once, writes
per-file immutable progress receipts for safe restart, and emits a narrowed
acquisition receipt with destination stat identities and the original pinned
Hugging Face SHA-256 values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Sequence


from agent1_v5_pipeline import canonical_json, load_config, sha256_file, utc_now, write_json_atomic


SCHEMA = "agent1_v5_capstor_acquisition_staging_v1"
ACQUISITION_SCHEMA = "full_cpt_acquisition_receipt_v1"
CHUNK_BYTES = 16 * 1024 * 1024


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def safe_relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe acquisition artifact path: {value!r}")
    return Path(*pure.parts)


def stat_binding(path: Path) -> dict[str, int]:
    stat_result = path.stat()
    return {
        "size": stat_result.st_size,
        "device": stat_result.st_dev,
        "inode": stat_result.st_ino,
        "mtime_ns": stat_result.st_mtime_ns,
        "ctime_ns": stat_result.st_ctime_ns,
    }


def copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != expected_sha256:
            raise ValueError(f"existing staged file differs from expected content: {destination}")
        return
    temporary = destination.with_name(f".{destination.name}.partial-{os.getpid()}")
    hasher = hashlib.sha256()
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(CHUNK_BYTES):
                hasher.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if hasher.hexdigest() != expected_sha256:
            raise ValueError(f"source bytes differ from pinned SHA-256: {source}")
        os.chmod(temporary, 0o600)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    source_receipt = read_object(args.acquisition_receipt)
    if source_receipt.get("schema_version") != ACQUISITION_SCHEMA or source_receipt.get("status") != "passed":
        raise ValueError("source acquisition receipt is not passed")
    by_id = {str(row["source_id"]): row for row in source_receipt["sources"]}
    selected_ids = [*config["sources"], "nanochat_base"]
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    staged_sources = []
    total_files = 0
    total_bytes = 0
    for source_id in selected_ids:
        if source_id not in by_id:
            raise ValueError(f"acquisition receipt lacks {source_id}")
        source = by_id[source_id]
        staged_files = []
        revision = str(source["revision"])
        local_root = output_root / source_id / revision
        for row in source["files"]:
            source_path = Path(str(row["local_path"])).resolve()
            if not source_path.is_file() or source_path.is_symlink():
                raise ValueError(f"source file is missing or is a symlink: {source_path}")
            source_stat = stat_binding(source_path)
            for name in ("size", "device", "inode", "mtime_ns", "ctime_ns"):
                if int(row.get(name, -1)) != source_stat[name]:
                    raise ValueError(f"source acquisition stat drift ({name}): {source_path}")
            expected = str(row["expected_hash"])
            if row.get("hash_kind") not in {"lfs_sha256", "sha256"} or len(expected) != 64:
                raise ValueError(f"unsupported source hash binding: {source_path}")
            destination = local_root / safe_relative(str(row["path"]))
            progress_key = hashlib.sha256(
                f"{source_id}\0{revision}\0{row['path']}".encode("utf-8")
            ).hexdigest()
            progress_path = output_root / ".receipts" / f"{progress_key}.json"
            if progress_path.exists():
                progress = read_object(progress_path)
                if (
                    progress.get("schema_version") != SCHEMA
                    or progress.get("expected_sha256") != expected
                    or progress.get("destination") != str(destination)
                    or progress.get("stat") != stat_binding(destination)
                ):
                    raise ValueError(f"staging progress receipt drift: {progress_path}")
            else:
                copy_verified(source_path, destination, expected)
                write_json_atomic(
                    progress_path,
                    {
                        "schema_version": SCHEMA,
                        "status": "passed",
                        "source_id": source_id,
                        "revision": revision,
                        "artifact_path": row["path"],
                        "expected_sha256": expected,
                        "destination": str(destination),
                        "stat": stat_binding(destination),
                    },
                )
            binding = {
                **row,
                "local_path": str(destination),
                **stat_binding(destination),
            }
            staged_files.append(binding)
            total_files += 1
            total_bytes += int(binding["size"])
        staged_sources.append(
            {
                **source,
                "local_root": str(local_root),
                "files": staged_files,
                "selected_file_count": len(staged_files),
                "selected_bytes": sum(int(row["size"]) for row in staged_files),
            }
        )
    result: dict[str, Any] = {
        "schema_version": ACQUISITION_SCHEMA,
        "status": "passed",
        "created_at": utc_now(),
        "destination": str(output_root),
        "sources": staged_sources,
        "sources_config": str(args.config.resolve()),
        "sources_config_sha256": sha256_file(args.config),
        "content_verification": "streamed source SHA-256 equals pinned HF LFS/blob SHA-256",
        "staging": {
            "schema_version": SCHEMA,
            "source_receipt": str(args.acquisition_receipt.resolve()),
            "source_receipt_sha256": sha256_file(args.acquisition_receipt),
            "selected_source_ids": selected_ids,
            "files": total_files,
            "bytes": total_bytes,
        },
    }
    write_json_atomic(args.output_receipt, result)
    print(canonical_json({"ok": True, "sources": len(staged_sources), "files": total_files, "bytes": total_bytes}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
