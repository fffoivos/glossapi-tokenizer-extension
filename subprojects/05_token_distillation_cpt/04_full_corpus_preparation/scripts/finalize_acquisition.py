#!/usr/bin/env python3
"""Create the immutable completion receipt for a locked Phase-04 acquisition.

The downloader performs the expensive LFS SHA-256 verification once. This
finalizer binds that verified download manifest, the schema audit and the exact
local paths/sizes into one small receipt. Downstream launch gates re-check the
receipt, paths and sizes without hashing the full corpus again.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import tempfile
from pathlib import Path


def load_object(path: Path) -> dict:
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


def under(root: Path, path: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def selected_by_config(path: str, includes: list[str], excludes: list[str]) -> bool:
    return (
        any(fnmatch.fnmatchcase(path, pattern) for pattern in includes)
        and not any(fnmatch.fnmatchcase(path, pattern) for pattern in excludes)
    )


def build_receipt(
    *,
    sources_path: Path,
    lock_path: Path,
    download_manifest_path: Path,
    schema_audit_path: Path,
    destination: Path,
    code_commit: str,
) -> dict:
    sources_config = load_object(sources_path)
    lock = load_object(lock_path)
    download = load_object(download_manifest_path)
    schema = load_object(schema_audit_path)
    destination = destination.resolve()
    errors: list[str] = []

    if len(code_commit) not in {40, 64} or any(character not in "0123456789abcdef" for character in code_commit):
        errors.append("code commit must be a lowercase full Git object ID")

    if lock.get("schema_version") != "full_cpt_sources_lock_v1":
        errors.append("unsupported source-lock schema")
    if download.get("schema_version") != "full_cpt_download_manifest_v1":
        errors.append("unsupported download-manifest schema")
    if schema.get("schema_version") != "full_cpt_staged_schema_audit_v1":
        errors.append("unsupported staged-schema schema")

    sources_sha256 = sha256_file(sources_path)
    lock_sha256 = sha256_file(lock_path)
    if lock.get("sources_config_sha256") != sources_sha256:
        errors.append("sources.json does not match the source lock")
    if download.get("source_lock_sha256") != lock_sha256:
        errors.append("download manifest does not match the source lock")
    if schema.get("source_lock_sha256") != lock_sha256:
        errors.append("schema audit does not match the source lock")
    if schema.get("sources_config_sha256") != sources_sha256:
        errors.append("schema audit does not match sources.json")
    if schema.get("ok") is not True:
        errors.append("schema audit did not pass")
    if Path(str(download.get("destination", ""))).resolve() != destination:
        errors.append("download manifest destination mismatch")
    if Path(str(schema.get("destination", ""))).resolve() != destination:
        errors.append("schema audit destination mismatch")

    locked_rows = lock.get("sources", [])
    downloaded_rows = download.get("sources", [])
    locked_sources = {str(row["source_id"]): row for row in locked_rows}
    downloaded_sources = {str(row["source_id"]): row for row in downloaded_rows}
    if len(locked_sources) != len(locked_rows):
        errors.append("source lock contains duplicate source IDs")
    if len(downloaded_sources) != len(downloaded_rows):
        errors.append("download manifest contains duplicate source IDs")
    if set(locked_sources) != set(downloaded_sources):
        errors.append(
            f"downloaded source IDs differ from lock: locked={sorted(locked_sources)}, "
            f"downloaded={sorted(downloaded_sources)}"
        )

    configured_sources = {
        "nanochat_base": sources_config.get("base", {}),
        "apertus_overlap_overlay": sources_config.get("apertus_overlap_overlay", {}),
        "modern_greek_148k_tokenizer": sources_config.get("tokenizer", {}),
        **{
            str(row["source_id"]): row
            for row in sources_config.get("sources", [])
            if isinstance(row, dict) and row.get("source_id")
        },
    }
    for source_id, locked in locked_sources.items():
        configured = configured_sources.get(source_id)
        if configured is None:
            errors.append(f"{source_id}: lock source is absent from sources.json")
            continue
        expected_metadata = {
            "repo_id": configured.get("repo_id"),
            "repo_type": configured.get("repo_type", "dataset"),
            "revision": configured.get("revision"),
            "role": configured.get("role", "artifact"),
        }
        for field, expected in expected_metadata.items():
            if locked.get(field) != expected:
                errors.append(f"{source_id}: locked {field} does not match sources.json")
        selected_files = locked.get("selected_files", [])
        selected_paths = [str(row.get("path")) for row in selected_files if isinstance(row, dict)]
        if len(selected_paths) != len(selected_files) or len(set(selected_paths)) != len(selected_paths):
            errors.append(f"{source_id}: lock has invalid or duplicate selected paths")
        includes = list(configured.get("include_globs", []))
        excludes = list(configured.get("exclude_globs", []))
        for row in selected_files:
            if not isinstance(row, dict):
                continue
            path = str(row.get("path", ""))
            relative = Path(path)
            if not path or relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{source_id}: lock contains an unsafe selected path {path!r}")
            elif not selected_by_config(path, includes, excludes):
                errors.append(f"{source_id}: locked path is outside tracked include/exclude globs: {path}")
            try:
                size = int(row.get("size", -1))
            except (TypeError, ValueError):
                size = -1
            if size < 0:
                errors.append(f"{source_id}:{path}: invalid locked size")
            if row.get("lfs_size") is not None and int(row["lfs_size"]) != size:
                errors.append(f"{source_id}:{path}: LFS size differs from locked file size")
            if not row.get("lfs_sha256") and not row.get("blob_id"):
                errors.append(f"{source_id}:{path}: no immutable content identifier")
        if locked.get("selected_file_count") != len(selected_files):
            errors.append(f"{source_id}: selected_file_count does not match selected_files")
        try:
            selected_bytes = sum(int(row["size"]) for row in selected_files)
        except (KeyError, TypeError, ValueError):
            selected_bytes = -1
        if locked.get("selected_bytes") != selected_bytes:
            errors.append(f"{source_id}: selected_bytes does not match selected_files")

    schema_reports = {
        str(row["source_id"]): row
        for row in schema.get("sources", [])
        if isinstance(row, dict) and row.get("source_id")
    }
    configured_data_ids = {
        "nanochat_base",
        *(
            str(row["source_id"])
            for row in sources_config.get("sources", [])
            if isinstance(row, dict) and row.get("source_id")
        ),
    }
    expected_schema_ids = set(locked_sources) & configured_data_ids
    if set(schema_reports) != expected_schema_ids:
        errors.append(
            f"schema-audit source IDs differ from locked data sources: expected={sorted(expected_schema_ids)}, "
            f"reported={sorted(schema_reports)}"
        )
    for source_id, report in schema_reports.items():
        if report.get("status") not in {"ok", "adapter_required_non_parquet"}:
            errors.append(f"{source_id}: schema audit has non-passing status {report.get('status')!r}")
    expected_skipped = set(locked_sources) - configured_data_ids
    actual_skipped = set(map(str, schema.get("skipped_artifacts", [])))
    if actual_skipped != expected_skipped:
        errors.append(
            f"schema-audit skipped artifacts differ from lock: expected={sorted(expected_skipped)}, "
            f"reported={sorted(actual_skipped)}"
        )

    receipt_sources: list[dict] = []
    for source_id, locked in sorted(locked_sources.items()):
        downloaded = downloaded_sources.get(source_id, {})
        selected = list(locked.get("selected_files", []))
        expected_lfs = sum(bool(row.get("lfs_sha256")) for row in selected)
        expected_git_blobs = sum(bool(row.get("blob_id")) and not row.get("lfs_sha256") for row in selected)
        if int(downloaded.get("files", -1)) != len(selected):
            errors.append(f"{source_id}: download file count does not match lock")
        if int(downloaded.get("bytes", -1)) != int(locked.get("selected_bytes", -2)):
            errors.append(f"{source_id}: download byte count does not match lock")
        if int(downloaded.get("lfs_sha256_verified", -1)) != expected_lfs:
            errors.append(f"{source_id}: not every locked LFS object was SHA-256 verified")
        if int(downloaded.get("git_blob_ids_verified", -1)) != expected_git_blobs:
            errors.append(f"{source_id}: not every locked non-LFS Git blob was verified")
        if downloaded.get("repo_id") != locked.get("repo_id"):
            errors.append(f"{source_id}: download repository does not match lock")
        if downloaded.get("repo_type") != locked.get("repo_type"):
            errors.append(f"{source_id}: download repository type does not match lock")
        if downloaded.get("revision") != locked.get("revision"):
            errors.append(f"{source_id}: download revision does not match lock")

        local_root = (destination / source_id / str(locked["revision"])).resolve()
        if Path(str(downloaded.get("local_dir", ""))).resolve() != local_root:
            errors.append(f"{source_id}: download local directory does not match locked destination")
        verified_rows = downloaded.get("verified_files", [])
        verified_by_path = {
            str(row["path"]): row
            for row in verified_rows
            if isinstance(row, dict) and row.get("path")
        }
        if len(verified_by_path) != len(verified_rows):
            errors.append(f"{source_id}: download manifest has duplicate or invalid verified-file rows")
        if set(verified_by_path) != {str(row.get("path")) for row in selected}:
            errors.append(f"{source_id}: verified-file path set does not match lock")
        files: list[dict] = []
        for row in selected:
            relative = Path(str(row["path"]))
            local_path = (local_root / relative).resolve()
            if not under(local_root, local_path):
                errors.append(f"{source_id}: locked path escapes source root: {relative}")
                continue
            if not local_path.is_file():
                errors.append(f"{source_id}: locked local file is missing: {local_path}")
                continue
            file_stat = local_path.stat()
            actual_size = file_stat.st_size
            expected_size = int(row.get("size", -1))
            if actual_size != expected_size:
                errors.append(
                    f"{source_id}: size changed after verification for {relative}: "
                    f"expected {expected_size}, got {actual_size}"
                )
            expected_hash = row.get("lfs_sha256") or row.get("blob_id")
            hash_kind = "lfs_sha256" if row.get("lfs_sha256") else "git_blob_id"
            if not expected_hash:
                errors.append(f"{source_id}: locked file has no immutable content identifier: {relative}")
            verified = verified_by_path.get(str(relative), {})
            if verified.get("size") != expected_size:
                errors.append(f"{source_id}: verified size does not match lock for {relative}")
            if verified.get("hash_kind") != hash_kind:
                errors.append(f"{source_id}: verified hash kind does not match lock for {relative}")
            if verified.get("expected_hash") != expected_hash or verified.get("actual_hash") != expected_hash:
                errors.append(f"{source_id}: payload hash was not verified against the lock for {relative}")
            stat_bindings = {
                "device": file_stat.st_dev,
                "inode": file_stat.st_ino,
                "mtime_ns": file_stat.st_mtime_ns,
                "ctime_ns": file_stat.st_ctime_ns,
            }
            for field, actual in stat_bindings.items():
                if verified.get(field) != actual:
                    errors.append(f"{source_id}: {field} changed after payload verification for {relative}")
            files.append(
                {
                    "path": str(relative),
                    "local_path": str(local_path),
                    "size": expected_size,
                    "device": file_stat.st_dev,
                    "inode": file_stat.st_ino,
                    "mtime_ns": file_stat.st_mtime_ns,
                    "ctime_ns": file_stat.st_ctime_ns,
                    "hash_kind": hash_kind,
                    "expected_hash": expected_hash,
                }
            )
        receipt_sources.append(
            {
                "source_id": source_id,
                "repo_id": locked["repo_id"],
                "repo_type": locked["repo_type"],
                "revision": locked["revision"],
                "role": locked.get("role"),
                "local_root": str(local_root),
                "selected_file_count": len(files),
                "selected_bytes": sum(int(row["size"]) for row in files),
                "files": files,
            }
        )

    if errors:
        raise ValueError("acquisition completion failed:\n- " + "\n- ".join(errors))

    return {
        "schema_version": "full_cpt_acquisition_receipt_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": code_commit,
        "destination": str(destination),
        "sources_config": str(sources_path.resolve()),
        "sources_config_sha256": sources_sha256,
        "source_lock": str(lock_path.resolve()),
        "source_lock_sha256": lock_sha256,
        "download_manifest": str(download_manifest_path.resolve()),
        "download_manifest_sha256": sha256_file(download_manifest_path),
        "schema_audit": str(schema_audit_path.resolve()),
        "schema_audit_sha256": sha256_file(schema_audit_path),
        "content_verification": (
            "LFS payloads SHA-256 verified once by downloader; Git blobs pinned by commit/blob ID; "
            "downstream gates re-check exact paths and sizes"
        ),
        "sources": receipt_sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--schema-audit", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite immutable acquisition receipt: {args.output}")
    receipt = build_receipt(
        sources_path=args.sources,
        lock_path=args.lock,
        download_manifest_path=args.download_manifest,
        schema_audit_path=args.schema_audit,
        destination=args.destination,
        code_commit=args.code_commit,
    )
    write_json_atomic(args.output, receipt)
    print(json.dumps({"ok": True, "sources": len(receipt["sources"]), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
