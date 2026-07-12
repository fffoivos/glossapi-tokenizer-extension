#!/usr/bin/env python3
"""Acquire receipt-bound Mozilla Data Collective sources without leaking URLs.

The MDC API returns short-lived presigned storage URLs.  They are held only in
memory: logs and receipts contain the public dataset identity, declared file
metadata, and verified SHA-256 values, never the API key, download token, or
presigned URL.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "full_cpt_mdc_acquisition_receipt_v1"
SOURCE_SCHEMA = "full_cpt_mdc_source_receipt_v1"
PAYLOAD_VALIDATION_SCHEMA = "full_cpt_mdc_payload_validation_v1"
DEFAULT_API_BASE = "https://mozilladatacollective.com/api"
HEX = frozenset("0123456789abcdef")


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_object_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


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


def canonical_checksum(value: object) -> str:
    checksum = str(value or "").strip().lower()
    if checksum.startswith("sha256:"):
        checksum = checksum.removeprefix("sha256:")
    if len(checksum) != 64 or set(checksum) - HEX:
        raise ValueError("MDC response did not provide a valid SHA-256 checksum")
    return checksum


class MdcClient:
    def __init__(self, api_base: str, api_key: str, timeout: int = 120) -> None:
        if not api_key:
            raise ValueError("MOZILLA_DATA_COLLECTIVE_API_KEY is required")
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def json_request(self, path: str, *, method: str = "GET") -> dict[str, Any]:
        url = f"{self.api_base}/{path.lstrip('/')}"
        request = urllib.request.Request(
            url,
            method=method,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
                "User-Agent": "GlossAPI-Phase04-MDC/1",
            },
        )
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.load(response)
                if not isinstance(value, dict):
                    raise ValueError("MDC API response must be a JSON object")
                return value
            except urllib.error.HTTPError as error:
                body = error.read().decode("utf-8", "replace")
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError:
                    payload = {}
                if error.code == 403:
                    message = str(payload.get("error") or payload.get("message") or "access denied")
                    raise PermissionError(f"MDC dataset access denied: {message}") from None
                if error.code == 401:
                    raise PermissionError("MDC API key was rejected") from None
                if error.code not in {429, 500, 502, 503, 504} or attempt == 5:
                    raise RuntimeError(f"MDC API request failed with HTTP {error.code}") from None
                delay = int(error.headers.get("Retry-After", "0") or 0) or min(60, 2**attempt)
                time.sleep(delay)
            except urllib.error.URLError:
                if attempt == 5:
                    raise RuntimeError("MDC API request failed after retries") from None
                time.sleep(min(60, 2**attempt))
        raise AssertionError("unreachable")


def download_storage(
    url: str,
    output: Path,
    *,
    expected_bytes: int,
    timeout: int,
    chunk_size: int = 8 * 1024 * 1024,
) -> None:
    """Resume a presigned storage download without exposing its URL."""

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected_bytes:
        partial.unlink()
        offset = 0
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "GlossAPI-Phase04-MDC/1",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"MDC storage download failed with HTTP {error.code}") from None
    except urllib.error.URLError:
        raise RuntimeError("MDC storage download failed") from None
    with response:
        status = int(getattr(response, "status", response.getcode()))
        if offset and status != 206:
            partial.unlink(missing_ok=True)
            return download_storage(
                url, output, expected_bytes=expected_bytes, timeout=timeout, chunk_size=chunk_size
            )
        mode = "ab" if offset else "wb"
        with partial.open(mode) as handle:
            while chunk := response.read(chunk_size):
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    if partial.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"MDC storage download is incomplete: {partial.stat().st_size} != {expected_bytes}"
        )
    os.replace(partial, output)


def safe_member_path(root: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member path: {name!r}")
    result = (root / relative).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"archive member escapes extraction root: {name!r}") from error
    return result


def safe_extract_tar(archive: Path, output: Path, *, maximum_bytes: int) -> None:
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{output.name}.partial-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    try:
        with tarfile.open(archive, "r:*") as bundle:
            members = bundle.getmembers()
            total = 0
            for member in members:
                safe_member_path(temporary, member.name)
                if member.issym() or member.islnk() or member.isdev():
                    raise ValueError(f"unsupported archive member type: {member.name!r}")
                if member.isfile():
                    total += int(member.size)
                    if total > maximum_bytes:
                        raise ValueError("archive exceeds the configured extraction size ceiling")
            for member in members:
                destination = safe_member_path(temporary, member.name)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                extracted = bundle.extractfile(member)
                if extracted is None:
                    raise ValueError(f"could not read archive member: {member.name!r}")
                with extracted, destination.open("wb") as handle:
                    shutil.copyfileobj(extracted, handle, length=8 * 1024 * 1024)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def selected_files(root: Path, includes: Iterable[str], excludes: Iterable[str]) -> list[Path]:
    include_patterns = list(includes)
    exclude_patterns = list(excludes)
    rows: list[Path] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        included = any(fnmatch.fnmatchcase(relative, pattern) for pattern in include_patterns)
        excluded = any(fnmatch.fnmatchcase(relative, pattern) for pattern in exclude_patterns)
        if included and not excluded:
            rows.append(path)
    if not rows:
        raise ValueError(f"no extracted files match include globs under {root}")
    return rows


def validate_parquet_payload(
    files: list[Path], source: dict[str, Any]
) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - Clariden runtime contract
        raise RuntimeError("pyarrow is required to validate MDC Parquet payloads") from error

    source_id = str(source["source_id"])
    candidate_text = sorted(
        {
            str(value)
            for value in (
                list(source.get("text_columns", []))
                + list(source.get("alternate_text_columns", []))
            )
            if str(value)
        }
    )
    required_text = sorted(
        {str(value) for value in source.get("required_text_columns", []) if str(value)}
    )
    candidate_ids = sorted(
        {str(value) for value in source.get("id_columns", []) if str(value)}
    )
    if not candidate_text:
        raise ValueError(
            f"{source_id}: PARQUET MDC route has no configured candidate text columns"
        )
    if not files:
        raise ValueError(f"{source_id}: PARQUET MDC route selected no payload files")

    audits: list[dict[str, Any]] = []
    total_rows = 0
    for path in files:
        try:
            parquet = pq.ParquetFile(path)
        except Exception as error:
            raise ValueError(
                f"{source_id}: selected payload is not readable Parquet: {path}: "
                f"{type(error).__name__}"
            ) from error
        columns = set(parquet.schema_arrow.names)
        present_text = sorted(columns & set(candidate_text))
        present_ids = sorted(columns & set(candidate_ids))
        missing_required = sorted(set(required_text) - columns)
        if missing_required:
            raise ValueError(
                f"{source_id}:{path.name}: missing required text columns {missing_required}"
            )
        if not present_text:
            raise ValueError(
                f"{source_id}:{path.name}: none of the candidate text columns are present: "
                f"{candidate_text}"
            )
        if candidate_ids and not present_ids:
            raise ValueError(
                f"{source_id}:{path.name}: none of the candidate id columns are present: "
                f"{candidate_ids}"
            )
        rows = int(parquet.metadata.num_rows)
        if rows < 1:
            raise ValueError(f"{source_id}:{path.name}: selected Parquet file has zero rows")
        total_rows += rows
        audits.append(
            {
                "local_path": str(path.resolve()),
                "rows": rows,
                "row_groups": int(parquet.num_row_groups),
                "columns": sorted(columns),
                "present_text_columns": present_text,
                "present_id_columns": present_ids,
            }
        )
    if total_rows < 1:
        raise ValueError(f"{source_id}: selected MDC payload has zero aggregate rows")
    return {
        "schema_version": PAYLOAD_VALIDATION_SCHEMA,
        "status": "passed",
        "format": "PARQUET",
        "source_config_sha256": canonical_object_sha256(source),
        "selected_file_count": len(files),
        "total_rows": total_rows,
        "candidate_text_columns": candidate_text,
        "required_text_columns": required_text,
        "candidate_id_columns": candidate_ids,
        "files": audits,
    }


def validate_payload(files: list[Path], source: dict[str, Any]) -> dict[str, Any]:
    payload_format = str(source.get("mdc_format", "")).upper()
    if payload_format == "PARQUET":
        return validate_parquet_payload(files, source)
    raise ValueError(
        f"{source.get('source_id')}: unsupported MDC payload format {payload_format!r}; "
        "add and test a format-specific validator before acquisition"
    )


def validate_source_receipt(path: Path, source: dict[str, Any]) -> dict[str, Any]:
    receipt = load_object(path)
    for field, expected in {
        "schema_version": SOURCE_SCHEMA,
        "source_id": source["source_id"],
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "mdc_dataset_id": source["mdc_dataset_id"],
        "source_config_sha256": canonical_object_sha256(source),
    }.items():
        if receipt.get(field) != expected:
            raise ValueError(f"{path}: source receipt drift for {field}")
    archive = receipt.get("archive")
    if not isinstance(archive, dict):
        raise ValueError(f"{path}: source receipt has no archive evidence")
    archive_path = Path(str(archive.get("local_path", "")))
    if (
        not archive_path.is_file()
        or archive_path.name != source["mdc_expected_filename"]
        or archive_path.stat().st_size != int(archive.get("bytes", -1))
        or archive_path.stat().st_size != int(source["mdc_expected_bytes"])
        or sha256_file(archive_path) != archive.get("sha256")
    ):
        raise ValueError(f"{path}: downloaded archive drift for {archive_path}")
    pinned = canonical_checksum(source.get("mdc_expected_sha256"))
    if (
        archive.get("sha256") != pinned
        or archive.get("registry_sha256") != pinned
        or archive.get("metadata_sha256") != pinned
    ):
        raise ValueError(f"{path}: archive differs from pinned source checksum")
    file_rows = receipt.get("files")
    if not isinstance(file_rows, list) or not file_rows:
        raise ValueError(f"{path}: source receipt has no selected payload files")
    if int(receipt.get("selected_file_count", -1)) != len(file_rows):
        raise ValueError(f"{path}: selected payload file-count drift")
    selected_bytes = 0
    local_files: list[Path] = []
    for row in file_rows:
        local = Path(str(row.get("local_path", "")))
        stat = local.stat() if local.is_file() else None
        if (
            stat is None
            or stat.st_size != int(row.get("size", -1))
            or sha256_file(local) != row.get("expected_hash")
        ):
            raise ValueError(f"{path}: extracted payload drift for {local}")
        for field, actual in {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        }.items():
            if int(row.get(field, -1)) != int(actual):
                raise ValueError(f"{path}: extracted payload {field} drift for {local}")
        selected_bytes += stat.st_size
        local_files.append(local)
    if int(receipt.get("selected_bytes", -1)) != selected_bytes:
        raise ValueError(f"{path}: selected payload byte-count drift")
    validation = validate_payload(local_files, source)
    if receipt.get("payload_validation") != validation:
        raise ValueError(f"{path}: payload_validation receipt drift")
    return receipt


def acquire_source(
    source: dict[str, Any],
    *,
    client: MdcClient,
    destination: Path,
    extraction_multiplier: int,
) -> dict[str, Any]:
    source_root = destination / str(source["source_id"]) / str(source["revision"])
    source_receipt = source_root / "source_receipt.json"
    if source_receipt.exists():
        return validate_source_receipt(source_receipt, source)

    dataset_id = str(source["mdc_dataset_id"])
    pinned = canonical_checksum(source.get("mdc_expected_sha256"))
    details = client.json_request(f"datasets/{dataset_id}")
    for field, expected in {
        "id": dataset_id,
        "slug": source["mdc_slug"],
        "name": source["mdc_name"],
        "format": source["mdc_format"],
    }.items():
        if str(details.get(field)) != str(expected):
            raise ValueError(f"{source['source_id']}: MDC metadata drift for {field}")
    if int(details.get("sizeBytes", -1)) != int(source["mdc_expected_bytes"]):
        raise ValueError(f"{source['source_id']}: MDC declared size drift")
    metadata_checksum = canonical_checksum(details.get("checksum"))
    if metadata_checksum != pinned:
        raise ValueError(f"{source['source_id']}: MDC metadata checksum differs from registry")

    download = client.json_request(f"datasets/{dataset_id}/download", method="POST")
    filename = str(download.get("filename") or "")
    size = int(download.get("sizeBytes", -1))
    checksum = canonical_checksum(download.get("checksum"))
    if filename != source["mdc_expected_filename"] or size != int(source["mdc_expected_bytes"]):
        raise ValueError(f"{source['source_id']}: MDC download metadata drift")
    if checksum != pinned:
        raise ValueError(f"{source['source_id']}: MDC archive checksum differs from registry")
    download_url = str(download.get("downloadUrl") or "")
    if not download_url.startswith("https://"):
        raise ValueError(f"{source['source_id']}: MDC did not return an HTTPS storage URL")

    archive = source_root / "archive" / filename
    if not archive.exists():
        download_storage(download_url, archive, expected_bytes=size, timeout=client.timeout)
    if archive.stat().st_size != size or sha256_file(archive) != checksum:
        raise ValueError(f"{source['source_id']}: downloaded archive failed SHA-256 verification")

    payload = source_root / "payload"
    if not payload.exists():
        safe_extract_tar(
            archive,
            payload,
            maximum_bytes=max(size, 1) * extraction_multiplier,
        )
    files = selected_files(
        payload,
        source.get("include_globs", []),
        source.get("exclude_globs", []),
    )
    payload_validation = validate_payload(files, source)
    file_rows: list[dict[str, Any]] = []
    for path in files:
        stat = path.stat()
        file_rows.append(
            {
                "path": path.relative_to(payload).as_posix(),
                "local_path": str(path.resolve()),
                "size": stat.st_size,
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
                "hash_kind": "sha256",
                "expected_hash": sha256_file(path),
            }
        )
    receipt = {
        "schema_version": SOURCE_SCHEMA,
        "source_id": source["source_id"],
        "repo_id": source["repo_id"],
        "revision": source["revision"],
        "role": source["role"],
        "mdc_dataset_id": dataset_id,
        "mdc_slug": details["slug"],
        "source_config_sha256": canonical_object_sha256(source),
        "archive": {
            "filename": filename,
            "local_path": str(archive.resolve()),
            "bytes": size,
            "sha256": checksum,
            "registry_sha256": pinned,
            "metadata_sha256": metadata_checksum,
            "content_type": download.get("contentType"),
        },
        "local_root": str(payload.resolve()),
        "selected_file_count": len(file_rows),
        "selected_bytes": sum(int(row["size"]) for row in file_rows),
        "payload_validation": payload_validation,
        "files": file_rows,
    }
    write_json_atomic(source_receipt, receipt)
    return receipt


def build_receipt(
    *,
    sources_path: Path,
    destination: Path,
    selected: set[str],
    client: MdcClient,
    code_commit: str,
    extraction_multiplier: int,
) -> dict[str, Any]:
    config = load_object(sources_path)
    candidates = [
        dict(source)
        for source in config.get("sources", [])
        if source.get("acquisition_kind") == "mozilla_data_collective"
        and (not selected or source.get("source_id") in selected)
    ]
    available = {str(source["source_id"]) for source in candidates}
    if selected - available:
        raise ValueError(f"unknown or non-MDC source IDs: {sorted(selected - available)}")
    if not candidates:
        raise ValueError("no MDC sources selected")
    rows = [
        acquire_source(
            source,
            client=client,
            destination=destination,
            extraction_multiplier=extraction_multiplier,
        )
        for source in sorted(candidates, key=lambda row: str(row["source_id"]))
    ]
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_commit": code_commit,
        "sources_config": str(sources_path.resolve()),
        "sources_config_sha256": sha256_file(sources_path),
        "destination": str(destination.resolve()),
        "secret_handling": "API key read from environment; download token and presigned URL never persisted",
        "sources": rows,
    }


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, default=here / "configs" / "sources.json")
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--api-base", default=DEFAULT_API_BASE)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--extraction-multiplier", type=int, default=20)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    if args.extraction_multiplier < 1:
        raise ValueError("--extraction-multiplier must be positive")
    client = MdcClient(
        args.api_base,
        os.environ.get("MOZILLA_DATA_COLLECTIVE_API_KEY", ""),
        timeout=args.timeout,
    )
    receipt = build_receipt(
        sources_path=args.sources,
        destination=args.destination,
        selected=set(args.source),
        client=client,
        code_commit=args.code_commit,
        extraction_multiplier=args.extraction_multiplier,
    )
    write_json_atomic(args.output, receipt)
    print(
        json.dumps(
            {
                "ok": True,
                "sources": len(receipt["sources"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
