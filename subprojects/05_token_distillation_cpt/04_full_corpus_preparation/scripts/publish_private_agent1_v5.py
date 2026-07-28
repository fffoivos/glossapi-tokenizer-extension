#!/usr/bin/env python3
"""Publish a receipt-bound Agent-1 v5 release to an explicitly scoped HF repo.

The command refuses visibility disagreement, unexpected remote files, local
symlinks, non-final public manifests, manifest/repository disagreement, and
unverifiable post-upload content. Its persistent hardlink staging tree lets
``upload_large_folder`` resume without modifying the immutable release.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from agent1_v5_datatrove import DEDUP_MANIFEST_SCHEMA
from agent1_v5_pipeline import (
    COMBINED_MANIFEST_SCHEMA,
    canonical_json,
    file_receipt,
    sha256_file,
    write_json_atomic,
)


PUBLICATION_SCHEMA = "agent1_v5_hf_publication_receipt_v2"
ALLOWED_REMOTE_SYSTEM_FILES = {".gitattributes"}
HUB_RATE_LIMIT_BACKOFF_SECONDS = 310
HUB_UPLOAD_WORKERS = 2


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected JSON object")
    return value


def _manifest_path(release: Path, kind: str) -> Path:
    name = (
        "combined_manifest.json"
        if kind == "pre-dedup"
        else "deduplicated_manifest.json"
    )
    return release / "manifests" / name


def validate_release(
    release: Path, kind: str, visibility: str
) -> tuple[dict[str, Any], Path]:
    if release.is_symlink():
        raise ValueError("release root may not be a symlink")
    release = release.resolve(strict=True)
    manifest_path = _manifest_path(release, kind)
    manifest = read_object(manifest_path)
    expected_schema = (
        COMBINED_MANIFEST_SCHEMA if kind == "pre-dedup" else DEDUP_MANIFEST_SCHEMA
    )
    if (
        manifest.get("schema_version") != expected_schema
        or manifest.get("status") != "passed"
    ):
        raise ValueError(f"{kind} release manifest is not passed")
    expected_private = visibility == "private"
    if bool(manifest.get("private_only")) != expected_private:
        raise ValueError("release manifest visibility differs from publication request")
    if visibility == "public" and manifest.get("publication_ready") is not True:
        raise ValueError("public release manifest has not passed finalization")
    if Path(str(manifest.get("root"))).resolve() != release:
        raise ValueError("release manifest is bound to a different local root")
    expected_data = {str(row["path"]): row for row in manifest["files"]}
    actual_data = {
        path.relative_to(release).as_posix(): path
        for path in sorted((release / "data").glob("*.parquet"))
    }
    if set(actual_data) != set(expected_data):
        raise ValueError("release data inventory differs from its manifest")
    for relative, path in actual_data.items():
        if path.is_symlink():
            raise ValueError(f"release contains symlink: {path}")
        row = expected_data[relative]
        if (
            path.stat().st_size != int(row["bytes"])
            or sha256_file(path) != row["sha256"]
        ):
            raise ValueError(f"release data drift: {path}")
    return manifest, manifest_path


def local_inventory(release: Path) -> list[dict[str, Any]]:
    files = []
    for current, directories, names in os.walk(release, followlinks=False):
        current_path = Path(current)
        directories[:] = sorted(name for name in directories if name != ".cache")
        for directory in directories:
            if (current_path / directory).is_symlink():
                raise ValueError(
                    f"release contains symlink: {current_path / directory}"
                )
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"release contains symlink: {path}")
            relative = path.relative_to(release).as_posix()
            files.append({**file_receipt(path, root=release), "remote_path": relative})
    if not files:
        raise ValueError("release contains no files")
    return files


def build_or_validate_staging(
    release: Path,
    staging: Path,
    inventory: Sequence[Mapping[str, Any]],
) -> None:
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    expected = {str(row["remote_path"]): row for row in inventory}
    actual = {
        path.relative_to(staging).as_posix(): path
        for path in staging.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(staging).parts
    }
    unexpected = set(actual) - set(expected)
    if unexpected:
        raise ValueError(
            f"publication staging has unexpected payload: {sorted(unexpected)[:20]}"
        )
    for relative, row in expected.items():
        source = release / relative
        destination = staging / relative
        if destination.exists():
            if destination.is_symlink() or destination.stat().st_size != int(
                row["bytes"]
            ):
                raise ValueError(f"publication staging drift: {destination}")
            if sha256_file(destination) != row["sha256"]:
                raise ValueError(f"publication staging checksum drift: {destination}")
            continue
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.link(source, destination)
        except OSError as exc:
            raise RuntimeError(
                "publication staging requires same-filesystem hardlinks"
            ) from exc


def remote_files(api: Any, repo_id: str, revision: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "recursive": True,
        "expand": True,
    }
    if revision:
        kwargs["revision"] = revision
    result = {}
    for row in api.list_repo_tree(**kwargs):
        path = _field(row, "path")
        if isinstance(path, str) and _field(row, "size") is not None:
            result[path] = row
    return result


def remote_lfs_sha(row: object) -> str | None:
    lfs = _field(row, "lfs")
    value = _field(lfs, "sha256") if lfs is not None else None
    return value if isinstance(value, str) and value else None


def _http_status_code(error: BaseException) -> int | None:
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _retry_after_seconds(error: BaseException) -> int:
    """Wait out Hub rate-limit windows instead of hot-looping on HTTP 429.

    ``upload_large_folder`` persists its own per-file state, but version 0.26
    immediately requeues a failed final commit when its other queues are
    empty.  That behaviour continuously refreshes a rolling Hub rate-limit
    window.  Wrapping only ``create_commit`` preserves that resumable state
    while making a 429 an in-place, bounded pause.
    """
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    retry_after = headers.get("Retry-After") if headers is not None else None
    try:
        hinted = int(float(retry_after))
    except (TypeError, ValueError):
        hinted = 0
    return max(HUB_RATE_LIMIT_BACKOFF_SECONDS, hinted)


def upload_large_folder_with_rate_limit_backoff(api: Any, **kwargs: Any) -> None:
    """Call the Hub uploader with deliberate recovery from commit API 429s."""
    original_create_commit = api.create_commit
    rate_limit_events = 0

    def create_commit_with_backoff(*args: Any, **commit_kwargs: Any) -> Any:
        nonlocal rate_limit_events
        while True:
            try:
                return original_create_commit(*args, **commit_kwargs)
            except Exception as error:
                if _http_status_code(error) != 429:
                    raise
                rate_limit_events += 1
                delay = _retry_after_seconds(error)
                print(
                    canonical_json(
                        {
                            "event": "huggingface_rate_limit_backoff",
                            "attempt": rate_limit_events,
                            "sleep_seconds": delay,
                        }
                    ),
                    flush=True,
                )
                time.sleep(delay)

    api.create_commit = create_commit_with_backoff
    api.upload_large_folder(num_workers=HUB_UPLOAD_WORKERS, **kwargs)


def verify_remote(
    api: Any,
    repo_id: str,
    revision: str,
    token: str,
    inventory: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected = {str(row["remote_path"]): row for row in inventory}
    remote = remote_files(api, repo_id, revision)
    extras = set(remote) - set(expected) - ALLOWED_REMOTE_SYSTEM_FILES
    missing = set(expected) - set(remote)
    if extras or missing:
        mismatch = f"extra={sorted(extras)}, missing={sorted(missing)}"
        raise RuntimeError(
            f"remote inventory mismatch: {mismatch}"
        )
    from huggingface_hub import hf_hub_download

    verified = []
    with tempfile.TemporaryDirectory(prefix="agent1-v5-hf-verify-") as temporary:
        for remote_path in sorted(expected):
            expected_row = expected[remote_path]
            remote_row = remote[remote_path]
            if int(_field(remote_row, "size", -1)) != int(expected_row["bytes"]):
                raise RuntimeError(f"remote byte-size mismatch: {remote_path}")
            actual_sha = remote_lfs_sha(remote_row)
            method = "lfs_sha256"
            if actual_sha is None:
                downloaded = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    revision=revision,
                    token=token,
                    local_dir=temporary,
                )
                actual_sha = sha256_file(Path(downloaded))
                method = "downloaded_sha256"
            if actual_sha != expected_row["sha256"]:
                raise RuntimeError(f"remote checksum mismatch: {remote_path}")
            verified.append(
                {
                    "path": remote_path,
                    "bytes": int(expected_row["bytes"]),
                    "sha256": actual_sha,
                    "verification": method,
                }
            )
    return verified


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--kind", choices=["pre-dedup", "deduplicated"], required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--visibility", choices=["private", "public"], default="private"
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable publication receipt exists: {args.output}")
    manifest, manifest_path = validate_release(args.release, args.kind, args.visibility)
    if args.repo_id != manifest["repository_id"]:
        raise ValueError("repository ID differs from the immutable release manifest")
    inventory = local_inventory(args.release.resolve())
    base_receipt: dict[str, Any] = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "dry_run" if not args.execute else "passed",
        "kind": args.kind,
        "private": args.visibility == "private",
        "visibility": args.visibility,
        "repo_id": args.repo_id,
        "release": str(args.release.resolve()),
        "release_manifest": file_receipt(manifest_path),
        "files": len(inventory),
        "bytes": sum(int(row["bytes"]) for row in inventory),
        "rows": int(manifest["rows"]),
        "local_inventory": inventory,
        "commit_sha": None,
        "remote_inventory": [],
    }
    if not args.execute:
        write_json_atomic(args.output, base_receipt)
        print(canonical_json({"ok": True, "dry_run": True, "repo_id": args.repo_id}))
        return 0
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("--execute requires --token or HF_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    identity = api.whoami(token=token)
    if not isinstance(identity, Mapping) or not identity.get("name"):
        raise RuntimeError("Hugging Face identity check failed")
    expected_private = args.visibility == "private"
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=expected_private,
        exist_ok=True,
    )
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset", files_metadata=True)
    if bool(_field(info, "private", False)) != expected_private:
        raise RuntimeError(
            "refusing to upload because Hugging Face repository visibility differs"
        )
    existing = remote_files(api, args.repo_id)
    extras = (
        set(existing)
        - {str(row["remote_path"]) for row in inventory}
        - ALLOWED_REMOTE_SYSTEM_FILES
    )
    if extras:
        raise RuntimeError(
            f"remote repository contains unexpected payload: {sorted(extras)}"
        )
    build_or_validate_staging(args.release.resolve(), args.staging.resolve(), inventory)
    upload_large_folder_with_rate_limit_backoff(
        api,
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(args.staging.resolve()),
    )
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset", files_metadata=True)
    commit_sha = _field(info, "sha")
    if (
        bool(_field(info, "private", False)) != expected_private
        or not isinstance(commit_sha, str)
        or not commit_sha
    ):
        raise RuntimeError("final repository visibility or commit SHA is invalid")
    remote_inventory = verify_remote(api, args.repo_id, commit_sha, token, inventory)
    base_receipt["commit_sha"] = commit_sha
    base_receipt["remote_inventory"] = remote_inventory
    write_json_atomic(args.output, base_receipt)
    print(
        canonical_json({"ok": True, "repo_id": args.repo_id, "commit_sha": commit_sha})
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
