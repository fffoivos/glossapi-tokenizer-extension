#!/usr/bin/env python3
"""Publish a receipt-bound metadata-only commit to a private Agent-1 v5 repo.

The original data publication commit remains immutable.  This command requires
the live repository head to equal the selected parent receipt, replaces or adds
only explicitly supplied small metadata files, and proves that every other
remote object is unchanged in the resulting commit.  A passed metadata receipt
can be supplied to append another metadata-only commit without weakening the
binding to the original data publication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from email.utils import parsedate_to_datetime
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


METADATA_PUBLICATION_SCHEMA = "agent1_v5_private_hf_metadata_publication_receipt_v1"
MAX_METADATA_FILE_BYTES = 16 * 1024 * 1024
ALLOWED_METADATA_PATHS = {"README.md", "manifests/provenance.json"}


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def remote_lfs_sha(row: object) -> str | None:
    lfs = _field(row, "lfs")
    value = _field(lfs, "sha256") if lfs is not None else None
    return value if isinstance(value, str) and value else None


def remote_object_id(row: object) -> str | None:
    for name in ("blob_id", "oid"):
        value = _field(row, name)
        if isinstance(value, str) and value:
            return value
    return None


def remote_files(api: Any, repo_id: str, revision: str) -> dict[str, Any]:
    result = {}
    for row in api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
    ):
        path = _field(row, "path")
        if isinstance(path, str) and _field(row, "size") is not None:
            result[path] = row
    return result


def retryable_hf_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return (
        status == 429
        or (isinstance(status, int) and status >= 500)
        or any(
            token in exc.__class__.__name__.lower()
            for token in (
                "connection",
                "connecterror",
                "networkerror",
                "protocolerror",
                "timeout",
            )
        )
    )


def retry_delay_seconds(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    if status != 429:
        return float(min(2**attempt, 30))
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    parsed = 0.0
    if value is not None:
        try:
            parsed = max(0.0, float(value))
        except (TypeError, ValueError):
            try:
                target = parsedate_to_datetime(str(value))
                parsed = max(0.0, target.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                parsed = 0.0
    # This repository has previously observed a roughly 310-second throttle
    # window. Honor a larger server Retry-After value, never a shorter one.
    return max(310.0, parsed)


def wait_retry(delay: float) -> None:
    deadline = time.monotonic() + delay
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        # Keep individual sleeps bounded so an orchestrator can yield status.
        time.sleep(min(30.0, remaining))


def retry_hf(operation: Any, *, attempts: int = 8) -> Any:
    """Retry only throttling, transient server, and transport failures."""

    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not retryable_hf_error(exc) or attempt + 1 == attempts:
                raise
            wait_retry(retry_delay_seconds(exc, attempt))


def parse_addition(value: str) -> tuple[Path, str]:
    if "=" not in value:
        raise ValueError("--add must be LOCAL_PATH=REPO_PATH")
    local_value, remote_value = value.split("=", 1)
    unresolved_local = Path(local_value)
    if unresolved_local.is_symlink():
        raise ValueError(f"unsafe metadata addition: {value}")
    local = unresolved_local.resolve(strict=True)
    remote = PurePosixPath(remote_value)
    if (
        not local.is_file()
        or local.stat().st_size > MAX_METADATA_FILE_BYTES
        or remote.is_absolute()
        or not remote.parts
        or any(part in {"", ".", ".."} for part in remote.parts)
        or remote.as_posix() not in ALLOWED_METADATA_PATHS
    ):
        raise ValueError(f"unsafe metadata addition: {value}")
    return local, remote.as_posix()


def local_additions(values: Sequence[str]) -> dict[str, dict[str, Any]]:
    result = {}
    for value in values:
        local, remote = parse_addition(value)
        if remote in result:
            raise ValueError(f"duplicate remote metadata path: {remote}")
        payload = local.read_bytes()
        if len(payload) > MAX_METADATA_FILE_BYTES:
            raise ValueError(f"metadata addition grew beyond its size limit: {local}")
        result[remote] = {
            "payload": payload,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if "README.md" not in result:
        raise ValueError("metadata commit must explicitly replace README.md")
    return result


def validated_receipt_additions(value: object) -> dict[str, dict[str, Any]]:
    """Validate the immutable byte bindings in a passed metadata receipt."""

    if not isinstance(value, Mapping) or "README.md" not in value:
        raise ValueError("previous metadata receipt has invalid additions")
    result: dict[str, dict[str, Any]] = {}
    for remote, descriptor in value.items():
        if remote not in ALLOWED_METADATA_PATHS or not isinstance(descriptor, Mapping):
            raise ValueError("previous metadata receipt has invalid additions")
        byte_count = descriptor.get("bytes")
        sha256 = descriptor.get("sha256")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or byte_count > MAX_METADATA_FILE_BYTES
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ValueError("previous metadata receipt has invalid additions")
        result[str(remote)] = {"bytes": byte_count, "sha256": sha256}
    return result


def same_remote_object(left: object, right: object) -> bool:
    if int(_field(left, "size", -1)) != int(_field(right, "size", -2)):
        return False
    left_lfs = remote_lfs_sha(left)
    right_lfs = remote_lfs_sha(right)
    if left_lfs is not None or right_lfs is not None:
        return left_lfs is not None and left_lfs == right_lfs
    left_id = remote_object_id(left)
    right_id = remote_object_id(right)
    return left_id is not None and left_id == right_id


def verify_download(
    *, repo_id: str, revision: str, remote_path: str, expected_sha256: str, token: str
) -> None:
    from huggingface_hub import hf_hub_download

    with tempfile.TemporaryDirectory(prefix="agent1-v5-hf-metadata-") as temporary:
        downloaded = retry_hf(
            lambda: hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                revision=revision,
                token=token,
                local_dir=temporary,
            )
        )
        if sha256_file(Path(downloaded)) != expected_sha256:
            raise RuntimeError(f"remote metadata checksum mismatch: {remote_path}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--publication-receipt", type=Path, required=True)
    parser.add_argument(
        "--previous-metadata-receipt",
        type=Path,
        help="passed receipt whose metadata commit is the parent of this update",
    )
    parser.add_argument("--add", action="append", default=[], metavar="LOCAL=REMOTE")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--token", default=None)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(
            f"immutable metadata publication receipt exists: {args.output}"
        )
    publication_path = args.publication_receipt.resolve(strict=True)
    publication = read_object(publication_path)
    if (
        publication.get("schema_version")
        != "agent1_v5_private_hf_publication_receipt_v1"
        or publication.get("status") != "passed"
        or publication.get("private") is not True
        or publication.get("repo_id") != args.repo_id
        or not publication.get("commit_sha")
    ):
        raise ValueError(
            "base publication receipt is not a passed matching private publication"
        )
    publication_receipt_sha256 = sha256_file(publication_path)
    base_revision = str(publication["commit_sha"])
    parent_revision = base_revision
    previous_metadata_path: Path | None = None
    previous_metadata: dict[str, Any] | None = None
    previous_receipt_additions: dict[str, dict[str, Any]] = {}
    previous_metadata_binding: dict[str, str] | None = None
    if args.previous_metadata_receipt is not None:
        previous_metadata_path = args.previous_metadata_receipt.resolve(strict=True)
        previous_metadata = read_object(previous_metadata_path)
        previous_base_binding = previous_metadata.get("base_publication_receipt")
        previous_commit_sha = previous_metadata.get("commit_sha")
        if (
            previous_metadata.get("schema_version") != METADATA_PUBLICATION_SCHEMA
            or previous_metadata.get("status") != "passed"
            or previous_metadata.get("private") is not True
            or previous_metadata.get("repo_id") != args.repo_id
            or previous_metadata.get("base_revision") != base_revision
            or not isinstance(previous_base_binding, Mapping)
            or previous_base_binding.get("sha256") != publication_receipt_sha256
            or not isinstance(previous_commit_sha, str)
            or not previous_commit_sha
        ):
            raise ValueError(
                "previous metadata receipt is not a passed matching publication"
            )
        previous_receipt_additions = validated_receipt_additions(
            previous_metadata.get("additions")
        )
        parent_revision = previous_commit_sha
        previous_metadata_binding = {"sha256": sha256_file(previous_metadata_path)}
    additions = local_additions(args.add)
    receipt_additions = {
        remote: {"bytes": value["bytes"], "sha256": value["sha256"]}
        for remote, value in additions.items()
    }
    receipt: dict[str, Any] = {
        "schema_version": METADATA_PUBLICATION_SCHEMA,
        "status": "dry_run" if not args.execute else "passed",
        "repo_id": args.repo_id,
        "private": True,
        "base_revision": base_revision,
        "parent_revision": parent_revision,
        "base_publication_receipt": {
            "sha256": publication_receipt_sha256,
        },
        "previous_metadata_receipt": previous_metadata_binding,
        "additions": receipt_additions,
        "commit_sha": None,
        "unchanged_files": 0,
        "unchanged_bytes": 0,
    }
    if not args.execute:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
        print(canonical_json({"ok": True, "dry_run": True, "repo_id": args.repo_id}))
        return 0

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("--execute requires --token or HF_TOKEN")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)
    info = retry_hf(
        lambda: api.repo_info(
            repo_id=args.repo_id, repo_type="dataset", files_metadata=True
        )
    )
    if not bool(_field(info, "private", False)):
        raise RuntimeError("refusing metadata update because repository is not private")
    base = retry_hf(lambda: remote_files(api, args.repo_id, base_revision))
    parent = base
    if previous_metadata is not None:
        parent = retry_hf(lambda: remote_files(api, args.repo_id, parent_revision))
        base_non_metadata = set(base) - ALLOWED_METADATA_PATHS
        parent_non_metadata = set(parent) - ALLOWED_METADATA_PATHS
        if parent_non_metadata != base_non_metadata:
            raise RuntimeError(
                "previous metadata commit changed non-metadata inventory: "
                f"extra={sorted(parent_non_metadata - base_non_metadata)}, "
                f"missing={sorted(base_non_metadata - parent_non_metadata)}"
            )
        prior_drifted = [
            path
            for path in sorted(base_non_metadata)
            if not same_remote_object(base[path], parent[path])
        ]
        if prior_drifted:
            raise RuntimeError(
                f"previous metadata commit changed data objects: {prior_drifted[:20]}"
            )
        for remote, value in previous_receipt_additions.items():
            if remote not in parent or int(_field(parent[remote], "size", -1)) != int(
                value["bytes"]
            ):
                raise RuntimeError(
                    f"previous metadata receipt byte-size mismatch: {remote}"
                )
            verify_download(
                repo_id=args.repo_id,
                revision=parent_revision,
                remote_path=remote,
                expected_sha256=str(value["sha256"]),
                token=token,
            )
    operations = [
        CommitOperationAdd(path_in_repo=remote, path_or_fileobj=value["payload"])
        for remote, value in sorted(additions.items())
    ]
    pending_path = args.output.with_name(f".{args.output.name}.pending")
    commit_sha: str | None = None
    recovered_existing_head = False
    if pending_path.exists():
        pending = read_object(pending_path)
        if (
            pending.get("schema_version") != METADATA_PUBLICATION_SCHEMA
            or pending.get("status") != "verifying"
            or pending.get("repo_id") != args.repo_id
            or pending.get("base_revision") != base_revision
            or pending.get("parent_revision", pending.get("base_revision"))
            != parent_revision
            or pending.get("previous_metadata_receipt") != previous_metadata_binding
            or pending.get("additions") != receipt_additions
            or not pending.get("commit_sha")
        ):
            raise RuntimeError(
                "metadata publication pending journal does not match this request"
            )
        commit_sha = str(pending["commit_sha"])
        recovered_existing_head = True
    else:
        live_head = _field(info, "sha")
        if live_head == parent_revision:
            for attempt in range(8):
                try:
                    commit = api.create_commit(
                        repo_id=args.repo_id,
                        repo_type="dataset",
                        operations=operations,
                        commit_message="Update Agent-1 v5 operational dataset status",
                        parent_commit=parent_revision,
                    )
                    commit_sha = _field(commit, "oid") or _field(commit, "commit_id")
                    break
                except Exception as exc:
                    # A timed-out request can still have committed. Re-read
                    # the head before any retry; exact verification below
                    # decides whether the changed head is the intended commit.
                    refreshed = retry_hf(
                        lambda: api.repo_info(
                            repo_id=args.repo_id,
                            repo_type="dataset",
                            files_metadata=True,
                        )
                    )
                    refreshed_head = _field(refreshed, "sha")
                    if refreshed_head != parent_revision:
                        commit_sha = refreshed_head
                        recovered_existing_head = True
                        break
                    if not retryable_hf_error(exc) or attempt == 7:
                        raise
                    wait_retry(retry_delay_seconds(exc, attempt))
        else:
            # Resume safely after an earlier process committed but failed
            # before writing its final receipt. Exact inventory, unchanged
            # object IDs, and downloaded metadata hashes are checked below.
            commit_sha = live_head
            recovered_existing_head = True
        if not isinstance(commit_sha, str) or not commit_sha:
            refreshed = retry_hf(
                lambda: api.repo_info(
                    repo_id=args.repo_id,
                    repo_type="dataset",
                    files_metadata=True,
                )
            )
            commit_sha = _field(refreshed, "sha")
        if not isinstance(commit_sha, str) or not commit_sha:
            raise RuntimeError("metadata commit did not return a commit SHA")
        pending_payload = dict(receipt)
        pending_payload.update({"status": "verifying", "commit_sha": commit_sha})
        pending_path.parent.mkdir(parents=True, exist_ok=True)
        pending_temporary = pending_path.with_name(
            f".{pending_path.name}.partial-{os.getpid()}"
        )
        pending_temporary.write_text(
            canonical_json(pending_payload) + "\n", encoding="utf-8"
        )
        pending_temporary.replace(pending_path)

    live_after = retry_hf(
        lambda: api.repo_info(
            repo_id=args.repo_id, repo_type="dataset", files_metadata=True
        )
    )
    if _field(live_after, "sha") != commit_sha:
        raise RuntimeError("repository head changed during metadata verification")
    updated = retry_hf(lambda: remote_files(api, args.repo_id, commit_sha))
    expected_paths = set(parent) | set(additions)
    if set(updated) != expected_paths:
        raise RuntimeError(
            f"metadata commit inventory mismatch: extra={sorted(set(updated) - expected_paths)}, "
            f"missing={sorted(expected_paths - set(updated))}"
        )
    unchanged = set(parent) - set(additions)
    drifted = [
        path
        for path in sorted(unchanged)
        if not same_remote_object(parent[path], updated[path])
    ]
    if drifted:
        raise RuntimeError(f"non-metadata remote objects changed: {drifted[:20]}")
    for remote, value in additions.items():
        if int(_field(updated[remote], "size", -1)) != int(value["bytes"]):
            raise RuntimeError(f"remote metadata byte-size mismatch: {remote}")
        verify_download(
            repo_id=args.repo_id,
            revision=commit_sha,
            remote_path=remote,
            expected_sha256=str(value["sha256"]),
            token=token,
        )
    final_info = retry_hf(
        lambda: api.repo_info(
            repo_id=args.repo_id, repo_type="dataset", files_metadata=True
        )
    )
    if not bool(_field(final_info, "private", False)):
        raise RuntimeError("repository became public during metadata verification")
    if _field(final_info, "sha") != commit_sha:
        raise RuntimeError(
            "repository head changed before metadata receipt finalization"
        )
    receipt["commit_sha"] = commit_sha
    receipt["recovered_existing_head"] = recovered_existing_head
    receipt["unchanged_files"] = len(unchanged)
    receipt["unchanged_bytes"] = sum(
        int(_field(updated[path], "size", 0)) for path in unchanged
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.partial-{os.getpid()}")
    temporary.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    pending_path.unlink(missing_ok=True)
    print(
        canonical_json({"ok": True, "repo_id": args.repo_id, "commit_sha": commit_sha})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
