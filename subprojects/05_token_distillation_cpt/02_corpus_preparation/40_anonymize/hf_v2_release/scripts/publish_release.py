#!/usr/bin/env python3
"""Publish the verified anonymized release through a checked Hugging Face PR."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from release_common import (
    FINAL_SCHEMA,
    PUBLICATION_SCHEMA,
    file_receipt,
    load_contract,
    read_json,
    sha256_file,
    utc_now,
    write_json_atomic,
)


def _field(value: object, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _remote_files(api: Any, repo_id: str, revision: str) -> dict[str, object]:
    rows = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
    )
    result: dict[str, object] = {}
    for row in rows:
        path = _field(row, "path")
        if isinstance(path, str) and _field(row, "size") is not None:
            result[path] = row
    return result


def _lfs_sha(row: object) -> str | None:
    lfs = _field(row, "lfs")
    value = _field(lfs, "sha256") if lfs is not None else None
    return value if isinstance(value, str) and len(value) == 64 else None


def _identity(row: object) -> tuple[int, str | None, str | None]:
    blob = _field(row, "blob_id") or _field(row, "oid")
    return int(_field(row, "size", -1)), _lfs_sha(row), str(blob) if blob else None


def _local_inventory(release_root: Path, manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in manifest.get("files", []):
        relative = str(row.get("path", ""))
        path = release_root / relative
        if relative in result or not relative.startswith("data/") or path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid local data inventory row: {relative!r}")
        actual = file_receipt(path, rows=int(row["rows"]), relative_to=release_root)
        if any(actual[key] != row[key] for key in ("path", "bytes", "sha256", "rows")):
            raise ValueError(f"release data drift after finalization: {relative}")
        result[relative] = actual
    for relative in ("README.md", "manifests/token_counts.json", "manifests/anonymization_manifest.json"):
        path = release_root / relative
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        result[relative] = file_receipt(path, relative_to=release_root)
    if len([path for path in result if path.startswith("data/")]) != 431:
        raise ValueError("publication does not contain exactly 431 Parquet shards")
    return result


def _stage_hardlinks(release_root: Path, staging: Path, inventory: Mapping[str, Mapping[str, Any]]) -> None:
    staging.mkdir(parents=True, exist_ok=True)
    for relative in sorted(inventory):
        source = release_root / relative
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.is_symlink() or not os.path.samefile(source, destination):
                raise ValueError(f"publication staging drift: {destination}")
            continue
        os.link(source, destination, follow_symlinks=False)


def _download_sha(repo_id: str, revision: str, path: str, token: str, root: Path) -> str:
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=path,
        token=token,
        local_dir=root,
    )
    return sha256_file(Path(downloaded))


def _verify_updated(
    api: Any,
    *,
    repo_id: str,
    revision: str,
    token: str,
    inventory: Mapping[str, Mapping[str, Any]],
    temporary: Path,
) -> dict[str, dict[str, Any]]:
    remote = _remote_files(api, repo_id, revision)
    for relative, expected in inventory.items():
        row = remote.get(relative)
        if row is None:
            raise RuntimeError(f"remote revision lacks {relative}")
        if int(_field(row, "size", -1)) != int(expected["bytes"]):
            raise RuntimeError(f"remote byte-size mismatch: {relative}")
        actual_sha = _lfs_sha(row)
        if actual_sha is None:
            if relative.startswith("data/"):
                raise RuntimeError(f"large remote file lacks SHA-verifiable LFS metadata: {relative}")
            actual_sha = _download_sha(repo_id, revision, relative, token, temporary)
        if actual_sha != expected["sha256"]:
            raise RuntimeError(f"remote checksum mismatch: {relative}")
    return remote


def _verify_no_unplanned_changes(
    base: Mapping[str, object], candidate: Mapping[str, object], updated: set[str]
) -> None:
    expected_paths = set(base) | (updated - set(base))
    if set(candidate) != expected_paths:
        raise RuntimeError(
            "PR remote inventory changed unexpectedly: "
            f"extra={sorted(set(candidate) - expected_paths)}, missing={sorted(expected_paths - set(candidate))}"
        )
    for path in sorted(set(base) - updated):
        if _identity(base[path]) != _identity(candidate[path]):
            raise RuntimeError(f"unplanned remote file change: {path}")


def _discussion_can_continue(status: object) -> bool:
    return str(status) in {"draft", "open", "merged"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    release_root = args.release_root.resolve()
    manifest_path = args.manifest.resolve()
    contract_path = args.contract.resolve()
    contract = load_contract(contract_path, executing_code_root=args.code_root.resolve())
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != FINAL_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("publication requires a passed anonymized release manifest")
    if not all(manifest.get("invariants", {}).values()):
        raise ValueError("publication manifest contains a failed invariant")
    if manifest.get("contract") != file_receipt(contract_path):
        raise ValueError("publication manifest is not bound to the executing run contract")
    if Path(str(contract["run_root"])).resolve() / "release" != release_root:
        raise ValueError("release root differs from the frozen run contract")
    repo_id = str(manifest["input"]["repository_id"])
    base_revision = str(manifest["input"]["revision"])
    inventory = _local_inventory(release_root, manifest)
    if args.receipt.exists():
        receipt = read_json(args.receipt)
        if receipt.get("schema_version") != PUBLICATION_SCHEMA or receipt.get("status") != "passed":
            raise ValueError("existing publication receipt is invalid")
        print(json.dumps({"ok": True, "resumed": True, "commit_sha": receipt["commit_sha"]}, sort_keys=True))
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required even for the remote-head gate")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    base = _remote_files(api, repo_id, base_revision)
    state: dict[str, Any] | None = None
    discussion: Any | None = None
    if args.state.exists():
        state = read_json(args.state)
        if state.get("repo_id") != repo_id or state.get("base_revision") != base_revision:
            raise ValueError("publication resume state binding drift")
        if state.get("manifest_sha256") != sha256_file(manifest_path):
            raise ValueError("publication manifest changed after PR creation")
        discussion = api.get_discussion_details(
            repo_id=repo_id,
            repo_type="dataset",
            discussion_num=int(state["discussion_num"]),
        )
    current_head = str(_field(api.repo_info(repo_id=repo_id, repo_type="dataset"), "sha", ""))
    merged_resume = discussion is not None and str(discussion.status) == "merged"
    if not merged_resume and current_head != base_revision:
        raise RuntimeError(f"HF main moved since the frozen input: {current_head} != {base_revision}")
    if not args.execute:
        print(
            json.dumps(
                {
                    "ok": True,
                    "dry_run": True,
                    "repo_id": repo_id,
                    "base_revision": base_revision,
                    "updated_files": len(inventory),
                },
                sort_keys=True,
            )
        )
        return 0

    args.state.parent.mkdir(parents=True, exist_ok=True)
    if state is None:
        discussion = api.create_pull_request(
            repo_id=repo_id,
            repo_type="dataset",
            title="Publish row-preserving Apertus-standard anonymized v2",
            description=(
                "Replaces only text values containing Apertus PII patterns; preserves all 51,839,746 rows, "
                "row order, multiplicity, schema, source identities, and non-text values."
            ),
        )
        state = {
            "schema_version": "glossapi_hf_v2_anonymized_publication_state_v1",
            "status": "uploading",
            "created_at": utc_now(),
            "repo_id": repo_id,
            "base_revision": base_revision,
            "manifest_sha256": sha256_file(manifest_path),
            "main_pinned_before_upload": True,
            "discussion_num": int(discussion.num),
            "pr_revision": f"refs/pr/{int(discussion.num)}",
        }
        write_json_atomic(args.state, state)
    pr_revision = str(state["pr_revision"])
    if discussion is None:
        discussion = api.get_discussion_details(
            repo_id=repo_id,
            repo_type="dataset",
            discussion_num=int(state["discussion_num"]),
        )
    if not _discussion_can_continue(discussion.status):
        raise RuntimeError(f"publication PR is not draft, open, or merged: {discussion.status}")
    if str(discussion.status) == "merged":
        if not all(
            state.get(name) is True
            for name in ("pr_verified", "no_unplanned_pr_changes", "main_pinned_before_merge")
        ):
            raise RuntimeError("merged PR resume lacks the pre-merge verification gates")
        with tempfile.TemporaryDirectory(prefix="hf-v2-published-verify-", dir=args.state.parent) as temporary:
            main = _verify_updated(
                api,
                repo_id=repo_id,
                revision="main",
                token=token,
                inventory=inventory,
                temporary=Path(temporary),
            )
        _verify_no_unplanned_changes(base, main, set(inventory))
        commit_sha = current_head
    else:
        staging = args.state.parent / "upload-staging"
        _stage_hardlinks(release_root, staging, inventory)
        # The release is about 131 GiB across 431 Parquet shards.  The Hub's
        # large-folder uploader persists task state under the staging folder,
        # so a debug-allocation timeout can resume without retransmitting
        # completed objects.  Xet may be disabled by the launcher on ARM64;
        # huggingface_hub then uses its supported HTTP/LFS transport.
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=staging,
            revision=pr_revision,
            allow_patterns=sorted(inventory),
            num_workers=16,
            print_report=True,
            print_report_every=60,
        )
        state["upload_mode"] = "upload_large_folder_http_lfs"
        state["upload_completed_at"] = utc_now()
        write_json_atomic(args.state, state)
        with tempfile.TemporaryDirectory(prefix="hf-v2-pr-verify-", dir=args.state.parent) as temporary:
            candidate = _verify_updated(
                api,
                repo_id=repo_id,
                revision=pr_revision,
                token=token,
                inventory=inventory,
                temporary=Path(temporary),
            )
        _verify_no_unplanned_changes(base, candidate, set(inventory))
        state["pr_verified"] = True
        state["no_unplanned_pr_changes"] = True
        state["pr_verified_at"] = utc_now()
        write_json_atomic(args.state, state)
        head_before_merge = str(_field(api.repo_info(repo_id=repo_id, repo_type="dataset"), "sha", ""))
        if head_before_merge != base_revision:
            raise RuntimeError("HF main changed while the anonymized release was uploading; PR left unmerged")
        state["main_pinned_before_merge"] = True
        state["main_pinned_before_merge_at"] = utc_now()
        write_json_atomic(args.state, state)
        # Hugging Face creates pull requests in draft state.  A verified draft
        # must be explicitly opened before the merge API accepts it.
        if str(discussion.status) == "draft":
            api.change_discussion_status(
                repo_id=repo_id,
                repo_type="dataset",
                discussion_num=int(state["discussion_num"]),
                new_status="open",
                comment="Opening after the checksum and no-unplanned-change gates passed.",
            )
            discussion = api.get_discussion_details(
                repo_id=repo_id,
                repo_type="dataset",
                discussion_num=int(state["discussion_num"]),
            )
            if str(discussion.status) != "open":
                raise RuntimeError(f"publication PR did not become open: {discussion.status}")
            state["pr_opened_for_merge_at"] = utc_now()
            write_json_atomic(args.state, state)
        api.merge_pull_request(
            repo_id=repo_id,
            repo_type="dataset",
            discussion_num=int(state["discussion_num"]),
            comment="All 431 shard checksums, row/schema invariants, source counts, token counts, and README scope passed.",
        )
        commit_sha = str(_field(api.repo_info(repo_id=repo_id, repo_type="dataset"), "sha", ""))
        if not commit_sha or commit_sha == base_revision:
            raise RuntimeError("Hugging Face main did not advance after PR merge")
        with tempfile.TemporaryDirectory(prefix="hf-v2-main-verify-", dir=args.state.parent) as temporary:
            main = _verify_updated(
                api,
                repo_id=repo_id,
                revision=commit_sha,
                token=token,
                inventory=inventory,
                temporary=Path(temporary),
            )
        _verify_no_unplanned_changes(base, main, set(inventory))

    receipt = {
        "schema_version": PUBLICATION_SCHEMA,
        "status": "passed",
        "completed_at": utc_now(),
        "repo_id": repo_id,
        "base_revision": base_revision,
        "commit_sha": commit_sha,
        "discussion_num": int(state["discussion_num"]),
        "pr_revision": pr_revision,
        "manifest": file_receipt(manifest_path),
        "publisher_overlay": {
            "executing_publisher": file_receipt(Path(__file__)),
            "executing_release_common": file_receipt(Path(__file__).resolve().parent / "release_common.py"),
            "contract_code_root": str(args.code_root.resolve()),
        },
        "updated_files": len(inventory),
        "updated_inventory": inventory,
        "main_inventory_files": len(main),
        "invariants": {
            "main_was_pinned_before_upload": state.get("main_pinned_before_upload") is True,
            "upload_used_a_pull_request": True,
            "all_updated_files_checksum_verified_on_pr": state.get("pr_verified") is True,
            "no_unplanned_pr_changes": state.get("no_unplanned_pr_changes") is True,
            "main_was_still_pinned_before_merge": state.get("main_pinned_before_merge") is True,
            "all_updated_files_checksum_verified_on_main": True,
            "no_unplanned_main_changes": True,
        },
    }
    write_json_atomic(args.receipt, receipt)
    state["status"] = "passed"
    state["commit_sha"] = commit_sha
    state["completed_at"] = utc_now()
    write_json_atomic(args.state, state)
    print(json.dumps({"ok": True, "repo_id": repo_id, "commit_sha": commit_sha}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
