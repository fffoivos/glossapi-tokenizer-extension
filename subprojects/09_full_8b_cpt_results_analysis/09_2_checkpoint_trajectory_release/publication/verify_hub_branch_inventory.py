#!/usr/bin/env python3
"""Verify the complete Hub inventory of one frozen private model branch.

The canonical checkpoint publisher checks that core model files reached the
Hub.  This companion checker additionally verifies every frozen file: LFS
objects by their Hub SHA-256 OID and ordinary files by content download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path}: expected JSON object")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def lfs_oid(sibling: Any) -> str | None:
    value = getattr(sibling, "lfs", None)
    if value is None:
        return None
    if isinstance(value, dict):
        candidate = value.get("oid")
    else:
        candidate = getattr(value, "oid", None)
    return str(candidate) if candidate else None


def inventory_check(
    expected: list[dict[str, Any]],
    siblings: list[Any],
    download: Callable[[str], Path],
) -> list[dict[str, Any]]:
    actual = {str(item.rfilename): item for item in siblings}
    wanted = {str(row["relative_path"]): row for row in expected}
    # The Hub creates this LFS routing file when a repository first receives
    # a large object. It is platform metadata rather than release payload;
    # record it separately while rejecting every other unexpected object.
    hub_generated = {".gitattributes"}
    missing = set(wanted) - set(actual)
    extra = set(actual) - set(wanted) - hub_generated
    require(not missing and not extra, f"Hub file inventory differs from frozen checkpoint: missing={sorted(missing)}, extra={sorted(extra)}")
    checked: list[dict[str, Any]] = []
    for relative, row in sorted(wanted.items()):
        sibling = actual[relative]
        expected_bytes = int(row["bytes"])
        observed_bytes = int(getattr(sibling, "size", 0) or 0)
        require(observed_bytes == expected_bytes, f"Hub byte size drift: {relative}")
        oid = lfs_oid(sibling)
        if oid:
            require(oid == row["sha256"], f"Hub LFS SHA-256 drift: {relative}")
            method = "hub_lfs_oid"
        else:
            local = download(relative)
            require(local.is_file(), f"Hub download missing: {relative}")
            require(sha256_file(local) == row["sha256"], f"Hub content SHA-256 drift: {relative}")
            method = "downloaded_content_sha256"
        checked.append({"relative_path": relative, "bytes": expected_bytes, "sha256": row["sha256"], "method": method})
    generated = sorted(set(actual) & hub_generated)
    if generated:
        checked.append({"relative_path": ".gitattributes", "bytes": int(getattr(actual[".gitattributes"], "size", 0) or 0), "sha256": None, "method": "hub_generated_lfs_routing_metadata"})
    return checked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--upload-receipt", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    token = os.environ.get("HF_TOKEN")
    require(bool(token), "HF_TOKEN must be injected per command")
    from huggingface_hub import HfApi, hf_hub_download

    frozen = read_json(args.freeze_receipt)
    upload = read_json(args.upload_receipt)
    require(frozen.get("status") == "passed", "freeze receipt is not passed")
    require(upload.get("status") == "completed", "upload receipt is not completed")
    require(upload.get("checkpoint_tree_sha256") == frozen.get("tree_sha256"), "upload/freeze tree drift")
    repo_id, revision = str(upload["repo_id"]), str(upload["revision"])
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type="model", revision=revision, files_metadata=True)
    require(info.sha == revision, "Hub revision drift")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    checked = inventory_check(
        list(frozen["files"]), list(info.siblings),
        lambda relative: Path(hf_hub_download(repo_id=repo_id, repo_type="model", revision=revision, filename=relative, token=token, local_dir=args.cache_dir)),
    )
    result = {"schema_version": "apertus_model_checkpoint_full_hub_inventory_v1", "status": "passed", "repo_id": repo_id, "revision": revision, "private": bool(info.private), "freeze_receipt_sha256": sha256_file(args.freeze_receipt), "files": checked}
    write_json(args.output, result)
    print(json.dumps({"ok": True, "files": len(checked), "revision": revision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
