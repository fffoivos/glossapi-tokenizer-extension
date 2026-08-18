#!/usr/bin/env python3
"""Resume-upload one manifest-bound frozen dataset to Hugging Face.

This is intentionally restricted to the two full-8B dataset manifest schemas.
It refuses a public upload unless the Modern-Greek snapshot is explicitly
verified, and refuses a private upload unless the D0 payload hash sweep passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


PUBLIC_SCHEMA = "apertus_full8_modern_greek_train_snapshot_v1"
PRIVATE_SCHEMA = "apertus_full8_d0_private_portable_dataset_v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def manifest_paths(stage: Path, manifest: dict[str, Any], *, verify_content_hashes: bool = True) -> list[str]:
    rows = manifest.get("upload_payload_inventory")
    require(isinstance(rows, list) and rows, "frozen dataset manifest has no upload inventory")
    paths: list[str] = []
    for row in rows:
        relative = Path(str(row.get("relative_path", "")))
        require(str(relative) and not relative.is_absolute() and ".." not in relative.parts, "nonportable manifest path")
        path = stage / relative
        require(path.is_file(), f"manifest file missing: {relative}")
        require(path.stat().st_size == int(row["bytes"]), f"manifest file byte drift: {relative}")
        if verify_content_hashes:
            require(sha256_file(path) == row["sha256"], f"manifest file checksum drift: {relative}")
        paths.append(relative.as_posix())
    return sorted(set(["manifest.json", *paths]))


def verify_private_payload_receipt(manifest: dict[str, Any]) -> None:
    verification = manifest.get("hash_verification", {})
    receipt = Path(str(verification.get("receipt", "")))
    require(receipt.is_file(), "private upload lacks payload hash verification receipt")
    require(sha256_file(receipt) == verification.get("sha256"), "private payload verification receipt drift")
    value = json.loads(receipt.read_text(encoding="utf-8"))
    require(value.get("schema_version") == "apertus_full8_d0_private_payload_hash_verification_v1" and value.get("status") == "passed", "private payload verification receipt is not passing")
    expected = {str(row["relative_path"]): (int(row["bytes"]), str(row["sha256"])) for row in manifest["upload_payload_inventory"]}
    observed = {str(row["relative_path"]): (int(row["bytes"]), str(row["sha256"])) for row in value.get("files", [])}
    require(observed == expected, "private payload verification receipt does not bind the upload inventory")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(1 <= args.workers <= 32, "workers must be in [1,32]")
    token = os.environ.get("HF_TOKEN")
    require(bool(token), "HF_TOKEN must be injected per command")
    stage = args.stage_root.resolve()
    manifest_path = stage / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema, status = manifest.get("schema_version"), manifest.get("status")
    if args.public:
        require(schema == PUBLIC_SCHEMA and status == "verified", "public upload requires a verified Modern-Greek snapshot")
        require(args.repo_id == "fffoivos/apertus-8b-greek-cpt-modern-greek-train", "unexpected public repository")
    else:
        require(schema == PRIVATE_SCHEMA and status == "verified_payload_hashes", "private upload requires a hash-verified exact D0 stage")
        require(args.repo_id == "fffoivos/apertus-8b-greek-cpt-d0-full-mix", "unexpected private repository")
        verify_private_payload_receipt(manifest)
    from huggingface_hub import HfApi

    paths = manifest_paths(stage, manifest, verify_content_hashes=args.public)
    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=stage, revision=args.revision, private=not args.public, allow_patterns=paths, ignore_patterns=["*.partial", "logs/**"], num_workers=args.workers, print_report=True, print_report_every=60)
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset", revision=args.revision, files_metadata=True)
    expected = set(paths)
    actual = {str(row.rfilename) for row in info.siblings}
    require(expected == actual, f"Hub dataset inventory differs: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}")
    result = {"schema_version": "apertus_full8_frozen_dataset_hf_upload_v1", "status": "completed", "repo_id": args.repo_id, "revision": info.sha, "private": bool(info.private), "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)}, "files": len(paths), "training_must_pin_revision": info.sha}
    write_json(args.output, result)
    print(json.dumps({"ok": True, "repo_id": args.repo_id, "revision": info.sha, "files": len(paths)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
