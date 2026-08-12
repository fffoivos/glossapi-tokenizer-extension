#!/usr/bin/env python3
"""Publish a verified contamination-audit payload beside an HF dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload-dir", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--commit-message", default="Add native Greek benchmark contamination evidence")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    args = parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    manifest_path = args.payload_dir / "publish_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "passed" or manifest.get("schema_version") != "greek_benchmark_contamination_hf_payload_v1":
        raise ValueError("invalid publish manifest")
    files = manifest["files"]
    for filename, expected in files.items():
        path = args.payload_dir / filename
        if not path.is_file() or path.stat().st_size != int(expected["bytes"]) or sha256(path) != expected["sha256"]:
            raise ValueError(f"payload drift: {path}")
    token = args.token_file.read_text().strip()
    if not token:
        raise ValueError("empty HF token file")
    api = HfApi(token=token)
    repo_id = manifest["repo_id"]
    before = api.dataset_info(repo_id).sha
    target = manifest["target_path"].rstrip("/")

    # If an identical payload is already present, verify and emit an idempotent
    # receipt. Otherwise require that main is still the audited dataset commit.
    remote_matches = True
    verified_remote: dict[str, dict[str, object]] = {}
    for filename, expected in files.items():
        remote_path = f"{target}/{filename}"
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                    revision=before,
                    token=token,
                )
            )
        except Exception:
            remote_matches = False
            break
        measured = sha256(cached)
        if measured != expected["sha256"] or cached.stat().st_size != int(expected["bytes"]):
            remote_matches = False
            break
        verified_remote[remote_path] = {"bytes": cached.stat().st_size, "sha256": measured}

    if remote_matches and len(verified_remote) == len(files):
        after = before
        mode = "already_present"
    else:
        if before != manifest["dataset_revision_audited"]:
            raise ValueError(
                f"dataset main moved since the audited revision: {before} != {manifest['dataset_revision_audited']}"
            )
        operations = [
            CommitOperationAdd(path_in_repo=f"{target}/{filename}", path_or_fileobj=args.payload_dir / filename)
            for filename in sorted(files)
        ]
        commit = api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=args.commit_message,
            parent_commit=before,
        )
        after = commit.oid
        mode = "uploaded"
        verified_remote = {}
        for filename, expected in files.items():
            remote_path = f"{target}/{filename}"
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=remote_path,
                    revision=after,
                    token=token,
                    force_download=True,
                )
            )
            measured = sha256(cached)
            if measured != expected["sha256"] or cached.stat().st_size != int(expected["bytes"]):
                raise ValueError(f"remote verification failed: {remote_path}")
            verified_remote[remote_path] = {"bytes": cached.stat().st_size, "sha256": measured}

    receipt = {
        "schema_version": "greek_benchmark_contamination_hf_publication_v1",
        "status": "passed",
        "mode": mode,
        "repo_id": repo_id,
        "dataset_revision_audited": manifest["dataset_revision_audited"],
        "main_before": before,
        "main_after": after,
        "target_path": target,
        "payload_manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "remote_files": verified_remote,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.receipt)
    print(json.dumps({"ok": True, "mode": mode, "main_after": after, "files": len(files)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
