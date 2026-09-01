#!/usr/bin/env python3
"""Re-hash the release and require exact agreement with its manifest and Hub commit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .contracts import PREFLIGHT_SCHEMA, atomic_write_json, load_json, sha256_file


def run(args: argparse.Namespace) -> dict:
    release = Path(args.release).resolve()
    manifest_path = release / "manifests" / "deduplicated_manifest.json"
    manifest = load_json(manifest_path)
    publication = load_json(args.publication_receipt)
    problems: list[str] = []
    if manifest["input_rows"] - manifest["removed_rows"] != manifest["rows"]:
        problems.append("release manifest row waterfall does not close")
    expected_names = sorted(Path(entry["path"]).name for entry in manifest["files"])
    actual_names = sorted(path.name for path in (release / "data").glob("*.parquet"))
    if actual_names != expected_names:
        problems.append("release data file set differs from the manifest")

    local_hashes: dict[str, str] = {}
    rows = 0
    bytes_total = 0
    drift: list[dict] = []
    for entry in manifest["files"]:
        path = release / entry["path"]
        if path.is_symlink() or not path.is_file():
            drift.append({"path": entry["path"], "problem": "missing or symlinked"})
            continue
        digest = sha256_file(path)
        size = path.stat().st_size
        local_hashes[entry["path"]] = digest
        rows += int(entry["rows"])
        bytes_total += size
        if digest != entry["sha256"] or size != int(entry["bytes"]):
            drift.append(
                {
                    "path": entry["path"],
                    "expected_sha256": entry["sha256"],
                    "actual_sha256": digest,
                    "expected_bytes": entry["bytes"],
                    "actual_bytes": size,
                }
            )
    if drift:
        problems.append(f"{len(drift)} release files drifted from the manifest")
    if rows != manifest["rows"]:
        problems.append(
            f"manifest file rows sum to {rows}, expected {manifest['rows']}"
        )

    hub_check: dict
    try:
        from huggingface_hub import HfApi, get_token

        token = os.environ.get("HF_TOKEN") or get_token()
        if not token:
            raise RuntimeError("HF_TOKEN is required for the Hub identity check")
        commit = publication["commit_sha"]
        tree = HfApi(token=token).list_repo_tree(
            manifest["repository_id"],
            repo_type="dataset",
            revision=commit,
            recursive=True,
            expand=True,
        )
        remote = {
            item.path: item.lfs.sha256
            for item in tree
            if getattr(item, "lfs", None) is not None
        }
        missing = sorted(set(local_hashes) - set(remote))
        mismatched = sorted(
            path
            for path, digest in local_hashes.items()
            if path in remote and remote[path] != digest
        )
        hub_check = {
            "ok": not missing and not mismatched,
            "repository_id": manifest["repository_id"],
            "commit_sha": commit,
            "compared": len(local_hashes),
            "missing": missing,
            "mismatched": mismatched,
        }
        if not hub_check["ok"]:
            problems.append("release differs from the pinned Hub commit")
    except Exception as error:  # noqa: BLE001 - every Hub failure must fail closed
        hub_check = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        problems.append("Hub identity check did not complete successfully")

    result = {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "passed" if not problems else "failed",
        "release": str(release),
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "repository_id": manifest["repository_id"],
            "files": len(manifest["files"]),
            "rows": rows,
            "bytes": bytes_total,
        },
        "local_vs_manifest": {"ok": not drift, "drift": drift},
        "local_vs_hub": hub_check,
        "problems": problems,
    }
    atomic_write_json(args.output, result)
    print(result["status"], args.output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", required=True)
    parser.add_argument("--publication-receipt", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    raise SystemExit(result["status"] != "passed")
