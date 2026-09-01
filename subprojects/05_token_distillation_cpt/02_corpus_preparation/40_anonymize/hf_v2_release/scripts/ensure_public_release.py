#!/usr/bin/env python3
"""Make the verified HF dataset public and ungated, then prove anonymous access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx
from huggingface_hub import HfApi


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--expected-sha", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required")
    api = HfApi(token=token)
    info = api.repo_info(repo_id=args.repo_id, repo_type="dataset")
    if str(info.sha) != args.expected_sha:
        raise RuntimeError(f"HF main drifted: {info.sha} != {args.expected_sha}")

    api_url = f"https://huggingface.co/api/datasets/{args.repo_id}"
    with httpx.Client(follow_redirects=True, timeout=60) as client:
        before_response = client.get(api_url)
        before_response.raise_for_status()
        before = before_response.json()

    if args.execute and (before.get("private") is not False or before.get("gated") is not False):
        api.update_repo_settings(
            repo_id=args.repo_id,
            repo_type="dataset",
            gated=False,
            visibility="public",
        )

    with httpx.Client(follow_redirects=True, timeout=60) as client:
        after_response = client.get(api_url)
        after_response.raise_for_status()
        after = after_response.json()
        manifest_url = (
            f"https://huggingface.co/datasets/{args.repo_id}/resolve/"
            f"{args.expected_sha}/manifests/anonymization_manifest.json"
        )
        manifest_response = client.get(manifest_url)

    if str(after.get("sha")) != args.expected_sha:
        raise RuntimeError("repository head changed while access settings were updated")
    if after.get("private") is not False or after.get("gated") is not False:
        raise RuntimeError(f"dataset is not public and ungated: private={after.get('private')}, gated={after.get('gated')}")
    manifest_response.raise_for_status()
    manifest_sha = hashlib.sha256(manifest_response.content).hexdigest()
    if manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError("anonymous manifest checksum mismatch")

    receipt = {
        "schema_version": "glossapi_hf_v2_public_access_v1",
        "status": "passed",
        "completed_at": utc_now(),
        "repo_id": args.repo_id,
        "commit_sha": args.expected_sha,
        "before": {"private": before.get("private"), "gated": before.get("gated")},
        "after": {"private": after.get("private"), "gated": after.get("gated")},
        "anonymous_manifest": {
            "http_status": manifest_response.status_code,
            "bytes": len(manifest_response.content),
            "sha256": manifest_sha,
        },
    }
    if args.execute:
        write_json_atomic(args.receipt, receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
