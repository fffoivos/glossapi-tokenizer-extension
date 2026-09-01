#!/usr/bin/env python3
"""Receipt the public visibility and exact revision of the training dataset."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import urllib.request
from pathlib import Path

from contract import atomic_write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    url = f"https://huggingface.co/api/datasets/{args.repo_id}"
    with urllib.request.urlopen(url, timeout=30) as response:
        value = json.load(response)
        status_code = response.status
    public = status_code == 200 and value.get("private") is False and value.get("disabled") is not True
    revision_matches = value.get("sha") == args.revision
    payload = {
        "schema_version": "apertus_full_8b_hf_visibility_v1",
        "status": "passed" if public and revision_matches else "failed",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "observed_revision": value.get("sha"),
        "public": public,
        "private": value.get("private"),
        "disabled": value.get("disabled"),
        "gated": value.get("gated"),
        "api_url": url,
        "http_status": status_code,
    }
    atomic_write_json(args.output, payload)
    if payload["status"] != "passed":
        raise RuntimeError(f"Hugging Face visibility/revision gate failed: {payload}")
    print(json.dumps({"ok": True, "repo_id": args.repo_id, "public": public, "gated": value.get("gated")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
