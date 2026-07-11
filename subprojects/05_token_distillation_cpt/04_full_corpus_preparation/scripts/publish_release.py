#!/usr/bin/env python3
"""Dry-run by default; publish only the validated redistribution release."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from finalization_io import discover_parquet, read_json_object, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--validation-receipt", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--gate-mode", choices=["manual", "auto"], default="manual")
    parser.add_argument("--execute", action="store_true", help="Actually create/update and upload")
    parser.add_argument("--token", default=None, help="Defaults to HF_TOKEN; never written to receipts")
    args = parser.parse_args()
    manifest = read_json_object(args.release_manifest)
    validation = read_json_object(args.validation_receipt)
    if manifest.get("schema_version") != "full_cpt_release_manifest_v1":
        raise ValueError("release manifest schema is unsupported")
    if validation.get("schema_version") != "full_cpt_release_validation_v1" or validation.get("status") != "passed":
        raise ValueError("publication requires a passed full-corpus validation receipt")
    if validation.get("release_manifest_sha256") != sha256_file(args.release_manifest):
        raise ValueError("validation receipt is bound to a different release manifest")
    if Path(str(manifest.get("output"))).resolve() != args.release.resolve():
        raise ValueError("release root differs from the immutable manifest")
    redistribution = args.release / str(manifest["redistribution_root"])
    if redistribution.resolve().is_relative_to((args.release / "training").resolve()):
        raise ValueError("refusing to publish a training-only path")
    files = discover_parquet(redistribution)
    inventory = {
        "repo_id": args.repo_id,
        "gate_mode": args.gate_mode,
        "redistribution_root": str(redistribution.resolve()),
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "rows": int(manifest["counts"]["redistribution_rows"]),
        "training_rows_not_uploaded": int(manifest["counts"]["training_rows"]) - int(manifest["counts"]["redistribution_rows"]),
        "execute": args.execute,
    }
    if not args.execute:
        print(json.dumps({"ok": True, "dry_run": True, **inventory}, sort_keys=True))
        return 0
    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("--execute requires --token or HF_TOKEN")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=False, exist_ok=True)
    api.update_repo_settings(repo_id=args.repo_id, repo_type="dataset", gated=args.gate_mode, private=False)
    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=str(redistribution),
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(args.release_manifest),
        path_in_repo="provenance/release_manifest.json",
    )
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(args.validation_receipt),
        path_in_repo="provenance/validation_receipt.json",
    )
    waterfall = Path(str(manifest["token_waterfall"]))
    api.upload_file(
        repo_id=args.repo_id,
        repo_type="dataset",
        path_or_fileobj=str(waterfall),
        path_in_repo="provenance/token_waterfall.json",
    )
    print(json.dumps({"ok": True, "dry_run": False, **inventory}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
