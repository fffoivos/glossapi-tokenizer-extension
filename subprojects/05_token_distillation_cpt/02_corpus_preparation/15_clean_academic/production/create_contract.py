#!/usr/bin/env python3
"""Create an immutable run contract after preflight and exact-commit parity pass."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from pathlib import Path

from .contracts import (
    CONTRACT_SCHEMA,
    PARITY_SCHEMA,
    PREFLIGHT_SCHEMA,
    atomic_write_json,
    canonical_bytes,
    load_json,
    require_schema,
    sha256_file,
)


def run(args: argparse.Namespace) -> Path:
    policy = load_json(args.policy)
    preflight = load_json(args.preflight)
    parity = load_json(args.parity)
    require_schema(preflight, PREFLIGHT_SCHEMA, args.preflight)
    require_schema(parity, PARITY_SCHEMA, args.parity)
    if preflight.get("status") != "passed" or parity.get("status") != "passed":
        raise ValueError("preflight and parity must both pass before contract creation")
    if policy["kallipos_apply_authorized"]:
        raise ValueError("reviewed policy must keep Kallipos outside apply scope")

    artifacts = load_json(args.artifact_manifest)
    if artifacts.get("schema_version") != "bibliography-line-model-export-v2":
        raise ValueError("the model artifact manifest must use the v2 hash contract")
    created = args.created_utc or dt.datetime.now(dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )
    payload = {
        "schema_version": CONTRACT_SCHEMA,
        "created_utc": created,
        "mode": args.mode,
        "release_root": str(Path(args.release_root).resolve()),
        "run_base": str(Path(args.run_base).resolve()),
        "policy": policy,
        "code": {
            "train_commit": args.train_commit,
            "glossapi_commit": args.glossapi_commit,
            "train_archive": {
                "path": str(Path(args.train_archive).resolve()),
                "sha256": sha256_file(args.train_archive),
            },
            "glossapi_archive": {
                "path": str(Path(args.glossapi_archive).resolve()),
                "sha256": sha256_file(args.glossapi_archive),
            },
            "glossapi_wheel": {
                "path": str(Path(args.glossapi_wheel).resolve()),
                "sha256": sha256_file(args.glossapi_wheel),
            },
        },
        "model_artifacts": {
            "path": str(Path(args.artifact_manifest).resolve()),
            "manifest_sha256": sha256_file(args.artifact_manifest),
            "stages": artifacts["stages"],
        },
        "evidence": {
            "preflight": {
                "path": str(Path(args.preflight).resolve()),
                "sha256": sha256_file(args.preflight),
            },
            "parity": {
                "path": str(Path(args.parity).resolve()),
                "sha256": sha256_file(args.parity),
            },
        },
        "size_column_policy": "recompute only where the source value is non-null",
        # This contract controls scoring/materialization only. Publication remains a
        # later receipt-bound action even when the owner policy authorizes its target.
        "publication_authorized": False,
    }
    identity_sha = hashlib.sha256(canonical_bytes(payload)).hexdigest()
    run_id = f"{created}-{identity_sha[:12]}"
    payload["identity_payload_sha256"] = identity_sha
    payload["run_id"] = run_id
    run_root = Path(args.run_base).resolve() / run_id
    payload["run_root"] = str(run_root)
    contract_path = run_root / "contract.json"
    if run_root.exists():
        raise FileExistsError(run_root)
    run_root.mkdir(parents=True)
    atomic_write_json(contract_path, payload)
    print(contract_path)
    return contract_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--parity", required=True)
    parser.add_argument("--artifact-manifest", required=True)
    parser.add_argument("--train-archive", required=True)
    parser.add_argument("--glossapi-archive", required=True)
    parser.add_argument("--glossapi-wheel", required=True)
    parser.add_argument("--train-commit", required=True)
    parser.add_argument("--glossapi-commit", required=True)
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--run-base", required=True)
    parser.add_argument("--mode", choices=("dry-run", "apply"), default="dry-run")
    parser.add_argument("--created-utc")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
