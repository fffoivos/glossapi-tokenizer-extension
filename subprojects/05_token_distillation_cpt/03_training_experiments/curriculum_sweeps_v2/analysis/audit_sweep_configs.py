#!/usr/bin/env python3
"""Verify the frozen curriculum-v2 sweep decisions against raw run metadata.

The checked-in manifest can be validated offline. Pass ``--run-root`` on
Clariden to additionally hash and compare every ``run_metadata.json`` and final
checkpoint marker. Comparability means that, after removing run-local fields
and the one intended sweep field, every normalized metadata object is equal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VOLATILE_FIELDS = frozenset({"init_ckpt", "output_dir", "slurm_job_id", "start_time"})


class AuditError(RuntimeError):
    """Raised when a decision or as-run artifact violates the audit contract."""


def normalize(value: Any) -> Any:
    """Normalize JSON values without hiding semantically meaningful changes."""

    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


def stable_json(value: Any) -> str:
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))


def common_payload(metadata: dict[str, Any], varied_fields: set[str]) -> dict[str, Any]:
    ignored = VOLATILE_FIELDS | varied_fields
    return normalize({key: value for key, value in metadata.items() if key not in ignored})


def fingerprint(metadata: dict[str, Any], varied_fields: set[str]) -> str:
    return hashlib.sha256(stable_json(common_payload(metadata, varied_fields)).encode()).hexdigest()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _equal(left: Any, right: Any) -> bool:
    return stable_json(left) == stable_json(right)


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1:
        raise AuditError("unsupported manifest schema_version")

    canonical = manifest.get("canonical", {})
    policy = manifest.get("beta2_policy", {})
    beta2 = manifest.get("sweeps", {}).get("beta2")
    if not beta2:
        raise AuditError("manifest is missing the beta2 sweep")

    comparable = bool(beta2.get("comparable"))
    expected_beta2 = policy.get("selected_if_comparable") if comparable else policy.get("fallback")
    if not _equal(canonical.get("ademamix_beta2"), expected_beta2):
        raise AuditError(
            f"beta2 policy requires {expected_beta2}, got {canonical.get('ademamix_beta2')}"
        )

    if canonical.get("lr_warmup_iters") != policy.get("fixed_lr_warmup_iters"):
        raise AuditError("canonical LR warmup does not match the beta2 comparison")

    for name, sweep in manifest.get("sweeps", {}).items():
        if name != "beta2" and not sweep.get("comparable"):
            raise AuditError(f"{name} sweep is not mechanically comparable")
        varied = sweep.get("varied_fields", [])
        if not varied:
            raise AuditError(f"{name} has no declared varied field")
        expected_fingerprint = sweep.get("normalized_common_sha256", "")
        if len(expected_fingerprint) != 64:
            raise AuditError(f"{name} has an invalid common fingerprint")
        if not sweep.get("runs"):
            raise AuditError(f"{name} has no runs")
        for run in sweep["runs"]:
            if len(run.get("metadata_sha256", "")) != 64:
                raise AuditError(f"{name}/{run.get('run_tag')} has an invalid metadata hash")
            if run.get("final_checkpoint") != 3218:
                raise AuditError(f"{name}/{run.get('run_tag')} did not reach checkpoint 3218")


def audit_live(manifest: dict[str, Any], run_root: Path) -> list[str]:
    """Verify the manifest against a mounted or remote Clariden run root."""

    messages: list[str] = []
    for name, sweep in manifest["sweeps"].items():
        varied_fields = set(sweep["varied_fields"])
        observed_fingerprints: list[str] = []
        for run in sweep["runs"]:
            run_dir = run_root / run["run_tag"]
            metadata_path = run_dir / "run_metadata.json"
            raw = metadata_path.read_bytes()
            observed_hash = sha256_bytes(raw)
            if observed_hash != run["metadata_sha256"]:
                raise AuditError(f"metadata hash mismatch: {metadata_path}")
            metadata = json.loads(raw)

            for field, expected in run.get("varied", {}).items():
                if not _equal(metadata.get(field), expected):
                    raise AuditError(
                        f"{name}/{run['run_tag']} expected {field}={expected!r}, "
                        f"got {metadata.get(field)!r}"
                    )
            for field, expected in sweep.get("required_common", {}).items():
                if not _equal(metadata.get(field), expected):
                    raise AuditError(
                        f"{name}/{run['run_tag']} drifted on {field}: "
                        f"expected {expected!r}, got {metadata.get(field)!r}"
                    )

            observed_fingerprints.append(fingerprint(metadata, varied_fields))
            latest = (run_dir / "checkpoints" / "latest_checkpointed_iteration.txt").read_text().strip()
            if latest != str(run["final_checkpoint"]):
                raise AuditError(f"checkpoint mismatch: {run_dir} expected 3218, got {latest}")

        comparable = len(set(observed_fingerprints)) == 1
        if comparable != bool(sweep["comparable"]):
            raise AuditError(f"{name} comparability changed: observed={comparable}")
        if observed_fingerprints[0] != sweep["normalized_common_sha256"]:
            raise AuditError(f"{name} normalized common fingerprint mismatch")
        messages.append(f"{name}: {len(observed_fingerprints)} mechanically comparable runs")

    return messages


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parent.parent / "results" / "sweep_config_audit_20260711.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=default_manifest_path())
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    validate_manifest(manifest)
    print(f"offline policy audit passed: {args.manifest}")
    if args.run_root:
        for message in audit_live(manifest, args.run_root):
            print(message)
        print("live as-run artifact audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
