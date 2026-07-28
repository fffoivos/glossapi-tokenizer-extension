"""Shared hashing, atomic-write, and run-contract validation helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA = "bibliography-cleaning-run-contract-v2"
PLAN_SCHEMA = "bibliography-cleaning-work-plan-v2"
RECEIPT_SCHEMA = "bibliography-cleaning-unit-receipt-v2"
LEDGER_SCHEMA = "bibliography-cleaning-document-ledger-v2"
PREFLIGHT_SCHEMA = "bibliography-cleaning-preflight-v2"
PARITY_SCHEMA = "bibliography-cleaning-parity-v2"
SUMMARY_SCHEMA = "bibliography-cleaning-dryrun-summary-v2"
APPLY_SUMMARY_SCHEMA = "bibliography-cleaning-apply-summary-v1"
QA_PACKET_SCHEMA = "bibliography-cleaning-qa-packet-v2"
QA_REVIEW_SCHEMA = "bibliography-cleaning-qa-review-v2"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".partial", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def require_schema(value: dict[str, Any], expected: str, path: str | Path) -> None:
    actual = value.get("schema_version")
    if actual != expected:
        raise ValueError(f"{path}: schema {actual!r}, expected {expected!r}")


def unit_id(rank: int, row_group_start: int, row_group_end: int) -> str:
    if rank < 0 or row_group_start < 0 or row_group_end <= row_group_start:
        raise ValueError("invalid unit coordinates")
    return f"{rank:06d}-rg{row_group_start:04d}-{row_group_end:04d}"


def validate_contract(path: str | Path) -> tuple[dict[str, Any], str]:
    contract_path = Path(path).resolve()
    contract = load_json(contract_path)
    require_schema(contract, CONTRACT_SCHEMA, contract_path)
    if contract["mode"] not in {"dry-run", "apply"}:
        raise ValueError(f"invalid run mode {contract['mode']!r}")
    if contract["publication_authorized"]:
        raise ValueError("this runner does not accept publication authorization")
    policy = contract["policy"]
    if not policy["model_policy"]["character_damage_measure_approved"]:
        raise ValueError("character-damage measure has not been approved")
    if policy["kallipos_apply_authorized"]:
        raise ValueError("Kallipos must remain outside the apply scope")
    override = policy["license_overrides"]["glossAPI/libduth"]
    if (
        override["public_redistribution"] is not True
        or "v2 public release" not in override["scope"]
        or not override.get("approved_by")
    ):
        raise ValueError("libduth requires the explicit owner-approved v2 public override")
    publication = policy["publication_target"]
    if (
        policy["publication_authorized"] is not True
        or publication["repo_id"]
        != "fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2"
        or publication["visibility"] != "public"
    ):
        raise ValueError("the owner-approved public v2 publication policy is absent")
    train_archive = contract["code"]["train_archive"]
    if sha256_file(train_archive["path"]) != train_archive["sha256"]:
        raise ValueError("train source archive hash mismatch")
    archive = contract["code"]["glossapi_archive"]
    if sha256_file(archive["path"]) != archive["sha256"]:
        raise ValueError("GlossAPI source archive hash mismatch")
    wheel = contract["code"]["glossapi_wheel"]
    if sha256_file(wheel["path"]) != wheel["sha256"]:
        raise ValueError("GlossAPI extension wheel hash mismatch")
    artifact_record = contract["model_artifacts"]
    artifact_manifest = Path(artifact_record["path"])
    if sha256_file(artifact_manifest) != artifact_record["manifest_sha256"]:
        raise ValueError("model artifact manifest hash mismatch")
    artifact_root = artifact_manifest.parent
    for stage_name, stage in artifact_record["stages"].items():
        stage_path = artifact_root / stage["file"]
        if stage_path.stat().st_size != stage["bytes"]:
            raise ValueError(f"{stage_name} artifact size mismatch")
        if sha256_file(stage_path) != stage["sha256"]:
            raise ValueError(f"{stage_name} artifact hash mismatch")
    evidence_records = {}
    for evidence_name in ("preflight", "parity"):
        evidence = contract["evidence"][evidence_name]
        evidence_path = Path(evidence["path"])
        if sha256_file(evidence_path) != evidence["sha256"]:
            raise ValueError(f"{evidence_name} receipt hash mismatch")
        record = load_json(evidence_path)
        if record.get("status") != "passed":
            raise ValueError(f"{evidence_name} did not pass")
        evidence_records[evidence_name] = record
    preflight = evidence_records["preflight"]
    require_schema(
        preflight, PREFLIGHT_SCHEMA, contract["evidence"]["preflight"]["path"]
    )
    if Path(preflight["release"]).resolve() != Path(contract["release_root"]).resolve():
        raise ValueError("preflight receipt covers a different release")
    parity = evidence_records["parity"]
    require_schema(parity, PARITY_SCHEMA, contract["evidence"]["parity"]["path"])
    if (
        parity["train_commit"] != contract["code"]["train_commit"]
        or parity["glossapi_commit"] != contract["code"]["glossapi_commit"]
        or parity["artifact_manifest_sha256"] != artifact_record["manifest_sha256"]
        or parity["equal_masks"] != parity["expected_lines"]
        or parity["candidate_positives"] != parity["expected_positives"]
    ):
        raise ValueError("parity receipt is not bound to this code/model contract")
    return contract, sha256_file(contract_path)


def validate_plan(
    contract_path: str | Path, plan_path: str | Path
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    contract, contract_sha = validate_contract(contract_path)
    plan = load_json(plan_path)
    require_schema(plan, PLAN_SCHEMA, plan_path)
    plan_sha = sha256_file(plan_path)
    if plan["contract_sha256"] != contract_sha:
        raise ValueError("plan is bound to a different run contract")
    expected = {
        unit_id(unit["rank"], unit["row_group_start"], unit["row_group_end"])
        for unit in plan["units"]
    }
    actual = {unit["unit_id"] for unit in plan["units"]}
    if actual != expected or len(actual) != len(plan["units"]):
        raise ValueError("work-plan unit IDs are invalid or duplicated")
    return contract, plan, contract_sha, plan_sha
