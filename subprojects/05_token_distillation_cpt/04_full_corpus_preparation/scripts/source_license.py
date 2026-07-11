#!/usr/bin/env python3
"""Validate and load the source-specific Phase-04 license adjudication.

The acquisition registry records what a repository declared.  This file is the
separate, default-deny decision layer used by cleaning and release code.  It is
deliberately source-specific: access gating and a coarse license category must
never silently grant either model-training or redistribution rights.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "full_cpt_source_license_adjudication_v1"
APPROVED = "allowed_for_pipeline"
EXCLUDED = "excluded"
DECISION_STATUSES = {APPROVED, EXCLUDED}
HEX = set("0123456789abcdef")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in HEX for character in value)
    )


def _validate_decision(
    row: object,
    *,
    label: str,
    errors: list[str],
) -> tuple[bool, str, list[str]]:
    if not isinstance(row, dict):
        errors.append(f"{label}: must be an object")
        return False, EXCLUDED, []
    status = row.get("status")
    eligible = row.get("eligible")
    conditions = row.get("conditions")
    if status not in DECISION_STATUSES:
        errors.append(f"{label}.status: expected one of {sorted(DECISION_STATUSES)}")
    if not isinstance(eligible, bool):
        errors.append(f"{label}.eligible: must be boolean")
        eligible = False
    if eligible != (status == APPROVED):
        errors.append(
            f"{label}: eligible must be true exactly when status is allowed_for_pipeline"
        )
    if (
        not isinstance(conditions, list)
        or not conditions
        or not all(isinstance(item, str) and item for item in conditions)
        or len(conditions) != len(set(conditions))
    ):
        errors.append(f"{label}.conditions: must be a non-empty unique string list")
        conditions = []
    return bool(eligible), str(status), list(conditions)


def validate_adjudication(
    payload: Mapping[str, Any],
    source_registry: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> list[str]:
    """Return all contract errors without making a legal inference."""

    errors: list[str] = []
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("license_adjudication: unsupported schema_version")
    if payload.get("status") != "technical_audit_complete":
        errors.append("license_adjudication: status must be technical_audit_complete")
    if payload.get("default_policy") != "deny_training_and_redistribution":
        errors.append("license_adjudication: default_policy must remain default-deny")
    if payload.get("purpose") != "noncommercial_research_continued_pretraining":
        errors.append("license_adjudication: purpose drift")

    registry_receipt = payload.get("source_registry")
    if not isinstance(registry_receipt, dict):
        errors.append("license_adjudication.source_registry: must be an object")
        registry_receipt = {}
    expected_registry_sha = registry_receipt.get("sha256")
    if not _is_sha256(expected_registry_sha):
        errors.append("license_adjudication.source_registry.sha256: invalid SHA-256")
    if registry_path is not None:
        if not registry_path.is_file():
            errors.append(f"license_adjudication: source registry is missing: {registry_path}")
        elif expected_registry_sha != sha256_file(registry_path):
            errors.append("license_adjudication: source registry checksum drift")

    registered_sources = source_registry.get("sources")
    if not isinstance(registered_sources, list):
        errors.append("source_registry.sources: must be a list")
        registered_sources = []
    expected: dict[str, dict[str, Any]] = {}
    for item in registered_sources:
        if not isinstance(item, dict) or not isinstance(item.get("source_id"), str):
            errors.append("source_registry.sources: invalid source entry")
            continue
        expected[str(item["source_id"])] = item

    rows = payload.get("sources")
    if not isinstance(rows, list):
        errors.append("license_adjudication.sources: must be a list")
        rows = []
    seen: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(rows):
        label = f"license_adjudication.sources[{index}]"
        if not isinstance(item, dict):
            errors.append(f"{label}: must be an object")
            continue
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            errors.append(f"{label}.source_id: required")
            continue
        if source_id in seen:
            errors.append(f"{label}.source_id: duplicate {source_id!r}")
            continue
        seen[source_id] = item
        registered = expected.get(source_id)
        if registered is None:
            errors.append(f"{label}: source_id is not in sources.json")
            continue
        for field in ("repo_id", "revision"):
            if item.get(field) != registered.get(field):
                errors.append(f"{label}.{field}: does not match sources.json")
        if item.get("registry_training_eligibility") != registered.get("training_eligibility"):
            errors.append(f"{label}.registry_training_eligibility: does not match sources.json")
        declared = item.get("declared_license")
        if declared is not None and (not isinstance(declared, str) or not declared):
            errors.append(f"{label}.declared_license: must be null or a non-empty string")

        training, _, training_conditions = _validate_decision(
            item.get("local_training"), label=f"{label}.local_training", errors=errors
        )
        redistribution, _, _ = _validate_decision(
            item.get("redistribution"), label=f"{label}.redistribution", errors=errors
        )
        if redistribution and not training:
            errors.append(f"{label}: redistribution approval requires local-training approval")
        if training and isinstance(declared, str) and "-nc" in declared:
            required = {"noncommercial_research_only", "no_public_redistribution"}
            if not required <= set(training_conditions):
                errors.append(
                    f"{label}: noncommercial training approval lacks conditions {sorted(required)}"
                )
        if redistribution and isinstance(declared, str) and "-nc" in declared:
            errors.append(f"{label}: noncommercial source cannot enter the public release")
        if training and isinstance(declared, str) and "-nd" in declared:
            errors.append(f"{label}: NoDerivatives source cannot be approved for cleaned CPT")

        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{label}.evidence: non-empty list required")
            evidence = []
        immutable_card = False
        supports_redistribution = False
        upstream_terms = False
        for evidence_index, record in enumerate(evidence):
            evidence_label = f"{label}.evidence[{evidence_index}]"
            if not isinstance(record, dict):
                errors.append(f"{evidence_label}: must be an object")
                continue
            kind = record.get("kind")
            url = record.get("url")
            if kind not in {"hf_repo_metadata", "hf_dataset_card", "upstream_terms"}:
                errors.append(f"{evidence_label}.kind: unsupported")
            if not isinstance(url, str) or not url.startswith("https://"):
                errors.append(f"{evidence_label}.url: HTTPS URL required")
            if kind in {"hf_repo_metadata", "hf_dataset_card"}:
                if record.get("revision") != item.get("revision"):
                    errors.append(f"{evidence_label}.revision: must match the pinned source")
                if isinstance(url, str) and str(item.get("revision")) not in url:
                    errors.append(f"{evidence_label}.url: must embed the pinned revision")
                if kind == "hf_dataset_card":
                    immutable_card = True
            if kind == "upstream_terms":
                upstream_terms = True
            if record.get("supports_redistribution") is True:
                supports_redistribution = True
        if not immutable_card:
            errors.append(f"{label}: immutable pinned HF dataset-card evidence is required")
        if redistribution and not supports_redistribution:
            errors.append(f"{label}: redistribution approval lacks explicit supporting evidence")
        if redistribution and declared in {None, "other"} and not upstream_terms:
            errors.append(f"{label}: nonstandard license approval requires upstream terms")

    if set(seen) != set(expected):
        errors.append(
            "license_adjudication: candidate coverage differs from sources.json; "
            f"missing={sorted(set(expected) - set(seen))}, "
            f"unexpected={sorted(set(seen) - set(expected))}"
        )

    base = payload.get("base")
    registered_base = source_registry.get("base")
    if not isinstance(base, dict) or not isinstance(registered_base, dict):
        errors.append("license_adjudication.base: invalid base binding")
    else:
        if base.get("source_id") != "nanochat_base":
            errors.append("license_adjudication.base.source_id must be nanochat_base")
        for field in ("repo_id", "revision"):
            if base.get(field) != registered_base.get(field):
                errors.append(f"license_adjudication.base.{field}: does not match sources.json")
        base_training, _, _ = _validate_decision(
            base.get("local_training"), label="license_adjudication.base.local_training", errors=errors
        )
        base_redistribution, _, _ = _validate_decision(
            base.get("redistribution"), label="license_adjudication.base.redistribution", errors=errors
        )
        if base_redistribution and not base_training:
            errors.append("license_adjudication.base: redistribution requires training approval")
        base_evidence = base.get("evidence")
        if not isinstance(base_evidence, list) or not any(
            isinstance(record, dict)
            and record.get("kind") == "hf_dataset_card"
            and record.get("revision") == base.get("revision")
            and str(base.get("revision")) in str(record.get("url", ""))
            for record in base_evidence or []
        ):
            errors.append("license_adjudication.base: immutable pinned HF card evidence required")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        errors.append("license_adjudication.summary: must be an object")
    else:
        expected_candidate_training = sorted(
            source_id
            for source_id, row in seen.items()
            if bool(row.get("local_training", {}).get("eligible"))
        )
        expected_candidate_redistribution = sorted(
            source_id
            for source_id, row in seen.items()
            if bool(row.get("redistribution", {}).get("eligible"))
        )
        expected_counts = {
            "candidate_sources": len(seen),
            "candidate_local_training_allowed_sources": len(expected_candidate_training),
            "candidate_redistribution_allowed_sources": len(expected_candidate_redistribution),
        }
        for key, value in expected_counts.items():
            if summary.get(key) != value:
                errors.append(f"license_adjudication.summary.{key}: expected {value}")
        if summary.get("base_local_training_allowed") is not bool(
            isinstance(base, dict) and base.get("local_training", {}).get("eligible")
        ):
            errors.append("license_adjudication.summary.base_local_training_allowed: drift")
        if summary.get("candidate_local_training_allowed_source_ids") != expected_candidate_training:
            errors.append(
                "license_adjudication.summary.candidate_local_training_allowed_source_ids: drift"
            )
        if summary.get("candidate_redistribution_allowed_source_ids") != expected_candidate_redistribution:
            errors.append(
                "license_adjudication.summary.candidate_redistribution_allowed_source_ids: drift"
            )
    return errors


def load_adjudication(
    path: Path,
    *,
    source_registry_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load a completed matrix and return exact per-acquisition-source decisions."""

    payload = read_json_object(path)
    registry = read_json_object(source_registry_path)
    errors = validate_adjudication(payload, registry, registry_path=source_registry_path)
    if errors:
        raise ValueError(f"{path}: invalid source-license adjudication: {'; '.join(errors)}")
    rows = [payload["base"], *payload["sources"]]
    return {
        str(row["source_id"]): {
            "registry_training_eligibility": (
                "inherited_base"
                if row["source_id"] == "nanochat_base"
                else str(row["registry_training_eligibility"])
            ),
            "training_eligible": bool(row["local_training"]["eligible"]),
            "redistribution_eligible": bool(row["redistribution"]["eligible"]),
            "training_status": str(row["local_training"]["status"]),
            "redistribution_status": str(row["redistribution"]["status"]),
        }
        for row in rows
    }


def decision_for(
    source_id: str,
    eligibility_category: str,
    decisions: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return a source decision, failing closed on missing or mismatched identity."""

    decision = decisions.get(source_id)
    if decision is None:
        raise ValueError(f"source {source_id!r} has no license adjudication")
    expected_category = decision.get("registry_training_eligibility")
    if eligibility_category != expected_category:
        raise ValueError(
            f"source {source_id!r} eligibility category drift: "
            f"{eligibility_category!r} != {expected_category!r}"
        )
    return decision
