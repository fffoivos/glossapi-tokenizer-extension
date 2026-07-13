#!/usr/bin/env python3
"""Validate the explicit human gate between v4 raw review and field discovery."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent1_v4_raw_review import (  # noqa: E402
    file_binding,
    read_json_object,
    sha256_file,
    validate_packet,
    write_json_no_replace,
)
from freeze_agent1_v4_review import FREEZE_SCHEMA  # noqa: E402


DECISION_SCHEMA = "agent1_v4_human_decision_bundle_v1"
RECEIPT_SCHEMA = "agent1_v4_human_review_gate_receipt_v1"
SOURCE_STATUSES = frozenset({"admit", "hold", "exclude"})
DOCUMENT_DISPOSITIONS = frozenset({"agree", "override", "flag", "hold"})


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: request must be an object")
        rows.append(value)
    return rows


def _strict_mapping(value: object, expected_keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        missing = sorted(expected_keys - set(value) if isinstance(value, Mapping) else expected_keys)
        extra = sorted(set(value) - expected_keys) if isinstance(value, Mapping) else []
        raise ValueError(f"{label} does not close selected sources; missing={missing}, extra={extra}")
    return value


def _validate_document_decisions(
    decisions: Mapping[str, object], requests: Mapping[str, Mapping[str, object]]
) -> Counter[str]:
    rows = _strict_mapping(decisions.get("documents"), set(requests), "document decisions")
    counts: Counter[str] = Counter()
    for request_id, request in requests.items():
        row = rows[request_id]
        if not isinstance(row, Mapping):
            raise ValueError(f"document decision is not an object: {request_id}")
        expected = {
            "source_id",
            "source_doc_id",
            "disposition",
            "cleanliness_score_override",
            "text_quality_score_override",
            "note",
        }
        if set(row) != expected:
            raise ValueError(f"document decision keys drift: {request_id}")
        if row.get("source_id") != request.get("source_id") or row.get("source_doc_id") != request.get("source_doc_id"):
            raise ValueError(f"document decision identity drift: {request_id}")
        disposition = row.get("disposition")
        if disposition not in DOCUMENT_DISPOSITIONS:
            raise ValueError(f"document has no final disposition: {request_id}")
        overrides = (row.get("cleanliness_score_override"), row.get("text_quality_score_override"))
        if any(value is not None and (not isinstance(value, int) or not 1 <= value <= 5) for value in overrides):
            raise ValueError(f"document score override is invalid: {request_id}")
        if disposition == "override" and overrides == (None, None):
            raise ValueError(f"override disposition lacks a score override: {request_id}")
        if disposition != "override" and overrides != (None, None):
            raise ValueError(f"non-override disposition carries a score override: {request_id}")
        if not isinstance(row.get("note"), str):
            raise ValueError(f"document note is invalid: {request_id}")
        counts[str(disposition)] += 1
    return counts


def validate_human_decisions(
    *,
    packet_root: Path,
    packet_manifest_path: Path,
    freeze_receipt_path: Path,
    decisions_path: Path,
    output: Path,
) -> dict[str, object]:
    """Close Stage 20 only after every frozen document and source is decided."""

    output = output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"human gate receipt already exists: {output}")
    manifest = validate_packet(packet_root, packet_manifest_path)
    freeze = read_json_object(freeze_receipt_path)
    if freeze.get("schema_version") != FREEZE_SCHEMA or freeze.get("status") != "passed":
        raise ValueError("freeze receipt is not passed v4 Stage-00 evidence")
    scope = freeze.get("review_scope")
    if not isinstance(scope, Mapping) or not isinstance(scope.get("source_ids"), list):
        raise ValueError("freeze receipt lacks source scope")
    source_ids = [str(value) for value in scope["source_ids"]]
    expected_count = manifest.get("logical_review_count")
    if (
        len(source_ids) != 18
        or len(set(source_ids)) != 18
        or not isinstance(expected_count, int)
        or scope.get("logical_review_count") != expected_count
    ):
        raise ValueError("freeze/packet scope closure drift")
    decisions = read_json_object(decisions_path)
    expected_keys = {
        "schema_version",
        "packet_manifest_sha256",
        "approval_to_begin_field_discovery",
        "source_status",
        "source_observations",
        "mapping_questions",
        "documents",
    }
    if set(decisions) != expected_keys or decisions.get("schema_version") != DECISION_SCHEMA:
        raise ValueError("unsupported human decision bundle schema")
    packet_sha256 = sha256_file(packet_manifest_path)
    if decisions.get("packet_manifest_sha256") != packet_sha256:
        raise ValueError("human decision bundle is bound to a different packet manifest")
    if decisions.get("approval_to_begin_field_discovery") is not True:
        raise ValueError("human gate lacks approval to begin field discovery")
    source_key_set = set(source_ids)
    statuses = _strict_mapping(decisions.get("source_status"), source_key_set, "source status")
    observations = _strict_mapping(decisions.get("source_observations"), source_key_set, "source observations")
    mapping_questions = _strict_mapping(decisions.get("mapping_questions"), source_key_set, "mapping questions")
    if any(statuses[source_id] not in SOURCE_STATUSES for source_id in source_ids):
        raise ValueError("one or more source statuses are invalid")
    if any(not isinstance(observations[source_id], str) or not isinstance(mapping_questions[source_id], str) for source_id in source_ids):
        raise ValueError("source observation/mapping question must be strings")
    frozen_sources = freeze.get("sources")
    if not isinstance(frozen_sources, list):
        raise ValueError("freeze receipt lacks per-source evidence")
    frozen_by_source = {
        str(row.get("source_id")): row for row in frozen_sources if isinstance(row, Mapping)
    }
    if set(frozen_by_source) != source_key_set:
        raise ValueError("freeze receipt per-source closure drift")
    admitted: list[str] = []
    for source_id in source_ids:
        if statuses[source_id] != "admit":
            continue
        license_row = frozen_by_source[source_id].get("license")
        if not isinstance(license_row, Mapping) or license_row.get("local_training_eligible") is not True:
            raise ValueError(f"cannot admit source lacking eligible local-training evidence: {source_id}")
        admitted.append(source_id)
    requests = _read_jsonl(packet_root / "requests.jsonl")
    by_request = {str(row.get("request_id")): row for row in requests}
    if len(by_request) != expected_count:
        raise ValueError("packet request closure drift")
    disposition_counts = _validate_document_decisions(decisions, by_request)
    receipt: dict[str, object] = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "passed",
        "packet_manifest": file_binding(packet_manifest_path),
        "freeze_receipt": file_binding(freeze_receipt_path),
        "human_decisions": file_binding(decisions_path),
        "logical_review_count": len(by_request),
        "document_disposition_counts": dict(sorted(disposition_counts.items())),
        "source_status": {source_id: statuses[source_id] for source_id in source_ids},
        "admitted_source_ids": admitted,
        "held_source_ids": [source_id for source_id in source_ids if statuses[source_id] == "hold"],
        "excluded_source_ids": [source_id for source_id in source_ids if statuses[source_id] == "exclude"],
    }
    write_json_no_replace(output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--packet-manifest", type=Path, required=True)
    parser.add_argument("--freeze-receipt", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = validate_human_decisions(
        packet_root=args.packet_root,
        packet_manifest_path=args.packet_manifest,
        freeze_receipt_path=args.freeze_receipt,
        decisions_path=args.decisions,
        output=args.output,
    )
    print(json.dumps({"ok": True, "admitted_sources": receipt["admitted_source_ids"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
