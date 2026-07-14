#!/usr/bin/env python3
"""Merge independent role reviews, score agreement, and write overlay v2."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_role_review_runner import (
    BOUNDARIES,
    REVIEW_SCHEMA,
    ROLES,
    load_packet,
    load_provenance,
)
from .contract import canonical_json_sha256, sha256_file


OVERLAY_SCHEMA = "bibliography-role-overlay-v2"
REPORT_SCHEMA = "bibliography-role-dual-review-agreement-v1"
ENTRY_ROLES = frozenset({"ENTRY_ANCHOR"})
ATTACHABLE_ROLES = frozenset({"ENTRY_ANCHOR", "CONTINUATION", "FILLER", "HEADER", "SUBHEADER"})


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_review(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != REVIEW_SCHEMA:
        raise ValueError(f"unsupported review: {path}")
    reviewer, cases = raw.get("reviewer"), raw.get("cases")
    if not isinstance(reviewer, str) or not reviewer or not isinstance(cases, list) or not cases:
        raise ValueError(f"malformed review: {path}")
    return raw


def _packet_inventory(packet: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    inventory: dict[tuple[str, str], dict[str, Any]] = {}
    for case in packet["cases"]:
        for line in case["lines"]:
            key = (str(case["document_id"]), str(line["line_id"]))
            expected = {
                "document_id": str(case["document_id"]),
                "work_id": str(case["work_id"]),
                "source": str(case["source"]),
                "line_id": str(line["line_id"]),
                "abs_idx": int(line["abs_idx"]),
                "text": str(line["text"]),
            }
            if key in inventory and inventory[key] != expected:
                raise ValueError(f"overlapping chunks disagree on line identity: {key}")
            inventory[key] = expected
    return inventory


def _source_labels(provenance: Mapping[str, Any]) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    for block in provenance["cases"]:
        document_id = str(block["document_id"])
        for chunk in block.get("review_chunks", []):
            for line in chunk.get("context_source_labels", []):
                key = (document_id, str(line["line_id"]))
                label = str(line["original_region_label"])
                if key in labels and labels[key] != label:
                    raise ValueError(f"provenance labels conflict for {key}")
                labels[key] = label
    return labels


def _validate_review_against_packet(
    review: Mapping[str, Any], packet: Mapping[str, Any]
) -> None:
    expected = {str(case["case_id"]): case for case in packet["cases"]}
    cases = review["cases"]
    if len(cases) != len(expected):
        raise ValueError("review case coverage is incomplete")
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("case_id", ""))
        if case_id not in expected or case_id in seen:
            raise ValueError("review repeats or invents a case")
        seen.add(case_id)
        lines = case.get("lines")
        expected_lines = expected[case_id]["lines"]
        if not isinstance(lines, list) or len(lines) != len(expected_lines):
            raise ValueError(f"case {case_id}: line coverage is incomplete")
        for line, wanted in zip(lines, expected_lines):
            if line.get("line_id") != wanted["line_id"] or line.get("abs_idx") != wanted["abs_idx"]:
                raise ValueError(f"case {case_id}: line identity/order mismatch")
            if line.get("role") not in ROLES or line.get("boundary_flag") not in BOUNDARIES:
                raise ValueError(f"case {case_id}: invalid role/boundary")


def collect_votes(
    review: Mapping[str, Any], packet: Mapping[str, Any]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Collect all overlap votes; duplicates are preserved for consistency checks."""

    _validate_review_against_packet(review, packet)
    case_inventory = {str(case["case_id"]): case for case in packet["cases"]}
    votes: dict[tuple[str, str], dict[str, Any]] = {}
    for case in review["cases"]:
        packet_case = case_inventory[str(case["case_id"])]
        document_id = str(packet_case["document_id"])
        for line in case["lines"]:
            key = (document_id, str(line["line_id"]))
            target = votes.setdefault(
                key,
                {"roles": [], "boundaries": [], "confidences": [], "case_ids": [], "reasons": []},
            )
            target["roles"].append(str(line["role"]))
            target["boundaries"].append(str(line["boundary_flag"]))
            target["confidences"].append(float(line["confidence"]))
            target["case_ids"].append(str(case["case_id"]))
            target["reasons"].append(str(line["reason"]))
    return votes


def _consensus(values: Sequence[str]) -> str | None:
    unique = set(values)
    return next(iter(unique)) if len(unique) == 1 else None


def merge_reviews(
    packet: Mapping[str, Any], provenance: Mapping[str, Any],
    review_a: Mapping[str, Any], review_b: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviewer_a, reviewer_b = str(review_a["reviewer"]), str(review_b["reviewer"])
    if reviewer_a == reviewer_b:
        raise ValueError("dual review requires distinct reviewer identities")
    inventory = _packet_inventory(packet)
    labels = _source_labels(provenance)
    if set(inventory) != set(labels):
        missing = set(inventory) - set(labels)
        extra = set(labels) - set(inventory)
        raise ValueError(f"packet/provenance inventory mismatch: missing={len(missing)} extra={len(extra)}")
    votes_a = collect_votes(review_a, packet)
    votes_b = collect_votes(review_b, packet)
    if set(votes_a) != set(inventory) or set(votes_b) != set(inventory):
        raise ValueError("reviews do not cover the complete unique-line inventory")

    role_confusion: collections.Counter[tuple[str, str]] = collections.Counter()
    boundary_confusion: collections.Counter[tuple[str, str]] = collections.Counter()
    source_stats: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    overlays: list[dict[str, Any]] = []
    exact_role = entry_agreement = attachable_agreement = exact_boundary = 0
    internally_consistent_a = internally_consistent_b = 0
    for key in sorted(inventory, key=lambda value: (value[0], inventory[value]["abs_idx"], value[1])):
        source_line = inventory[key]
        a, b = votes_a[key], votes_b[key]
        role_a, role_b = _consensus(a["roles"]), _consensus(b["roles"])
        boundary_a, boundary_b = _consensus(a["boundaries"]), _consensus(b["boundaries"])
        internally_consistent_a += int(role_a is not None and boundary_a is not None)
        internally_consistent_b += int(role_b is not None and boundary_b is not None)
        agreed_role = role_a is not None and role_a == role_b
        agreed_boundary = boundary_a is not None and boundary_a == boundary_b
        if role_a is not None and role_b is not None:
            role_confusion[(role_a, role_b)] += 1
            exact_role += int(agreed_role)
            entry_agreement += int((role_a in ENTRY_ROLES) == (role_b in ENTRY_ROLES))
            attachable_agreement += int(
                (role_a in ATTACHABLE_ROLES) == (role_b in ATTACHABLE_ROLES)
            )
        if boundary_a is not None and boundary_b is not None:
            boundary_confusion[(boundary_a, boundary_b)] += 1
            exact_boundary += int(agreed_boundary)
        source = str(source_line["source"])
        source_stats[source]["lines"] += 1
        source_stats[source]["role_agreements"] += int(agreed_role)
        source_stats[source]["boundary_agreements"] += int(agreed_boundary)
        role_confidence = min(a["confidences"] + b["confidences"]) if agreed_role else 0.0
        boundary_confidence = min(a["confidences"] + b["confidences"]) if agreed_boundary else 0.0
        overlays.append(
            {
                "schema_version": OVERLAY_SCHEMA,
                "document_id": source_line["document_id"],
                "work_id": source_line["work_id"],
                "line_id": source_line["line_id"],
                "abs_idx": source_line["abs_idx"],
                "text_sha256": hashlib.sha256(source_line["text"].encode()).hexdigest(),
                "original_region_label": labels[key],
                "role": role_a if agreed_role else "UNKNOWN",
                "role_status": "AGREED_REVIEW" if agreed_role else "UNRESOLVED",
                "boundary_flag": boundary_a if agreed_boundary else "NONE",
                "boundary_status": "AGREED_REVIEW" if agreed_boundary else "UNRESOLVED",
                "role_confidence": role_confidence,
                "boundary_confidence": boundary_confidence,
                "label_origin": "codex_dual_context_review",
                "reviewers": [reviewer_a, reviewer_b],
                "review_case_ids": sorted(set(a["case_ids"] + b["case_ids"])),
                "raw_role_votes": {reviewer_a: a["roles"], reviewer_b: b["roles"]},
                "raw_boundary_votes": {
                    reviewer_a: a["boundaries"], reviewer_b: b["boundaries"]
                },
            }
        )
    denominator = len(inventory)
    exact_role_rate = exact_role / denominator
    entry_rate = entry_agreement / denominator
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "calibration_passed" if exact_role_rate >= 0.85 and entry_rate >= 0.95 else "calibration_failed",
        "gates": {
            "complete_line_coverage_required": True,
            "exact_seven_role_agreement_minimum": 0.85,
            "entry_seed_eligibility_agreement_minimum": 0.95,
        },
        "reviewers": [reviewer_a, reviewer_b],
        "unique_line_count": denominator,
        "case_count": len(packet["cases"]),
        "exact_role_agreement": exact_role_rate,
        "entry_seed_eligibility_agreement": entry_rate,
        "attachable_operational_agreement": attachable_agreement / denominator,
        "exact_boundary_agreement": exact_boundary / denominator,
        "role_agreed_line_count": exact_role,
        "boundary_agreed_line_count": exact_boundary,
        "reviewer_internal_consistent_line_counts": {
            reviewer_a: internally_consistent_a,
            reviewer_b: internally_consistent_b,
        },
        "role_confusion": {
            f"{left} -> {right}": value for (left, right), value in sorted(role_confusion.items())
        },
        "boundary_confusion": {
            f"{left} -> {right}": value
            for (left, right), value in sorted(boundary_confusion.items())
        },
        "per_source": {
            source: {
                "line_count": stats["lines"],
                "exact_role_agreement": stats["role_agreements"] / stats["lines"],
                "exact_boundary_agreement": stats["boundary_agreements"] / stats["lines"],
            }
            for source, stats in sorted(source_stats.items())
        },
        "overlay_content_sha256": canonical_json_sha256(overlays),
    }
    return overlays, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--provenance", required=True)
    parser.add_argument("--review-a", required=True)
    parser.add_argument("--review-b", required=True)
    parser.add_argument("--overlay-out", required=True)
    parser.add_argument("--report-out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    packet_path, provenance_path = Path(args.packet).resolve(), Path(args.provenance).resolve()
    review_a_path, review_b_path = Path(args.review_a).resolve(), Path(args.review_b).resolve()
    overlays, report = merge_reviews(
        load_packet(packet_path), load_provenance(provenance_path),
        load_review(review_a_path), load_review(review_b_path),
    )
    report["inputs"] = {
        "packet_sha256": sha256_file(packet_path),
        "provenance_sha256": sha256_file(provenance_path),
        "review_a_sha256": sha256_file(review_a_path),
        "review_b_sha256": sha256_file(review_b_path),
    }
    overlay_path, report_path = Path(args.overlay_out).resolve(), Path(args.report_out).resolve()
    _write_jsonl_new(overlay_path, overlays)
    report["overlay_sha256"] = sha256_file(overlay_path)
    _write_json_new(report_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
