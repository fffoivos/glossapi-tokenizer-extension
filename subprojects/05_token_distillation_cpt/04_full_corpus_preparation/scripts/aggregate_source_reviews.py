#!/usr/bin/env python3
"""Validate Codex source reviews and aggregate conservative admission decisions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from source_lineage import iter_jsonl, load_json, write_json


RESPONSE_SCHEMA = "source_quality_review_response_v1"
AGGREGATE_SCHEMA = "source_quality_review_admission_v1"
ACTIONS = {"include", "include_after_cleaning", "quarantine", "exclude"}
VALUES = {"high", "medium", "low", "none"}
CONFIDENCE = {"high", "medium", "low"}
REGISTERS = {
    "modern_greek",
    "polytonic_or_ancient_greek",
    "mixed_greek_foreign",
    "foreign_language",
    "code_or_data",
    "unknown",
}
SEVERITIES = {"none", "minor", "major", "severe"}
DEFECT_KEYS = {
    "html_or_markup",
    "boilerplate",
    "ocr_corruption",
    "mojibake",
    "fragmentation",
    "tables_or_loops",
    "pii",
    "non_greek_drift",
}


def validate_response(value: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "review_id",
        "sample_id",
        "reviewer_slot",
        "source_dataset",
        "substantive_training_value",
        "quality_score",
        "language_register",
        "defects",
        "variability",
        "action",
        "defects_deterministically_repairable",
        "safety_or_license_blocker",
        "confidence",
        "evidence",
    }
    missing = sorted(required - set(value))
    if missing:
        errors.append(f"missing required fields: {missing}")
    if value.get("schema_version") != RESPONSE_SCHEMA:
        errors.append("unsupported schema_version")
    for field in ("review_id", "source_dataset", "evidence"):
        if not isinstance(value.get(field), str) or not value.get(field):
            errors.append(f"{field} must be a non-empty string")
    sample_id = value.get("sample_id")
    if not isinstance(sample_id, str) or len(sample_id) != 64 or any(
        char not in "0123456789abcdef" for char in sample_id
    ):
        errors.append("sample_id must be lowercase SHA-256")
    if value.get("reviewer_slot") not in {"primary", "secondary", "adjudicator"}:
        errors.append("invalid reviewer_slot")
    if value.get("substantive_training_value") not in VALUES:
        errors.append("invalid substantive_training_value")
    score = value.get("quality_score")
    if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 4:
        errors.append("quality_score must be an integer in [0,4]")
    if value.get("language_register") not in REGISTERS:
        errors.append("invalid language_register")
    if value.get("action") not in ACTIONS:
        errors.append("invalid action")
    if value.get("confidence") not in CONFIDENCE:
        errors.append("invalid confidence")
    for field in ("defects_deterministically_repairable", "safety_or_license_blocker"):
        if not isinstance(value.get(field), bool):
            errors.append(f"{field} must be boolean")
    defects = value.get("defects")
    if not isinstance(defects, dict) or set(defects) != DEFECT_KEYS:
        errors.append("defects must contain exactly the eight required dimensions")
    elif any(severity not in SEVERITIES for severity in defects.values()):
        errors.append("invalid defect severity")
    variability = value.get("variability")
    if not isinstance(variability, dict) or set(variability) != {
        "template_similarity",
        "substantive_variation",
    }:
        errors.append("variability must contain template_similarity and substantive_variation")
    return errors


def load_requests(path: Path) -> tuple[dict[str, dict], dict[str, dict[str, dict]]]:
    by_id: dict[str, dict] = {}
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for _, line_number, request in iter_jsonl([path]):
        if request.get("schema_version") != "source_quality_review_request_v1":
            raise ValueError(f"{path}:{line_number}: unsupported request schema")
        review_id = str(request.get("review_id", ""))
        sample_id = str(request.get("sample_id", ""))
        slot = str(request.get("reviewer_slot", ""))
        if not review_id or review_id in by_id:
            raise ValueError(f"{path}:{line_number}: duplicate/empty review_id")
        if slot not in {"primary", "secondary"} or slot in by_sample[sample_id]:
            raise ValueError(f"{path}:{line_number}: invalid/duplicate requested reviewer slot")
        by_id[review_id] = request
        by_sample[sample_id][slot] = request
    return by_id, by_sample


def load_responses(
    path: Path,
    requests_by_id: Mapping[str, dict],
    requests_by_sample: Mapping[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    responses: dict[str, dict[str, dict]] = defaultdict(dict)
    seen_review_ids: set[str] = set()
    for _, line_number, response in iter_jsonl([path]):
        errors = validate_response(response)
        if errors:
            raise ValueError(f"{path}:{line_number}: {'; '.join(errors)}")
        review_id = response["review_id"]
        sample_id = response["sample_id"]
        slot = response["reviewer_slot"]
        if review_id in seen_review_ids:
            raise ValueError(f"{path}:{line_number}: duplicate review_id {review_id!r}")
        seen_review_ids.add(review_id)
        if sample_id not in requests_by_sample:
            raise ValueError(f"{path}:{line_number}: unknown sample_id {sample_id!r}")
        if slot in {"primary", "secondary"}:
            request = requests_by_id.get(review_id)
            if request is None:
                raise ValueError(f"{path}:{line_number}: review_id was not requested")
            if request["sample_id"] != sample_id or request["reviewer_slot"] != slot:
                raise ValueError(f"{path}:{line_number}: response/request identity mismatch")
        if response["source_dataset"] != next(iter(requests_by_sample[sample_id].values()))[
            "source_dataset"
        ]:
            raise ValueError(f"{path}:{line_number}: source_dataset drift")
        if slot in responses[sample_id]:
            raise ValueError(f"{path}:{line_number}: duplicate {slot} response for sample")
        responses[sample_id][slot] = response
    return responses


def resolution_signature(review: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        review["action"],
        review["substantive_training_value"],
        review["defects_deterministically_repairable"],
        review["safety_or_license_blocker"],
    )


def resolve_sample(
    requested: Mapping[str, dict], responses: Mapping[str, dict], policy: Mapping[str, Any]
) -> tuple[dict | None, str | None]:
    adjudicator = responses.get("adjudicator")
    if adjudicator is not None:
        return dict(adjudicator), None
    primary = responses.get("primary")
    if primary is None:
        return None, "missing_primary"
    if "secondary" in requested and "secondary" not in responses:
        return None, "missing_secondary"
    if policy["admission"].get("low_confidence_requires_adjudication") and primary[
        "confidence"
    ] == "low":
        return None, "low_confidence"
    secondary = responses.get("secondary")
    if secondary is not None:
        if policy["admission"].get("low_confidence_requires_adjudication") and secondary[
            "confidence"
        ] == "low":
            return None, "low_confidence"
        if (
            policy["admission"].get("disagreement_requires_adjudication")
            and resolution_signature(primary) != resolution_signature(secondary)
        ):
            return None, "reviewer_disagreement"
        resolved = dict(primary)
        resolved["quality_score"] = min(primary["quality_score"], secondary["quality_score"])
        resolved["safety_or_license_blocker"] = (
            primary["safety_or_license_blocker"] or secondary["safety_or_license_blocker"]
        )
        resolved["defects_deterministically_repairable"] = (
            primary["defects_deterministically_repairable"]
            and secondary["defects_deterministically_repairable"]
        )
        return resolved, None
    return dict(primary), None


def novelty_by_source(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    value = load_json(path)
    entries = value.get("sources", value)
    result: dict[str, float] = {}
    if isinstance(entries, list):
        iterable = ((entry.get("source_dataset"), entry.get("novel_token_fraction")) for entry in entries)
    elif isinstance(entries, dict):
        iterable = (
            (name, entry.get("novel_token_fraction") if isinstance(entry, dict) else entry)
            for name, entry in entries.items()
        )
    else:
        raise ValueError("novelty summary sources must be a list or mapping")
    for name, fraction in iterable:
        if not isinstance(name, str) or not isinstance(fraction, (float, int)) or not 0 <= fraction <= 1:
            raise ValueError("invalid novelty summary entry")
        result[name] = float(fraction)
    return result


def admission_decision(
    resolved: list[dict],
    *,
    review_phase: str,
    policy: Mapping[str, Any],
    novel_fraction: float | None,
) -> tuple[str, list[str], dict[str, Any]]:
    admission = policy["admission"]
    action_counts = Counter(review["action"] for review in resolved)
    value_counts = Counter(review["substantive_training_value"] for review in resolved)
    total = len(resolved)
    usable = action_counts["include"] + action_counts["include_after_cleaning"]
    useful = value_counts["high"] + value_counts["medium"]
    usable_fraction = usable / total if total else 0.0
    useful_fraction = useful / total if total else 0.0
    blockers = sum(bool(review["safety_or_license_blocker"]) for review in resolved)
    unrepairable_cleaning = sum(
        review["action"] == "include_after_cleaning"
        and not review["defects_deterministically_repairable"]
        for review in resolved
    )
    reasons: list[str] = []
    if blockers and admission.get("safety_or_license_blocker_forces_quarantine"):
        decision = "quarantine"
        reasons.append("reviewed_safety_or_license_blocker")
    elif novel_fraction is not None and novel_fraction < float(admission["minimum_novel_token_fraction"]):
        decision = "exclude"
        reasons.append("novel_token_fraction_below_gate")
    elif unrepairable_cleaning:
        decision = "quarantine"
        reasons.append("claimed_cleaning_action_is_not_deterministically_repairable")
    elif review_phase == "post_clean":
        if usable_fraction >= float(admission["post_clean_min_usable_fraction"]):
            decision = "include"
            reasons.append("post_clean_usability_gate_passed")
        else:
            decision = "exclude"
            reasons.append("post_clean_usability_gate_failed")
    elif usable_fraction >= float(admission["direct_include_min_usable_fraction"]):
        if action_counts["include_after_cleaning"]:
            decision = "include_after_cleaning"
            reasons.append("usability_gate_passed_but_cleaning_is_required")
        else:
            decision = "include"
            reasons.append("direct_include_usability_gate_passed")
    elif useful_fraction >= float(admission["cleanable_min_useful_fraction"]):
        decision = "include_after_cleaning"
        reasons.append("cleanable_usefulness_gate_passed")
    else:
        decision = "exclude"
        reasons.append("usefulness_gate_failed")
    metrics = {
        "resolved_documents": total,
        "action_counts": dict(sorted(action_counts.items())),
        "substantive_value_counts": dict(sorted(value_counts.items())),
        "usable_fraction": round(usable_fraction, 6),
        "useful_fraction": round(useful_fraction, 6),
        "safety_or_license_blockers": blockers,
        "unrepairable_cleaning_reviews": unrepairable_cleaning,
        "novel_token_fraction": novel_fraction,
    }
    return decision, reasons, metrics


def aggregate(args: argparse.Namespace) -> int:
    policy = load_json(args.review_policy)
    packet_summary = load_json(args.packet_summary)
    requests_by_id, requests_by_sample = load_requests(args.requests)
    responses = load_responses(args.reviews, requests_by_id, requests_by_sample)
    novelty = novelty_by_source(args.novelty_summary)

    samples_by_source: dict[str, list[str]] = defaultdict(list)
    for sample_id, requested in requests_by_sample.items():
        samples_by_source[next(iter(requested.values()))["source_dataset"]].append(sample_id)
    packet_sources = {
        entry["source_dataset"]: entry for entry in packet_summary.get("sources", [])
    }
    source_results: list[dict[str, Any]] = []
    pending_total = 0
    for source_dataset in sorted(samples_by_source):
        resolved: list[dict] = []
        pending: list[dict[str, str]] = []
        for sample_id in sorted(samples_by_source[source_dataset]):
            review, reason = resolve_sample(
                requests_by_sample[sample_id], responses.get(sample_id, {}), policy
            )
            if reason:
                pending.append({"sample_id": sample_id, "reason": reason})
            else:
                assert review is not None
                resolved.append(review)
        expected = int(packet_sources.get(source_dataset, {}).get("unique_sampled_documents", 0))
        if expected != len(samples_by_source[source_dataset]):
            pending.append(
                {
                    "sample_id": "source_summary",
                    "reason": "packet_summary_unique_document_count_mismatch",
                }
            )
        if pending:
            decision = "pending_adjudication"
            reasons = ["reviews_incomplete_or_unresolved"]
            metrics = {"resolved_documents": len(resolved)}
        else:
            decision, reasons, metrics = admission_decision(
                resolved,
                review_phase=str(packet_summary.get("review_phase", "pre_clean")),
                policy=policy,
                novel_fraction=novelty.get(source_dataset),
            )
        pending_total += len(pending)
        source_results.append(
            {
                "source_dataset": source_dataset,
                "decision": decision,
                "reasons": reasons,
                "metrics": metrics,
                "pending": pending,
                "post_clean_review_required": decision == "include_after_cleaning",
            }
        )

    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "review_phase": packet_summary.get("review_phase"),
        "request_rows": len(requests_by_id),
        "response_rows": sum(len(value) for value in responses.values()),
        "unique_documents": len(requests_by_sample),
        "pending_adjudications": pending_total,
        "sources": source_results,
    }
    write_json(args.output, result)
    return 0 if not pending_total or args.allow_incomplete else 2


def main() -> int:
    here = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--packet-summary", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--novelty-summary", type=Path)
    parser.add_argument(
        "--review-policy", type=Path, default=here / "configs" / "source_review_policy.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return aggregate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
