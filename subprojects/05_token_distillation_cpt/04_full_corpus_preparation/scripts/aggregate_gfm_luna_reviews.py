#!/usr/bin/env python3
"""Prepare adjudications and aggregate final Luna GFM transformation verdicts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from run_gfm_luna_reviews import read_jsonl, validate_response, write_json_atomic, write_jsonl_atomic
from source_lineage import canonical_json


FINAL_SCHEMA = "gfm_luna_validation_summary_v1"
COMPARE_FIELDS = (
    "text_preservation",
    "artifact_removal",
    "gfm_validity",
    "table_outcome",
    "unintended_change",
    "verdict",
)


def strict_pass(response: Mapping[str, object]) -> bool:
    return (
        response.get("text_preservation") == "pass"
        and response.get("artifact_removal") in {"pass", "not_applicable"}
        and response.get("gfm_validity") == "pass"
        and response.get("table_outcome") in {"valid_gfm", "readable_fallback", "not_applicable"}
        and response.get("unintended_change") == "none"
        and response.get("verdict") == "pass"
        and response.get("confidence") in {"medium", "high"}
    )


def logical_region_identity(region: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(region["opaque_id"]),
        str(region["transformation_family"]),
        tuple(str(rule) for rule in region["rule_ids"]),
        str(region["risk_tier"]),
        int(region["ordinal"]),
    )


def load_bound(requests_path: Path, responses_path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, object]]]:
    requests = read_jsonl(requests_path)
    responses = read_jsonl(responses_path) if responses_path.is_file() and responses_path.stat().st_size else []
    by_request = {str(row["review_id"]): row for row in requests}
    by_response: dict[str, dict[str, object]] = {}
    for response in responses:
        request = by_request.get(str(response.get("review_id")))
        if request is None:
            raise ValueError("response does not bind to a request")
        validated = validate_response(response, request)
        by_response[str(validated["review_id"])] = validated
    if responses and len(by_response) != len(requests):
        raise ValueError("request/response closure failed")
    return requests, by_response


def prepare(
    *,
    requests_path: Path,
    responses_path: Path,
    output_path: Path,
) -> dict[str, object]:
    requests, responses = load_bound(requests_path, responses_path)
    by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for request in requests:
        by_region[str(request["region_id"])].append(request)
    adjudications: list[dict[str, object]] = []
    reasons: Counter[str] = Counter()
    for region_id, region_requests in sorted(by_region.items()):
        region_responses = [responses[str(request["review_id"])] for request in region_requests]
        high = any(request["risk_tier"] == "high" for request in region_requests)
        disagreement = high and len({tuple(response[field] for field in COMPARE_FIELDS) for response in region_responses}) > 1
        nonpass = any(not strict_pass(response) for response in region_responses)
        if not disagreement and not nonpass:
            continue
        if disagreement:
            reasons["high_risk_disagreement"] += 1
        if nonpass:
            reasons["nonpass_or_low_confidence"] += 1
        base = dict(region_requests[0])
        base["reviewer_slot"] = "adjudicator"
        base["prior_reviews"] = [
            {field: response[field] for field in ("reviewer_slot", *COMPARE_FIELDS, "confidence", "evidence")}
            for response in region_responses
        ]
        base["review_id"] = hashlib.sha256(
            canonical_json(
                {
                    "region_id": region_id,
                    "reviewer_slot": "adjudicator",
                    "normalizer_sha256": base["normalizer_sha256"],
                    "prior_reviews": base["prior_reviews"],
                }
            ).encode("utf-8")
        ).hexdigest()
        adjudications.append(base)
    write_jsonl_atomic(output_path, adjudications)
    return {
        "status": "adjudication_required" if adjudications else "no_adjudication_required",
        "regions": len(by_region),
        "adjudication_requests": len(adjudications),
        "reasons": dict(sorted(reasons.items())),
    }


def finalize(
    *,
    regions_path: Path,
    requests_path: Path,
    responses_path: Path,
    adjudication_requests_path: Path,
    adjudication_responses_path: Path,
    output_path: Path,
) -> dict[str, object]:
    regions_payload = json.loads(regions_path.read_text(encoding="utf-8"))
    regions = {str(row["region_id"]): dict(row) for row in regions_payload["regions"]}
    requests, responses = load_bound(requests_path, responses_path)
    adjudication_requests, adjudication_responses = (
        load_bound(adjudication_requests_path, adjudication_responses_path)
        if adjudication_requests_path.is_file() and adjudication_requests_path.stat().st_size
        else ([], {})
    )
    all_requests = [*requests, *adjudication_requests]
    all_responses = {**responses, **adjudication_responses}
    by_region: dict[str, list[dict[str, object]]] = defaultdict(list)
    for request in all_requests:
        response = all_responses.get(str(request["review_id"]))
        if response is None:
            raise ValueError("missing final response")
        by_region[str(request["region_id"])].append(response)

    rows: list[dict[str, object]] = []
    failures = 0
    for region_id, region in sorted(regions.items()):
        reviews = by_region.get(region_id, [])
        adjudicator = next((row for row in reviews if row["reviewer_slot"] == "adjudicator"), None)
        if adjudicator is not None:
            passed = strict_pass(adjudicator)
            final_verdict = str(adjudicator["verdict"])
        else:
            passed = bool(reviews) and all(strict_pass(row) for row in reviews)
            final_verdict = "pass" if passed else "needs_human"
        failures += int(not passed)
        rows.append(
            {
                **region,
                "reviews": reviews,
                "final_verdict": final_verdict,
                "validated": passed,
            }
        )
    summary = {
        "schema_version": FINAL_SCHEMA,
        "status": "passed" if failures == 0 else "failed",
        "region_count": len(rows),
        "validated_regions": len(rows) - failures,
        "failed_or_unresolved_regions": failures,
        "review_count": sum(len(row["reviews"]) for row in rows),
        "adjudicated_regions": sum(any(review["reviewer_slot"] == "adjudicator" for review in row["reviews"]) for row in rows),
        "families": dict(sorted(Counter(str(row["transformation_family"]) for row in rows).items())),
        "regions": rows,
    }
    write_json_atomic(output_path, summary)
    return summary


def merge_revalidation(
    *, baseline_path: Path, revalidation_path: Path, output_path: Path
) -> dict[str, object]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    revalidation = json.loads(revalidation_path.read_text(encoding="utf-8"))
    rows = [dict(row) for row in baseline["regions"]]
    positions: dict[tuple[object, ...], int] = {}
    for index, row in enumerate(rows):
        identity = logical_region_identity(row)
        if identity in positions:
            raise ValueError(f"duplicate baseline logical region: {identity}")
        positions[identity] = index
    replaced = 0
    for row_value in revalidation["regions"]:
        row = dict(row_value)
        identity = logical_region_identity(row)
        if identity not in positions:
            raise ValueError(f"revalidated logical region is absent from baseline: {identity}")
        rows[positions[identity]] = row
        replaced += 1
    failures = sum(not bool(row["validated"]) for row in rows)
    summary = {
        "schema_version": FINAL_SCHEMA,
        "status": "passed" if failures == 0 else "failed",
        "region_count": len(rows),
        "validated_regions": len(rows) - failures,
        "failed_or_unresolved_regions": failures,
        "review_count": sum(len(row["reviews"]) for row in rows),
        "adjudicated_regions": sum(
            any(review["reviewer_slot"] == "adjudicator" for review in row["reviews"])
            for row in rows
        ),
        "families": dict(sorted(Counter(str(row["transformation_family"]) for row in rows).items())),
        "revalidated_regions": replaced,
        "reused_validated_regions": len(rows) - replaced,
        "regions": rows,
    }
    write_json_atomic(output_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--requests", type=Path, required=True)
    prepare_parser.add_argument("--responses", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    finalize_parser = commands.add_parser("finalize")
    finalize_parser.add_argument("--regions", type=Path, required=True)
    finalize_parser.add_argument("--requests", type=Path, required=True)
    finalize_parser.add_argument("--responses", type=Path, required=True)
    finalize_parser.add_argument("--adjudication-requests", type=Path, required=True)
    finalize_parser.add_argument("--adjudication-responses", type=Path, required=True)
    finalize_parser.add_argument("--output", type=Path, required=True)
    merge_parser = commands.add_parser("merge-revalidation")
    merge_parser.add_argument("--baseline", type=Path, required=True)
    merge_parser.add_argument("--revalidation", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(requests_path=args.requests, responses_path=args.responses, output_path=args.output)
    elif args.command == "finalize":
        result = finalize(
            regions_path=args.regions,
            requests_path=args.requests,
            responses_path=args.responses,
            adjudication_requests_path=args.adjudication_requests,
            adjudication_responses_path=args.adjudication_responses,
            output_path=args.output,
        )
    else:
        result = merge_revalidation(
            baseline_path=args.baseline,
            revalidation_path=args.revalidation,
            output_path=args.output,
        )
    print(json.dumps({key: value for key, value in result.items() if key != "regions"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
