#!/usr/bin/env python3
"""Evaluate a completed human/agent QA review against the frozen run gate."""

from __future__ import annotations

import argparse
import datetime as dt

from .contracts import (
    QA_PACKET_SCHEMA,
    QA_REVIEW_SCHEMA,
    atomic_write_json,
    load_json,
    require_schema,
    sha256_file,
)

ALLOWED = {"acceptable", "body_only", "catastrophic", "uncertain"}


def run(args: argparse.Namespace) -> dict:
    packet = load_json(args.packet)
    review = load_json(args.review)
    require_schema(packet, QA_PACKET_SCHEMA, args.packet)
    require_schema(review, QA_REVIEW_SCHEMA, args.review)
    if review["packet_sha256"] != sha256_file(args.packet):
        raise ValueError("review is bound to a different QA packet")
    packet_items = {item["item_id"]: item for item in packet["items"]}
    decisions = {decision["item_id"]: decision for decision in review["decisions"]}
    if set(decisions) != set(packet_items) or len(decisions) != len(
        review["decisions"]
    ):
        raise ValueError("review decisions do not exactly match packet items")
    incomplete = [
        item_id
        for item_id, decision in decisions.items()
        if decision["classification"] not in ALLOWED
        or not isinstance(decision["primarily_bibliography"], bool)
        or not decision["rationale"]
    ]
    if not review.get("reviewer") or not review.get("reviewed_utc"):
        incomplete.append("<review-metadata>")
    bad = [
        item_id
        for item_id, decision in decisions.items()
        if decision["classification"] in {"body_only", "catastrophic", "uncertain"}
    ]
    kallipos = [
        decisions[item_id]
        for item_id, item in packet_items.items()
        if "kallipos_median" in item["reasons"]
    ]
    openarchives = [
        decisions[item_id]
        for item_id, item in packet_items.items()
        if "openarchives_over_50pct" in item["reasons"]
    ]
    gate = packet["gate"]
    checks = {
        "review_complete": not incomplete,
        "zero_catastrophic_body_only_or_uncertain": not bad,
        "kallipos_sample_size": len(kallipos) == gate["kallipos_sample_size"],
        "kallipos_primarily_bibliography": sum(
            decision["primarily_bibliography"] is True for decision in kallipos
        )
        >= gate["kallipos_primarily_bibliography_min"],
        "all_openarchives_over_50pct_acceptable": all(
            decision["classification"] == "acceptable"
            and decision["primarily_bibliography"]
            for decision in openarchives
        ),
    }
    result = {
        "schema_version": "bibliography-cleaning-qa-gate-v2",
        "status": "passed" if all(checks.values()) else "failed",
        "run_id": packet["run_id"],
        "packet_sha256": sha256_file(args.packet),
        "review_sha256": sha256_file(args.review),
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checks": checks,
        "counts": {
            "items": len(packet_items),
            "incomplete": len(incomplete),
            "bad_or_uncertain": len(bad),
            "kallipos": len(kallipos),
            "kallipos_primarily_bibliography": sum(
                decision["primarily_bibliography"] is True for decision in kallipos
            ),
            "openarchives_over_50pct": len(openarchives),
        },
        "incomplete_item_ids": incomplete,
        "failed_item_ids": bad,
    }
    atomic_write_json(args.output, result)
    print(result["status"], checks)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    raise SystemExit(result["status"] != "passed")
