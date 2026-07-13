#!/usr/bin/env python3
"""Validate and summarize independent bibliography-header adjudications.

This module deliberately separates three decisions:

* whether a silver ``BIB`` line is an entry-training positive;
* whether it must be masked from the entry-classifier loss; and
* whether it is reliable positive boundary evidence for the block model.

Disagreement never creates a negative example or a boundary cue.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LABELS = {
    "ENTRY",
    "BIB_HEADER",
    "BIB_SUBHEADER",
    "OTHER_STRUCTURE",
    "UNCERTAIN",
}
HEADER_LABELS = {"BIB_HEADER", "BIB_SUBHEADER"}


def _load_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _unique_by_id(rows: Any, *, name: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{name}.cases must be an array")
    result: dict[str, Mapping[str, Any]] = {}
    for offset, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"{name}.cases[{offset}] must be an object")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError(f"{name}.cases[{offset}] has no candidate_id")
        if candidate_id in result:
            raise ValueError(f"{name} repeats candidate_id {candidate_id}")
        result[candidate_id] = row
    return result


def _training_action(label_a: str, label_b: str) -> str:
    if label_a != label_b:
        return "MASK_DISAGREEMENT_NO_CUE"
    if label_a == "ENTRY":
        return "KEEP_ENTRY_POSITIVE"
    if label_a in HEADER_LABELS:
        return "MASK_HEADER_ENABLE_BOUNDARY_CUE"
    if label_a == "OTHER_STRUCTURE":
        return "MASK_OTHER_STRUCTURE_NO_CUE"
    return "MASK_UNCERTAIN_NO_CUE"


def adjudicate(
    packet: Mapping[str, Any],
    review_a: Mapping[str, Any],
    review_b: Mapping[str, Any],
) -> dict[str, Any]:
    cases = _unique_by_id(packet.get("cases"), name="packet")
    reviews_a = _unique_by_id(review_a.get("cases"), name="review_a")
    reviews_b = _unique_by_id(review_b.get("cases"), name="review_b")
    if set(cases) != set(reviews_a) or set(cases) != set(reviews_b):
        raise ValueError("packet and reviewer candidate_id sets differ")
    if review_a.get("reviewer") == review_b.get("reviewer"):
        raise ValueError("reviewers must have distinct identities")

    decisions: list[dict[str, Any]] = []
    for candidate_id, case in cases.items():
        a, b = reviews_a[candidate_id], reviews_b[candidate_id]
        label_a, label_b = a.get("label"), b.get("label")
        if label_a not in LABELS or label_b not in LABELS:
            raise ValueError(f"invalid label for {candidate_id}")
        action = _training_action(str(label_a), str(label_b))
        decisions.append(
            {
                "candidate_id": candidate_id,
                "stratum": str(case.get("stratum")),
                "source": str(case.get("source")),
                "document_id": str(case.get("document_id")),
                "abs_idx": int(case.get("abs_idx")),
                "text": str(case.get("text")),
                "review_a_label": label_a,
                "review_a_confidence": a.get("confidence"),
                "review_b_label": label_b,
                "review_b_confidence": b.get("confidence"),
                "agreement": label_a == label_b,
                "training_action": action,
                "boundary_cue": label_a if label_a == label_b and label_a in HEADER_LABELS else None,
            }
        )

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        action_counts = collections.Counter(row["training_action"] for row in rows)
        labels_a = collections.Counter(row["review_a_label"] for row in rows)
        labels_b = collections.Counter(row["review_b_label"] for row in rows)
        return {
            "count": len(rows),
            "agreement_count": sum(row["agreement"] for row in rows),
            "agreement_rate": sum(row["agreement"] for row in rows) / len(rows),
            "review_a_labels": dict(sorted(labels_a.items())),
            "review_b_labels": dict(sorted(labels_b.items())),
            "training_actions": dict(sorted(action_counts.items())),
        }

    strata = sorted({row["stratum"] for row in decisions})
    exact = [
        row
        for row in decisions
        if row["stratum"] in {"exact_heading", "exact_subheading"}
    ]
    exact_entry_votes = sum(
        row["review_a_label"] == "ENTRY" or row["review_b_label"] == "ENTRY"
        for row in exact
    )
    return {
        "schema_version": "bibliography-header-adjudication-summary-v1",
        "reviewers": [review_a.get("reviewer"), review_b.get("reviewer")],
        "overall": summarize(decisions),
        "by_stratum": {
            stratum: summarize([row for row in decisions if row["stratum"] == stratum])
            for stratum in strata
        },
        "conservative_exact_mask_audit": {
            "sample_count": len(exact),
            "entry_vote_count_from_either_reviewer": exact_entry_votes,
            "observed_entry_false_exclusion_rate": exact_entry_votes / len(exact),
            "verified_scope": "sampled exact heading and exact subheading rules only",
        },
        "decisions": decisions,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--review-a", type=Path, required=True)
    parser.add_argument("--review-b", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = adjudicate(
        _load_object(args.packet),
        _load_object(args.review_a),
        _load_object(args.review_b),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
