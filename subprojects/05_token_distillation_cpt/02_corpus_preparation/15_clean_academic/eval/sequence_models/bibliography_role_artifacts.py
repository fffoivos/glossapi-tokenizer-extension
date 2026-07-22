#!/usr/bin/env python3
"""Subset blind role packets and union provenance-bound role overlays."""

from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .bibliography_role_adjudication import OVERLAY_SCHEMA
from .bibliography_role_review_runner import load_packet, load_provenance
from .contract import canonical_json_sha256, sha256_file


SUBSET_SCHEMA = "bibliography-role-targeted-subset-v1"
UNION_SCHEMA = "bibliography-role-overlay-union-v1"
IMMUTABLE_FIELDS = (
    "schema_version", "document_id", "work_id", "line_id", "abs_idx",
    "text_sha256", "original_region_label",
)
STATUS_RANK = {
    "UNRESOLVED": 0,
    "PROVISIONAL": 1,
    "SINGLE_REVIEW": 2,
    "AGREED_REVIEW": 3,
    "ADJUDICATED": 4,
}


def _write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_new(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def subset_packet(
    packet: Mapping[str, Any], provenance: Mapping[str, Any], *, source: str,
    strata: Sequence[str], expected_block_count: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    wanted = frozenset(strata)
    if not source or not wanted:
        raise ValueError("source and at least one stratum are required")
    selected_blocks = [
        row for row in provenance["cases"]
        if row.get("source") == source and row.get("bootstrap_stratum") in wanted
    ]
    if not selected_blocks:
        raise ValueError("targeted selection is empty")
    if expected_block_count is not None and len(selected_blocks) != expected_block_count:
        raise ValueError(
            f"expected {expected_block_count} targeted blocks, got {len(selected_blocks)}"
        )
    block_ids = {str(row["block_case_id"]) for row in selected_blocks}
    selected_cases = [
        row for row in packet["cases"] if str(row["block_case_id"]) in block_ids
    ]
    case_ids = {str(row["case_id"]) for row in selected_cases}
    provenance_case_ids = {
        str(chunk["case_id"])
        for block in selected_blocks for chunk in block.get("review_chunks", [])
    }
    if case_ids != provenance_case_ids:
        raise ValueError("targeted packet/provenance case inventory mismatch")
    if {str(row["block_case_id"]) for row in selected_cases} != block_ids:
        raise ValueError("one or more selected blocks lack blind review cases")

    selection = {
        "purpose": "targeted sparse-role bootstrap review",
        "source": source,
        "strata": sorted(wanted),
        "block_count": len(selected_blocks),
        "case_count": len(selected_cases),
        "line_response_count": sum(len(row["lines"]) for row in selected_cases),
        "parent_packet_content_sha256": canonical_json_sha256(packet),
        "parent_provenance_content_sha256": canonical_json_sha256(provenance),
    }
    subset = dict(packet)
    subset["selection"] = selection
    subset["cases"] = selected_cases
    subset_provenance = {
        "schema_version": provenance["schema_version"],
        "selection_schema_version": SUBSET_SCHEMA,
        "warning": provenance.get("warning", "Contains blinded-source provenance."),
        "selection": selection,
        "cases": selected_blocks,
    }
    # Re-run the strict blind-packet validator without touching disk.
    seen_cases: set[str] = set()
    for case in subset["cases"]:
        case_id = str(case["case_id"])
        if case_id in seen_cases:
            raise ValueError("targeted packet repeats a case")
        seen_cases.add(case_id)
    report = {
        "schema_version": SUBSET_SCHEMA,
        "status": "passed_exact_targeted_subset",
        **selection,
        "bootstrap_stratum_counts": dict(
            sorted(collections.Counter(str(row["bootstrap_stratum"]) for row in selected_blocks).items())
        ),
        "work_count": len({str(row["work_id"]) for row in selected_blocks}),
    }
    return subset, subset_provenance, report


def _iter_overlay(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema_version") != OVERLAY_SCHEMA:
                raise ValueError(f"{path}:{number}: expected overlay v2 row")
            for dimension in ("role", "boundary"):
                status = row.get(f"{dimension}_status")
                if status not in STATUS_RANK:
                    raise ValueError(f"{path}:{number}: invalid {dimension} status")
            yield row


def _merge_vote_maps(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, list[str]]:
    result = {str(name): list(values) for name, values in left.items()}
    for name, values in right.items():
        incoming = list(values)
        if name in result and result[name] != incoming:
            raise ValueError(f"reviewer {name} has conflicting duplicate votes")
        result[str(name)] = incoming
    return result


def _choose_dimension(
    left: Mapping[str, Any], right: Mapping[str, Any], *, dimension: str,
) -> tuple[Mapping[str, Any], bool]:
    status_field = f"{dimension}_status"
    value_field = "role" if dimension == "role" else "boundary_flag"
    confidence_field = f"{dimension}_confidence"
    left_rank, right_rank = STATUS_RANK[str(left[status_field])], STATUS_RANK[str(right[status_field])]
    if left_rank == right_rank:
        if left[value_field] != right[value_field] or left[status_field] != right[status_field]:
            raise ValueError(f"equal-trust duplicate overlays conflict on {dimension}")
        return left, False
    chosen = left if left_rank > right_rank else right
    rejected = right if chosen is left else left
    if (
        STATUS_RANK[str(chosen[status_field])] >= STATUS_RANK["AGREED_REVIEW"]
        and STATUS_RANK[str(rejected[status_field])] >= STATUS_RANK["AGREED_REVIEW"]
        and chosen[value_field] != rejected[value_field]
        and chosen[status_field] != "ADJUDICATED"
    ):
        raise ValueError(f"trusted duplicate overlays conflict on {dimension}")
    if confidence_field not in chosen:
        raise ValueError(f"chosen overlay lacks {confidence_field}")
    return chosen, chosen[status_field] == "ADJUDICATED"


def union_overlays(
    inputs: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not inputs:
        raise ValueError("at least one overlay is required")
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_count = adjudicated_precedence_count = 0
    input_counts: dict[str, int] = {}
    for name, overlay in inputs:
        input_counts[name] = len(overlay)
        for incoming_raw in overlay:
            incoming = dict(incoming_raw)
            key = (str(incoming.get("document_id", "")), str(incoming.get("line_id", "")))
            if not all(key):
                raise ValueError("overlay contains an empty identity")
            current = rows.get(key)
            if current is None:
                rows[key] = incoming
                continue
            duplicate_count += 1
            for field in IMMUTABLE_FIELDS:
                if current.get(field) != incoming.get(field):
                    raise ValueError(f"duplicate overlay immutable mismatch for {key}: {field}")
            role_choice, role_human = _choose_dimension(current, incoming, dimension="role")
            boundary_choice, boundary_human = _choose_dimension(
                current, incoming, dimension="boundary"
            )
            adjudicated_precedence_count += int(role_human or boundary_human)
            merged = dict(current)
            for field in ("role", "role_status", "role_confidence"):
                merged[field] = role_choice[field]
            for field in ("boundary_flag", "boundary_status", "boundary_confidence"):
                merged[field] = boundary_choice[field]
            merged["reviewers"] = list(dict.fromkeys([*current["reviewers"], *incoming["reviewers"]]))
            merged["review_case_ids"] = sorted(
                set(current.get("review_case_ids", [])) | set(incoming.get("review_case_ids", []))
            )
            merged["raw_role_votes"] = _merge_vote_maps(
                current["raw_role_votes"], incoming["raw_role_votes"]
            )
            merged["raw_boundary_votes"] = _merge_vote_maps(
                current["raw_boundary_votes"], incoming["raw_boundary_votes"]
            )
            merged["label_origin"] = (
                "overlay_union_human_precedence"
                if role_human or boundary_human else "overlay_union"
            )
            rows[key] = merged
    ordered = sorted(rows.values(), key=lambda row: (row["document_id"], row["abs_idx"], row["line_id"]))
    report = {
        "schema_version": UNION_SCHEMA,
        "status": "passed_fail_closed_overlay_union",
        "input_line_counts": input_counts,
        "output_line_count": len(ordered),
        "duplicate_line_count": duplicate_count,
        "adjudicated_precedence_count": adjudicated_precedence_count,
        "overlay_content_sha256": canonical_json_sha256(ordered),
    }
    return ordered, report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subset = subparsers.add_parser("subset")
    subset.add_argument("--packet", required=True)
    subset.add_argument("--provenance", required=True)
    subset.add_argument("--source", required=True)
    subset.add_argument("--stratum", action="append", required=True)
    subset.add_argument("--expected-block-count", type=int)
    subset.add_argument("--packet-out", required=True)
    subset.add_argument("--provenance-out", required=True)
    subset.add_argument("--receipt-out", required=True)
    union = subparsers.add_parser("union")
    union.add_argument("--overlay", action="append", required=True)
    union.add_argument("--overlay-out", required=True)
    union.add_argument("--receipt-out", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "subset":
        packet_path, provenance_path = Path(args.packet).resolve(), Path(args.provenance).resolve()
        packet, provenance, receipt = subset_packet(
            load_packet(packet_path), load_provenance(provenance_path),
            source=args.source, strata=args.stratum,
            expected_block_count=args.expected_block_count,
        )
        packet_out, provenance_out = Path(args.packet_out).resolve(), Path(args.provenance_out).resolve()
        _write_json_new(packet_out, packet)
        _write_json_new(provenance_out, provenance)
        # Validate the serialized packet as the reviewer will see it.
        load_packet(packet_out)
        load_provenance(provenance_out)
        receipt["inputs"] = {
            "packet_sha256": sha256_file(packet_path),
            "provenance_sha256": sha256_file(provenance_path),
        }
        receipt["outputs"] = {
            "packet": {"path": str(packet_out), "sha256": sha256_file(packet_out)},
            "provenance": {
                "path": str(provenance_out), "sha256": sha256_file(provenance_out)
            },
        }
        _write_json_new(Path(args.receipt_out).resolve(), receipt)
        return 0

    paths = [Path(value).resolve() for value in args.overlay]
    inputs = [(str(path), list(_iter_overlay(path))) for path in paths]
    rows, receipt = union_overlays(inputs)
    output = Path(args.overlay_out).resolve()
    _write_jsonl_new(output, rows)
    receipt["inputs"] = {
        str(path): {"sha256": sha256_file(path), "line_count": len(rows_local)}
        for path, (_, rows_local) in zip(paths, inputs, strict=True)
    }
    receipt["output"] = {"path": str(output), "sha256": sha256_file(output)}
    _write_json_new(Path(args.receipt_out).resolve(), receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
