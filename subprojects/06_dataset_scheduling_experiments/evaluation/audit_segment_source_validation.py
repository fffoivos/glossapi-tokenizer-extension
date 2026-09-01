#!/usr/bin/env python3
"""Fail closed unless one segment has every finite source-validation binding."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.collect_validation_trajectory import parse_log
from production.campaign_contract import ARMS, atomic_write_json, read_json, sha256_file


FINITE_FIELDS = (
    "lm_loss",
    "bpb",
    "base_target_loss",
    "base_target_bpb",
    "base_target_count",
    "base_target_bytes",
    "added_target_count",
    "added_target_bytes",
)


def validate_metrics(row: dict, key: tuple[str, int, str]) -> None:
    for field in FINITE_FIELDS:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"nonfinite or absent {field} for {key}: {value!r}")
    if float(row["base_target_count"]) <= 0 or float(row["base_target_bytes"]) <= 0:
        raise ValueError(f"empty base-token validation stratum for {key}")
    if float(row["added_target_count"]) > 0:
        for field in ("added_target_loss", "added_target_bpb"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"nonfinite or absent {field} for {key}: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--segment-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-validation-receipt", type=Path, required=True)
    parser.add_argument("--authoritative-attempt-boundary", type=int)
    parser.add_argument("--post-boundary-attempt", type=int)
    parser.add_argument(
        "--attempt-authority-through",
        action="append",
        default=[],
        metavar="ITERATION:ATTEMPT",
        help=(
            "Select one authoritative attempt through each inclusive checkpoint "
            "boundary; may be repeated for multi-stage recovery."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    campaign = read_json(args.campaign_manifest)
    if (
        campaign.get("schema_version") != "apertus_mini_campaign_manifest_v1"
        or campaign.get("status") != "frozen"
    ):
        raise ValueError("campaign manifest is not frozen")
    segment = campaign["segments"][args.segment_id]
    start, end = int(segment["start_iteration"]), int(segment["end_iteration"])
    iterations = tuple(
        int(value)
        for value in campaign["evaluation"]["checkpoint_iterations"]
        if (start < int(value) <= end)
        or (args.segment_id == 0 and int(value) == 0)
    )
    panels = tuple(campaign["evaluation"]["validation_panels"])
    if len(panels) != 13 or len(set(panels)) != 13:
        raise ValueError("campaign does not freeze exactly 13 validation panels")
    legacy_recovery_policy = args.authoritative_attempt_boundary is not None
    if legacy_recovery_policy != (args.post_boundary_attempt is not None):
        raise ValueError(
            "authoritative attempt boundary and post-boundary attempt must be supplied together"
        )
    if legacy_recovery_policy and args.attempt_authority_through:
        raise ValueError("legacy and multi-stage attempt authority policies are exclusive")
    if legacy_recovery_policy and (
        args.authoritative_attempt_boundary not in iterations
        or args.post_boundary_attempt <= 0
    ):
        raise ValueError("invalid source-validation recovery authority policy")
    attempt_authority: list[tuple[int, int]] = []
    if legacy_recovery_policy:
        attempt_authority = [
            (int(args.authoritative_attempt_boundary), 0),
            (max(iterations), int(args.post_boundary_attempt)),
        ]
    elif args.attempt_authority_through:
        for item in args.attempt_authority_through:
            fields = item.split(":", 1)
            if len(fields) != 2:
                raise ValueError(f"invalid attempt authority item: {item!r}")
            boundary, attempt = map(int, fields)
            if boundary not in iterations or attempt < 0:
                raise ValueError(f"invalid attempt authority item: {item!r}")
            attempt_authority.append((boundary, attempt))
        if attempt_authority != sorted(set(attempt_authority)):
            raise ValueError("attempt authority boundaries must be unique and increasing")
        if attempt_authority[-1][0] != max(iterations):
            raise ValueError("attempt authority policy must cover the segment endpoint")

    def authoritative_attempt(iteration: int) -> int | None:
        for boundary, attempt in attempt_authority:
            if iteration <= boundary:
                return attempt
        return None

    initial = read_json(args.initial_validation_receipt)
    if (
        initial.get("schema_version")
        != "apertus_mini_initial_validation_receipt_v1"
        or initial.get("status") != "completed"
    ):
        raise ValueError("initial validation receipt is incomplete")
    initial_rows = {row["panel"]: row for row in initial.get("panels", [])}
    if set(initial_rows) != set(panels):
        raise ValueError("initial validation panel inventory drift")

    collected: dict[tuple[str, int, str], dict] = {}
    source_logs: dict[Path, str] = {}
    for arm in ARMS:
        logs = sorted(
            args.run_root.glob(
                f"segments/segment_{args.segment_id}/attempt_*/{arm}/driver.out"
            )
        )
        for log in logs:
            source_logs[log.resolve()] = sha256_file(log)
            match = re.fullmatch(r"attempt_(\d+)", log.parents[1].name)
            if match is None:
                raise ValueError(f"cannot resolve attempt from source log: {log}")
            attempt = int(match.group(1))
            for row in parse_log(log):
                iteration, panel = int(row["iteration"]), str(row["panel"])
                if iteration not in iterations or panel not in panels:
                    continue
                selected_attempt = authoritative_attempt(iteration)
                if selected_attempt is not None:
                    if attempt != selected_attempt:
                        continue
                key = (arm, iteration, panel)
                validate_metrics(row, key)
                previous = collected.get(key)
                if previous is not None:
                    numeric = (
                        "lm_loss",
                        "bpb",
                        "base_target_loss",
                        "base_target_bpb",
                        "base_target_count",
                        "base_target_bytes",
                        "added_target_count",
                        "added_target_bytes",
                    )
                    if any(
                        abs(float(previous[field]) - float(row[field])) > 1.0e-6
                        for field in numeric
                    ):
                        raise ValueError(f"conflicting duplicate validation binding: {key}")
                else:
                    collected[key] = row
        if args.segment_id == 0:
            for panel, row in initial_rows.items():
                key = (arm, 0, panel)
                normalized = {
                    "iteration": 0,
                    "panel": panel,
                    **{name: value for name, value in row.items() if name != "panel"},
                }
                validate_metrics(normalized, key)
                collected[key] = normalized

    expected_keys = {
        (arm, iteration, panel)
        for arm in ARMS
        for iteration in iterations
        for panel in panels
    }
    if set(collected) != expected_keys:
        missing = sorted(expected_keys - set(collected))
        extra = sorted(set(collected) - expected_keys)
        raise ValueError(
            f"source validation coverage drift: missing={missing[:5]} extra={extra[:5]}"
        )
    payload = {
        "schema_version": "apertus_mini_segment_source_validation_gate_v1",
        "status": "passed",
        "segment_id": args.segment_id,
        "campaign_manifest": {
            "path": str(args.campaign_manifest.resolve()),
            "sha256": sha256_file(args.campaign_manifest),
        },
        "iterations": list(iterations),
        "arms": list(ARMS),
        "panels": list(panels),
        "binding_count": len(collected),
        "expected_binding_count": len(expected_keys),
        "all_metrics_finite": True,
        "complete_panels_per_arm_iteration": True,
        "attempt_authority": (
            {
                "inclusive_upper_bound_attempts": [
                    {"through_iteration": boundary, "attempt": attempt}
                    for boundary, attempt in attempt_authority
                ],
                "legacy_policy": (
                    {
                        "authoritative_attempt_boundary": args.authoritative_attempt_boundary,
                        "post_boundary_attempt": args.post_boundary_attempt,
                    }
                    if legacy_recovery_policy
                    else None
                ),
            }
            if attempt_authority
            else {"single_attempt_campaign": True}
        ),
        "initial_validation_receipt": {
            "path": str(args.initial_validation_receipt.resolve()),
            "sha256": sha256_file(args.initial_validation_receipt),
        },
        "source_logs": [
            {"path": str(path), "sha256": digest}
            for path, digest in sorted(source_logs.items(), key=lambda item: str(item[0]))
        ],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "bindings": len(collected)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
