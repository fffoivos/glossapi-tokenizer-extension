#!/usr/bin/env python3
"""Collect complete source-conditioned NLL/BPB trajectories from trainer logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from production.campaign_contract import ARMS, atomic_write_json, read_json


LINE = re.compile(
    r"validation loss at iteration (\d+)(?: on validation set)? \[([^\]]+)\]\s*\|\s*(.*)"
)
METRIC = re.compile(r"(?:^|\|)\s*([^|]+?) value:\s*([-+0-9.eE]+)")
REQUIRED = {
    "lm loss",
    "base-token target loss",
    "base-token target count",
    "base-token target bytes",
}


def parse_log(path: Path) -> list[dict]:
    rows = []
    lines = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        rank_match = re.match(r"^(\d+):\s?(.*)$", raw)
        lines.append(
            (
                int(rank_match.group(1)) if rank_match is not None else None,
                rank_match.group(2) if rank_match is not None else raw,
            )
        )
    for index, (rank, line) in enumerate(lines):
        match = LINE.search(line)
        if match is None:
            continue
        block = match.group(3)
        for continuation_rank, continuation in lines[index + 1 :]:
            if LINE.search(continuation) or re.fullmatch(r"\s*-{10,}\s*", continuation):
                break
            # Slurm multiplexes rank output. A checkpoint message from rank 0
            # can land inside a wrapped validation record from another rank;
            # skip it instead of corrupting the split metric name.
            if (
                rank is not None
                and continuation_rank is not None
                and continuation_rank != rank
            ):
                continue
            block += continuation
        metrics = {name.strip(): float(value) for name, value in METRIC.findall(block)}
        if not REQUIRED.issubset(metrics):
            continue
        base_count = metrics["base-token target count"]
        added_count = metrics.get("added-token target count", 0.0)
        base_bytes = metrics["base-token target bytes"]
        added_bytes = metrics.get("added-token target bytes", 0.0)
        base_nll = metrics["base-token target loss"] * base_count
        added_nll = metrics.get("added-token target loss", 0.0) * added_count
        rows.append(
            {
                "iteration": int(match.group(1)),
                "panel": match.group(2),
                "lm_loss": metrics["lm loss"],
                "base_target_loss": metrics["base-token target loss"],
                "added_target_loss": metrics.get("added-token target loss"),
                "base_target_count": base_count,
                "added_target_count": added_count,
                "base_target_bytes": base_bytes,
                "added_target_bytes": added_bytes,
                "bpb": (base_nll + added_nll) / max(base_bytes + added_bytes, 1.0) / math.log(2),
                "base_target_bpb": base_nll / max(base_bytes, 1.0) / math.log(2),
                "added_target_bpb": (
                    added_nll / added_bytes / math.log(2) if added_bytes > 0 else None
                ),
                "source_log": str(path.resolve()),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--initial-validation-receipt", type=Path, required=True)
    parser.add_argument("--attempt-authority-through", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    campaign = read_json(args.campaign_manifest)
    expected_iterations = tuple(int(value) for value in campaign["evaluation"]["checkpoint_iterations"])
    expected_panels = tuple(campaign["evaluation"]["validation_panels"])
    initial = read_json(args.initial_validation_receipt)
    if initial.get("schema_version") != "apertus_mini_initial_validation_receipt_v1" or initial.get("status") != "completed":
        raise ValueError("initial validation receipt is incomplete")
    initial_rows = {row["panel"]: row for row in initial["panels"]}
    if set(initial_rows) != set(expected_panels):
        raise ValueError("initial validation panel inventory drift")

    authority: list[tuple[int, int]] = []
    for item in args.attempt_authority_through:
        fields = item.split(":", 1)
        if len(fields) != 2:
            raise ValueError(f"invalid attempt authority: {item!r}")
        upper, attempt = map(int, fields)
        if upper <= 0 or attempt < 0:
            raise ValueError(f"invalid attempt authority: {item!r}")
        authority.append((upper, attempt))
    if authority:
        if authority != sorted(authority) or len({upper for upper, _ in authority}) != len(authority):
            raise ValueError("attempt authority thresholds must be unique and increasing")
        if authority[-1][0] < max(expected_iterations):
            raise ValueError("attempt authority does not cover the final checkpoint")

    def authoritative_attempt(iteration: int) -> int | None:
        for upper, attempt in authority:
            if iteration <= upper:
                return attempt
        return None

    collected: dict[tuple[str, int, str], dict] = {}
    for arm in ARMS:
        for log in sorted(args.run_root.glob(f"segments/segment_*/attempt_*/{arm}/driver.out")):
            attempt = int(log.parents[1].name.removeprefix("attempt_"))
            for row in parse_log(log):
                if row["iteration"] not in expected_iterations or row["panel"] not in expected_panels:
                    continue
                selected_attempt = authoritative_attempt(row["iteration"])
                if authority and attempt != selected_attempt:
                    continue
                key = (arm, row["iteration"], row["panel"])
                previous = collected.get(key)
                if previous is not None:
                    numeric = ("lm_loss", "base_target_loss", "bpb", "base_target_bpb")
                    if any(abs(float(previous[name]) - float(row[name])) > 1.0e-6 for name in numeric):
                        raise ValueError(f"conflicting duplicate validation result: {key}")
                else:
                    collected[key] = {"arm_id": arm, **row}
        for panel, row in initial_rows.items():
            collected[(arm, 0, panel)] = {
                "arm_id": arm,
                "iteration": 0,
                "panel": panel,
                **{key: value for key, value in row.items() if key != "panel"},
                "source_log": str(args.initial_validation_receipt.resolve()),
            }

    expected = len(ARMS) * len(expected_iterations) * len(expected_panels)
    missing = [
        (arm, iteration, panel)
        for arm in ARMS
        for iteration in expected_iterations
        for panel in expected_panels
        if (arm, iteration, panel) not in collected
    ]
    if missing:
        raise ValueError(f"missing {len(missing)} validation bindings; first={missing[:5]}")
    rows = [collected[key] for key in sorted(collected)]
    payload = {
        "schema_version": "apertus_mini_validation_trajectory_v1",
        "status": "completed",
        "rows": rows,
        "row_count": len(rows),
        "expected_row_count": expected,
        "attempt_authority": [
            {"inclusive_upper_iteration": upper, "attempt": attempt}
            for upper, attempt in authority
        ],
    }
    atomic_write_json(args.output_json, payload)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(args.output_csv) + ".partial")
    fields = [
        "arm_id",
        "iteration",
        "panel",
        "lm_loss",
        "bpb",
        "base_target_loss",
        "base_target_bpb",
        "added_target_loss",
        "added_target_bpb",
        "base_target_count",
        "added_target_count",
        "base_target_bytes",
        "added_target_bytes",
        "source_log",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.output_csv)
    print(json.dumps({"ok": True, "rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
