#!/usr/bin/env python3
"""Freeze one benchmark-blind 1.5B LR pilot arm from its exact run evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from contract_utils import (
    executing_code_bundle,
    file_binding,
    read_json,
    require,
    write_json_atomic,
)
from finalize_profile_benchmark import parse_log
from materialize_phase_cache import validate_overlay_receipt

LINE = re.compile(r"validation loss at iteration (\d+)(?: on validation set)? \[([^\]]+)\]\s*\|\s*(.*)")
METRIC = re.compile(r"(?:^|\|)\s*([^|]+?) value:\s*([-+0-9.eE]+)")
SELECTED_PANELS = ("hplt", "english", "de", "ru", "zh", "code", "old_greek")
PANEL_ROLES = {
    "hplt": ["hplt"],
    "foreign": ["english", "de", "ru", "zh", "code"],
    "old_greek": ["old_greek"],
}
FLOORS = {"7.5e-5": "7.5e-6", "1.0e-4": "1.0e-5", "1.25e-4": "1.25e-5"}


def parse_validation_log(path: Path) -> dict[tuple[int, str], float]:
    require(path.is_file(), f"LR-pilot log missing: {path}")
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines: list[tuple[int | None, str]] = []
    for raw in raw_lines:
        rank_match = re.match(r"^(\d+):\s?(.*)$", raw)
        lines.append((int(rank_match.group(1)), rank_match.group(2)) if rank_match else (None, raw))
    rows: dict[tuple[int, str], float] = {}
    for index, (rank, line) in enumerate(lines):
        match = LINE.search(line)
        if match is None:
            continue
        block = match.group(3)
        for continuation_rank, continuation in lines[index + 1 :]:
            if LINE.search(continuation) or re.fullmatch(r"\s*-{10,}\s*", continuation):
                break
            if rank is not None and continuation_rank is not None and continuation_rank != rank:
                continue
            block += continuation
        metrics = {name.strip(): float(value) for name, value in METRIC.findall(block)}
        if "lm loss" not in metrics:
            continue
        iteration, panel = int(match.group(1)), match.group(2)
        if iteration not in {0, 238} or panel not in SELECTED_PANELS:
            continue
        value = float(metrics["lm loss"])
        require(math.isfinite(value) and value > 0, f"invalid {panel} loss at update {iteration}")
        key = (iteration, panel)
        require(key not in rows or abs(rows[key] - value) <= 1e-8, f"conflicting validation row: {key}")
        rows[key] = value
    expected = {(iteration, panel) for iteration in (0, 238) for panel in SELECTED_PANELS}
    require(set(rows) == expected, f"LR-pilot validation panel coverage drift: missing={sorted(expected - set(rows))}")
    return rows


def stable_identifier(value: dict[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-contract", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--pilot-log", type=Path, required=True)
    parser.add_argument("--run-metadata", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable LR-pilot receipt exists: {args.output}")
    contract = read_json(args.benchmark_contract)
    require(contract.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_contract_v1", "LR-pilot contract schema drift")
    require(contract.get("status") == "frozen" and contract.get("kind") == "lr" and contract.get("scale") == "1p5b", "LR-pilot contract identity drift")
    peak = str(contract.get("peak_lr"))
    require(peak in FLOORS and str(contract.get("floor_lr")) == FLOORS[peak] and int(contract.get("updates", -1)) == 238, "LR-pilot LR/horizon drift")
    access = contract.get("benchmark_access")
    require(isinstance(access, dict) and access.get("greekmmlu") is False and access.get("native_suite") is False, "LR-pilot benchmark-access policy drift")
    current = executing_code_bundle()
    bundle = contract.get("executing_code_bundle")
    require(isinstance(bundle, dict) and bundle.get("root") == current["root"] and bundle.get("tree_sha256") == current["tree_sha256"], "LR-pilot code-bundle drift")
    output_root = Path(str(contract.get("output_root", ""))).resolve()
    expected_paths = {
        args.preflight.resolve(): output_root / "preflight.json",
        args.pilot_log.resolve(): output_root / "pilot/driver.out",
        args.run_metadata.resolve(): output_root / "pilot/run_metadata.json",
        args.checkpoint_root.resolve(): output_root / "pilot/checkpoints",
    }
    require(all(actual == expected.resolve() for actual, expected in expected_paths.items()), "LR-pilot evidence path drift")
    preflight = read_json(args.preflight)
    require(preflight.get("schema_version") == "apertus_hard_h_to_g_prelaunch_benchmark_preflight_v1" and preflight.get("status") == "passed", "LR-pilot preflight drift")
    require(preflight.get("contract") == file_binding(args.benchmark_contract), "LR-pilot preflight contract binding drift")
    overlay_audits: dict[str, dict[str, object]] = {}
    for phase, binding_field, root_field in (
        (1, "phase_cache_overlay_receipt", "phase_cache_root"),
        (2, "phase2_cache_overlay_receipt", "phase2_cache_root"),
    ):
        binding = contract.get(binding_field)
        require(isinstance(binding, dict), f"Phase-{phase} cache overlay binding missing")
        receipt_path = Path(str(binding.get("path", "")))
        require(receipt_path.is_file() and binding == file_binding(receipt_path), f"Phase-{phase} cache overlay binding drift")
        receipt = read_json(receipt_path)
        overlay_audits[str(phase)] = validate_overlay_receipt(
            receipt,
            phase=phase,
            overlay_root=Path(str(contract.get(root_field, ""))),
            require_pristine=False,
        )
    metadata = read_json(args.run_metadata)
    require(metadata.get("resume_training") == 0 and metadata.get("init_ckpt") == contract.get("initialization_root"), "LR-pilot did not start independently from frozen initialization")
    require(str(metadata.get("lr_peak")) == peak and str(metadata.get("lr_final")) == FLOORS[peak], "LR-pilot realized LR drift")
    require(int(metadata.get("global_batch_samples", -1)) == 1024 and int(metadata.get("slurm_nodes", -1)) == int(contract.get("nodes", -2)), "LR-pilot realized geometry drift")
    optimizer_rows = parse_log(args.pilot_log)
    require(set(range(1, 239)) <= set(optimizer_rows), "LR-pilot lacks optimizer updates 1..238")
    require(max(optimizer_rows) == 238, "LR-pilot ran beyond frozen horizon")
    tracker = args.checkpoint_root / "latest_checkpointed_iteration.txt"
    checkpoint_metadata = args.checkpoint_root / "iter_0000238/.metadata"
    require(tracker.is_file() and tracker.read_text(encoding="utf-8").strip() == "238", "LR-pilot final checkpoint tracker drift")
    require(checkpoint_metadata.is_file() and checkpoint_metadata.stat().st_size > 0, "LR-pilot final DCP metadata missing")
    validations = parse_validation_log(args.pilot_log)
    initial = {panel: validations[(0, panel)] for panel in SELECTED_PANELS}
    final = {panel: validations[(238, panel)] for panel in SELECTED_PANELS}
    initialization_binding = stable_identifier(contract["initialization_permit"])
    optimizer_state_evidence = {
        "output_root": str(output_root),
        "benchmark_contract": file_binding(args.benchmark_contract),
        "checkpoint_metadata": file_binding(checkpoint_metadata),
        "tracker": file_binding(tracker),
    }
    payload = {
        "schema_version": "apertus_hard_h_to_g_lr_pilot_arm_v1",
        "status": "completed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": "1p5b",
        "peak_lr": peak,
        "floor_lr": FLOORS[peak],
        "updates": 238,
        "skipped_updates": 0,
        "nonfinite_updates": 0,
        "greekmmlu_accessed": False,
        "downstream_benchmarks_accessed": False,
        "initialization_binding": initialization_binding,
        "data_prefix_trajectory_sha256": str(contract["phase_cache_tree_sha256"]),
        "optimizer_state_id": stable_identifier(optimizer_state_evidence),
        "panel_losses": {"initial": initial, "final": final, "roles": PANEL_ROLES},
        "qualification_cache_overlay_audits": overlay_audits,
        "benchmark_contract": file_binding(args.benchmark_contract),
        "optimizer_state_evidence": optimizer_state_evidence,
        "executing_code_bundle": current,
        "evidence": [
            file_binding(args.benchmark_contract), file_binding(args.preflight),
            file_binding(args.pilot_log), file_binding(args.run_metadata),
            file_binding(checkpoint_metadata), file_binding(tracker),
        ],
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
