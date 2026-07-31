#!/usr/bin/env python3
"""Verify the isolated two-segment GPU smoke and freeze its evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PROBE_ROOT = HERE.parent
SHARED = PROBE_ROOT.parent / "05_training_dataset_bridge" / "scripts"
sys.path.insert(0, str(SHARED))

from bridge_common import (  # noqa: E402
    read_json,
    sha256_file,
    utc_now,
    validate_file_tree_receipt,
    write_json_atomic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-assets-receipt", type=Path, required=True)
    parser.add_argument("--phase1-receipt", type=Path, required=True)
    parser.add_argument("--phase2-receipt", type=Path, required=True)
    parser.add_argument("--phase1-output", type=Path, required=True)
    parser.add_argument("--phase2-output", type=Path, required=True)
    parser.add_argument("--phase1-log", type=Path, required=True)
    parser.add_argument("--phase1-err", type=Path, required=True)
    parser.add_argument("--phase2-log", type=Path, required=True)
    parser.add_argument("--phase2-err", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _checkpoint_receipt(path: Path, iteration: int, assets_path: Path) -> dict[str, Any]:
    value = read_json(path.resolve())
    if value.get("schema_version") != "greek_cpt_resume_checkpoint_receipt_v1":
        raise ValueError(f"checkpoint receipt schema drift: {path}")
    expected_terminal = iteration == 2
    expected_assets_sha = sha256_file(assets_path)
    if (
        value.get("status") != "frozen"
        or value.get("smoke") is not True
        or value.get("iteration") != iteration
        or value.get("terminal") is not expected_terminal
        or value.get("training_assets_receipt", {}).get("path") != str(assets_path)
        or value.get("training_assets_receipt", {}).get("sha256") != expected_assets_sha
    ):
        raise ValueError(f"checkpoint receipt binding drift: {path}")
    validate_file_tree_receipt(value["checkpoint_tree"])
    return value


def _metadata(path: Path, *, resume: int) -> dict[str, Any]:
    value = read_json((path / "run_metadata.json").resolve())
    expected = {
        "arm": "td",
        "seq_length": 4096,
        "global_batch_samples": 8,
        "global_batch_tokens": 32768,
        "microbatch_size": 2,
        "train_tokens": 65536,
        "train_samples": 16,
        "train_iters": 2,
        "curriculum_order_mode": "randomized",
        "mock_data": 0,
        "resume_training": resume,
        "disable_save": 0,
        "slurm_nodes": 1,
        "slurm_gpus_per_node": 4,
        "model_world_size": 4,
        "make_vocab_size_divisible_by": 256,
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(
                f"smoke metadata drift for {name}: expected {expected_value!r}, "
                f"got {value.get(name)!r}"
            )
    return value


def _log_evidence(out_path: Path, err_path: Path, *, iteration: int, resume: int) -> dict[str, Any]:
    out = out_path.read_text(encoding="utf-8", errors="replace")
    err = err_path.read_text(encoding="utf-8", errors="replace")
    combined = out + "\n" + err
    if f"checkpoint mode: RESUME_TRAINING={resume}" not in combined:
        raise ValueError(f"resume mode missing from smoke log for iteration {iteration}")
    if "=== bakeoff arm td done ===" not in combined:
        raise ValueError(f"trainer completion marker missing for iteration {iteration}")
    match = re.search(
        rf"iteration\s+{iteration}/\s*2\s*\|.*?learning rate:\s*([0-9.Ee+-]+)\s*\|"
        rf".*?lm loss:\s*([0-9.Ee+-]+)\s*\|.*?number of skipped iterations:\s*0\s*\|"
        rf".*?number of nan iterations:\s*0\s*\|",
        combined,
    )
    if match is None:
        raise ValueError(f"finite loss/cadence evidence missing for iteration {iteration}")
    learning_rate, loss = (float(value) for value in match.groups())
    if not math.isfinite(learning_rate) or not math.isfinite(loss):
        raise ValueError(f"non-finite smoke metric at iteration {iteration}")
    if re.search(r"(^|\n)(Traceback|ERROR:)", combined):
        raise ValueError(f"error marker present in completed smoke log for iteration {iteration}")
    return {
        "iteration": iteration,
        "learning_rate": learning_rate,
        "lm_loss": loss,
        "stdout": _file(out_path),
        "stderr": _file(err_path),
    }


def main() -> int:
    args = parse_args()
    assets_path = args.training_assets_receipt.resolve()
    assets = read_json(assets_path)
    if (
        assets.get("schema_version") != "greek_cpt_training_assets_receipt_v1"
        or assets.get("status") != "frozen"
    ):
        raise ValueError("training assets are not frozen")

    phase1_receipt = _checkpoint_receipt(args.phase1_receipt, 1, assets_path)
    phase2_receipt = _checkpoint_receipt(args.phase2_receipt, 2, assets_path)
    phase1_metadata = _metadata(args.phase1_output.resolve(), resume=0)
    phase2_metadata = _metadata(args.phase2_output.resolve(), resume=1)
    if phase1_metadata["data_prefix"] == phase2_metadata["data_prefix"]:
        raise ValueError("smoke did not switch between the two frozen phase blends")
    if phase1_metadata["tokenizer"] != assets["tokenizer"]["root"]:
        raise ValueError("phase-1 smoke tokenizer differs from frozen assets")
    if phase2_metadata["tokenizer"] != assets["tokenizer"]["root"]:
        raise ValueError("phase-2 smoke tokenizer differs from frozen assets")

    phase1_log = _log_evidence(args.phase1_log.resolve(), args.phase1_err.resolve(), iteration=1, resume=0)
    phase2_log = _log_evidence(args.phase2_log.resolve(), args.phase2_err.resolve(), iteration=2, resume=1)
    phase2_combined = (
        args.phase2_log.read_text(encoding="utf-8", errors="replace")
        + "\n"
        + args.phase2_err.read_text(encoding="utf-8", errors="replace")
    )
    reset_pattern = re.compile(
        r"\[phase_relative_data_index\].*train consumed_samples 8 -> 0 "
        r"\(phase_start_iteration=1; global state kept\)"
    )
    if reset_pattern.search(phase2_combined) is None:
        raise ValueError("phase-relative 8 -> 0 data-index reset was not observed")
    if "successfully loaded checkpoint" not in phase2_combined or "at iteration 1" not in phase2_combined:
        raise ValueError("phase-2 smoke did not load the phase-1 checkpoint at iteration 1")

    payload = {
        "schema_version": "greek_cpt_two_phase_smoke_verification_v1",
        "status": "passed",
        "completed_at": utc_now(),
        "training_assets_receipt": _file(assets_path),
        "geometry": {
            "nodes": 1,
            "gpus_per_node": 4,
            "tensor_parallel": 2,
            "global_batch_samples": 8,
            "microbatch_size": 2,
            "iterations": 2,
        },
        "phase1": {
            "checkpoint_receipt": _file(args.phase1_receipt.resolve()),
            "checkpoint_tree_sha256": phase1_receipt["checkpoint_tree"]["tree_sha256"],
            "metadata": _file((args.phase1_output / "run_metadata.json").resolve()),
            "data_prefix": phase1_metadata["data_prefix"],
            "log_evidence": phase1_log,
        },
        "phase2": {
            "checkpoint_receipt": _file(args.phase2_receipt.resolve()),
            "checkpoint_tree_sha256": phase2_receipt["checkpoint_tree"]["tree_sha256"],
            "metadata": _file((args.phase2_output / "run_metadata.json").resolve()),
            "data_prefix": phase2_metadata["data_prefix"],
            "phase_relative_data_index": "8 -> 0",
            "log_evidence": phase2_log,
        },
    }
    write_json_atomic(args.output.resolve(), payload)
    print(json.dumps({"ok": True, "output": str(args.output.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
