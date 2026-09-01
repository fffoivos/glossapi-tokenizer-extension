#!/usr/bin/env python3
"""Finalize one actual-checkpoint Phase-3 cursor/LR resume smoke."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import re
from decimal import Decimal
from pathlib import Path

from contract_utils import executing_code_bundle, file_binding, read_json, require, write_json_atomic


LR = re.compile(r"learning rate:\s*([+\-0-9.Ee]+)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--start-update", type=int, choices=(3218, 3456), required=True)
    parser.add_argument("--floor-lr", required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--checkpoint-audit", type=Path, required=True)
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"immutable Phase-3 resume-smoke receipt exists: {args.output}")
    end_update = args.start_update + 1
    preflight = read_json(args.preflight)
    require(
        preflight.get("schema_version") == "apertus_hard_h_to_g_train_segment_preflight_v1"
        and preflight.get("status") == "passed"
        and preflight.get("scale") == args.scale
        and preflight.get("phase") == 3
        and preflight.get("start_update") == args.start_update
        and preflight.get("exit_update") == end_update
        and preflight.get("one_update_resume_smoke") is True,
        "Phase-3 smoke preflight drift",
    )
    audit = read_json(args.checkpoint_audit)
    require(
        audit.get("schema_version") == "apertus_hard_h_to_g_checkpoint_state_audit_v1"
        and audit.get("status") == "passed"
        and audit.get("scale") == args.scale
        and audit.get("source_phase") == 3
        and audit.get("update") == end_update
        and all(audit.get("checks", {}).values()),
        "Phase-3 smoke checkpoint audit drift",
    )
    require(audit.get("segment_preflight") == file_binding(args.preflight), "Phase-3 smoke audit/preflight binding drift")
    require(audit.get("training_log") == file_binding(args.training_log), "Phase-3 smoke audit/log binding drift")
    expected_cursor = (args.start_update - 3218) * 1024
    text = args.training_log.read_text(encoding="utf-8", errors="replace")
    require(f"phase_local_samples={expected_cursor}" in text, "Phase-3 local cursor evidence missing")
    require("[h2g_constant_floor]" in text, "constant-floor runtime guard evidence missing")
    values = [float(match.group(1)) for match in LR.finditer(text)]
    require(values and all(math.isfinite(value) for value in values), "Phase-3 learning-rate evidence missing")
    floor = Decimal(args.floor_lr)
    require(all(Decimal(str(value)) == floor for value in values), "Phase-3 learning rate reanchored")
    scheduler = audit.get("scheduler", {})
    require(Decimal(str(scheduler.get("max_lr"))) == floor, "saved scheduler max LR drift")
    require(Decimal(str(scheduler.get("min_lr"))) == floor, "saved scheduler min LR drift")
    require(str(scheduler.get("lr_decay_style")).lower() == "constant", "saved scheduler decay style drift")
    require(audit.get("data_cursor", {}).get("phase_local_consumed_samples") == (end_update - 3218) * 1024, "saved Phase-3 cursor drift")
    payload = {
        "schema_version": "apertus_hard_h_to_g_phase3_resume_smoke_v1",
        "status": "passed",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scale": args.scale,
        "start_update": args.start_update,
        "end_update": end_update,
        "entry_phase_local_samples": expected_cursor,
        "saved_phase_local_samples": (end_update - 3218) * 1024,
        "floor_lr": args.floor_lr,
        "preflight": file_binding(args.preflight),
        "checkpoint_audit": file_binding(args.checkpoint_audit),
        "training_log": file_binding(args.training_log),
        "checks": {
            "exact_checkpoint_permit_was_preflighted": True,
            "phase3_cache_was_preflighted": True,
            "phase_local_cursor_matches_history": True,
            "one_optimizer_update_completed_without_skip_or_nonfinite": True,
            "constant_floor_guard_executed": True,
            "learning_rate_remained_at_exact_floor": True,
            "saved_scheduler_is_constant_at_exact_floor": True,
            "saved_checkpoint_contains_model_optimizer_rng_and_scheduler": True,
        },
        "executing_code_bundle": executing_code_bundle(),
    }
    write_json_atomic(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
