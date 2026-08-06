#!/usr/bin/env python3
"""Write the training-complete receipt after the terminal checkpoint gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected = read_json(args.selected_profile)
    boundaries = selected["selection"]["segment_boundaries"]
    receipts = []
    for iteration in boundaries[1:]:
        path = args.run_root / "checkpoint_receipts" / f"iter_{iteration:07d}.json"
        value = read_json(path)
        if value.get("status") != "frozen" or value.get("iteration") != iteration:
            raise ValueError(f"checkpoint gate drift at {iteration}")
        receipts.append(file_binding(path))
    terminal = read_json(Path(receipts[-1]["path"]))
    recipe = read_json(args.recipe)
    terminal_iteration = int(recipe["batch_and_parallelism"]["training_updates"])
    active_tokens = int(recipe["data"]["planning_post_dedup_active_tokens"])
    token_slots = int(recipe["data"]["planning_training_slots_tokens"])
    if terminal.get("terminal") is not True or terminal.get("iteration") != terminal_iteration:
        raise ValueError("terminal checkpoint gate absent")
    payload = {
        "schema_version": "apertus_full_8b_training_completion_v1",
        "status": "completed",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "terminal_iteration": terminal_iteration,
        "active_tokens_consumed": active_tokens,
        "scheduled_token_slots": token_slots,
        "recipe": file_binding(args.recipe),
        "selected_profile": file_binding(args.selected_profile),
        "launch_gate": file_binding(args.launch_gate),
        "checkpoint_receipts": receipts,
        "terminal_checkpoint": receipts[-1],
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({"ok": True, "terminal_iteration": terminal_iteration, "segments": len(receipts)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
