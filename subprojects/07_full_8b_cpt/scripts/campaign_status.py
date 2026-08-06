#!/usr/bin/env python3
"""Report exact full-8B training, evaluation and dependency progress."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from contract import read_json


def passing(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return str(read_json(path).get("status", "")).lower() in {"completed", "passed", "frozen", "promoted"}
    except Exception:
        return False


def jobs() -> list[dict]:
    result = subprocess.run(
        ["squeue", "-h", "-u", "fffoivos", "-o", "%i|%j|%T|%M|%R"],
        text=True, capture_output=True, check=False,
    )
    rows = []
    for line in result.stdout.splitlines():
        fields = line.split("|", 4)
        if len(fields) == 5 and (fields[1].startswith("full8b") or fields[1].startswith("mini_ckpt") or fields[1].startswith("mini_native")):
            rows.append(dict(zip(("job_id", "name", "state", "elapsed", "reason_or_nodes"), fields)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--prelaunch-root", type=Path, required=True)
    parser.add_argument(
        "--initial-greekmmlu",
        type=Path,
        help=(
            "completed iteration-zero GreekMMLU receipt; supply this when the "
            "validated model anchor predates a later prelaunch-stage rebuild"
        ),
    )
    args = parser.parse_args()
    selected = read_json(args.selected_profile)
    recipe_path = Path(
        selected.get("recipe", {}).get(
            "path",
            Path(__file__).resolve().parents[1]
            / "configs"
            / "recipe_8b_full_mixed.json",
        )
    )
    recipe = read_json(recipe_path)
    boundaries = [int(value) for value in selected["selection"]["segment_boundaries"]]
    completed = [iteration for iteration in boundaries[1:] if passing(args.run_root / "checkpoint_receipts" / f"iter_{iteration:07d}.json")]
    completed_update = max(completed, default=0)
    authoritative = list(args.run_root.glob("checkpoint_evaluations/iter_*/authoritative_attempt.json"))
    initial_docs = [path for path in (args.prelaunch_root / "per_document_initial").glob("*.receipt.json") if passing(path)]
    endpoint_docs = []
    per_document_updates = [
        int(value)
        for value in recipe["evaluation"]["per_document_validation"]["milestone_updates"]
        if int(value) != 0
    ]
    for iteration in per_document_updates:
        canonical = args.run_root / "checkpoint_evaluations" / f"iter_{iteration:07d}" / "authoritative_attempt.json"
        if passing(canonical):
            root = Path(read_json(canonical)["per_document_root"])
            endpoint_docs.extend(path for path in root.glob("*.receipt.json") if passing(path))
    current_jobs = jobs()
    initial_greekmmlu = (
        args.initial_greekmmlu
        if args.initial_greekmmlu is not None
        else args.prelaunch_root / "initial_greekmmlu" / "initial_greekmmlu_receipt.json"
    )
    latest = args.run_root / "orchestration/latest.json"
    result = {
        "schema_version": "apertus_full_8b_campaign_status_v2",
        "observed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "profile_id": selected["selection"]["profile_id"],
        "nodes": selected["selection"]["nodes"],
        "progress": {
            "completed_checkpoint_gated_updates": completed_update,
            "required_updates": int(recipe["batch_and_parallelism"]["training_updates"]),
            "completed_token_slots": completed_update * 4_194_304,
            "required_token_slots": int(recipe["data"]["planning_training_slots_tokens"]),
            "segments_completed": len(completed),
            "segments_required": len(boundaries) - 1,
        },
        "evidence": {
            "greekmmlu_completed": (1 if passing(initial_greekmmlu) else 0) + sum(passing(path) for path in authoritative),
            "greekmmlu_required": len(recipe["evaluation"]["greekmmlu"]["checkpoint_updates"]),
            "per_document_panels_completed": len(initial_docs) + len(endpoint_docs),
            "per_document_panels_required": 13 * len(recipe["evaluation"]["per_document_validation"]["milestone_updates"]),
            "training_complete": passing(args.run_root / "training_completion_receipt.json"),
            "evidence_complete": passing(args.run_root / "campaign_evidence_completion_receipt.json"),
        },
        "active_jobs": current_jobs,
        "latest_event": read_json(latest) if latest.is_file() else None,
        "next_dependency": "campaign complete" if passing(args.run_root / "campaign_evidence_completion_receipt.json") else (current_jobs[0] if current_jobs else "inspect latest event or failed dependency"),
        "eta_contract": "Recompute compute-only, training-complete and evidence-complete clocks after every start, gate, failure or evaluation-backlog change.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
