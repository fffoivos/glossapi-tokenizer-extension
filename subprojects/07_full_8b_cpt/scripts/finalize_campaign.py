#!/usr/bin/env python3
"""Wait for and freeze the complete 8B training and evaluation evidence set."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path

from contract import atomic_write_json, file_binding, read_json


def passing(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return str(read_json(path).get("status", "")).lower() in {"completed", "passed", "frozen", "promoted"}
    except Exception:
        return False


def active_job_ids(submissions: list[Path]) -> list[str]:
    ids = set()
    for path in submissions:
        try:
            jobs = read_json(path).get("jobs", {})
        except Exception:
            continue
        if isinstance(jobs, dict):
            ids.update(str(value) for value in jobs.values() if value)
    if not ids:
        return []
    result = subprocess.run(
        ["squeue", "-h", "-j", ",".join(sorted(ids)), "-o", "%i"],
        text=True, capture_output=True, check=False,
    )
    current = os.environ.get("SLURM_JOB_ID")
    return sorted({row.strip() for row in result.stdout.splitlines() if row.strip() and row.strip() != current})


def inspect(args: argparse.Namespace) -> dict:
    recipe = read_json(args.recipe)
    greek = [int(value) for value in recipe["evaluation"]["greekmmlu"]["checkpoint_updates"]]
    per_document_updates = {
        int(value)
        for value in recipe["evaluation"]["per_document_validation"]["milestone_updates"]
    }
    launch_gate = read_json(args.launch_gate)
    if (
        launch_gate.get("schema_version") != "apertus_full_8b_launch_gate_v1"
        or launch_gate.get("status") != "passed"
    ):
        raise ValueError("campaign finalizer requires a passing launch gate")
    evidence_bindings = launch_gate.get("evidence", {})
    initial_greek = Path(evidence_bindings["initial_greekmmlu"]["path"])
    greek_receipts = [file_binding(initial_greek)] if passing(initial_greek) else []
    doc_receipts = [
        Path(binding["path"])
        for binding in evidence_bindings.get("initial_per_document", [])
        if passing(Path(binding["path"])) and file_binding(Path(binding["path"])) == binding
    ]
    authoritative = []
    for iteration in greek[1:]:
        canonical = args.run_root / "checkpoint_evaluations" / f"iter_{iteration:07d}" / "authoritative_attempt.json"
        if not passing(canonical):
            continue
        choice = read_json(canonical)
        receipt = Path(choice["greekmmlu_receipt"])
        if not passing(receipt):
            continue
        greek_receipts.append(file_binding(receipt))
        authoritative.append(file_binding(canonical))
        if iteration in per_document_updates:
            root = Path(choice["per_document_root"])
            doc_receipts.extend(path for path in root.glob("*.receipt.json") if passing(path))
    selected = read_json(args.selected_profile)
    segments = len(selected["selection"]["segment_boundaries"]) - 1
    audits = [path for path in (args.run_root / "training_audits").glob("segment_*_attempt_*.json") if passing(path)]
    terminal_iteration = greek[-1]
    terminal_canonical = (
        args.run_root
        / "checkpoint_evaluations"
        / f"iter_{terminal_iteration:07d}"
        / "authoritative_attempt.json"
    )
    terminal_export = None
    if passing(terminal_canonical):
        attempt = Path(read_json(terminal_canonical)["attempt_root"])
        export_receipt = attempt / "export/checkpoint_eval_export_receipt.json"
        hf_root = attempt / "export/hf"
        if passing(export_receipt) and hf_root.is_dir() and any(hf_root.glob("*.safetensors")):
            terminal_export = {"receipt": file_binding(export_receipt), "hf_root": str(hf_root.resolve())}
    submissions = list(args.run_root.glob("checkpoint_evaluations/iter_*/attempt_*/submission.json"))
    return {
        "greekmmlu": greek_receipts,
        "authoritative": authoritative,
        "per_document": [file_binding(path) for path in sorted(set(doc_receipts))],
        "training_audits": [file_binding(path) for path in sorted(audits)],
        "required_training_segments": segments,
        "terminal_export": terminal_export,
        "active_evaluation_jobs": active_job_ids(submissions),
        "expected_greekmmlu": len(greek),
        "expected_authoritative": len(greek) - 1,
        "expected_per_document": 13 * len(per_document_updates),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--launch-gate", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--selected-profile", type=Path, required=True)
    parser.add_argument("--training-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--max-seconds", type=int, default=41400)
    args = parser.parse_args()
    if not passing(args.training_receipt):
        raise ValueError("training completion receipt is absent or invalid")
    started = time.monotonic()
    while True:
        evidence = inspect(args)
        complete = (
            len(evidence["greekmmlu"]) == evidence["expected_greekmmlu"]
            and len(evidence["authoritative"]) == evidence["expected_authoritative"]
            and len(evidence["per_document"]) == evidence["expected_per_document"]
            and len(evidence["training_audits"]) >= evidence["required_training_segments"]
            and evidence["terminal_export"] is not None
            and not evidence["active_evaluation_jobs"]
        )
        if complete:
            payload = {
                "schema_version": "apertus_full_8b_campaign_evidence_completion_v1",
                "status": "completed",
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                "training_completion": file_binding(args.training_receipt),
                "launch_gate": file_binding(args.launch_gate),
                "selected_profile": file_binding(args.selected_profile),
                "greekmmlu_receipts": evidence["greekmmlu"],
                "authoritative_checkpoint_evaluations": evidence["authoritative"],
                "per_document_panel_receipts": evidence["per_document"],
                "training_attempt_audits": evidence["training_audits"],
                "terminal_model_export": evidence["terminal_export"],
                "counts": {
                    "greekmmlu": evidence["expected_greekmmlu"],
                    "per_document_panels": evidence["expected_per_document"],
                    "source_validated_segments": evidence["required_training_segments"],
                },
                "checkpoint_averaging": False,
            }
            atomic_write_json(args.output, payload)
            print(json.dumps({
                "ok": True,
                "greekmmlu": evidence["expected_greekmmlu"],
                "per_document": evidence["expected_per_document"],
            }))
            return 0
        if time.monotonic() - started > args.max_seconds:
            state = args.run_root / "orchestration/evidence_finalizer_timeout.json"
            atomic_write_json(state, {"schema_version":"apertus_full_8b_evidence_wait_v1","status":"pending","observed_at":dt.datetime.now(dt.timezone.utc).isoformat(),"evidence":evidence}, exclusive=False)
            raise TimeoutError("evidence did not complete inside one controller allocation")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
