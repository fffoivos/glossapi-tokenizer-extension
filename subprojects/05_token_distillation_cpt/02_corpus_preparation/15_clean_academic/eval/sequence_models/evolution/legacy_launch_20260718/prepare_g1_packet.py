#!/usr/bin/env python3
"""Materialize, but never submit, the exact parent-bound G1 anchor queue."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_COMMIT = "931a56d119b9ab44e79c23fa82a16cd2edf0c4b7"
EXPECTED_PARENT_ID = "g0-34a7dde38689d1b15e8afd42"
EXPECTED_PARENT_RECEIPT_SHA = "95eb0f5a855f780c606d87844fa680e889dc5b70b7eae752b051fa3c48902955"
EXPECTED_PARENT_PREDICTION_SHA = "84179ec0d45dbbf5c5ad1c78fd20b7ed2b82ead70c7287180bf3ed024e9a6d5a"
EXPECTED_FAMILY_COUNTS = {
    "anchor_threshold": 5,
}


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--g0-receipt", type=Path, required=True)
    args = parser.parse_args()

    launch_root = args.launch_root.resolve()
    repo_root = args.repo_root.resolve()
    g0_receipt_path = args.g0_receipt.resolve()
    eval_root = (
        repo_root
        / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
    )
    sequence_root = eval_root / "sequence_models"
    evolution_root = sequence_root / "evolution"
    sys.path.insert(0, str(eval_root))

    from sequence_models.bibliography_evolution import (
        _make_input_row,
        _make_parent_inputs,
        _write_jsonl_exclusive,
    )
    from sequence_models.bibliography_evolution_contract import (
        canonical_json_bytes,
        enforce_leakage_barrier,
        expand_template,
        load_json,
        sha256_file,
        validate_candidate_spec,
        verify_finalized_receipt,
        write_json_exclusive,
    )

    if git(repo_root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise RuntimeError("deployed checkout is not the audited commit")
    if git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("deployed checkout is dirty")
    if sha256_file(g0_receipt_path) != EXPECTED_PARENT_RECEIPT_SHA:
        raise RuntimeError("G0 receipt bytes differ from the admitted parent")
    parent_verification = verify_finalized_receipt(g0_receipt_path)
    if parent_verification.get("candidate_id") != EXPECTED_PARENT_ID:
        raise RuntimeError("G0 finalized-receipt verification returned another parent")
    g0_receipt = load_json(g0_receipt_path)
    g0_spec = load_json(g0_receipt_path.parent / "spec.json")
    prediction = g0_receipt_path.parent / g0_receipt["predictions"]["main"]["path"]
    if sha256_file(prediction) != EXPECTED_PARENT_PREDICTION_SHA:
        raise RuntimeError("G0 parent prediction is not byte-identical baseline")
    if (
        g0_receipt.get("status") != "passed"
        or not g0_receipt.get("selection", {}).get("eligible_for_pareto")
        or not g0_receipt.get("selection", {}).get("acceptance", {}).get("passed")
    ):
        raise RuntimeError("G0 is not an accepted parent")

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("G1 packet preparation must run on a Slurm compute node")
    packet_root = launch_root / "packets" / f"g1-{EXPECTED_COMMIT[:8]}-prep-{job_id}"
    packet_root.mkdir(parents=True)

    def path_for_input(name: str) -> Path:
        return Path(str(g0_spec["input_receipts"][name]["path"])).resolve()

    input_receipts = {
        "validation_table": _make_input_row(
            path_for_input("validation_table"),
            data_class="development_table",
            split="validation",
            document_scope="prediction_blind_extraction_qualified_268",
            contains_labels=True,
        ),
        "validation_signal_probability": _make_input_row(
            path_for_input("validation_signal_probability"),
            data_class="validation_signal_probability",
            split="validation",
            document_scope="retrospective_validation_274",
            contains_labels=False,
        ),
        "validation_line_probability": _make_input_row(
            path_for_input("validation_line_probability"),
            data_class="validation_line_probability",
            split="validation",
            document_scope="retrospective_validation_274",
            contains_labels=False,
        ),
        "validation_scope_mask": _make_input_row(
            path_for_input("validation_scope_mask"),
            data_class="validation_scope_mask",
            split="validation",
            document_scope="retrospective_validation_274",
            contains_labels=False,
        ),
        "qualified_inventory": _make_input_row(
            path_for_input("qualified_inventory"),
            data_class="qualified_development_inventory",
            split="development",
            document_scope="prediction_blind_extraction_qualified_268",
            contains_labels=False,
        ),
        "baseline_work": _make_input_row(
            path_for_input("baseline_work"),
            data_class="baseline_work_objectives",
            split="development",
            document_scope="prediction_blind_extraction_qualified_268",
            contains_labels=True,
        ),
        "code_tests": _make_input_row(
            path_for_input("code_tests"),
            data_class="code_test_receipt",
            split="development",
            document_scope="aggregate_no_rows",
            contains_labels=False,
        ),
    }
    input_receipts.update(
        _make_parent_inputs(g0_receipt_path, prefix="g0", receipt_only=True)
    )
    inputs_path = packet_root / "g1.inputs.json"
    write_json_exclusive(inputs_path, input_receipts)

    policy_path = evolution_root / "leakage.policy.json"
    policy = load_json(policy_path)
    policy_sha = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
    bindings = {
        "CODE_COMMIT": EXPECTED_COMMIT,
        "LEAKAGE_POLICY_SHA256": policy_sha,
        "G0_CANDIDATE_ID": EXPECTED_PARENT_ID,
        "VALIDATION_TABLE_DIR": str(path_for_input("validation_table")),
        "VALIDATION_SIGNAL_PROBABILITY": str(path_for_input("validation_signal_probability")),
        "VALIDATION_LINE_PROBABILITY": str(path_for_input("validation_line_probability")),
        "VALIDATION_SCOPE_MASK": str(path_for_input("validation_scope_mask")),
        "QUALIFIED_268_IDS": str(path_for_input("qualified_inventory")),
        "G1_INPUT_RECEIPTS": input_receipts,
    }
    bindings_path = packet_root / "bindings.g1.json"
    write_json_exclusive(bindings_path, bindings)

    templates_packet = load_json(evolution_root / "experiment_templates.json")
    queue_rows = []
    for template in templates_packet["templates"]:
        if (
            template["generation"] == "G1"
            and template["parameter_family"] == "anchor_threshold"
        ):
            queue_rows.extend(expand_template(template, bindings))
    family_counts = dict(collections.Counter(row["parameter_family"] for row in queue_rows))
    if len(queue_rows) != 5 or family_counts != EXPECTED_FAMILY_COUNTS:
        raise RuntimeError(f"unexpected G1 queue shape: {len(queue_rows)}, {family_counts}")
    candidate_ids = [str(row["candidate_id"]) for row in queue_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("G1 candidate IDs are not unique")
    for row in queue_rows:
        validate_candidate_spec(row)
        if row["parent_candidate_ids"] != [EXPECTED_PARENT_ID]:
            raise RuntimeError("G1 queue row has the wrong parent")
    queue_path = packet_root / "queue.g1.jsonl"
    _write_jsonl_exclusive(queue_path, queue_rows)

    leakage_rows = []
    for index, row in enumerate(queue_rows):
        result = enforce_leakage_barrier(row, policy)
        leakage_rows.append(
            {
                "queue_index": index,
                "candidate_id": row["candidate_id"],
                "parameter_family": row["parameter_family"],
                "status": result["status"],
                "checked_input_count": len(result["checked_inputs"]),
                "verified_parent_ids": [
                    parent["candidate_id"]
                    for parent in result["parent_lineage"]["parents"]
                ],
            }
        )
    leakage_path = packet_root / "leakage_preflight.g1.jsonl"
    _write_jsonl_exclusive(leakage_path, leakage_rows)

    queue_sha = sha256_file(queue_path)
    candidate_root = launch_root / "candidates"
    launcher = sequence_root / "clariden/run_bibliography_evolution_cpu.sbatch"
    intended_command = (
        "sbatch --array=0-4%5 "
        f"--export=ALL,CODE_ROOT={eval_root},QUEUE_JSONL={queue_path},"
        f"QUEUE_SHA256={queue_sha},LEAKAGE_POLICY={policy_path},"
        f"CANDIDATE_ROOT={candidate_root},WORKING_DIR={repo_root} {launcher}"
    )
    (packet_root / "intended_sbatch_command.txt").write_text(
        intended_command + "\n", encoding="utf-8"
    )
    write_json_exclusive(
        packet_root / "git_attestation.json",
        {
            "status": "passed",
            "head": git(repo_root, "rev-parse", "HEAD"),
            "clean": not bool(git(repo_root, "status", "--porcelain", "--untracked-files=all")),
            "repo_root": str(repo_root),
        },
    )
    artifact_rows = {}
    for path in sorted(packet_root.rglob("*")):
        if path.is_file() and path.name != "packet_receipt.json":
            artifact_rows[path.relative_to(packet_root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    receipt_path = packet_root / "packet_receipt.json"
    write_json_exclusive(
        receipt_path,
        {
            "schema_version": "bibliography-evolution-g1-launch-packet-v1",
            "status": "passed_preflight_not_submitted_parent_admitted",
            "code_commit": EXPECTED_COMMIT,
            "slurm_preparation_job_id": job_id,
            "packet_root": str(packet_root),
            "parent": {
                "candidate_id": EXPECTED_PARENT_ID,
                "receipt_path": str(g0_receipt_path),
                "receipt_sha256": EXPECTED_PARENT_RECEIPT_SHA,
                "prediction_sha256": EXPECTED_PARENT_PREDICTION_SHA,
                "finalized_verification": parent_verification,
                "independent_post_run_audit_verdict": "ADMIT_AS_G1_PARENT",
            },
            "queue": {
                "path": str(queue_path),
                "rows": len(queue_rows),
                "sha256": queue_sha,
                "family_counts": family_counts,
                "candidate_ids": candidate_ids,
            },
            "policy_sha256": policy_sha,
            "bindings_sha256": sha256_file(bindings_path),
            "inputs_sha256": sha256_file(inputs_path),
            "leakage_preflight_sha256": sha256_file(leakage_path),
            "intended_sbatch_command_not_executed": intended_command,
            "g1_submitted": False,
            "artifacts": artifact_rows,
        },
    )
    print(
        json.dumps(
            {
                "status": "passed_preflight_not_submitted_parent_admitted",
                "packet_root": str(packet_root),
                "packet_receipt": str(receipt_path),
                "queue_rows": len(queue_rows),
                "queue_sha256": queue_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
