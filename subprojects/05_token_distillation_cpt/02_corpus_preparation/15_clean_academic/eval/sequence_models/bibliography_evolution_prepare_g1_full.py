#!/usr/bin/env python3
"""Materialize, but never submit, the complete five-family G1 queue."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_PARENT_ID = "g0-34a7dde38689d1b15e8afd42"
EXPECTED_PARENT_RECEIPT_SHA = "95eb0f5a855f780c606d87844fa680e889dc5b70b7eae752b051fa3c48902955"
EXPECTED_PARENT_PREDICTION_SHA = "84179ec0d45dbbf5c5ad1c78fd20b7ed2b82ead70c7287180bf3ed024e9a6d5a"
EXPECTED_FAMILY_COUNTS = {
    "anchor_threshold": 5,
    "anchor_density": 8,
    "maximum_bridge_gap": 4,
    "inside_probability": 5,
    "adjacent_expansion": 5,
}
EXPECTED_TOTAL_ROWS = 27
EXPECTED_PILOT_PACKET_SHA = "dd1025d33315201e6eb5be840779ff4ed9fde2ef786c401411fadbefa2d6e13e"
EXPECTED_PILOT_QUEUE_SHA = "3896242cb960a4f9a9dd234d3417c4b49eec7666f035c53294087c2e8253f501"
EXPECTED_PILOT_LAUNCH_SHA = "26fb3a8de94bf4601ca96550d08013b33ae8dc189b0dbcc82cd1fb700e969382"


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def select_full_g1_templates(
    templates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Return the five predeclared G1 families in file order, fail closed."""

    selected = [row for row in templates if row.get("generation") == "G1"]
    counts: dict[str, int] = {}
    for template in selected:
        family = str(template.get("parameter_family"))
        grid = template.get("sweep_grid")
        if not isinstance(grid, Mapping) or not grid:
            raise ValueError(f"G1 family has no sweep grid: {family}")
        count = 1
        for values in grid.values():
            if not isinstance(values, list) or not values:
                raise ValueError(f"G1 family has an empty sweep dimension: {family}")
            count *= len(values)
        counts[family] = counts.get(family, 0) + count
    if counts != EXPECTED_FAMILY_COUNTS or sum(counts.values()) != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"predeclared G1 template inventory changed: {counts}")
    return selected


def candidate_store_snapshot(candidate_root: Path) -> dict[str, Any]:
    """Prove that packet preparation has not executed a G2 candidate."""

    candidates = []
    for path in sorted(candidate_root.iterdir() if candidate_root.is_dir() else ()):
        if not path.is_dir() or path.is_symlink():
            continue
        spec_path = path / "spec.json"
        execution_path = path / "execution.json"
        if not spec_path.is_file() or not execution_path.is_file():
            raise RuntimeError(f"candidate directory is incomplete: {path}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        candidates.append(
            {
                "candidate_id": path.name,
                "generation": spec.get("generation"),
                "parameter_family": spec.get("parameter_family"),
                "slurm_job_id": execution.get("slurm_job_id"),
                "slurm_array_task_id": execution.get("slurm_array_task_id"),
            }
        )
    generations = dict(collections.Counter(str(row["generation"]) for row in candidates))
    g2 = [row for row in candidates if row["generation"] == "G2"]
    if len(candidates) != 6 or generations != {"G0": 1, "G1": 5} or g2:
        raise RuntimeError(
            f"candidate store is not the G0 plus five-row G1 pilot: {generations}"
        )
    return {
        "schema_version": "bibliography-evolution-preparation-candidate-store-snapshot-v1",
        "status": "passed_no_g2_candidates_or_executions",
        "candidate_count": len(candidates),
        "generation_counts": generations,
        "g2_candidate_count": 0,
        "g2_execution_count": 0,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--g0-receipt", type=Path, required=True)
    parser.add_argument("--pilot-packet-receipt", type=Path, required=True)
    parser.add_argument("--pilot-queue", type=Path, required=True)
    parser.add_argument("--pilot-launch-receipt", type=Path, required=True)
    args = parser.parse_args()

    launch_root = args.launch_root.resolve()
    repo_root = args.repo_root.resolve()
    code_commit = str(args.code_commit)
    g0_receipt_path = args.g0_receipt.resolve()
    pilot_packet_receipt_path = args.pilot_packet_receipt.resolve()
    pilot_queue_path = args.pilot_queue.resolve()
    pilot_launch_receipt_path = args.pilot_launch_receipt.resolve()
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
    from sequence_models.bibliography_evolution_prepare_g2 import (
        materialize_code_test_attestation,
    )

    if git(repo_root, "rev-parse", "HEAD") != code_commit:
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

    expected_pilot_hashes = {
        pilot_packet_receipt_path: EXPECTED_PILOT_PACKET_SHA,
        pilot_queue_path: EXPECTED_PILOT_QUEUE_SHA,
        pilot_launch_receipt_path: EXPECTED_PILOT_LAUNCH_SHA,
    }
    if any(not path.is_file() or path.is_symlink() for path in expected_pilot_hashes):
        raise RuntimeError("a G1 pilot artifact is missing or a symlink")
    for path, expected_sha in expected_pilot_hashes.items():
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"G1 pilot artifact drift: {path}")
    pilot_packet = load_json(pilot_packet_receipt_path)
    pilot_launch = load_json(pilot_launch_receipt_path)
    pilot_rows = [
        json.loads(raw)
        for raw in pilot_queue_path.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    if (
        len(pilot_rows) != 5
        or {row.get("parameter_family") for row in pilot_rows} != {"anchor_threshold"}
        or any(row.get("parent_candidate_ids") != [EXPECTED_PARENT_ID] for row in pilot_rows)
        or pilot_packet.get("queue", {}).get("sha256") != EXPECTED_PILOT_QUEUE_SHA
        or pilot_launch.get("queue", {}).get("sha256") != EXPECTED_PILOT_QUEUE_SHA
        or pilot_launch.get("slurm_array_job_id") != "2790040"
        or pilot_launch.get("other_g1_template_families_submitted") is not False
    ):
        raise RuntimeError("prior G1 anchor run is not the exact five-row pilot")

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("G1 packet preparation must run on a Slurm compute node")
    code_test_receipt = materialize_code_test_attestation(
        launch_root=launch_root,
        repo_root=repo_root,
        eval_root=eval_root,
        code_commit=code_commit,
        job_id=job_id,
        lane="g1-full",
    )
    candidate_root = launch_root / "candidates"
    initial_store_snapshot = candidate_store_snapshot(candidate_root)
    packet_name = f"g1-full-{code_commit[:8]}-prep-{job_id}"
    final_packet_root = launch_root / "packets" / packet_name
    if final_packet_root.exists() or final_packet_root.is_symlink():
        raise FileExistsError(final_packet_root)
    final_packet_root.parent.mkdir(parents=True, exist_ok=True)
    packet_root = Path(
        tempfile.mkdtemp(prefix=f".{packet_name}.partial-", dir=final_packet_root.parent)
    )

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
            code_test_receipt,
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
        "CODE_COMMIT": code_commit,
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
    for template in select_full_g1_templates(templates_packet["templates"]):
        queue_rows.extend(expand_template(template, bindings))
    family_counts = dict(collections.Counter(row["parameter_family"] for row in queue_rows))
    if len(queue_rows) != EXPECTED_TOTAL_ROWS or family_counts != EXPECTED_FAMILY_COUNTS:
        raise RuntimeError(f"unexpected G1 queue shape: {len(queue_rows)}, {family_counts}")
    candidate_ids = [str(row["candidate_id"]) for row in queue_rows]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("G1 candidate IDs are not unique")
    for row in queue_rows:
        validate_candidate_spec(row)
        if row["parent_candidate_ids"] != [EXPECTED_PARENT_ID]:
            raise RuntimeError("G1 queue row has the wrong parent")
    queue_path = packet_root / "queue.g1.full.jsonl"
    _write_jsonl_exclusive(queue_path, queue_rows)

    leakage_rows = []
    for index, row in enumerate(queue_rows):
        result = enforce_leakage_barrier(row, policy)
        verified_parent_ids = [
            parent["candidate_id"]
            for parent in result["parent_lineage"]["parents"]
        ]
        if result.get("status") != "passed" or verified_parent_ids != [EXPECTED_PARENT_ID]:
            raise RuntimeError(f"G1 leakage/parent preflight failed at row {index}")
        leakage_rows.append(
            {
                "queue_index": index,
                "candidate_id": row["candidate_id"],
                "parameter_family": row["parameter_family"],
                "status": result["status"],
                "checked_input_count": len(result["checked_inputs"]),
                "verified_parent_ids": verified_parent_ids,
            }
        )
    leakage_path = packet_root / "leakage_preflight.g1.jsonl"
    _write_jsonl_exclusive(leakage_path, leakage_rows)

    queue_sha = sha256_file(queue_path)
    launcher = sequence_root / "clariden/run_bibliography_evolution_cpu.sbatch"
    intended_command = (
        "sbatch --array=0-26%8 "
        f"--export=ALL,CODE_ROOT={eval_root},QUEUE_JSONL={final_packet_root / queue_path.name},"
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
    write_json_exclusive(
        packet_root / "prior_anchor_pilot.json",
        {
            "schema_version": "bibliography-evolution-g1-anchor-pilot-lineage-v1",
            "status": "recorded_as_incomplete_five_row_pilot_not_full_g1",
            "pilot_scope": {"anchor_threshold": 5},
            "missing_from_pilot": {
                family: count
                for family, count in EXPECTED_FAMILY_COUNTS.items()
                if family != "anchor_threshold"
            },
            "pilot_packet_receipt": {
                "path": str(pilot_packet_receipt_path),
                "sha256": EXPECTED_PILOT_PACKET_SHA,
            },
            "pilot_queue": {
                "path": str(pilot_queue_path),
                "sha256": EXPECTED_PILOT_QUEUE_SHA,
                "rows": 5,
            },
            "pilot_launch_receipt": {
                "path": str(pilot_launch_receipt_path),
                "sha256": EXPECTED_PILOT_LAUNCH_SHA,
                "slurm_array_job_id": "2790040",
            },
            "full_packet_replays_anchor_family_on_one_new_exact_commit": True,
            "pilot_candidates_reused_in_full_queue": False,
        },
    )
    snapshot_path = packet_root / "candidate_store_snapshot.json"
    write_json_exclusive(snapshot_path, initial_store_snapshot)
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
            "schema_version": "bibliography-evolution-g1-full-launch-packet-v1",
            "status": "passed_full_27_row_preflight_not_submitted_parent_admitted",
            "code_commit": code_commit,
            "slurm_preparation_job_id": job_id,
            "packet_root": str(final_packet_root),
            "parent": {
                "candidate_id": EXPECTED_PARENT_ID,
                "receipt_path": str(g0_receipt_path),
                "receipt_sha256": EXPECTED_PARENT_RECEIPT_SHA,
                "prediction_sha256": EXPECTED_PARENT_PREDICTION_SHA,
                "finalized_verification": parent_verification,
                "independent_post_run_audit_verdict": "ADMIT_AS_G1_PARENT",
            },
            "queue": {
                "path": str(final_packet_root / queue_path.name),
                "rows": len(queue_rows),
                "sha256": queue_sha,
                "family_counts": family_counts,
                "candidate_ids": candidate_ids,
            },
            "policy_sha256": policy_sha,
            "bindings_sha256": sha256_file(bindings_path),
            "inputs_sha256": sha256_file(inputs_path),
            "leakage_preflight_sha256": sha256_file(leakage_path),
            "fresh_code_test_receipt": {
                "path": str(code_test_receipt),
                "sha256": sha256_file(code_test_receipt),
            },
            "prior_anchor_run_classification": "incomplete_five_row_pilot",
            "candidate_store_snapshot": {
                "path": str(final_packet_root / snapshot_path.name),
                "sha256": sha256_file(snapshot_path),
                "g2_candidate_count": 0,
                "g2_execution_count": 0,
            },
            "intended_sbatch_command_not_executed": intended_command,
            "g1_submitted": False,
            "g2_submitted": False,
            "artifacts": artifact_rows,
        },
    )
    if candidate_store_snapshot(candidate_root) != initial_store_snapshot:
        raise RuntimeError("candidate store changed during packet preparation")
    os.replace(packet_root, final_packet_root)
    for path in final_packet_root.rglob("*"):
        if path.is_file():
            path.chmod(0o440)
    final_packet_root.chmod(0o550)
    receipt_path = final_packet_root / "packet_receipt.json"
    print(
        json.dumps(
            {
                "status": "passed_full_27_row_preflight_not_submitted_parent_admitted",
                "packet_root": str(final_packet_root),
                "packet_receipt": str(receipt_path),
                "packet_receipt_sha256": sha256_file(receipt_path),
                "queue_rows": len(queue_rows),
                "queue_sha256": queue_sha,
                "family_counts": family_counts,
                "g1_submitted": False,
                "g2_candidate_count": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
