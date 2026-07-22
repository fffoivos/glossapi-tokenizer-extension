#!/usr/bin/env python3
"""Materialize and fail-closed preflight the exact bibliography G0 packet."""

from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


EXPECTED_COMMIT = "931a56d119b9ab44e79c23fa82a16cd2edf0c4b7"
EXPECTED_SOURCES = {"greek_phd": 92, "kallipos": 92, "openarchives": 84}
ACCEPTED_TEST_SUMMARIES = ("36 passed", "35 passed, 1 skipped")


def run_checked(command: list[str], *, cwd: Path, stdout: Path, stderr: Path) -> None:
    with stdout.open("x", encoding="utf-8") as out, stderr.open("x", encoding="utf-8") as err:
        completed = subprocess.run(command, cwd=cwd, stdout=out, stderr=err, text=True)
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {command!r}")


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
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()

    launch_root = args.launch_root.resolve()
    repo_root = args.repo_root.resolve()
    baseline_root = args.baseline_root.resolve()
    eval_root = (
        repo_root
        / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
    )
    sequence_root = eval_root / "sequence_models"
    evolution_root = sequence_root / "evolution"

    if git(repo_root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise RuntimeError("deployed checkout is not the audited commit")
    if git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("deployed checkout is dirty before preparation")
    if not baseline_root.is_dir() or baseline_root.is_symlink():
        raise RuntimeError("baseline root is missing or a symlink")

    sys.path.insert(0, str(eval_root))
    import numpy
    import sklearn
    import torch

    from sequence_models.bibliography_evolution import (
        _make_input_row,
        _write_jsonl_exclusive,
    )
    from sequence_models.bibliography_evolution_contract import (
        CandidateStore,
        canonical_json_bytes,
        expand_template,
        load_json,
        sha256_file,
        validate_candidate_spec,
        verify_g0,
        write_json_exclusive,
    )
    from sequence_models.bibliography_evolution_metrics import (
        parse_args as metrics_args,
        run as run_metrics,
    )

    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("this preparation must run on a Slurm compute node")
    packet_root = launch_root / "packets" / f"g0-{EXPECTED_COMMIT[:8]}-prep-{job_id}"
    packet_root.mkdir(parents=True)

    compile_command = [sys.executable, "-m", "compileall", "-q", "sequence_models"]
    run_checked(
        compile_command,
        cwd=eval_root,
        stdout=packet_root / "compile.stdout.log",
        stderr=packet_root / "compile.stderr.log",
    )
    test_paths = [
        "sequence_models/tests/test_bibliography_entry_blocks.py",
        "sequence_models/tests/test_bibliography_signal_barrier_decode.py",
        "sequence_models/tests/test_bibliography_signal_validation.py",
        "sequence_models/tests/test_bibliography_signal_tcn.py",
        "sequence_models/tests/test_bibliography_evolution_contract.py",
        "sequence_models/tests/test_bibliography_evolution_headers.py",
    ]
    test_command = [sys.executable, "-m", "pytest", "-q", *test_paths]
    test_stdout = packet_root / "pytest.stdout.log"
    run_checked(
        test_command,
        cwd=eval_root,
        stdout=test_stdout,
        stderr=packet_root / "pytest.stderr.log",
    )
    test_output = test_stdout.read_text(encoding="utf-8")
    observed_test_summary = next(
        (summary for summary in ACCEPTED_TEST_SUMMARIES if summary in test_output),
        None,
    )
    if observed_test_summary is None:
        raise RuntimeError("the complete audited test suite did not pass")
    if git(repo_root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise RuntimeError("checkout commit drifted during tests")
    if git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("checkout became dirty during tests")

    baseline_lock_path = evolution_root / "baseline.lock.json"
    leakage_policy_path = evolution_root / "leakage.policy.json"
    templates_path = evolution_root / "experiment_templates.json"
    quality_path = sequence_root / "bibliography_validation_quality_decisions_20260714.json"
    baseline_lock = load_json(baseline_lock_path)
    if baseline_lock["authoritative_root"] != str(baseline_root):
        raise RuntimeError("baseline lock points to a different authoritative root")
    if sha256_file(quality_path) != baseline_lock["input_hashes"]["validation_quality_decisions"]:
        raise RuntimeError("validation quality decisions do not match the baseline lock")

    table_dir = baseline_root / "validation_r3/validation_table"
    documents_path = table_dir / "documents.jsonl"
    documents = [json.loads(line) for line in documents_path.read_text(encoding="utf-8").splitlines() if line]
    decisions = load_json(quality_path)
    excluded = set(decisions["prediction_blind_excluded_document_ids"])
    if len(excluded) != 6:
        raise RuntimeError("prediction-blind exclusion inventory is not exactly six")
    table_ids = [str(row["document_id"]) for row in documents]
    if len(table_ids) != 274 or len(set(table_ids)) != 274 or not excluded.issubset(table_ids):
        raise RuntimeError("validation table identity inventory is inconsistent")
    qualified_rows = [row for row in documents if row["document_id"] not in excluded]
    source_counts = dict(collections.Counter(str(row["source"]) for row in qualified_rows))
    if len(qualified_rows) != 268 or source_counts != EXPECTED_SOURCES:
        raise RuntimeError(f"qualified inventory mismatch: {len(qualified_rows)}, {source_counts}")
    qualified_path = packet_root / "qualified_268.json"
    write_json_exclusive(
        qualified_path,
        {
            "schema_version": "bibliography-evolution-qualified-development-inventory-v1",
            "derivation": "validation table order minus prediction-blind quality exclusions only",
            "validation_document_count": 274,
            "qualified_document_count": 268,
            "source_document_counts": source_counts,
            "document_ids": [str(row["document_id"]) for row in qualified_rows],
            "excluded_document_ids": sorted(excluded),
            "quality_decisions_sha256": sha256_file(quality_path),
            "validation_documents_sha256": sha256_file(documents_path),
        },
    )

    baseline_prediction = baseline_root / baseline_lock["g0_replay"]["prediction_relative_path"]
    if sha256_file(baseline_prediction) != baseline_lock["g0_replay"]["prediction_sha256"]:
        raise RuntimeError("locked baseline prediction changed")
    baseline_rows_path = packet_root / "baseline_work_objectives.jsonl"
    baseline_metrics_path = packet_root / "baseline_work_metrics.json"
    baseline_metrics = run_metrics(
        metrics_args(
            [
                "--table-dir", str(table_dir),
                "--prediction", str(baseline_prediction),
                "--qualified-documents", str(qualified_path),
                "--output-rows", str(baseline_rows_path),
                "--output-report", str(baseline_metrics_path),
            ]
        )
    )
    expected_metrics = baseline_lock["headline_metrics_268"]
    actual_metrics = baseline_metrics["metrics"]
    for name, expected in expected_metrics.items():
        actual = actual_metrics.get(name)
        if isinstance(expected, float):
            if actual is None or abs(float(actual) - expected) > 1e-12:
                raise RuntimeError(f"baseline metric mismatch for {name}: {actual} != {expected}")
        elif actual != expected:
            raise RuntimeError(f"baseline metric mismatch for {name}: {actual} != {expected}")

    code_test_receipt_path = packet_root / "code_test_receipt.json"
    write_json_exclusive(
        code_test_receipt_path,
        {
            "schema_version": "bibliography-evolution-code-test-receipt-v1",
            "status": "passed",
            "code_commit": EXPECTED_COMMIT,
            "slurm_job_id": job_id,
            "commands": {
                "compile": compile_command,
                "pytest": test_command,
            },
            "pytest_summary": observed_test_summary,
            "accepted_environment_dependent_summaries": list(ACCEPTED_TEST_SUMMARIES),
            "pytest_stdout_sha256": sha256_file(test_stdout),
            "environment": {
                "python": platform.python_version(),
                "numpy": numpy.__version__,
                "sklearn": sklearn.__version__,
                "torch": torch.__version__,
            },
            "invariants": {
                "physical_gap_walls": True,
                "header_roles_non_seed": True,
            },
        },
    )

    validation_signal = baseline_root / "signal_validation_r4/validation_signal_ensemble_probability.npy"
    validation_line = baseline_root / "validation_r3/validation_line_probability.npy"
    validation_scope = baseline_root / "signal_validation_r4/validation_auxiliary_scope.npy"
    input_receipts = {
        "baseline_lock": _make_input_row(
            baseline_lock_path,
            data_class="baseline_lock",
            split="development",
            document_scope="aggregate_no_rows",
            contains_labels=False,
        ),
        "baseline_root": _make_input_row(
            baseline_root,
            data_class="baseline_authoritative_root",
            split="development",
            document_scope="aggregate_no_rows",
            contains_labels=True,
        ),
        "validation_table": _make_input_row(
            table_dir,
            data_class="development_table",
            split="validation",
            document_scope="prediction_blind_extraction_qualified_268",
            contains_labels=True,
        ),
        "validation_signal_probability": _make_input_row(
            validation_signal,
            data_class="validation_signal_probability",
            split="validation",
            document_scope="retrospective_validation_274",
            contains_labels=False,
        ),
        "validation_line_probability": _make_input_row(
            validation_line,
            data_class="validation_line_probability",
            split="validation",
            document_scope="retrospective_validation_274",
            contains_labels=False,
        ),
        "validation_scope_mask": _make_input_row(
            validation_scope,
            data_class="validation_scope_mask",
            split="validation",
            document_scope="retrospective_validation_274",
            contains_labels=False,
        ),
        "qualified_inventory": _make_input_row(
            qualified_path,
            data_class="qualified_development_inventory",
            split="development",
            document_scope="prediction_blind_extraction_qualified_268",
            contains_labels=False,
        ),
        "quality_decisions": _make_input_row(
            quality_path,
            data_class="validation_quality_decisions",
            split="development",
            document_scope="retrospective_validation_274",
            contains_labels=False,
        ),
        "baseline_prediction": _make_input_row(
            baseline_prediction,
            data_class="baseline_prediction_for_objectives",
            split="development",
            document_scope="prediction_blind_extraction_qualified_268",
            contains_labels=False,
        ),
        "baseline_work": _make_input_row(
            baseline_rows_path,
            data_class="baseline_work_objectives",
            split="development",
            document_scope="prediction_blind_extraction_qualified_268",
            contains_labels=True,
        ),
        "code_tests": _make_input_row(
            code_test_receipt_path,
            data_class="code_test_receipt",
            split="development",
            document_scope="aggregate_no_rows",
            contains_labels=False,
        ),
    }
    inputs_path = packet_root / "g0.inputs.json"
    write_json_exclusive(inputs_path, input_receipts)

    policy = load_json(leakage_policy_path)
    policy_sha = __import__("hashlib").sha256(canonical_json_bytes(policy)).hexdigest()
    bindings = {
        "CODE_COMMIT": EXPECTED_COMMIT,
        "LEAKAGE_POLICY_SHA256": policy_sha,
        "BASELINE_LOCK": str(baseline_lock_path.resolve()),
        "BASELINE_ROOT": str(baseline_root),
        "VALIDATION_TABLE_DIR": str(table_dir),
        "VALIDATION_SIGNAL_PROBABILITY": str(validation_signal),
        "VALIDATION_LINE_PROBABILITY": str(validation_line),
        "VALIDATION_SCOPE_MASK": str(validation_scope),
        "QUALIFIED_268_IDS": str(qualified_path),
        "G0_INPUT_RECEIPTS": input_receipts,
    }
    bindings_path = packet_root / "bindings.g0.json"
    write_json_exclusive(bindings_path, bindings)
    template_packet = load_json(templates_path)
    templates = [row for row in template_packet["templates"] if row["generation"] == "G0"]
    queue_rows = []
    for template in templates:
        queue_rows.extend(expand_template(template, bindings))
    if len(queue_rows) != 1:
        raise RuntimeError(f"G0 rendered {len(queue_rows)} rows instead of one")
    spec = queue_rows[0]
    validate_candidate_spec(spec)
    queue_path = packet_root / "queue.g0.jsonl"
    _write_jsonl_exclusive(queue_path, queue_rows)

    g0_preflight_path = packet_root / "g0_lock_preflight.json"
    write_json_exclusive(
        g0_preflight_path,
        verify_g0(baseline_lock, root=baseline_root, replay_prediction=baseline_prediction),
    )
    dry_store = packet_root / "dry_run_candidate_store"
    dry_candidate = CandidateStore(dry_store).create(spec, policy)
    leakage_preflight = load_json(dry_candidate / "leakage.json")
    if leakage_preflight.get("status") != "passed":
        raise RuntimeError("leakage preflight did not pass")

    queue_sha = sha256_file(queue_path)
    launcher = sequence_root / "clariden/run_bibliography_evolution_cpu.sbatch"
    candidate_root = launch_root / "candidates"
    intended_command = (
        "sbatch --array=0-0 "
        f"--export=ALL,CODE_ROOT={eval_root},QUEUE_JSONL={queue_path},"
        f"QUEUE_SHA256={queue_sha},LEAKAGE_POLICY={leakage_policy_path},"
        f"CANDIDATE_ROOT={candidate_root},WORKING_DIR={repo_root} {launcher}"
    )
    (packet_root / "intended_sbatch_command.txt").write_text(intended_command + "\n", encoding="utf-8")
    git_attestation_path = packet_root / "git_attestation.json"
    write_json_exclusive(
        git_attestation_path,
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
            "schema_version": "bibliography-evolution-g0-launch-packet-v1",
            "status": "passed_preflight_not_submitted",
            "code_commit": EXPECTED_COMMIT,
            "slurm_preparation_job_id": job_id,
            "packet_root": str(packet_root),
            "candidate_id": spec["candidate_id"],
            "queue": {"path": str(queue_path), "rows": 1, "sha256": queue_sha},
            "qualified_inventory": {
                "path": str(qualified_path),
                "sha256": sha256_file(qualified_path),
                "document_count": 268,
                "source_document_counts": source_counts,
            },
            "baseline_work": {
                "path": str(baseline_rows_path),
                "sha256": sha256_file(baseline_rows_path),
                "work_count": baseline_metrics["work_count"],
            },
            "code_test_receipt": {
                "path": str(code_test_receipt_path),
                "sha256": sha256_file(code_test_receipt_path),
                "summary": observed_test_summary,
            },
            "policy_sha256": policy_sha,
            "bindings_sha256": sha256_file(bindings_path),
            "inputs_sha256": sha256_file(inputs_path),
            "intended_sbatch_command": intended_command,
            "artifacts": artifact_rows,
        },
    )
    print(
        json.dumps(
            {
                "status": "passed_preflight_not_submitted",
                "packet_root": str(packet_root),
                "packet_receipt": str(receipt_path),
                "candidate_id": spec["candidate_id"],
                "queue_sha256": queue_sha,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
