#!/usr/bin/env python3
"""Materialize, but never submit, the exact deterministic-header G2 queue."""

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
from typing import Any

import numpy as np


EXPECTED_COMMIT = "931a56d119b9ab44e79c23fa82a16cd2edf0c4b7"
EXPECTED_PARENT_ID = "g1-e47e74768f652e91f3cd7f73"
EXPECTED_PARENT_RECEIPT_SHA = "7a1cebdc322f75e76e5d7ca9271dc9cece94d9dc5521fe0ffa81f4df9ac4453a"
EXPECTED_PARENT_PREDICTION_SHA = "58c4f0a4108f1c7c461782c81274363bb29e83fea2b9151dcd7751aecd6da684"
EXPECTED_PARENT_BARRIER_SHA = "540ad1326cc282aed13f4c13458d002bc13288fba3d765cbb78ff48fbb2c09b1"
EXPECTED_REGISTRY_SHA = "b47274355800fae4a8b8fdcfaaf2dce11fb5104ba8c7ea1b5ad3c7b6b7949ea8"
EXPECTED_REGISTRY_RECEIPT_SHA = "dcf3ae403bf231589b2a8239a9259275224cfdf11754fc81fca8a286d7b8991c"
EXPECTED_RECOMMENDATION_SHA = "6d4eed0ac5172474d6fab3d807ad24e63bf23894161ad42aaa093e41bcb4f473"
EXPECTED_WINDOWS = [1, 2, 3, 4]
EXPECTED_ROLE_COUNTS = {
    "NONE": 258672,
    "BIB_HEADER": 144,
    "BIB_SUBHEADER": 55,
    "NON_BIB_HEADER": 196,
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--header-role-dir", type=Path, required=True)
    parser.add_argument("--development-registry", type=Path, required=True)
    parser.add_argument("--registry-receipt", type=Path, required=True)
    parser.add_argument("--parent-recommendation", type=Path, required=True)
    args = parser.parse_args()

    launch_root = args.launch_root.resolve()
    repo_root = args.repo_root.resolve()
    parent_receipt_path = args.parent_receipt.resolve()
    role_root = args.header_role_dir.resolve()
    registry_path = args.development_registry.resolve()
    registry_receipt_path = args.registry_receipt.resolve()
    recommendation_path = args.parent_recommendation.resolve()
    role_path = role_root / "deterministic_header_roles.npy"
    role_receipt_path = role_root / "receipt.json"
    role_policy_path = role_root / "generation_policy.json"
    role_generator_path = role_root / "generate_deterministic_header_roles.py"
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
        verify_parent_lineage,
        write_json_exclusive,
    )

    if git(repo_root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise RuntimeError("deployed checkout is not the audited commit")
    if git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("deployed checkout is dirty")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("G2 packet preparation must run on a Slurm compute node")
    required_files = (
        parent_receipt_path,
        role_path,
        role_receipt_path,
        role_policy_path,
        role_generator_path,
        registry_path,
        registry_receipt_path,
        recommendation_path,
    )
    if any(not path.is_file() or path.is_symlink() for path in required_files):
        raise RuntimeError("a G2 input is missing or a symlink")
    expected_hashes = {
        parent_receipt_path: EXPECTED_PARENT_RECEIPT_SHA,
        registry_path: EXPECTED_REGISTRY_SHA,
        registry_receipt_path: EXPECTED_REGISTRY_RECEIPT_SHA,
        recommendation_path: EXPECTED_RECOMMENDATION_SHA,
    }
    for path, expected_sha in expected_hashes.items():
        if sha256_file(path) != expected_sha:
            raise RuntimeError(f"G2 admitted input drift: {path}")

    parent_verification = verify_finalized_receipt(parent_receipt_path)
    if parent_verification.get("candidate_id") != EXPECTED_PARENT_ID:
        raise RuntimeError("finalized parent verification returned another candidate")
    parent = load_json(parent_receipt_path)
    parent_spec = load_json(parent_receipt_path.parent / "spec.json")
    if (
        parent.get("status") != "passed"
        or not parent.get("selection", {}).get("eligible_for_pareto")
        or not parent.get("selection", {}).get("acceptance", {}).get("passed")
    ):
        raise RuntimeError("G1 parent is not accepted")
    if parent_spec.get("sweep_point") != {"anchor_probability": 0.3}:
        raise RuntimeError("G2 parent is not the exact G1 0.30 control")
    parent_inputs = _make_parent_inputs(
        parent_receipt_path, prefix="g1", receipt_only=False
    )
    if (
        parent_inputs["g1_prediction"]["sha256"] != EXPECTED_PARENT_PREDICTION_SHA
        or parent_inputs["g1_barriers"]["sha256"] != EXPECTED_PARENT_BARRIER_SHA
    ):
        raise RuntimeError("G1 parent-owned inference artifact drift")

    registry = load_json(registry_path)
    registry_receipt = load_json(registry_receipt_path)
    recommendation = load_json(recommendation_path)
    if (
        registry.get("candidate_count") != 6
        or recommendation.get("recommended_parent_candidate_id") != EXPECTED_PARENT_ID
        or registry_receipt.get("g2_parent_recommendation", {}).get("candidate_id")
        != EXPECTED_PARENT_ID
        or recommendation.get("sealed_data_opened") is not False
        or registry_receipt.get("sealed_data_opened") is not False
    ):
        raise RuntimeError("frozen G1 registry does not admit the requested G2 parent")

    role_receipt = load_json(role_receipt_path)
    role_array = np.load(role_path, allow_pickle=False)
    if (
        role_receipt.get("schema_version")
        != "bibliography-deterministic-header-role-lineage-v1"
        or role_receipt.get("status")
        != "passed_text_only_generation_byte_identical_reference_gates"
        or role_receipt.get("code_commit") != EXPECTED_COMMIT
        or role_receipt.get("role_counts") != EXPECTED_ROLE_COUNTS
        or role_receipt.get("inference_contract", {}).get("uses_labels") is not False
        or role_receipt.get("inference_contract", {}).get(
            "reproducible_on_unlabeled_sealed_documents"
        )
        is not True
        or role_array.shape != (259_067,)
        or role_array.dtype != np.uint8
        or set(np.unique(role_array).tolist()) != {0, 1, 2, 3}
        or role_receipt.get("outputs", {})
        .get("deterministic_header_roles.npy", {})
        .get("sha256")
        != sha256_file(role_path)
    ):
        raise RuntimeError("deterministic header-role lineage gate failed")

    packet_name = f"g2-deterministic-{EXPECTED_COMMIT[:8]}-prep-{job_id}"
    packet_root = launch_root / "packets" / packet_name
    if packet_root.exists() or packet_root.is_symlink():
        raise FileExistsError(packet_root)
    packet_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{packet_name}.partial-", dir=packet_root.parent)
    )
    try:
        def parent_input_path(name: str) -> Path:
            return Path(str(parent_spec["input_receipts"][name]["path"])).resolve()

        input_receipts = {
            "validation_table": _make_input_row(
                parent_input_path("validation_table"),
                data_class="development_table",
                split="validation",
                document_scope="prediction_blind_extraction_qualified_268",
                contains_labels=True,
            ),
            "validation_signal_probability": _make_input_row(
                parent_input_path("validation_signal_probability"),
                data_class="validation_signal_probability",
                split="validation",
                document_scope="retrospective_validation_274",
                contains_labels=False,
            ),
            "validation_scope_mask": _make_input_row(
                parent_input_path("validation_scope_mask"),
                data_class="validation_scope_mask",
                split="validation",
                document_scope="retrospective_validation_274",
                contains_labels=False,
            ),
            "qualified_inventory": _make_input_row(
                parent_input_path("qualified_inventory"),
                data_class="qualified_development_inventory",
                split="development",
                document_scope="prediction_blind_extraction_qualified_268",
                contains_labels=False,
            ),
            "baseline_work": _make_input_row(
                parent_input_path("baseline_work"),
                data_class="baseline_work_objectives",
                split="development",
                document_scope="prediction_blind_extraction_qualified_268",
                contains_labels=True,
            ),
            "code_tests": _make_input_row(
                parent_input_path("code_tests"),
                data_class="code_test_receipt",
                split="development",
                document_scope="aggregate_no_rows",
                contains_labels=False,
            ),
            "deterministic_header_roles": _make_input_row(
                role_path,
                data_class="deterministic_header_role_ids",
                split="validation",
                document_scope="retrospective_validation_274",
                contains_labels=False,
            ),
            "deterministic_header_role_receipt": _make_input_row(
                role_receipt_path,
                data_class="deterministic_header_role_generation_receipt",
                split="development",
                document_scope="aggregate_no_rows",
                contains_labels=False,
            ),
            "deterministic_header_role_policy": _make_input_row(
                role_policy_path,
                data_class="deterministic_header_role_generation_policy",
                split="development",
                document_scope="aggregate_no_rows",
                contains_labels=False,
            ),
            "deterministic_header_role_generator": _make_input_row(
                role_generator_path,
                data_class="deterministic_header_role_generator_code",
                split="development",
                document_scope="aggregate_no_rows",
                contains_labels=False,
            ),
            "g1_development_registry": _make_input_row(
                registry_path,
                data_class="development_selection_registry",
                split="development",
                document_scope="aggregate_no_rows",
                contains_labels=True,
            ),
            "g2_parent_recommendation": _make_input_row(
                recommendation_path,
                data_class="development_parent_recommendation",
                split="development",
                document_scope="aggregate_no_rows",
                contains_labels=True,
            ),
        }
        input_receipts.update(parent_inputs)
        inputs_path = partial / "g2.inputs.json"
        write_json_exclusive(inputs_path, input_receipts)

        policy_path = evolution_root / "leakage.policy.json"
        policy = load_json(policy_path)
        policy_sha = hashlib.sha256(canonical_json_bytes(policy)).hexdigest()
        bindings = {
            "CODE_COMMIT": EXPECTED_COMMIT,
            "LEAKAGE_POLICY_SHA256": policy_sha,
            "G1_PARENT_ID": EXPECTED_PARENT_ID,
            "PARENT_PREDICTION": input_receipts["g1_prediction"]["path"],
            "PARENT_BARRIER_ARTIFACT": input_receipts["g1_barriers"]["path"],
            "VALIDATION_TABLE_DIR": input_receipts["validation_table"]["path"],
            "VALIDATION_SIGNAL_PROBABILITY": input_receipts[
                "validation_signal_probability"
            ]["path"],
            "VALIDATION_SCOPE_MASK": input_receipts["validation_scope_mask"]["path"],
            "DETERMINISTIC_HEADER_ROLES": str(role_path),
            "QUALIFIED_268_IDS": input_receipts["qualified_inventory"]["path"],
            "G2_INPUT_RECEIPTS": input_receipts,
        }
        bindings_path = partial / "bindings.g2.deterministic.json"
        write_json_exclusive(bindings_path, bindings)

        templates = load_json(evolution_root / "experiment_templates.json")
        selected_templates = [
            template
            for template in templates["templates"]
            if template.get("generation") == "G2"
            and template.get("parameter_family") == "deterministic_heading_windows"
        ]
        excluded_learned = [
            template
            for template in templates["templates"]
            if template.get("generation") == "G2"
            and template.get("parameter_family") == "learned_heading_windows"
        ]
        if len(selected_templates) != 1 or len(excluded_learned) != 1:
            raise RuntimeError("G2 heading-template inventory changed")
        queue_rows = expand_template(selected_templates[0], bindings)
        if (
            len(queue_rows) != 4
            or collections.Counter(row["parameter_family"] for row in queue_rows)
            != {"deterministic_heading_windows": 4}
            or sorted(row["sweep_point"]["header_window"] for row in queue_rows)
            != EXPECTED_WINDOWS
            or any(row["parent_candidate_ids"] != [EXPECTED_PARENT_ID] for row in queue_rows)
            or any(
                row["changes"]["headers.role_controller"]["parameters"].get("backend")
                != "deterministic"
                for row in queue_rows
            )
            or any("LEARNED_HEADER_ROLE_IDS" in json.dumps(row) for row in queue_rows)
        ):
            raise RuntimeError("deterministic-only G2 queue shape changed")
        candidate_ids = []
        lineage_rows = []
        leakage_rows = []
        for index, row in enumerate(queue_rows):
            validate_candidate_spec(row)
            lineage = verify_parent_lineage(row)
            leakage = enforce_leakage_barrier(row, policy)
            if (
                lineage.get("status") != "passed"
                or [parent["candidate_id"] for parent in lineage["parents"]]
                != [EXPECTED_PARENT_ID]
                or leakage.get("status") != "passed"
            ):
                raise RuntimeError("G2 lineage/leakage preflight failed")
            candidate_ids.append(row["candidate_id"])
            lineage_rows.append(
                {
                    "queue_index": index,
                    "candidate_id": row["candidate_id"],
                    "status": lineage["status"],
                    "parent_candidate_ids": [
                        parent["candidate_id"] for parent in lineage["parents"]
                    ],
                    "bound_parent_artifacts": lineage["bound_parent_artifacts"],
                }
            )
            leakage_rows.append(
                {
                    "queue_index": index,
                    "candidate_id": row["candidate_id"],
                    "status": leakage["status"],
                    "checked_input_count": len(leakage["checked_inputs"]),
                    "sealed_test_status": leakage["sealed_test_status"],
                }
            )
        if len(set(candidate_ids)) != 4:
            raise RuntimeError("G2 candidate IDs are not unique")
        queue_path = partial / "queue.g2.deterministic.jsonl"
        _write_jsonl_exclusive(queue_path, queue_rows)
        lineage_path = partial / "parent_lineage_preflight.g2.jsonl"
        _write_jsonl_exclusive(lineage_path, lineage_rows)
        leakage_path = partial / "leakage_preflight.g2.jsonl"
        _write_jsonl_exclusive(leakage_path, leakage_rows)

        queue_sha = sha256_file(queue_path)
        candidate_root = launch_root / "candidates"
        launcher = sequence_root / "clariden/run_bibliography_evolution_cpu.sbatch"
        intended_command = (
            "sbatch --array=0-3%4 "
            f"--export=ALL,CODE_ROOT={eval_root},QUEUE_JSONL={packet_root / queue_path.name},"
            f"QUEUE_SHA256={queue_sha},LEAKAGE_POLICY={policy_path},"
            f"CANDIDATE_ROOT={candidate_root},WORKING_DIR={repo_root} {launcher}"
        )
        (partial / "intended_sbatch_command.txt").write_text(
            intended_command + "\n", encoding="utf-8"
        )
        write_json_exclusive(
            partial / "git_attestation.json",
            {
                "status": "passed",
                "head": git(repo_root, "rev-parse", "HEAD"),
                "clean": not bool(
                    git(repo_root, "status", "--porcelain", "--untracked-files=all")
                ),
                "repo_root": str(repo_root),
            },
        )
        write_json_exclusive(
            partial / "template_selection.json",
            {
                "status": "passed_exact_deterministic_only",
                "selected_parameter_family": "deterministic_heading_windows",
                "selected_windows": EXPECTED_WINDOWS,
                "learned_parameter_family_present_but_excluded": True,
                "reason_learned_excluded": (
                    "No receipt-owned deployable learned heading inference artifact is "
                    "admitted; learned branch remains fail-closed."
                ),
            },
        )
        artifact_rows = {}
        for path in sorted(partial.rglob("*")):
            if path.is_file() and path.name != "packet_receipt.json":
                artifact_rows[path.relative_to(partial).as_posix()] = {
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
        receipt = {
            "schema_version": "bibliography-evolution-g2-deterministic-launch-packet-v1",
            "status": "passed_all_preflight_gates_not_submitted",
            "code_commit": EXPECTED_COMMIT,
            "slurm_preparation_job_id": job_id,
            "packet_root": str(packet_root),
            "parent": {
                "candidate_id": EXPECTED_PARENT_ID,
                "receipt_path": str(parent_receipt_path),
                "receipt_sha256": EXPECTED_PARENT_RECEIPT_SHA,
                "prediction_sha256": EXPECTED_PARENT_PREDICTION_SHA,
                "barrier_sha256": EXPECTED_PARENT_BARRIER_SHA,
                "finalized_verification": parent_verification,
                "registry_audit_verdict": "ADMIT_AS_G2_PARENT",
                "registry_sha256": EXPECTED_REGISTRY_SHA,
                "recommendation_sha256": EXPECTED_RECOMMENDATION_SHA,
            },
            "deterministic_header_roles": {
                "path": str(role_path),
                "sha256": sha256_file(role_path),
                "receipt_path": str(role_receipt_path),
                "receipt_sha256": sha256_file(role_receipt_path),
                "policy_sha256": sha256_file(role_policy_path),
                "generator_sha256": sha256_file(role_generator_path),
                "role_counts": EXPECTED_ROLE_COUNTS,
                "text_only_and_sealed_reproducible": True,
                "byte_identical_reference_gates": True,
            },
            "queue": {
                "path": str(packet_root / queue_path.name),
                "rows": 4,
                "sha256": queue_sha,
                "parameter_family": "deterministic_heading_windows",
                "windows": EXPECTED_WINDOWS,
                "candidate_ids": candidate_ids,
            },
            "gates": {
                "git_clean_exact_commit": True,
                "parent_finalized_and_artifacts_owned": True,
                "parent_registry_admitted": True,
                "deterministic_role_lineage_passed": True,
                "candidate_schema_validation": True,
                "parent_lineage_preflight": True,
                "leakage_preflight": True,
                "learned_heading_template_excluded": True,
                "sealed_data_opened": False,
            },
            "policy_sha256": policy_sha,
            "bindings_sha256": sha256_file(bindings_path),
            "inputs_sha256": sha256_file(inputs_path),
            "lineage_preflight_sha256": sha256_file(lineage_path),
            "leakage_preflight_sha256": sha256_file(leakage_path),
            "intended_sbatch_command_not_executed": intended_command,
            "g2_submitted": False,
            "artifacts": artifact_rows,
        }
        write_json_exclusive(partial / "packet_receipt.json", receipt)
        os.replace(partial, packet_root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    for path in packet_root.rglob("*"):
        if path.is_file():
            path.chmod(0o440)
    packet_root.chmod(0o550)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "packet_root": str(packet_root),
                "packet_receipt": str(packet_root / "packet_receipt.json"),
                "packet_receipt_sha256": sha256_file(packet_root / "packet_receipt.json"),
                "queue_sha256": receipt["queue"]["sha256"],
                "queue_rows": receipt["queue"]["rows"],
                "candidate_ids": receipt["queue"]["candidate_ids"],
                "g2_submitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
