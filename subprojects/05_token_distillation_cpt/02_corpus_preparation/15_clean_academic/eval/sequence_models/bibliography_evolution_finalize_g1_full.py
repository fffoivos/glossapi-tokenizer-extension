#!/usr/bin/env python3
"""Freeze the audited full-G1 registry and select the deterministic G2 parent."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_G1_CODE_COMMIT = "c5b79d98bf58cf9111d49103d889d990be9018c6"
EXPECTED_QUEUE_SHA256 = "b56cb01317dd8408aaefef4a1ce6f1fe1cd1c221281a09f46169a855597cafca"
EXPECTED_REGISTRY_SHA256 = "eb6f229e8ce0b8df68c06461d580a9db807840a1f546b7955c961cfd0ec7285b"
EXPECTED_PARENT_ID = "g1-1909806a497053bb7ac4c964"
EXPECTED_PARENT_RECEIPT_SHA256 = (
    "9ae3ce4f3d80676ef7d561e429c835e12c65690f9004da15e0dc4e0a0e4479fb"
)
EXPECTED_PARENT_PREDICTION_SHA256 = (
    "58c4f0a4108f1c7c461782c81274363bb29e83fea2b9151dcd7751aecd6da684"
)
EXPECTED_PARENT_BARRIER_SHA256 = (
    "540ad1326cc282aed13f4c13458d002bc13288fba3d765cbb78ff48fbb2c09b1"
)
OBJECTIVES = (
    "token_fp",
    "token_fn",
    "spurious_blocks_per_zero_block_document",
    "mean_boundary_error_emitted_lines",
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout.strip()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_full_g1_parent(
    registry: Mapping[str, Any], queue_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen no-scalar strict-component-isolation rule."""

    queued_ids = [str(row["candidate_id"]) for row in queue_rows]
    if len(queued_ids) != 27 or len(set(queued_ids)) != 27:
        raise RuntimeError("full G1 queue must contain 27 unique candidates")
    controls = [
        str(row["candidate_id"])
        for row in queue_rows
        if row.get("sweep_point") == {"anchor_probability": 0.3}
    ]
    if controls != [EXPECTED_PARENT_ID]:
        raise RuntimeError("full G1 queue does not contain the expected 0.30 control")

    indexed = {str(row["candidate_id"]): row for row in registry["candidates"]}
    if any(candidate_id not in indexed for candidate_id in queued_ids):
        raise RuntimeError("full G1 queue and registry candidate sets differ")
    reference = indexed[EXPECTED_PARENT_ID]["objective_vector"]
    g0_rows = [row for row in registry["candidates"] if row["generation"] == "G0"]
    if len(g0_rows) != 1 or any(
        reference[name] != g0_rows[0]["objective_vector"][name]
        for name in OBJECTIVES
    ):
        raise RuntimeError("full G1 0.30 control is not G0-equivalent")

    comparisons: dict[str, Any] = {}
    qualifying: list[str] = []
    for candidate_id in queued_ids:
        if candidate_id == EXPECTED_PARENT_ID:
            continue
        objective_vector = indexed[candidate_id]["objective_vector"]
        deltas = {
            name: objective_vector[name] - reference[name] for name in OBJECTIVES
        }
        weakly_dominates = all(delta <= 0 for delta in deltas.values()) and any(
            delta < 0 for delta in deltas.values()
        )
        comparisons[candidate_id] = {
            "candidate_minus_reference": deltas,
            "weakly_dominates_with_at_least_one_strict_improvement": weakly_dominates,
        }
        if weakly_dominates:
            qualifying.append(candidate_id)
    if qualifying:
        raise RuntimeError(
            "a full G1 child weakly dominates the control; parent selection requires "
            "a new independently audited rule"
        )
    return {
        "reference_candidate_id": EXPECTED_PARENT_ID,
        "qualifying_weak_dominators": qualifying,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    from sequence_models.bibliography_evolution_contract import (
        load_json,
        sha256_file,
        verify_finalized_receipt,
        write_json_exclusive,
    )

    repo_root = args.repo_root.resolve()
    candidate_root = args.candidate_root.resolve()
    queue_path = args.queue.resolve()
    registry_path = args.registry.resolve()
    output_root = args.output_root.resolve()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("full G1 finalization must run on a Slurm compute node")
    finalizer_commit = _git(repo_root, "rev-parse", "HEAD")
    ancestor_check = subprocess.run(
        [
            "git",
            "-C",
            str(repo_root),
            "merge-base",
            "--is-ancestor",
            EXPECTED_G1_CODE_COMMIT,
            finalizer_commit,
        ],
        check=False,
    )
    if ancestor_check.returncode:
        raise RuntimeError("finalizer checkout does not descend from audited full G1 code")
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("deployed full G1 checkout is dirty")
    if sha256_file(queue_path) != EXPECTED_QUEUE_SHA256:
        raise RuntimeError("full G1 queue hash drift")
    if sha256_file(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("audited full G1 registry hash drift")

    queue_rows = _load_jsonl(queue_path)
    if any(
        row.get("generation") != "G1"
        or row.get("code_commit") != EXPECTED_G1_CODE_COMMIT
        for row in queue_rows
    ):
        raise RuntimeError("full G1 queue contract drift")
    registry = load_json(registry_path)
    if (
        registry.get("candidate_count") != 33
        or registry.get("eligible_count") != 33
        or len(registry.get("pareto_candidate_ids", [])) != 12
    ):
        raise RuntimeError("audited full G1 registry cardinality drift")

    receipt_hashes: dict[str, str] = {}
    expected_ids = set()
    for row in registry["candidates"]:
        candidate_id = str(row["candidate_id"])
        receipt_path = Path(str(row["receipt_path"])).resolve()
        if receipt_path != candidate_root / candidate_id / "receipt.json":
            raise RuntimeError(f"registry receipt path drift: {candidate_id}")
        digest = sha256_file(receipt_path)
        if digest != row["receipt_sha256"]:
            raise RuntimeError(f"registry receipt hash drift: {candidate_id}")
        verification = verify_finalized_receipt(receipt_path)
        if verification.get("candidate_id") != candidate_id:
            raise RuntimeError(f"finalized receipt identity drift: {candidate_id}")
        receipt_hashes[candidate_id] = digest
        expected_ids.add(candidate_id)
    actual_ids = {path.parent.name for path in candidate_root.glob("*/receipt.json")}
    if actual_ids != expected_ids:
        raise RuntimeError("candidate root contains an unregistered finalized receipt")

    selection = select_full_g1_parent(registry, queue_rows)
    parent_receipt_path = candidate_root / EXPECTED_PARENT_ID / "receipt.json"
    if sha256_file(parent_receipt_path) != EXPECTED_PARENT_RECEIPT_SHA256:
        raise RuntimeError("selected full G1 parent receipt drift")
    parent_prediction = parent_receipt_path.parent / "backend" / "prediction.npy"
    parent_barriers = parent_receipt_path.parent / "backend" / "combined_barriers.npz"
    if (
        sha256_file(parent_prediction) != EXPECTED_PARENT_PREDICTION_SHA256
        or sha256_file(parent_barriers) != EXPECTED_PARENT_BARRIER_SHA256
    ):
        raise RuntimeError("selected full G1 parent inference artifact drift")

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.partial-", dir=output_root.parent)
    )
    try:
        frozen_registry = partial / "development_registry.json"
        shutil.copyfile(registry_path, frozen_registry)
        recommendation = {
            "schema_version": "bibliography-evolution-g2-parent-recommendation-v2",
            "status": "passed_full_g1_strict_component_isolation",
            "development_registry_path": str(output_root / frozen_registry.name),
            "development_registry_sha256": EXPECTED_REGISTRY_SHA256,
            "full_g1_queue_sha256": EXPECTED_QUEUE_SHA256,
            "sealed_data_opened": False,
            "full_frontier_preserved_for_final_sealed_evaluation": registry[
                "pareto_candidate_ids"
            ],
            "rule": {
                "name": "strict_component_isolation",
                "reference_candidate_id": EXPECTED_PARENT_ID,
                "reference_anchor_probability": 0.3,
                "objectives_minimized": list(OBJECTIVES),
                "advance_rule": (
                    "A full-G1 child must be no worse on all four registry objectives "
                    "and strictly better on at least one; otherwise retain the full-run "
                    "0.30 baseline-equivalent child."
                ),
                "post_hoc_scalar_tradeoff_used": False,
            },
            "recommended_parent_candidate_id": EXPECTED_PARENT_ID,
            "recommended_anchor_probability": 0.3,
            "qualifying_weak_dominators": selection["qualifying_weak_dominators"],
            "g1_objective_comparisons_to_0_30_reference": selection["comparisons"],
            "reference_matches_g0_on_all_four_registry_objectives": True,
        }
        recommendation_path = partial / "g2_parent_recommendation.json"
        write_json_exclusive(recommendation_path, recommendation)
        receipt = {
            "schema_version": "bibliography-evolution-full-g1-registry-receipt-v1",
            "status": "passed_full_receipt_verification_registry_frozen",
            "full_g1_code_commit": EXPECTED_G1_CODE_COMMIT,
            "finalizer_code_commit": finalizer_commit,
            "slurm_job_id": job_id,
            "candidate_receipt_sha256": receipt_hashes,
            "full_g1_queue": {
                "path": str(queue_path),
                "sha256": EXPECTED_QUEUE_SHA256,
                "candidate_count": 27,
            },
            "development_registry": {
                "path": str(output_root / frozen_registry.name),
                "sha256": EXPECTED_REGISTRY_SHA256,
                "candidate_count": 33,
                "eligible_count": 33,
                "pareto_candidate_ids": registry["pareto_candidate_ids"],
            },
            "g2_parent_recommendation": {
                "path": str(output_root / recommendation_path.name),
                "sha256": sha256_file(recommendation_path),
                "candidate_id": EXPECTED_PARENT_ID,
            },
            "selected_parent": {
                "receipt_path": str(parent_receipt_path),
                "receipt_sha256": EXPECTED_PARENT_RECEIPT_SHA256,
                "prediction_sha256": EXPECTED_PARENT_PREDICTION_SHA256,
                "barrier_sha256": EXPECTED_PARENT_BARRIER_SHA256,
            },
            "sealed_data_opened": False,
            "g2_packet_materialized": False,
            "g2_submitted": False,
        }
        receipt_path = partial / "receipt.json"
        write_json_exclusive(receipt_path, receipt)
        os.replace(partial, output_root)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    for path in output_root.iterdir():
        path.chmod(0o440)
    output_root.chmod(0o550)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "registry_sha256": EXPECTED_REGISTRY_SHA256,
                "receipt_sha256": sha256_file(output_root / "receipt.json"),
                "recommended_g2_parent": EXPECTED_PARENT_ID,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
