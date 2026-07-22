#!/usr/bin/env python3
"""Verify G0/G1 receipts, freeze their dev registry, and recommend no G2 launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXPECTED_COMMIT = "931a56d119b9ab44e79c23fa82a16cd2edf0c4b7"
EXPECTED_RECEIPTS = {
    "g0-34a7dde38689d1b15e8afd42": "95eb0f5a855f780c606d87844fa680e889dc5b70b7eae752b051fa3c48902955",
    "g1-fb6b73f987ac39b8970bbb60": "9dd049c86f4f7f043da5e4e7de74ecc1bc320e7fd6ff5af4d9c5c88bd5d5308d",
    "g1-0c0ebff3b3d08309268e4c8d": "7a8b13500f92c8c9decd21fb3147be2e2d47f620f5c35a5064976d4f2e3b0280",
    "g1-e47e74768f652e91f3cd7f73": "7a1cebdc322f75e76e5d7ca9271dc9cece94d9dc5521fe0ffa81f4df9ac4453a",
    "g1-a5097ce7f3cfc473998b5e72": "cdb4aa63678dc8fba4936b41c5cfc166031e320e4d90997e889a66c959d7efff",
    "g1-8fd4f245e298cf85a01681a6": "4533880d1698ea4fb68b080b0ed1822f0455d0659a43dbdfe25d0527bb5be7b9",
}
EXPECTED_PARETO = {
    "g0-34a7dde38689d1b15e8afd42",
    "g1-fb6b73f987ac39b8970bbb60",
    "g1-0c0ebff3b3d08309268e4c8d",
    "g1-a5097ce7f3cfc473998b5e72",
    "g1-8fd4f245e298cf85a01681a6",
}
BASELINE_EQUIVALENT_G1 = "g1-e47e74768f652e91f3cd7f73"
RECOMMENDED_PARENT = BASELINE_EQUIVALENT_G1


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
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    candidate_root = args.candidate_root.resolve()
    registry_root = args.registry_root.resolve()
    eval_root = (
        repo_root
        / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
    )
    sys.path.insert(0, str(eval_root))

    from sequence_models.bibliography_evolution_contract import (
        build_registry,
        load_json,
        sha256_file,
        write_json_exclusive,
    )

    if git(repo_root, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise RuntimeError("registry checkout differs from the audited commit")
    if git(repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("registry checkout is dirty")
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("registry construction must run on a Slurm compute node")

    receipt_paths = []
    for candidate_id, digest in EXPECTED_RECEIPTS.items():
        path = candidate_root / candidate_id / "receipt.json"
        if sha256_file(path) != digest:
            raise RuntimeError(f"candidate receipt drift: {candidate_id}")
        receipt_paths.append(path)
    actual_receipts = {
        path.parent.name: sha256_file(path)
        for path in candidate_root.glob("*/receipt.json")
    }
    if actual_receipts != EXPECTED_RECEIPTS:
        raise RuntimeError("candidate root contains an unexpected finalized receipt")

    output_root = registry_root / f"g0-g1-anchor-{job_id}"
    output_root.mkdir(parents=True)
    registry = build_registry(receipt_paths)
    if (
        registry["candidate_count"] != 6
        or registry["eligible_count"] != 6
        or set(registry["pareto_candidate_ids"]) != EXPECTED_PARETO
    ):
        raise RuntimeError("development registry has an unexpected frontier")
    indexed = {row["candidate_id"]: row for row in registry["candidates"]}
    tied = indexed["g1-e47e74768f652e91f3cd7f73"]
    if tied["pareto"] or tied["dominated_by"] != ["g0-34a7dde38689d1b15e8afd42"]:
        raise RuntimeError("the exact G0-equivalent G1 control did not collapse by tie-break")
    registry_path = output_root / "development_registry.json"
    write_json_exclusive(registry_path, registry)

    receipts = {
        candidate_id: load_json(candidate_root / candidate_id / "receipt.json")
        for candidate_id in EXPECTED_RECEIPTS
    }
    g0_id = "g0-34a7dde38689d1b15e8afd42"
    g0 = receipts[g0_id]["metrics"]
    reference = receipts[BASELINE_EQUIVALENT_G1]["metrics"]
    objective_names = [
        "token_fp",
        "token_fn",
        "spurious_blocks_per_zero_block_document",
        "mean_boundary_error_emitted_lines",
    ]
    if any(reference[name] != g0[name] for name in objective_names):
        raise RuntimeError("the G1 0.30 control is not baseline-equivalent on the registry objectives")
    g1_ids = [
        candidate_id for candidate_id in EXPECTED_RECEIPTS if candidate_id.startswith("g1-")
    ]
    qualifying_dominators = []
    objective_comparisons = {}
    for candidate_id in g1_ids:
        if candidate_id == BASELINE_EQUIVALENT_G1:
            continue
        metrics = receipts[candidate_id]["metrics"]
        deltas = {name: metrics[name] - reference[name] for name in objective_names}
        weakly_dominates = all(delta <= 0 for delta in deltas.values()) and any(
            delta < 0 for delta in deltas.values()
        )
        objective_comparisons[candidate_id] = {
            "candidate_minus_reference": deltas,
            "weakly_dominates_with_at_least_one_strict_improvement": weakly_dominates,
        }
        if weakly_dominates:
            qualifying_dominators.append(candidate_id)
    if qualifying_dominators or RECOMMENDED_PARENT != BASELINE_EQUIVALENT_G1:
        raise RuntimeError("strict component-isolation rule should retain the 0.30 control")
    nondominated_g1 = [
        candidate_id
        for candidate_id in registry["pareto_candidate_ids"]
        if candidate_id.startswith("g1-")
    ]
    recommendation = {
        "schema_version": "bibliography-evolution-g2-parent-recommendation-v1",
        "status": "validation_only_recommendation_no_g2_packet_or_launch",
        "development_registry_path": str(registry_path),
        "development_registry_sha256": sha256_file(registry_path),
        "sealed_data_opened": False,
        "full_frontier_preserved_for_final_sealed_evaluation": registry["pareto_candidate_ids"],
        "nondominated_g1_candidates_preserved_for_final_sealed_evaluation": nondominated_g1,
        "rule": {
            "name": "strict_component_isolation",
            "reference_candidate_id": BASELINE_EQUIVALENT_G1,
            "reference_anchor_probability": 0.3,
            "objectives_minimized": objective_names,
            "advance_rule": "A G1 anchor child must be no worse on all four registry objectives and strictly better on at least one; otherwise retain the 0.30 baseline-equivalent child.",
            "post_hoc_scalar_tradeoff_used": False,
            "chosen_after_one_registered_anchor-threshold family only": True,
        },
        "recommended_parent_candidate_id": RECOMMENDED_PARENT,
        "recommended_anchor_probability": 0.3,
        "qualifying_weak_dominators": qualifying_dominators,
        "g1_objective_comparisons_to_0_30_reference": objective_comparisons,
        "reference_matches_g0_on_all_four_registry_objectives": True,
        "caveat": "This is a development-set sequencing recommendation, not a final model selection. No G2 packet is materialized until the parent is agreed.",
    }
    recommendation_path = output_root / "g2_parent_recommendation.json"
    write_json_exclusive(recommendation_path, recommendation)
    registry_receipt = {
        "schema_version": "bibliography-evolution-g0-g1-registry-receipt-v1",
        "status": "passed_full_receipt_verification_registry_frozen",
        "code_commit": EXPECTED_COMMIT,
        "slurm_job_id": job_id,
        "candidate_receipt_sha256": EXPECTED_RECEIPTS,
        "development_registry": {
            "path": str(registry_path),
            "sha256": sha256_file(registry_path),
            "candidate_count": registry["candidate_count"],
            "eligible_count": registry["eligible_count"],
            "pareto_candidate_ids": registry["pareto_candidate_ids"],
        },
        "g2_parent_recommendation": {
            "path": str(recommendation_path),
            "sha256": sha256_file(recommendation_path),
            "candidate_id": RECOMMENDED_PARENT,
        },
        "sealed_data_opened": False,
        "g2_packet_materialized": False,
        "g2_submitted": False,
    }
    receipt_path = output_root / "receipt.json"
    write_json_exclusive(receipt_path, registry_receipt)
    for path in output_root.rglob("*"):
        if path.is_file():
            path.chmod(0o440)
    print(
        json.dumps(
            {
                "status": registry_receipt["status"],
                "registry_path": str(registry_path),
                "registry_sha256": sha256_file(registry_path),
                "receipt_path": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "pareto_candidate_ids": registry["pareto_candidate_ids"],
                "recommended_g2_parent": RECOMMENDED_PARENT,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
