#!/usr/bin/env python3
"""Freeze the audited G2 registry and record the no-admission decision."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_G2_CODE_COMMIT = "ad0fd422658327034a2aaf10b4e9fa77ea25e825"
EXPECTED_QUEUE_SHA256 = "253ca9edc87958490cc302b38774347b9a964ff8f545f59ea777c0c96a06ba63"
EXPECTED_REGISTRY_SHA256 = "c629daa33b88ed32a7afd2353935ba90c2efb03de39856a59d83a13fe5e444e9"
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
EXPECTED_G2_RECEIPTS = {
    "g2-e0da48d37e3d357b45603366": (
        "4a10639f49dcdd957a2ea20bd3ea5acc7f6c02bb4afad6ce5669cfa6d1b447ed"
    ),
    "g2-f555b4bafa3befb2c6ed94e9": (
        "ef734cae8dc03c8dfd65c165174b1d2a87bcf21361fdc7efa0275ff93e1dab80"
    ),
    "g2-aa052a5aa2898cf11a7596e2": (
        "f809837611ee51f7a410f5ac339fbd0701b5b27a4881a0fb24d2a2d6a30a42ea"
    ),
    "g2-7c00e09c62c7745aa5906af4": (
        "96a3de93ad05916a723013c462b228952e3cce13773e032179e97de5b56eed9d"
    ),
}
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


def select_g2_parent(
    registry: Mapping[str, Any], queue_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply the frozen no-scalar strict-component-isolation rule to G2."""

    queued_ids = [str(row["candidate_id"]) for row in queue_rows]
    if len(queued_ids) != 4 or len(set(queued_ids)) != 4:
        raise RuntimeError("G2 queue must contain four unique candidates")
    if set(queued_ids) != set(EXPECTED_G2_RECEIPTS):
        raise RuntimeError("G2 queue candidate inventory drift")
    if any(
        row.get("generation") != "G2"
        or row.get("code_commit") != EXPECTED_G2_CODE_COMMIT
        or row.get("parent_candidate_ids") != [EXPECTED_PARENT_ID]
        for row in queue_rows
    ):
        raise RuntimeError("G2 queue lineage drift")

    indexed = {str(row["candidate_id"]): row for row in registry["candidates"]}
    if EXPECTED_PARENT_ID not in indexed or any(
        candidate_id not in indexed for candidate_id in queued_ids
    ):
        raise RuntimeError("G2 queue or reference parent is absent from registry")
    reference = indexed[EXPECTED_PARENT_ID]["objective_vector"]

    comparisons: dict[str, Any] = {}
    qualifying: list[str] = []
    for row in queue_rows:
        candidate_id = str(row["candidate_id"])
        objective_vector = indexed[candidate_id]["objective_vector"]
        deltas = {
            name: objective_vector[name] - reference[name] for name in OBJECTIVES
        }
        weakly_dominates = all(delta <= 0 for delta in deltas.values()) and any(
            delta < 0 for delta in deltas.values()
        )
        comparisons[candidate_id] = {
            "header_window": int(row["sweep_point"]["header_window"]),
            "candidate_minus_reference": deltas,
            "weakly_dominates_with_at_least_one_strict_improvement": weakly_dominates,
        }
        if weakly_dominates:
            qualifying.append(candidate_id)
    if qualifying:
        raise RuntimeError(
            "a G2 child weakly dominates the reference; this no-admission finalizer "
            "must not choose among qualifying candidates"
        )
    return {
        "reference_candidate_id": EXPECTED_PARENT_ID,
        "qualifying_weak_dominators": qualifying,
        "comparisons": comparisons,
        "promoted_g2_candidate_id": None,
        "g3_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalizer-repo-root", type=Path, required=True)
    parser.add_argument("--g2-repo-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--g2-slurm-job-id", default="2793872")
    args = parser.parse_args()

    from sequence_models.bibliography_evolution_contract import (
        load_json,
        sha256_file,
        verify_finalized_receipt,
        write_json_exclusive,
    )

    finalizer_repo = args.finalizer_repo_root.resolve()
    g2_repo = args.g2_repo_root.resolve()
    candidate_root = args.candidate_root.resolve()
    queue_path = args.queue.resolve()
    registry_path = args.registry.resolve()
    output_root = args.output_root.resolve()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        raise RuntimeError("G2 finalization must run on a Slurm compute node")
    finalizer_commit = _git(finalizer_repo, "rev-parse", "HEAD")
    if _git(finalizer_repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("finalizer checkout is dirty")
    if _git(g2_repo, "rev-parse", "HEAD") != EXPECTED_G2_CODE_COMMIT:
        raise RuntimeError("deployed G2 checkout differs from the audited code")
    if _git(g2_repo, "status", "--porcelain", "--untracked-files=all"):
        raise RuntimeError("deployed G2 checkout is dirty")
    if sha256_file(queue_path) != EXPECTED_QUEUE_SHA256:
        raise RuntimeError("G2 queue hash drift")
    if sha256_file(registry_path) != EXPECTED_REGISTRY_SHA256:
        raise RuntimeError("audited G2 registry hash drift")

    queue_rows = _load_jsonl(queue_path)
    registry = load_json(registry_path)
    if (
        registry.get("candidate_count") != 37
        or registry.get("eligible_count") != 37
        or len(registry.get("pareto_candidate_ids", [])) != 12
    ):
        raise RuntimeError("audited G2 registry cardinality drift")
    if any(
        candidate_id in registry["pareto_candidate_ids"]
        for candidate_id in EXPECTED_G2_RECEIPTS
    ):
        raise RuntimeError("a G2 candidate unexpectedly entered the Pareto frontier")

    receipt_hashes: dict[str, str] = {}
    expected_ids: set[str] = set()
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
    if any(
        receipt_hashes[candidate_id] != digest
        for candidate_id, digest in EXPECTED_G2_RECEIPTS.items()
    ):
        raise RuntimeError("audited G2 candidate receipt hash drift")

    selection = select_g2_parent(registry, queue_rows)
    parent_receipt_path = candidate_root / EXPECTED_PARENT_ID / "receipt.json"
    parent_prediction = parent_receipt_path.parent / "backend" / "prediction.npy"
    parent_barriers = parent_receipt_path.parent / "backend" / "combined_barriers.npz"
    if (
        sha256_file(parent_receipt_path) != EXPECTED_PARENT_RECEIPT_SHA256
        or sha256_file(parent_prediction) != EXPECTED_PARENT_PREDICTION_SHA256
        or sha256_file(parent_barriers) != EXPECTED_PARENT_BARRIER_SHA256
    ):
        raise RuntimeError("retained G1 parent artifact drift")

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.partial-", dir=output_root.parent))
    try:
        frozen_registry = partial / "development_registry.json"
        shutil.copyfile(registry_path, frozen_registry)
        decision = {
            "schema_version": "bibliography-evolution-g2-selection-v1",
            "status": "passed_g2_no_admission",
            "development_registry_path": str(output_root / frozen_registry.name),
            "development_registry_sha256": EXPECTED_REGISTRY_SHA256,
            "g2_queue_sha256": EXPECTED_QUEUE_SHA256,
            "sealed_data_opened": False,
            "rule": {
                "name": "strict_component_isolation",
                "reference_candidate_id": EXPECTED_PARENT_ID,
                "objectives_minimized": list(OBJECTIVES),
                "advance_rule": (
                    "A G2 child must be no worse on all four registry objectives and "
                    "strictly better on at least one; otherwise retain G1 and do not "
                    "advance this branch to G3."
                ),
                "post_hoc_scalar_tradeoff_used": False,
            },
            "retained_parent_candidate_id": EXPECTED_PARENT_ID,
            **selection,
        }
        decision_path = partial / "g2_selection.json"
        write_json_exclusive(decision_path, decision)
        receipt = {
            "schema_version": "bibliography-evolution-g2-registry-receipt-v1",
            "status": "passed_g2_receipt_verification_no_admission",
            "g2_code_commit": EXPECTED_G2_CODE_COMMIT,
            "finalizer_code_commit": finalizer_commit,
            "finalizer_source_sha256": sha256_file(Path(__file__).resolve()),
            "g2_slurm_job_id": str(args.g2_slurm_job_id),
            "finalizer_slurm_job_id": job_id,
            "candidate_receipt_sha256": receipt_hashes,
            "g2_queue": {
                "path": str(queue_path),
                "sha256": EXPECTED_QUEUE_SHA256,
                "candidate_count": 4,
            },
            "development_registry": {
                "path": str(output_root / frozen_registry.name),
                "sha256": EXPECTED_REGISTRY_SHA256,
                "candidate_count": 37,
                "eligible_count": 37,
                "pareto_candidate_ids": registry["pareto_candidate_ids"],
            },
            "g2_selection": {
                "path": str(output_root / decision_path.name),
                "sha256": sha256_file(decision_path),
                "promoted_g2_candidate_id": None,
                "g3_authorized": False,
            },
            "retained_parent": {
                "candidate_id": EXPECTED_PARENT_ID,
                "receipt_path": str(parent_receipt_path),
                "receipt_sha256": EXPECTED_PARENT_RECEIPT_SHA256,
                "prediction_sha256": EXPECTED_PARENT_PREDICTION_SHA256,
                "barrier_sha256": EXPECTED_PARENT_BARRIER_SHA256,
            },
            "sealed_data_opened": False,
        }
        write_json_exclusive(partial / "receipt.json", receipt)
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
                "retained_parent": EXPECTED_PARENT_ID,
                "g3_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
