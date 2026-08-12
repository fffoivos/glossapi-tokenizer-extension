#!/usr/bin/env python3
"""Rebind a passed FP32 fallback gate across operational-only bundle changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ALLOWED_OPERATIONAL_CHANGES = {
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/README.md",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/aggregate_checkpoint_shards.py",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/rebind_execution_gate.py",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/run_three_checkpoint_matrix.sbatch",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/test_native_greek_evaluation.py",
}
SCIENTIFIC_FILES = {
    "subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/run_native_greek_mcq_eval.py",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/compare_batch_parity.py",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/native_greek_3cp_contract.json",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/run_checkpoint_suite.py",
    "subprojects/09_full_8b_cpt_results_analysis/evaluation/run_fp32_batch_parity.sbatch",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-bundle-receipt", type=Path, required=True)
    parser.add_argument("--new-bundle-receipt", type=Path, required=True)
    parser.add_argument("--old-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    old_bundle = json.loads(args.old_bundle_receipt.read_text())
    new_bundle = json.loads(args.new_bundle_receipt.read_text())
    old_gate = json.loads(args.old_gate.read_text())
    for bundle in (old_bundle, new_bundle):
        if bundle.get("schema_version") != "native_greek_eval_code_bundle_v1" or bundle.get("status") != "frozen":
            raise ValueError("invalid code-bundle receipt")
    if old_gate.get("schema_version") != "apertus_full8_native_greek_execution_gate_v1" or old_gate.get("status") != "passed":
        raise ValueError("old execution gate is not passed")
    expected_selection = {"dtype": "float32", "scorer_mode": "legacy", "candidate_batch_size": 1, "example_batch_size": 16}
    if old_gate.get("selected") != expected_selection or old_gate.get("selection_status") != "proven_fallback" or old_gate.get("candidate_status") != "rejected":
        raise ValueError("old execution gate is not the rejected-candidate FP32 fallback")
    if old_gate.get("code_tree_sha256") != old_bundle.get("tree_sha256"):
        raise ValueError("old gate is not bound to the old bundle")

    old_files = {row["path"]: row for row in old_bundle["files"]}
    new_files = {row["path"]: row for row in new_bundle["files"]}
    changed = {
        path
        for path in old_files.keys() | new_files.keys()
        if old_files.get(path) != new_files.get(path)
    }
    if not changed or not changed <= ALLOWED_OPERATIONAL_CHANGES:
        raise ValueError(f"bundle changes are not the expected operational-only set: {sorted(changed)}")
    for path in SCIENTIFIC_FILES:
        if old_files.get(path) != new_files.get(path):
            raise ValueError(f"scientific evaluation file changed: {path}")

    output = dict(old_gate)
    output["code_tree_sha256"] = new_bundle["tree_sha256"]
    output["rebound_from"] = {
        "execution_gate_path": str(args.old_gate.resolve()),
        "execution_gate_sha256": sha256(args.old_gate),
        "code_tree_sha256": old_bundle["tree_sha256"],
    }
    output["rebind"] = {
        "reason": "matrix sharding and aggregation only; scorer, model contract and parity logic are byte-identical",
        "changed_paths": sorted(changed),
        "scientific_files_verified_identical": sorted(SCIENTIFIC_FILES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "changed_paths": sorted(changed), "selected": output["selected"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
