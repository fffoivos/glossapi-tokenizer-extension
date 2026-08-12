#!/usr/bin/env python3
"""Rebind the proven native-Greek scorer contract to one new HF export."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from contract_utils import atomic_json, file_binding, read_json, require, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--source-execution-gate", type=Path, required=True)
    parser.add_argument("--eval-code-receipt", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = read_json(args.source_contract)
    gate = read_json(args.source_execution_gate)
    bundle = read_json(args.eval_code_receipt)
    manifest = read_json(args.manifest)
    export = read_json(args.export_receipt)
    require(args.iteration == 13193, "native endpoint iteration drift")
    require(source.get("schema_version") == "apertus_full8_native_greek_3cp_contract_v1", "source contract drift")
    require(gate.get("schema_version") == "apertus_full8_native_greek_execution_gate_v1" and gate.get("status") == "passed", "native execution gate drift")
    require(bundle.get("schema_version") == "native_greek_eval_code_bundle_v1" and bundle.get("status") == "frozen" and gate.get("code_tree_sha256") == bundle.get("tree_sha256"), "native code/gate binding drift")
    require(gate.get("selected") == {"candidate_batch_size": 1, "dtype": "float32", "example_batch_size": 16, "scorer_mode": "legacy"}, "native scorer selection drift")
    require(gate.get("contract_sha256") == sha256_file(args.source_contract), "source contract/gate binding drift")
    require(gate.get("manifest_sha256") == sha256_file(args.manifest), "manifest/gate binding drift")
    require(manifest.get("status") == "completed" and args.model.is_dir(), "native inputs incomplete")
    require(export.get("schema_version") == "native_greekmmlu_exact_checkpoint_export_v1" and export.get("status") == "completed" and export.get("ready_for_frozen_native_greekmmlu") is True, "HF export receipt drift")
    require(int(export.get("source", {}).get("iteration", -1)) == args.iteration, "HF export iteration drift")
    result = copy.deepcopy(source)
    result["schema_version"] = "apertus_full8_native_greek_endpoint_contract_v1"
    result["status"] = "frozen_except_protipa_access"
    result["checkpoint_scope"] = [{
        "label": f"iter_{args.iteration:07d}",
        "iteration": args.iteration,
        "token_slots": args.iteration * 1024 * 4096,
        "model_path": str(args.model.resolve()),
        "evidence_receipt": str(args.export_receipt.resolve()),
    }]
    result["rebind_evidence"] = {
        "source_contract": file_binding(args.source_contract),
        "source_execution_gate": file_binding(args.source_execution_gate),
        "eval_code_receipt": file_binding(args.eval_code_receipt),
        "frozen_example_manifest": file_binding(args.manifest),
        "only_changed_contract_field": "checkpoint_scope",
        "scoring_model_and_benchmark_contract_byte_identical": True,
    }
    atomic_json(args.output, result)
    print(json.dumps({"ok": True, "iteration": args.iteration, "model": str(args.model)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
