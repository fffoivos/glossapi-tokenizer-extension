#!/usr/bin/env python3
"""Bind the frozen native-Greek scorer/examples to one verified study export."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from contract_utils import (
    file_binding,
    read_json,
    require,
    sha256_file,
    write_json_atomic,
)

SELECTED_PROFILE = {
    "candidate_batch_size": 1,
    "dtype": "float32",
    "example_batch_size": 16,
    "scorer_mode": "legacy",
}
TOKENIZER_HASHES = [
    "358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394",
    "37c110e765160f64a22bf913f714a40744c84208d0ec22d8f22b8232b1923c34",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-execution-gate", type=Path, required=True)
    parser.add_argument("--eval-code-receipt", type=Path, required=True)
    parser.add_argument("--checkpoint-export", type=Path, required=True)
    parser.add_argument("--scale", choices=("8b", "1p5b"), required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require(
        not args.output_dir.exists(),
        f"immutable native-suite assets exist: {args.output_dir}",
    )
    source_contract = read_json(args.source_contract)
    source_manifest = read_json(args.source_manifest)
    source_gate = read_json(args.source_execution_gate)
    code_receipt = read_json(args.eval_code_receipt)
    export = read_json(args.checkpoint_export)
    source_contract_sha = sha256_file(args.source_contract)
    source_manifest_sha = sha256_file(args.source_manifest)
    require(
        source_contract.get("schema_version")
        == "apertus_full8_native_greek_3cp_contract_v1",
        "source native-suite contract drift",
    )
    require(
        source_manifest.get("contract", {}).get("sha256") == source_contract_sha,
        "source manifest/contract drift",
    )
    examples_path = Path(str(source_manifest["examples"]["path"])).resolve()
    require(
        sha256_file(examples_path) == source_manifest["examples"]["sha256"],
        "frozen native-suite examples drift",
    )
    require(
        source_gate.get("schema_version")
        == "apertus_full8_native_greek_execution_gate_v1"
        and source_gate.get("status") == "passed"
        and source_gate.get("contract_sha256") == source_contract_sha
        and source_gate.get("manifest_sha256") == source_manifest_sha
        and source_gate.get("selected") == SELECTED_PROFILE,
        "source native-suite execution gate drift",
    )
    require(
        code_receipt.get("schema_version") == "native_greek_eval_code_bundle_v1"
        and code_receipt.get("status") == "frozen"
        and source_gate.get("code_tree_sha256") == code_receipt.get("tree_sha256"),
        "source native-suite code/gate drift",
    )
    require(
        export.get("schema_version") == "apertus_hard_h_to_g_checkpoint_export_v1"
        and export.get("status") == "completed"
        and export.get("scale") == args.scale
        and int(export.get("iteration", -1)) == args.iteration
        and export.get("ready_for_frozen_evaluators") is True,
        "study checkpoint export drift",
    )
    model = Path(str(export["hf_export"]["path"])).resolve()
    config = read_json(model / "config.json")
    require(
        int(config.get("vocab_size", -1)) == 148480
        and float(config.get("rope_theta", -1)) == 500000.0
        and int(config.get("max_position_embeddings", -1)) == 4096
        and config.get("tie_word_embeddings") is False,
        "native-suite study model geometry drift",
    )
    tokenizer_sha = sha256_file(model / "tokenizer.json")
    require(tokenizer_sha in TOKENIZER_HASHES, "native-suite study tokenizer drift")

    label = f"{args.scale}_iter_{args.iteration:07d}"
    contract = copy.deepcopy(source_contract)
    contract.update(
        {
            "schema_version": "apertus_hard_h_to_g_native_greek_checkpoint_contract_v1",
            "status": "frozen_except_protipa_access",
            "checkpoint_scope": [
                {
                    "label": label,
                    "iteration": args.iteration,
                    "token_slots": args.iteration * 4_194_304,
                    "model_path": str(model),
                    "evidence_receipt": str(args.checkpoint_export.resolve()),
                    "evidence_receipt_sha256": sha256_file(args.checkpoint_export),
                }
            ],
            "model_contract": {
                "vocab_size": 148480,
                "tokenizer_json_sha256_allowed": TOKENIZER_HASHES,
                "rope_theta": 500000,
                "max_position_embeddings": 4096,
                "tie_word_embeddings": False,
            },
            "rebind_evidence": {
                "source_contract": file_binding(args.source_contract),
                "source_manifest": file_binding(args.source_manifest),
                "source_execution_gate": file_binding(args.source_execution_gate),
                "eval_code_receipt": file_binding(args.eval_code_receipt),
                "checkpoint_export": file_binding(args.checkpoint_export),
                "unchanged_fields": ["scoring", "benchmarks"],
                "changed_fields": [
                    "schema_version",
                    "status",
                    "checkpoint_scope",
                    "model_contract",
                    "rebind_evidence",
                ],
            },
        }
    )
    args.output_dir.mkdir(parents=True)
    contract_path = args.output_dir / "contract.json"
    write_json_atomic(contract_path, contract)
    manifest = copy.deepcopy(source_manifest)
    manifest.update(
        {
            "schema_version": "apertus_hard_h_to_g_native_greek_examples_rebind_v1",
            "contract": file_binding(contract_path),
            "rebind_evidence": {
                "source_manifest": file_binding(args.source_manifest),
                "frozen_examples_unchanged": True,
                "examples_sha256": source_manifest["examples"]["sha256"],
            },
        }
    )
    manifest_path = args.output_dir / "manifest.json"
    write_json_atomic(manifest_path, manifest)
    write_json_atomic(
        args.output_dir / "rebind_receipt.json",
        {
            "schema_version": "apertus_hard_h_to_g_native_greek_rebind_v1",
            "status": "passed",
            "scale": args.scale,
            "iteration": args.iteration,
            "contract": file_binding(contract_path),
            "manifest": file_binding(manifest_path),
            "checkpoint_export": file_binding(args.checkpoint_export),
            "checks": {
                "source_execution_gate_passed": True,
                "scorer_profile_unchanged": True,
                "benchmark_contract_unchanged": contract["benchmarks"]
                == source_contract["benchmarks"],
                "scoring_contract_unchanged": contract["scoring"]
                == source_contract["scoring"],
                "frozen_examples_byte_identical": True,
                "checkpoint_export_valid": True,
                "model_geometry_matches_study_contract": True,
            },
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "scale": args.scale,
                "iteration": args.iteration,
                "label": label,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
