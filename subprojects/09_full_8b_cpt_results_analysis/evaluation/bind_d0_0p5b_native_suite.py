#!/usr/bin/env python3
"""Rebind the frozen native-Greek examples and scorer to three D0 0.5B exports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-execution-gate", type=Path, required=True)
    parser.add_argument("--eval-code-receipt", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require(not args.output_dir.exists(), f"refusing to overwrite {args.output_dir}")
    source_contract = read_json(args.source_contract)
    source_manifest = read_json(args.source_manifest)
    source_gate = read_json(args.source_execution_gate)
    code_receipt = read_json(args.eval_code_receipt)
    bindings = read_json(args.bindings)

    source_contract_sha = sha256(args.source_contract)
    source_manifest_sha = sha256(args.source_manifest)
    require(
        source_contract.get("schema_version") == "apertus_full8_native_greek_3cp_contract_v1",
        "source native-Greek contract drift",
    )
    require(
        source_manifest.get("contract", {}).get("sha256") == source_contract_sha,
        "source example manifest is not bound to the source contract",
    )
    examples_path = Path(source_manifest["examples"]["path"])
    require(sha256(examples_path) == source_manifest["examples"]["sha256"], "frozen example hash drift")
    require(
        source_gate.get("schema_version") == "apertus_full8_native_greek_execution_gate_v1"
        and source_gate.get("status") == "passed",
        "source execution gate is not passed",
    )
    require(source_gate.get("contract_sha256") == source_contract_sha, "execution gate/contract drift")
    require(source_gate.get("manifest_sha256") == source_manifest_sha, "execution gate/manifest drift")
    require(
        source_gate.get("selected")
        == {
            "candidate_batch_size": 1,
            "dtype": "float32",
            "example_batch_size": 16,
            "scorer_mode": "legacy",
        },
        "authoritative scorer profile drift",
    )
    require(
        code_receipt.get("schema_version") == "native_greek_eval_code_bundle_v1"
        and code_receipt.get("status") == "frozen"
        and source_gate.get("code_tree_sha256") == code_receipt.get("tree_sha256"),
        "execution gate/evaluation code drift",
    )
    require(
        bindings.get("schema_version") == "apertus_d0_0p5b_native_greek_checkpoint_bindings_v1"
        and bindings.get("status") == "frozen",
        "D0 checkpoint binding drift",
    )

    rows = bindings.get("checkpoints", [])
    require([row.get("iteration") for row in rows] == [0, 18944, 38496], "D0 checkpoint selection drift")
    tokens_per_update = int(bindings["tokens_per_update"])
    expected_geometry = bindings["model_contract"]
    checkpoint_scope = []
    for row in rows:
        iteration = int(row["iteration"])
        require(int(row["token_slots"]) == iteration * tokens_per_update, "D0 token-slot arithmetic drift")
        model_path = Path(row["model_path"])
        export_receipt_path = Path(row["export_receipt"])
        require(model_path.is_dir() and export_receipt_path.is_file(), f"missing D0 checkpoint inputs: {row['label']}")
        export = read_json(export_receipt_path)
        require(
            export.get("schema_version") == "native_greekmmlu_exact_checkpoint_export_v1"
            and export.get("status") == "completed"
            and export.get("ready_for_frozen_native_greekmmlu") is True,
            f"invalid D0 export receipt: {row['label']}",
        )
        require(int(export.get("source", {}).get("iteration", -1)) == iteration, f"iteration drift: {row['label']}")
        require(Path(export["hf_export"]["path"]).resolve() == model_path.resolve(), f"model path drift: {row['label']}")
        geometry = export["hf_export"]["geometry"]
        for key in (
            "vocab_size",
            "rope_theta",
            "tie_word_embeddings",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
        ):
            require(geometry.get(key) == expected_geometry[key], f"{key} drift: {row['label']}")
        require(
            export["hf_export"]["tokenizer_json_sha256"]
            in expected_geometry["tokenizer_json_sha256_allowed"],
            f"tokenizer drift: {row['label']}",
        )
        config = read_json(model_path / "config.json")
        require(int(config["max_position_embeddings"]) == 4096, f"context geometry drift: {row['label']}")
        checkpoint_scope.append(
            {
                "label": row["label"],
                "iteration": iteration,
                "token_slots": int(row["token_slots"]),
                "model_path": str(model_path.resolve()),
                "evidence_receipt": str(export_receipt_path.resolve()),
                "evidence_receipt_sha256": sha256(export_receipt_path),
            }
        )

    contract = copy.deepcopy(source_contract)
    contract["schema_version"] = "apertus_d0_0p5b_native_greek_3cp_contract_v1"
    contract["status"] = "frozen_except_protipa_access"
    contract["checkpoint_scope"] = checkpoint_scope
    contract["model_contract"] = copy.deepcopy(expected_geometry)
    contract["rebind_evidence"] = {
        "source_contract": {"path": str(args.source_contract.resolve()), "sha256": source_contract_sha},
        "source_manifest": {"path": str(args.source_manifest.resolve()), "sha256": source_manifest_sha},
        "source_execution_gate": {
            "path": str(args.source_execution_gate.resolve()),
            "sha256": sha256(args.source_execution_gate),
        },
        "eval_code_receipt": {"path": str(args.eval_code_receipt.resolve()), "sha256": sha256(args.eval_code_receipt)},
        "bindings": {"path": str(args.bindings.resolve()), "sha256": sha256(args.bindings)},
        "unchanged_fields": ["scoring", "benchmarks"],
        "changed_fields": ["schema_version", "status", "checkpoint_scope", "model_contract", "rebind_evidence"],
    }

    args.output_dir.mkdir(parents=True)
    contract_path = args.output_dir / "d0_0p5b_native_greek_3cp_contract.json"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    manifest = copy.deepcopy(source_manifest)
    manifest["schema_version"] = "apertus_d0_0p5b_native_greek_frozen_examples_rebind_v1"
    manifest["contract"] = {"path": str(contract_path.resolve()), "sha256": sha256(contract_path)}
    manifest["rebind_evidence"] = {
        "source_manifest": {"path": str(args.source_manifest.resolve()), "sha256": source_manifest_sha},
        "frozen_examples_unchanged": True,
        "examples_sha256": source_manifest["examples"]["sha256"],
    }
    manifest_path = args.output_dir / "d0_0p5b_frozen_examples_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    receipt = {
        "schema_version": "apertus_d0_0p5b_native_greek_rebind_receipt_v1",
        "status": "passed",
        "contract": {"path": str(contract_path.resolve()), "sha256": sha256(contract_path)},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "checkpoints": checkpoint_scope,
        "checks": {
            "source_execution_gate_passed": True,
            "scorer_profile_unchanged": True,
            "benchmark_contract_unchanged": contract["benchmarks"] == source_contract["benchmarks"],
            "scoring_contract_unchanged": contract["scoring"] == source_contract["scoring"],
            "frozen_examples_byte_identical": True,
            "all_checkpoint_export_receipts_valid": True,
            "all_model_geometries_match_d0_contract": True,
        },
    }
    (args.output_dir / "rebind_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"ok": True, "checkpoints": len(checkpoint_scope), "examples": source_manifest["examples"]["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
