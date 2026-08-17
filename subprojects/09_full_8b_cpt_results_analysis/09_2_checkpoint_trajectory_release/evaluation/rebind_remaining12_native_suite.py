#!/usr/bin/env python3
"""Bind the existing clean native-Greek population to twelve saved 8B exports."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ITERATIONS = [400, 1192, 2384, 3576, 4768, 5960, 13112, 14304, 14627, 15496, 16688, 17880]
EXPECTED_PROFILE = {
    "candidate_batch_size": 1,
    "dtype": "float32",
    "example_batch_size": 16,
    "scorer_mode": "legacy",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def source_assets(receipt_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    receipt = read_json(receipt_path)
    require(
        receipt.get("schema_version") == "apertus_full8_native_greek_peak_window_rebind_v1"
        and receipt.get("status") == "passed"
        and all(receipt.get("checks", {}).values()),
        "source clean-subset receipt is not passing",
    )
    contract_path = Path(receipt["contract"]["path"])
    manifest_path = Path(receipt["manifest"]["path"])
    require(sha256(contract_path) == receipt["contract"]["sha256"], "source contract hash drift")
    require(sha256(manifest_path) == receipt["manifest"]["sha256"], "source manifest hash drift")
    contract = read_json(contract_path)
    manifest = read_json(manifest_path)
    require(
        manifest.get("contract", {}).get("sha256") == sha256(contract_path),
        "source clean manifest contract drift",
    )
    examples_path = Path(manifest["examples"]["path"])
    require(
        examples_path.is_file() and sha256(examples_path) == manifest["examples"]["sha256"],
        "source clean examples drift",
    )
    return receipt, contract_path, contract, manifest_path, manifest


def verify_checkpoint(row: dict[str, Any], run_root: Path, model_contract: dict[str, Any]) -> dict[str, Any]:
    label = str(row["label"])
    iteration = int(row["iteration"])
    attempt = int(row["attempt"])
    export_root = run_root / "checkpoint_evaluations" / label / f"attempt_{attempt}" / "export"
    export_receipt = export_root / "checkpoint_eval_export_receipt.json"
    model_path = export_root / "hf"
    require(model_path.is_dir() and export_receipt.is_file(), f"missing checkpoint export: {label}")
    export = read_json(export_receipt)
    require(
        export.get("schema_version") == "native_greekmmlu_exact_checkpoint_export_v1"
        and export.get("ready_for_frozen_native_greekmmlu") is True,
        f"invalid checkpoint export receipt: {label}",
    )
    require(int(export.get("source", {}).get("iteration", -1)) == iteration, f"iteration drift: {label}")
    hf_export = export.get("hf_export", {})
    require(Path(hf_export.get("path", "")).resolve() == model_path.resolve(), f"HF path drift: {label}")
    geometry = hf_export.get("geometry", {})
    for key in ("vocab_size", "rope_theta", "max_position_embeddings", "tie_word_embeddings"):
        require(geometry.get(key) == model_contract[key], f"export {key} drift: {label}")
    tokenizer_hash = str(hf_export.get("tokenizer_json_sha256", ""))
    require(
        tokenizer_hash in model_contract["tokenizer_json_sha256_allowed"],
        f"export tokenizer drift: {label}",
    )
    config_path = model_path / "config.json"
    tokenizer_path = model_path / "tokenizer.json"
    require(config_path.is_file() and tokenizer_path.is_file(), f"checkpoint files missing: {label}")
    config = read_json(config_path)
    for key in ("vocab_size", "rope_theta", "max_position_embeddings", "tie_word_embeddings"):
        require(config.get(key) == model_contract[key], f"live {key} drift: {label}")
    require(sha256(tokenizer_path) == tokenizer_hash, f"live tokenizer drift: {label}")
    return {
        "label": label,
        "iteration": iteration,
        "token_slots": int(row["token_slots"]),
        "model_path": str(model_path.resolve()),
        "evidence_receipt": str(export_receipt.resolve()),
        "evidence_receipt_sha256": sha256(export_receipt),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output_dir.exists(), f"refusing to overwrite {args.output_dir}")
    bindings = read_json(args.bindings)
    require(
        bindings.get("schema_version") == "apertus_full8_native_greek_remaining12_bindings_v1"
        and bindings.get("status") == "frozen",
        "remaining checkpoint bindings drift",
    )
    rows = bindings.get("checkpoints_to_evaluate")
    require(isinstance(rows, list), "remaining checkpoint rows are missing")
    require([int(row["iteration"]) for row in rows] == EXPECTED_ITERATIONS, "checkpoint selection drift")
    tokens_per_update = int(bindings["tokens_per_update"])
    for row in rows:
        require(
            int(row["token_slots"]) == int(row["iteration"]) * tokens_per_update,
            f"token-slot arithmetic drift: {row['label']}",
        )

    subset = bindings["source_clean_subset"]
    source_receipt_path = Path(subset["rebind_receipt"])
    source_receipt, source_contract_path, source_contract, source_manifest_path, source_manifest = source_assets(
        source_receipt_path
    )
    clean_subset = source_manifest.get("clean_subset", {})
    require(
        clean_subset.get("exclusions", {}).get("sha256") == subset["exclusions_sha256"],
        "clean exclusion identity drift",
    )
    require(
        clean_subset.get("contamination_audit_receipt", {}).get("sha256")
        == subset["audit_receipt_sha256"],
        "clean audit identity drift",
    )
    require(
        sum(source_manifest.get("counts", {}).values()) == int(subset["retained_examples"])
        and int(clean_subset.get("exclusions", {}).get("rows", -1)) == int(subset["excluded_examples"]),
        "clean population count drift",
    )
    require(source_receipt.get("clean_subset") == clean_subset, "clean receipt/manifest drift")
    require(source_contract.get("scoring", {}).get("dtype_policy"), "source scorer contract is missing")
    source_gate = source_contract.get("rebind_evidence", {}).get("source_execution_gate", {})
    require(bool(source_gate.get("sha256")), "source execution gate binding is absent")

    checkpoint_scope = [
        verify_checkpoint(row, args.run_root, source_contract["model_contract"]) for row in rows
    ]
    contract = copy.deepcopy(source_contract)
    contract["schema_version"] = "apertus_full8_native_greek_remaining12_contract_v1"
    contract["checkpoint_scope"] = checkpoint_scope
    contract["remaining12"] = {"scored_here": EXPECTED_ITERATIONS, "tokens_per_update": tokens_per_update}
    contract["rebind_evidence"] = {
        "source_clean_rebind_receipt": {
            "path": str(source_receipt_path.resolve()),
            "sha256": sha256(source_receipt_path),
        },
        "source_clean_contract": {
            "path": str(source_contract_path.resolve()),
            "sha256": sha256(source_contract_path),
        },
        "source_clean_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": sha256(source_manifest_path),
        },
        "bindings": {"path": str(args.bindings.resolve()), "sha256": sha256(args.bindings)},
        "unchanged_fields": ["model_contract", "scoring", "benchmarks", "clean_subset"],
        "changed_fields": ["schema_version", "checkpoint_scope", "remaining12", "rebind_evidence"],
    }

    args.output_dir.mkdir(parents=True)
    contract_path = args.output_dir / "remaining12_contract.json"
    write_json(contract_path, contract)
    manifest = copy.deepcopy(source_manifest)
    manifest["schema_version"] = "apertus_full8_native_greek_remaining12_manifest_v1"
    manifest["contract"] = {"path": str(contract_path.resolve()), "sha256": sha256(contract_path)}
    manifest["rebind_evidence"] = {
        "source_clean_manifest": {
            "path": str(source_manifest_path.resolve()),
            "sha256": sha256(source_manifest_path),
        },
        "examples_byte_identical": True,
        "examples_sha256": source_manifest["examples"]["sha256"],
    }
    manifest_path = args.output_dir / "remaining12_manifest.json"
    write_json(manifest_path, manifest)
    receipt = {
        "schema_version": "apertus_full8_native_greek_remaining12_rebind_v1",
        "status": "passed",
        "contract": {"path": str(contract_path.resolve()), "sha256": sha256(contract_path)},
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256(manifest_path)},
        "checkpoints": checkpoint_scope,
        "clean_subset": clean_subset,
        "scorer_profile": EXPECTED_PROFILE,
        "checks": {
            "source_clean_subset_receipt_passed": True,
            "source_clean_examples_byte_identical": True,
            "contamination_identity_unchanged": True,
            "scorer_contract_unchanged": True,
            "all_checkpoint_exports_valid": True,
            "all_model_geometries_match": True,
        },
    }
    receipt_path = args.output_dir / "rebind_receipt.json"
    write_json(receipt_path, receipt)
    print(json.dumps({"ok": True, "checkpoints": len(checkpoint_scope), "rows": 73894}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
