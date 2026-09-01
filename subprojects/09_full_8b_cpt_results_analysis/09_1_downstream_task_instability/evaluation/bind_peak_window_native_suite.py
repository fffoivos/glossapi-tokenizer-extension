#!/usr/bin/env python3
"""Rebind frozen native-Greek examples to four full-8B peak-window exports."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ITERATIONS = [7152, 8344, 10728, 11920]
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-execution-gate", type=Path, required=True)
    parser.add_argument("--eval-code-receipt", type=Path, required=True)
    parser.add_argument("--bindings", type=Path, required=True)
    parser.add_argument("--contamination-audit-receipt", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    require(not args.output_dir.exists(), f"refusing to overwrite {args.output_dir}")
    source_contract = read_json(args.source_contract)
    source_manifest = read_json(args.source_manifest)
    source_gate = read_json(args.source_execution_gate)
    code_receipt = read_json(args.eval_code_receipt)
    bindings = read_json(args.bindings)
    contamination_audit = read_json(args.contamination_audit_receipt)

    source_contract_sha = sha256(args.source_contract)
    source_manifest_sha = sha256(args.source_manifest)
    require(
        source_contract.get("schema_version")
        == "apertus_full8_native_greek_3cp_contract_v1",
        "source native-Greek contract drift",
    )
    require(
        source_manifest.get("contract", {}).get("sha256") == source_contract_sha,
        "source example manifest is not bound to the source contract",
    )
    examples_path = Path(source_manifest["examples"]["path"])
    require(
        sha256(examples_path) == source_manifest["examples"]["sha256"],
        "frozen example hash drift",
    )
    require(
        source_gate.get("schema_version")
        == "apertus_full8_native_greek_execution_gate_v1"
        and source_gate.get("status") == "passed",
        "source execution gate is not passed",
    )
    require(
        source_gate.get("contract_sha256") == source_contract_sha,
        "execution gate/contract drift",
    )
    require(
        source_gate.get("manifest_sha256") == source_manifest_sha,
        "execution gate/manifest drift",
    )
    require(
        source_gate.get("selected") == EXPECTED_PROFILE,
        "authoritative scorer profile drift",
    )
    require(
        code_receipt.get("schema_version") == "native_greek_eval_code_bundle_v1"
        and code_receipt.get("status") == "frozen"
        and source_gate.get("code_tree_sha256") == code_receipt.get("tree_sha256"),
        "execution gate/evaluation code drift",
    )
    require(
        bindings.get("schema_version")
        == "apertus_full8_native_greek_peak_window_bindings_v1"
        and bindings.get("status") == "frozen",
        "peak-window checkpoint binding drift",
    )
    subset = bindings.get("contamination_subset", {})
    require(
        sha256(args.contamination_audit_receipt)
        == subset.get("audit_receipt_sha256"),
        "contamination audit receipt hash drift",
    )
    require(
        sha256(args.exclusions) == subset.get("exclusions_sha256"),
        "contamination exclusion hash drift",
    )
    require(
        contamination_audit.get("schema_version")
        == "greek_benchmark_contamination_audit_v1"
        and contamination_audit.get("dataset", {}).get("repository_id")
        == subset.get("training_dataset")
        and contamination_audit.get("dataset", {}).get("revision")
        == subset.get("training_dataset_revision"),
        "contamination audit dataset identity drift",
    )

    rows = bindings.get("checkpoints_to_evaluate", [])
    require(
        [row.get("iteration") for row in rows] == EXPECTED_ITERATIONS,
        "peak-window checkpoint selection drift",
    )
    best = bindings.get("best_checkpoint", {})
    require(
        best.get("iteration") == 9536
        and best.get("policy") == "reuse_authoritative_existing_evaluation",
        "best-checkpoint reuse policy drift",
    )
    tokens_per_update = int(bindings["tokens_per_update"])
    expected_geometry = source_contract["model_contract"]
    checkpoint_scope = []
    for row in rows:
        iteration = int(row["iteration"])
        require(
            int(row["token_slots"]) == iteration * tokens_per_update,
            f"token-slot arithmetic drift: {row['label']}",
        )
        model_path = Path(row["model_path"])
        export_receipt_path = Path(row["export_receipt"])
        require(
            model_path.is_dir() and export_receipt_path.is_file(),
            f"missing checkpoint inputs: {row['label']}",
        )
        export = read_json(export_receipt_path)
        require(
            export.get("schema_version")
            == "native_greekmmlu_exact_checkpoint_export_v1"
            and export.get("ready_for_frozen_native_greekmmlu") is True,
            f"invalid export receipt: {row['label']}",
        )
        require(
            int(export.get("source", {}).get("iteration", -1)) == iteration,
            f"iteration drift: {row['label']}",
        )
        hf_export = export["hf_export"]
        require(
            Path(hf_export["path"]).resolve() == model_path.resolve(),
            f"model path drift: {row['label']}",
        )
        geometry = hf_export["geometry"]
        for key in (
            "vocab_size",
            "rope_theta",
            "max_position_embeddings",
            "tie_word_embeddings",
        ):
            require(
                geometry.get(key) == expected_geometry[key],
                f"{key} drift: {row['label']}",
            )
        require(
            hf_export["tokenizer_json_sha256"]
            in expected_geometry["tokenizer_json_sha256_allowed"],
            f"tokenizer drift: {row['label']}",
        )
        config_path = model_path / "config.json"
        tokenizer_path = model_path / "tokenizer.json"
        require(
            config_path.is_file() and tokenizer_path.is_file(),
            f"missing HF config/tokenizer: {row['label']}",
        )
        config = read_json(config_path)
        for key in (
            "vocab_size",
            "rope_theta",
            "max_position_embeddings",
            "tie_word_embeddings",
        ):
            require(
                config.get(key) == expected_geometry[key],
                f"live config {key} drift: {row['label']}",
            )
        require(
            sha256(tokenizer_path) == hf_export["tokenizer_json_sha256"],
            f"live tokenizer hash drift: {row['label']}",
        )
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
    contract["schema_version"] = (
        "apertus_full8_native_greek_peak_window_missing4_contract_v1"
    )
    contract["checkpoint_scope"] = checkpoint_scope
    contract["peak_window"] = {
        "best_checkpoint": best,
        "complete_order": [7152, 8344, 9536, 10728, 11920],
        "scored_here": EXPECTED_ITERATIONS,
    }
    contract["rebind_evidence"] = {
        "source_contract": {
            "path": str(args.source_contract.resolve()),
            "sha256": source_contract_sha,
        },
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": source_manifest_sha,
        },
        "source_execution_gate": {
            "path": str(args.source_execution_gate.resolve()),
            "sha256": sha256(args.source_execution_gate),
        },
        "eval_code_receipt": {
            "path": str(args.eval_code_receipt.resolve()),
            "sha256": sha256(args.eval_code_receipt),
        },
        "bindings": {
            "path": str(args.bindings.resolve()),
            "sha256": sha256(args.bindings),
        },
        "unchanged_fields": ["model_contract", "scoring", "benchmarks"],
        "changed_fields": [
            "schema_version",
            "checkpoint_scope",
            "peak_window",
            "rebind_evidence",
        ],
    }

    args.output_dir.mkdir(parents=True)
    exclusion_rows = [
        json.loads(line)
        for line in args.exclusions.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exclusions = {
        (str(row["benchmark"]), str(row["example_id"]))
        for row in exclusion_rows
    }
    require(
        len(exclusions) == len(exclusion_rows) == int(subset["excluded_examples"]),
        "contamination exclusion identities are duplicated or incomplete",
    )
    source_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    retained_counts: Counter[str] = Counter()
    observed_exclusions: set[tuple[str, str]] = set()
    clean_examples_path = args.output_dir / "clean_examples.jsonl"
    with examples_path.open(encoding="utf-8") as source, clean_examples_path.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            row = json.loads(line)
            key = (str(row["benchmark"]), str(row["example_id"]))
            source_counts[key[0]] += 1
            if key in exclusions:
                observed_exclusions.add(key)
                excluded_counts[key[0]] += 1
                continue
            retained_counts[key[0]] += 1
            target.write(line if line.endswith("\n") else line + "\n")
    require(
        observed_exclusions == exclusions,
        "one or more audited exclusion identities are absent from frozen examples",
    )
    require(
        sum(source_counts.values()) == int(subset["source_scored_examples"])
        and sum(excluded_counts.values()) == int(subset["excluded_examples"])
        and sum(retained_counts.values()) == int(subset["retained_examples"]),
        "clean-subset total counts drift",
    )
    require(
        dict(sorted(excluded_counts.items())) == subset["excluded_by_benchmark"],
        "clean-subset excluded benchmark counts drift",
    )
    require(
        dict(sorted(retained_counts.items())) == subset["retained_by_benchmark"],
        "clean-subset retained benchmark counts drift",
    )
    contract_path = args.output_dir / "peak_window_missing4_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = copy.deepcopy(source_manifest)
    manifest["schema_version"] = (
        "apertus_full8_native_greek_peak_window_examples_rebind_v1"
    )
    manifest["contract"] = {
        "path": str(contract_path.resolve()),
        "sha256": sha256(contract_path),
    }
    manifest["examples"] = {
        "path": str(clean_examples_path.resolve()),
        "sha256": sha256(clean_examples_path),
        "rows": sum(retained_counts.values()),
    }
    manifest["counts"] = dict(sorted(retained_counts.items()))
    manifest["clean_subset"] = {
        "policy": subset["policy"],
        "source_examples": {
            "path": str(examples_path.resolve()),
            "sha256": source_manifest["examples"]["sha256"],
            "rows": sum(source_counts.values()),
        },
        "contamination_audit_receipt": {
            "path": str(args.contamination_audit_receipt.resolve()),
            "sha256": sha256(args.contamination_audit_receipt),
        },
        "exclusions": {
            "path": str(args.exclusions.resolve()),
            "sha256": sha256(args.exclusions),
            "rows": len(exclusions),
            "counts_by_benchmark": dict(sorted(excluded_counts.items())),
        },
        "retained_by_benchmark": dict(sorted(retained_counts.items())),
    }
    manifest["rebind_evidence"] = {
        "source_manifest": {
            "path": str(args.source_manifest.resolve()),
            "sha256": source_manifest_sha,
        },
        "source_examples_unchanged": True,
        "source_examples_sha256": source_manifest["examples"]["sha256"],
        "only_audited_strong_match_identities_removed": True,
    }
    manifest_path = args.output_dir / "peak_window_examples_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = {
        "schema_version": "apertus_full8_native_greek_peak_window_rebind_v1",
        "status": "passed",
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": sha256(contract_path),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256(manifest_path),
        },
        "checkpoints": checkpoint_scope,
        "clean_subset": manifest["clean_subset"],
        "checks": {
            "source_execution_gate_passed": True,
            "scorer_profile_unchanged": True,
            "benchmark_contract_unchanged": (
                contract["benchmarks"] == source_contract["benchmarks"]
            ),
            "scoring_contract_unchanged": (
                contract["scoring"] == source_contract["scoring"]
            ),
            "model_contract_unchanged": (
                contract["model_contract"] == source_contract["model_contract"]
            ),
            "source_examples_byte_identical": True,
            "clean_subset_exactly_matches_audit_exclusions": True,
            "all_checkpoint_export_receipts_valid": True,
            "all_model_geometries_match_source_contract": True,
        },
    }
    receipt_path = args.output_dir / "rebind_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "checkpoints": len(checkpoint_scope),
                "examples": manifest["examples"]["rows"],
                "excluded": len(exclusions),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
