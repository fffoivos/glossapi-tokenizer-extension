#!/usr/bin/env python3
"""Regression tests for the full-8B native-Greek peak-window adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from finalize_peak_window import BENCHMARKS, NEW_LABELS, ORDER


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class BinderTest(unittest.TestCase):
    def test_rebind_preserves_science_and_validates_four_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = b"frozen-tokenizer"
            tokenizer_sha = hashlib.sha256(tokenizer).hexdigest()
            model_contract = {
                "vocab_size": 148992,
                "tokenizer_json_sha256_allowed": [tokenizer_sha],
                "rope_theta": 500000,
                "max_position_embeddings": 4096,
                "tie_word_embeddings": False,
            }
            source_contract = root / "source_contract.json"
            write_json(
                source_contract,
                {
                    "schema_version": "apertus_full8_native_greek_3cp_contract_v1",
                    "model_contract": model_contract,
                    "checkpoint_scope": [],
                    "scoring": {"method": "frozen"},
                    "benchmarks": [{"id": "demosqa"}],
                },
            )
            examples = root / "examples.jsonl"
            examples.write_text(
                '{"benchmark":"demosqa","example_id":"a"}\n'
                '{"benchmark":"demosqa","example_id":"b"}\n',
                encoding="utf-8",
            )
            source_manifest = root / "source_manifest.json"
            write_json(
                source_manifest,
                {
                    "contract": {"sha256": sha256(source_contract)},
                    "examples": {
                        "path": str(examples),
                        "sha256": sha256(examples),
                        "rows": 2,
                    },
                    "counts": {"demosqa": 2},
                },
            )
            source_gate = root / "source_gate.json"
            write_json(
                source_gate,
                {
                    "schema_version": "apertus_full8_native_greek_execution_gate_v1",
                    "status": "passed",
                    "contract_sha256": sha256(source_contract),
                    "manifest_sha256": sha256(source_manifest),
                    "code_tree_sha256": "tree",
                    "selected": {
                        "candidate_batch_size": 1,
                        "dtype": "float32",
                        "example_batch_size": 16,
                        "scorer_mode": "legacy",
                    },
                },
            )
            code_receipt = root / "code_receipt.json"
            write_json(
                code_receipt,
                {
                    "schema_version": "native_greek_eval_code_bundle_v1",
                    "status": "frozen",
                    "tree_sha256": "tree",
                },
            )
            audit = root / "audit.json"
            write_json(
                audit,
                {
                    "schema_version": "greek_benchmark_contamination_audit_v1",
                    "dataset": {
                        "repository_id": "dataset",
                        "revision": "revision",
                    },
                },
            )
            exclusions = root / "exclusions.jsonl"
            exclusions.write_text(
                '{"benchmark":"demosqa","evaluation_unit_id":"a","example_id":"a"}\n',
                encoding="utf-8",
            )

            rows = []
            for iteration in (7152, 8344, 10728, 11920):
                model = root / f"model_{iteration}"
                model.mkdir()
                write_json(
                    model / "config.json",
                    {
                        "vocab_size": 148992,
                        "rope_theta": 500000,
                        "max_position_embeddings": 4096,
                        "tie_word_embeddings": False,
                    },
                )
                (model / "tokenizer.json").write_bytes(tokenizer)
                receipt = root / f"export_{iteration}.json"
                write_json(
                    receipt,
                    {
                        "schema_version": "native_greekmmlu_exact_checkpoint_export_v1",
                        "ready_for_frozen_native_greekmmlu": True,
                        "source": {"iteration": iteration},
                        "hf_export": {
                            "path": str(model),
                            "tokenizer_json_sha256": tokenizer_sha,
                            "geometry": model_contract,
                        },
                    },
                )
                rows.append(
                    {
                        "label": f"iter_{iteration:07d}",
                        "iteration": iteration,
                        "token_slots": iteration * 4_194_304,
                        "model_path": str(model),
                        "export_receipt": str(receipt),
                    }
                )
            bindings = root / "bindings.json"
            write_json(
                bindings,
                {
                    "schema_version": "apertus_full8_native_greek_peak_window_bindings_v1",
                    "status": "frozen",
                    "tokens_per_update": 4_194_304,
                    "best_checkpoint": {
                        "iteration": 9536,
                        "policy": "reuse_authoritative_existing_evaluation",
                    },
                    "contamination_subset": {
                        "policy": "strong_match_only",
                        "training_dataset": "dataset",
                        "training_dataset_revision": "revision",
                        "audit_receipt_sha256": sha256(audit),
                        "exclusions_sha256": sha256(exclusions),
                        "source_scored_examples": 2,
                        "excluded_examples": 1,
                        "retained_examples": 1,
                        "excluded_by_benchmark": {"demosqa": 1},
                        "retained_by_benchmark": {"demosqa": 1},
                    },
                    "checkpoints_to_evaluate": rows,
                },
            )
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "bind_peak_window_native_suite.py"),
                    "--source-contract",
                    str(source_contract),
                    "--source-manifest",
                    str(source_manifest),
                    "--source-execution-gate",
                    str(source_gate),
                    "--eval-code-receipt",
                    str(code_receipt),
                    "--bindings",
                    str(bindings),
                    "--contamination-audit-receipt",
                    str(audit),
                    "--exclusions",
                    str(exclusions),
                    "--output-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            receipt = json.loads((output / "rebind_receipt.json").read_text())
            rebound = json.loads((output / "peak_window_missing4_contract.json").read_text())
            self.assertTrue(all(receipt["checks"].values()))
            self.assertEqual(
                [row["iteration"] for row in receipt["checkpoints"]],
                [7152, 8344, 10728, 11920],
            )
            self.assertEqual(rebound["scoring"], {"method": "frozen"})
            self.assertEqual(rebound["benchmarks"], [{"id": "demosqa"}])
            self.assertEqual(receipt["clean_subset"]["retained_by_benchmark"], {"demosqa": 1})


class FinalizerTest(unittest.TestCase):
    def write_matrix(self, root: Path, labels: list[str]) -> None:
        bindings = []
        for label in labels:
            receipt = root / label / "combined/receipt.json"
            write_json(receipt, {"status": "completed", "model": label})
            metrics = receipt.parent / "metrics.csv"
            self.write_metrics(metrics, accuracy=0.5)
            bindings.append(
                {"model": label, "path": str(receipt), "sha256": sha256(receipt)}
            )
        write_json(
            root / "matrix_receipt.json",
            {
                "schema_version": "legacy_dynamic_checkpoint_matrix",
                "status": "completed",
                "checkpoint_receipts": bindings,
            },
        )

    def write_metrics(self, path: Path, *, accuracy: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = [
            "benchmark",
            "subject",
            "n",
            "accuracy",
            "choice_nll",
            "correct_answer_bpb",
            "binary_macro_f1",
            "balanced_accuracy",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for benchmark in BENCHMARKS:
                writer.writerow(
                    {
                        "benchmark": benchmark,
                        "subject": "__all__",
                        "n": 10,
                        "accuracy": accuracy,
                        "choice_nll": "" if benchmark.endswith("exact_set") else 1.0,
                        "correct_answer_bpb": "" if benchmark.endswith("exact_set") else 0.2,
                        "binary_macro_f1": 0.4,
                        "balanced_accuracy": 0.5,
                    }
                )

    def write_filtered(self, root: Path, labels: list[str]) -> None:
        checkpoints = []
        for label in labels:
            metrics = root / label / "strict_filtered_metrics.csv"
            self.write_metrics(metrics, accuracy=0.6)
            checkpoints.append(
                {
                    "model": label,
                    "strict_filtered_metrics_sha256": sha256(metrics),
                }
            )
        write_json(root / "receipt.json", {"status": "passed", "checkpoints": checkpoints})

    def test_finalizer_orders_new_and_reused_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            new_matrix = root / "new_matrix"
            reference_matrix = root / "reference_matrix"
            reference_filtered = root / "reference_filtered"
            reference_labels = ["iter_0000000", "iter_0009536", "iter_0018284"]
            self.write_matrix(new_matrix, NEW_LABELS)
            self.write_matrix(reference_matrix, reference_labels)
            self.write_filtered(reference_filtered, reference_labels)
            subset_receipt = root / "subset_receipt.json"
            write_json(
                subset_receipt,
                {
                    "status": "passed",
                    "checks": {"clean": True},
                    "clean_subset": {
                        "exclusions": {"rows": 10076},
                        "retained_by_benchmark": {"all": 73894},
                    },
                },
            )
            output = root / "result.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "finalize_peak_window.py"),
                    "--new-matrix-root",
                    str(new_matrix),
                    "--reference-matrix-root",
                    str(reference_matrix),
                    "--reference-filtered-root",
                    str(reference_filtered),
                    "--subset-receipt",
                    str(subset_receipt),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            result = json.loads(output.read_text())
            self.assertEqual(result["checkpoint_order"], ORDER)
            self.assertEqual(
                [row["label"] for row in result["table"]], ORDER
            )
            self.assertEqual(
                result["table"][2]["provenance"],
                "reused_authoritative_3cp_evaluation",
            )
            self.assertEqual(
                result["table"][0]["benchmarks"]["demosqa"]["accuracy"],
                0.5,
            )


if __name__ == "__main__":
    unittest.main()
