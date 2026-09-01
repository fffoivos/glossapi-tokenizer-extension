#!/usr/bin/env python3
"""Regression coverage for the remaining-twelve clean-suite binder."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "rebind_remaining12_native_suite.py"
BINDINGS = ROOT / "remaining12_checkpoint_bindings.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class RebindRemainingTwelveTest(unittest.TestCase):
    def test_binder_preserves_clean_population_and_checks_all_exports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bindings = json.loads(BINDINGS.read_text())
            tokenizer = b"frozen-extension-tokenizer"
            tokenizer_sha = hashlib.sha256(tokenizer).hexdigest()
            model_contract = {
                "vocab_size": 148992,
                "rope_theta": 500000,
                "max_position_embeddings": 4096,
                "tie_word_embeddings": False,
                "tokenizer_json_sha256_allowed": [tokenizer_sha],
            }
            examples = root / "clean_examples.jsonl"
            examples.write_text('{"benchmark":"demosqa","example_id":"a"}\n', encoding="utf-8")
            source_contract = root / "source_contract.json"
            write_json(
                source_contract,
                {
                    "model_contract": model_contract,
                    "checkpoint_scope": [],
                    "scoring": {"dtype_policy": "float32"},
                    "benchmarks": [{"id": "demosqa"}],
                    "rebind_evidence": {"source_execution_gate": {"sha256": "gate"}},
                },
            )
            subset = {
                "exclusions": {
                    "rows": bindings["source_clean_subset"]["excluded_examples"],
                    "sha256": bindings["source_clean_subset"]["exclusions_sha256"],
                },
                "contamination_audit_receipt": {
                    "sha256": bindings["source_clean_subset"]["audit_receipt_sha256"],
                },
                "retained_by_benchmark": {"demosqa": bindings["source_clean_subset"]["retained_examples"]},
            }
            source_manifest = root / "source_manifest.json"
            write_json(
                source_manifest,
                {
                    "contract": {"sha256": sha256(source_contract)},
                    "examples": {"path": str(examples), "sha256": sha256(examples)},
                    "counts": {"demosqa": bindings["source_clean_subset"]["retained_examples"]},
                    "clean_subset": subset,
                },
            )
            source_receipt = root / "source_rebind_receipt.json"
            write_json(
                source_receipt,
                {
                    "schema_version": "apertus_full8_native_greek_peak_window_rebind_v1",
                    "status": "passed",
                    "checks": {"clean": True},
                    "contract": {"path": str(source_contract), "sha256": sha256(source_contract)},
                    "manifest": {"path": str(source_manifest), "sha256": sha256(source_manifest)},
                    "clean_subset": subset,
                },
            )
            run_root = root / "run"
            for row in bindings["checkpoints_to_evaluate"]:
                export = run_root / "checkpoint_evaluations" / row["label"] / f"attempt_{row['attempt']}" / "export"
                model = export / "hf"
                model.mkdir(parents=True)
                write_json(model / "config.json", {key: model_contract[key] for key in model_contract if key != "tokenizer_json_sha256_allowed"})
                (model / "tokenizer.json").write_bytes(tokenizer)
                write_json(
                    export / "checkpoint_eval_export_receipt.json",
                    {
                        "schema_version": "native_greekmmlu_exact_checkpoint_export_v1",
                        "ready_for_frozen_native_greekmmlu": True,
                        "source": {"iteration": row["iteration"]},
                        "hf_export": {
                            "path": str(model),
                            "geometry": {key: model_contract[key] for key in ("vocab_size", "rope_theta", "max_position_embeddings", "tie_word_embeddings")},
                            "tokenizer_json_sha256": tokenizer_sha,
                        },
                    },
                )
            local_bindings = dict(bindings)
            local_bindings["source_clean_subset"] = dict(bindings["source_clean_subset"])
            local_bindings["source_clean_subset"]["rebind_receipt"] = str(source_receipt)
            local_bindings_path = root / "bindings.json"
            write_json(local_bindings_path, local_bindings)
            output = root / "output"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--bindings", str(local_bindings_path),
                    "--run-root", str(run_root),
                    "--output-dir", str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            receipt = json.loads((output / "rebind_receipt.json").read_text())
            contract = json.loads((output / "remaining12_contract.json").read_text())
            manifest = json.loads((output / "remaining12_manifest.json").read_text())
            self.assertTrue(all(receipt["checks"].values()))
            self.assertEqual(len(receipt["checkpoints"]), 12)
            self.assertEqual([row["iteration"] for row in receipt["checkpoints"]], [row["iteration"] for row in bindings["checkpoints_to_evaluate"]])
            self.assertEqual(contract["scoring"], {"dtype_policy": "float32"})
            self.assertEqual(manifest["examples"]["sha256"], sha256(examples))

    def test_slurm_logs_cannot_mutate_the_frozen_wrapper(self) -> None:
        for name in ("freeze_and_preflight_remaining12.sbatch", "run_remaining12_native_segment.sbatch"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --output=/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/logs/", text)
            self.assertIn("#SBATCH --error=/iopsstor/scratch/cscs/fffoivos/evals/full8_remaining12_checkpoint_release_20260817/logs/", text)

    def test_segment_can_attach_to_a_held_salloc_step(self) -> None:
        for name in ("freeze_and_preflight_remaining12.sbatch", "run_remaining12_native_segment.sbatch"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("SLURM_STEP_NUM_NODES", text)


if __name__ == "__main__":
    unittest.main()
