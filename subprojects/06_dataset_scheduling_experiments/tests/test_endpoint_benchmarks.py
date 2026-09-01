from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EndpointBenchmarkTests(unittest.TestCase):
    def test_demosqa_four_choice_serialization(self) -> None:
        path = ROOT / "evaluation" / "run_greek_endpoint_mcq_eval.py"
        spec = importlib.util.spec_from_file_location("endpoint_eval_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        value = 'Α. "πρώτη"\n\nΒ. "δεύτερη"\n\nΓ. "τρίτη"\n\nΔ."τέταρτη"'
        self.assertEqual(
            module.demos_choices(value), ["πρώτη", "δεύτερη", "τρίτη", "τέταρτη"]
        )

    def test_dataset_revisions_are_pinned(self) -> None:
        text = (ROOT / "evaluation" / "endpoint_benchmark_contract.json").read_text()
        self.assertIn("7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", text)
        self.assertIn("c06a8fe1f672eb57d011e52831586b2713ab956f", text)

    def test_greek_endpoint_uses_frozen_evaluation_venv(self) -> None:
        text = (ROOT / "clariden" / "run_greek_endpoint_wave.sbatch").read_text()
        self.assertIn("native_greek_eval_py312_aug2026", text)
        self.assertIn("uenv run pytorch/v2.9.1:v2 --view=default -- test -x", text)
        self.assertIn('"$EVAL_VENV/bin/python" "$runner"', text)
        self.assertNotIn('PYTHONPATH="$compat" python3 "$runner"', text)

    def test_retention_recovery_is_offline_and_cache_receipted(self) -> None:
        wave = (ROOT / "clariden" / "run_retention_endpoint_wave.sbatch").read_text()
        worker = (ROOT / "clariden" / "run_retention_endpoint_one.sh").read_text()
        freezer = (ROOT / "evaluation" / "freeze_retention_dataset_cache.py").read_text()
        self.assertIn("RETENTION_DATASET_CACHE_RECEIPT", wave)
        self.assertIn("HF_HUB_OFFLINE=1", wave)
        self.assertIn("HF_DATASETS_OFFLINE=1", worker)
        self.assertIn("dataset fingerprint missing", freezer)
        self.assertIn('"global_mmlu"', freezer)

    def test_retention_shards_preserve_the_frozen_wave(self) -> None:
        shard = (ROOT / "clariden" / "run_retention_endpoint_shard.sbatch").read_text()
        finalizer = (ROOT / "clariden" / "finalize_retention_endpoint_shards.sbatch").read_text()
        self.assertIn("SHARD_ARMS", shard)
        self.assertIn("HF_HUB_OFFLINE=1", shard)
        self.assertIn("HF_DATASETS_OFFLINE=1", shard)
        self.assertIn("--expected-receipt", shard)
        self.assertIn("finalize_retention_endpoint_shard.py", shard)
        self.assertIn("finalize_retention_endpoint_wave.py", finalizer)

    def test_retention_shard_receipt_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks = []
            for arm in ("initial_shared", "D0_mixed"):
                output = root / "tasks" / arm
                output.mkdir(parents=True)
                (output / "receipt.json").write_text(
                    json.dumps(
                        {
                            "schema_version": "apertus_mini_retention_endpoint_v1",
                            "status": "completed",
                        }
                    )
                )
                tasks.append({"arm_id": arm, "output_root": str(output)})
            wave = root / "wave.json"
            wave.write_text(json.dumps({"tasks": tasks}))
            output = root / "shard.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "evaluation" / "finalize_retention_endpoint_shard.py"),
                    "--wave-manifest",
                    str(wave),
                    "--arm",
                    "initial_shared",
                    "--arm",
                    "D0_mixed",
                    "--output",
                    str(output),
                ],
                check=True,
            )
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(
                [row["model_label"] for row in receipt["models"]],
                ["initial_shared", "D0_mixed"],
            )

    def test_completion_supports_an_in_run_recovery_receipt(self) -> None:
        finalizer = (ROOT / "production" / "finalize_campaign_evidence.py").read_text()
        wrapper = (ROOT / "clariden" / "finalize_campaign_evidence.sbatch").read_text()
        self.assertIn('"--greek-endpoint-receipt"', finalizer)
        self.assertIn("must remain inside the run root", finalizer)
        self.assertIn('"--full-endpoint-validation-receipt"', finalizer)
        self.assertIn('"--retention-endpoint-receipt"', finalizer)
        self.assertIn("GREEK_ENDPOINT_RECEIPT", wrapper)
        self.assertIn("FULL_ENDPOINT_VALIDATION_RECEIPT", wrapper)
        self.assertIn("RETENTION_ENDPOINT_RECEIPT", wrapper)

    def test_skip_train_does_not_construct_an_exhausted_schedule(self) -> None:
        text = (ROOT / "training" / "pretrain_scheduled_gpt.py").read_text()
        self.assertIn("if args.skip_train:", text)
        self.assertIn("return None, None, None", text)
        full = (ROOT / "clariden" / "run_full_endpoint_validation.sbatch").read_text()
        self.assertIn("RECOVERY_SCIENTIFIC_BUNDLE_RECEIPT", full)
        self.assertIn("runtime-scientific-bundle-receipt", full)


if __name__ == "__main__":
    unittest.main()
