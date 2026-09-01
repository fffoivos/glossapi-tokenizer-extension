from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PANELS = (
    "code",
    "de",
    "english",
    "math",
    "ru",
    "zh",
    "greek_phd",
    "historical_polytonic",
    "hplt",
    "non_hplt",
    "neutral_external_modern_greek",
    "openarchives",
    "old_greek",
)


def validation_lines(iteration: int, loss: float) -> list[str]:
    return [
        (
            f"validation loss at iteration {iteration} on validation set [{name}] | "
            f"lm loss value: {loss:.6E} | base-token target loss value: {loss:.6E} | "
            f"base-token target count value: 1000 | added-token target loss value: {loss:.6E} | "
            "added-token target count value: 20 |"
        )
        for name in PANELS
    ]


def wrapped_validation_lines(iteration: int, loss: float) -> list[str]:
    rows = []
    for line in validation_lines(iteration, loss):
        split_at = line.index("added-token") + 6
        rows.extend((f"3: {line[:split_at]}", f"3: {line[split_at:]}", "3: " + "-" * 80))
    return rows


class StabilitySmokeTests(unittest.TestCase):
    def test_decision_consumes_baseline_endpoint_retention_and_added_token_losses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.log"
            driver = root / "driver.log"
            endpoint = root / "endpoint.log"
            manifest = root / "validation.json"
            checkpoint = root / "checkpoints" / "iter_0001024"
            checkpoint.mkdir(parents=True)
            (checkpoint / ".metadata").write_text("complete")
            initial.write_text("\n".join(validation_lines(0, 2.0)) + "\n")
            training = [
                (
                    f"iteration {iteration:8d}/   38496 | lm loss: {2.0 - iteration / 2048:.6E} | "
                    "grad norm: 2.000000E+00 |"
                )
                for iteration in range(1, 1025)
            ]
            driver.write_text("\n".join(training) + "\n")
            endpoint.write_text("\n".join(validation_lines(1024, 1.9)) + "\n")
            manifest.write_text(json.dumps({"panels": [{"name": name} for name in PANELS]}))
            output = root / "receipt.json"
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / "training" / "evaluate_common_stability_smoke.py"),
                    "--driver-log",
                    str(driver),
                    "--initial-validation-log",
                    str(initial),
                    "--endpoint-validation-log",
                    str(endpoint),
                    "--validation-manifest",
                    str(manifest),
                    "--checkpoint-root",
                    str(root / "checkpoints"),
                    "--peak-lr",
                    "3e-4",
                    "--process-exit-code",
                    "0",
                    "--output",
                    str(output),
                ],
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["schema_version"], "apertus_mini_common_stability_smoke_v2")
            self.assertEqual(set(receipt["validation"]["endpoint"]), set(PANELS))
            self.assertTrue(receipt["checks"]["retention_panels_within_predeclared_relative_margin"])

    def test_validation_parser_reassembles_slurm_wrapped_metric_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial.log"
            driver = root / "driver.log"
            endpoint = root / "endpoint.log"
            manifest = root / "validation.json"
            checkpoint = root / "checkpoints" / "iter_0001024"
            checkpoint.mkdir(parents=True)
            (checkpoint / ".metadata").write_text("complete")
            initial.write_text("\n".join(wrapped_validation_lines(0, 2.0)) + "\n")
            endpoint.write_text("\n".join(wrapped_validation_lines(1024, 1.9)) + "\n")
            driver.write_text(
                "\n".join(
                    f"iteration {iteration:8d}/   38496 | lm loss: 2.0 | grad norm: 2.0 |"
                    for iteration in range(1, 1025)
                )
                + "\n"
            )
            manifest.write_text(json.dumps({"panels": [{"name": name} for name in PANELS]}))
            output = root / "receipt.json"
            result = subprocess.run(
                [
                    "python3",
                    str(ROOT / "training" / "evaluate_common_stability_smoke.py"),
                    "--driver-log", str(driver),
                    "--initial-validation-log", str(initial),
                    "--endpoint-validation-log", str(endpoint),
                    "--validation-manifest", str(manifest),
                    "--checkpoint-root", str(root / "checkpoints"),
                    "--peak-lr", "3e-4",
                    "--process-exit-code", "0",
                    "--output", str(output),
                ],
                check=False,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            receipt = json.loads(output.read_text())
            self.assertTrue(receipt["checks"]["added_token_target_loss_present_and_stable_on_four_greek_probes"])
            self.assertEqual(receipt["validation"]["endpoint"]["hplt"]["added_target_count"], 20.0)


if __name__ == "__main__":
    unittest.main()
