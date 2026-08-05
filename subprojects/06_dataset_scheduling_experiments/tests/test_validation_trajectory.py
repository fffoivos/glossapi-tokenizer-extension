from __future__ import annotations

import importlib.util
import math
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ValidationTrajectoryTests(unittest.TestCase):
    def test_parser_reconstructs_total_base_and_added_bpb(self) -> None:
        path = ROOT / "evaluation" / "collect_validation_trajectory.py"
        spec = importlib.util.spec_from_file_location("validation_trajectory_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        line = (
            "validation loss at iteration 512 [hplt] | lm loss value: 1.5 | "
            "base-token target loss value: 1.0 | base-token target count value: 3 | "
            "base-token target bytes value: 6 | added-token target loss value: 2.0 | "
            "added-token target count value: 1 | added-token target bytes value: 2\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "driver.out"
            log.write_text(line)
            rows = module.parse_log(log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["iteration"], 512)
        self.assertEqual(rows[0]["panel"], "hplt")
        self.assertAlmostEqual(rows[0]["bpb"], 5.0 / 8.0 / math.log(2))
        self.assertAlmostEqual(rows[0]["base_target_bpb"], 3.0 / 6.0 / math.log(2))
        self.assertAlmostEqual(rows[0]["added_target_bpb"], 2.0 / 2.0 / math.log(2))

    def test_parser_accepts_current_wrapped_megatron_validation_format(self) -> None:
        path = ROOT / "evaluation" / "collect_validation_trajectory.py"
        spec = importlib.util.spec_from_file_location("validation_trajectory_wrapped_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        lines = [
            "3: validation loss at iteration 128 on validation set [hplt] | lm loss value: 1.5 | base-token target lo",
            "3: ss value: 1.0 | base-token target count value: 3 | base-token target bytes value: 6 | added-token target loss value: 2.0 | added-token target count value: 1 | added-token target bytes value: 2 |",
            "3: " + "-" * 80,
        ]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "driver.out"
            log.write_text("\n".join(lines) + "\n")
            rows = module.parse_log(log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["iteration"], 128)
        self.assertEqual(rows[0]["panel"], "hplt")
        self.assertAlmostEqual(rows[0]["bpb"], 5.0 / 8.0 / math.log(2))

    def test_parser_skips_interleaved_output_from_another_rank(self) -> None:
        path = ROOT / "evaluation" / "collect_validation_trajectory.py"
        spec = importlib.util.spec_from_file_location("validation_trajectory_interleaved_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        lines = [
            "3: validation loss at iteration 1024 [zh] | lm loss value: 2.5 | base-token target loss value: 2.4 | base-token target count value: 10 | base-token target bytes value: 20 | added-token ",
            "0: saving checkpoint at iteration 1024 to /tmp/checkpoints",
            "3: target loss value: 7.0 | added-token target count value: 2 | added-token target bytes value: 4 |",
            "3: " + "-" * 80,
        ]
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "driver.out"
            log.write_text("\n".join(lines) + "\n")
            rows = module.parse_log(log)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["added_target_loss"], 7.0)
        self.assertEqual(rows[0]["added_target_count"], 2.0)
        self.assertEqual(rows[0]["added_target_bytes"], 4.0)
