from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "evaluation" / "build_checkpoint_evaluation_plan.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_evaluation_plan", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckpointEvaluationPlanTests(unittest.TestCase):
    def test_submitter_validates_plan_iteration_field(self) -> None:
        text = (ROOT / "clariden" / "submit_checkpoint_native_greekmmlu.sh").read_text()
        self.assertIn('row["iteration"]', text)
        self.assertNotIn('row["optimizer_step"]', text)

    def test_native_greekmmlu_wave_uses_four_isolated_gpu_lanes(self) -> None:
        wave = (ROOT / "clariden" / "run_checkpoint_native_greekmmlu_wave.sbatch").read_text()
        self.assertIn("--gpus-per-task=1", wave)
        self.assertIn("--gres=gpu:1", wave)
        self.assertIn("--exact", wave)
        self.assertIn("--cpus-per-task=64", wave)
        self.assertIn("--mem=105G", wave)
        self.assertIn("base+=4", wave)
        worker = (ROOT / "clariden" / "run_checkpoint_native_greekmmlu_one.sh").read_text()
        self.assertIn("dascim/GreekMMLU", worker)
        self.assertIn("not a required native-GreekMMLU checkpoint", worker)
        builder = (ROOT / "evaluation" / "build_greekmmlu_wave_manifest.py").read_text()
        self.assertIn("checkpoint_evaluation_plan_sha256", builder)
        submitter = (ROOT / "clariden" / "submit_checkpoint_native_greekmmlu_wave.sh").read_text()
        self.assertIn("does not bind the supplied checkpoint evaluation plan", submitter)

    def test_hard_transition_update_uses_first_destination_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ids.u64"
            h = np.arange(700, dtype=np.uint64)
            g = (np.uint64(1) << np.uint64(62)) | np.arange(700, dtype=np.uint64)
            np.concatenate((h, g)).tofile(path)
            ids = np.memmap(path, mode="r", dtype=np.uint64)
            slot, update = MODULE.hard_transition_update(ids, "D1_hard_h_to_g")
            self.assertEqual(slot, 700)
            self.assertEqual(update, 2)

    def test_generated_plan_requires_native_greekmmlu_for_every_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            slots = 1024 * 512
            h_count = slots // 2
            h = np.arange(h_count, dtype=np.uint64)
            g = (np.uint64(1) << np.uint64(62)) | np.arange(
                slots - h_count, dtype=np.uint64
            )
            arm_arrays = {
                "D0_mixed": np.concatenate((h, g)),
                "D1_hard_h_to_g": np.concatenate((h, g)),
                "D2_hard_g_to_h": np.concatenate((g, h)),
                "D3_gradual_h_to_g": np.concatenate((h, g)),
                "D4_gradual_g_to_h": np.concatenate((g, h)),
            }
            arms = []
            for arm_id, values in arm_arrays.items():
                path = root / f"{arm_id}.u64"
                values.tofile(path)
                arms.append(
                    {
                        "arm_id": arm_id,
                        "training_slots": slots,
                        "sequence_ids": {
                            "path": str(path),
                            "bytes": path.stat().st_size,
                            "sha256": sha(path),
                        },
                    }
                )
            schedule = root / "schedule.json"
            schedule.write_text(
                json.dumps(
                    {
                        "schema_version": "apertus_mini_five_data_order_schedules_v1",
                        "status": "completed",
                        "arms": arms,
                    }
                )
                + "\n"
            )
            matrix = root / "matrix.json"
            matrix_value = json.loads(
                (ROOT / "configs" / "experiment_matrix.json").read_text()
            )
            matrix_value["training_control"]["parallelism"][
                "selected_normal_partition_segment_boundary_iterations"
            ] = [512]
            matrix.write_text(json.dumps(matrix_value) + "\n")
            output = root / "plan.json"
            original = MODULE.parse_args
            try:
                MODULE.parse_args = lambda: type(
                    "Args",
                    (),
                    {
                        "schedule_manifest": schedule,
                        "experiment_matrix": matrix,
                        "output": output,
                    },
                )()
                self.assertEqual(MODULE.main(), 0)
            finally:
                MODULE.parse_args = original
            plan = json.loads(output.read_text())
            self.assertEqual(plan["greekmmlu_origin"], "natively_authored_greek")
            self.assertEqual(
                plan["native_greekmmlu_evaluations_total"],
                plan["checkpoint_count_per_arm"] * 5,
            )
            self.assertTrue(
                all(row["native_greekmmlu_required"] for row in plan["checkpoint_rows"])
            )
            self.assertIn(800, [row["iteration"] for row in plan["checkpoint_rows"]])
            row_512 = next(
                row for row in plan["checkpoint_rows"] if row["iteration"] == 512
            )
            self.assertIn("normal_partition_segment_boundary", row_512["reasons"])


if __name__ == "__main__":
    unittest.main()
