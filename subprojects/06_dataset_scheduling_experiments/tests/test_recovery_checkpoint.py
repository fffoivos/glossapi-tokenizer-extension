from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "D0_mixed",
    "D1_hard_h_to_g",
    "D2_hard_g_to_h",
    "D3_gradual_h_to_g",
    "D4_gradual_g_to_h",
)


class RecoveryCheckpointTests(unittest.TestCase):
    def test_freezes_latest_checkpoint_common_to_all_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign = root / "campaign.json"
            campaign.write_text(
                json.dumps(
                    {
                        "segments": [
                            {"segment_id": 0, "start_iteration": 0, "end_iteration": 19456},
                            {"segment_id": 1, "start_iteration": 19456, "end_iteration": 38496},
                        ]
                    }
                )
            )
            attempt = root / "segments" / "segment_0" / "attempt_0"
            attempt.mkdir(parents=True)
            (attempt / "segment_state.json").write_text(json.dumps({"status": "failed"}))
            for index, arm in enumerate(ARMS):
                arm_root = attempt / arm
                checkpoints = arm_root / "checkpoints"
                for iteration in (512, 1024):
                    point = checkpoints / f"iter_{iteration:07d}"
                    point.mkdir(parents=True)
                    (point / ".metadata").write_text("metadata")
                    (point / "state.bin").write_bytes(f"{arm}:{iteration}".encode())
                if index < 4:
                    point = checkpoints / "iter_0001536"
                    point.mkdir(parents=True)
                    (point / ".metadata").write_text("metadata")
                (arm_root / "driver.out").write_text("iteration completed\n")
            output = root / "recovery.json"
            subprocess.run(
                [
                    "python3",
                    str(ROOT / "production" / "freeze_common_recovery_checkpoint.py"),
                    "--campaign-manifest",
                    str(campaign),
                    "--segment-id",
                    "0",
                    "--run-root",
                    str(root),
                    "--segment-attempt",
                    "0",
                    "--start-iteration",
                    "0",
                    "--output",
                    str(output),
                ],
                check=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt["iteration"], 1024)
            self.assertEqual(receipt["purpose"], "common_infrastructure_failure_recovery")
            self.assertEqual([row["arm_id"] for row in receipt["arms"]], list(ARMS))


if __name__ == "__main__":
    unittest.main()
