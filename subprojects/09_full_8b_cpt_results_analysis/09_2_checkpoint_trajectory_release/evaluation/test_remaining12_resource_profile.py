#!/usr/bin/env python3
"""Guard the approved one-node fallback without changing evaluation science."""

from __future__ import annotations

import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("run_remaining12_native_segment.sbatch")


class RemainingTwelveResourceProfileTest(unittest.TestCase):
    def test_one_node_profile_has_four_gpu_workers_and_bounded_time(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("#SBATCH --nodes=1", text)
        self.assertIn("#SBATCH --ntasks=4", text)
        self.assertIn("#SBATCH --gpus-per-node=4", text)
        self.assertIn("#SBATCH --mem=640G", text)
        self.assertIn("#SBATCH --time=01:20:00", text)
        self.assertIn(': "${REMAINING12_EXPECTED_NNODES:=1}"', text)

    def test_geometry_guard_accepts_only_explicit_one_or_two_node_profiles(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('"$REMAINING12_EXPECTED_NNODES" == 1 || "$REMAINING12_EXPECTED_NNODES" == 2', text)
        self.assertIn('"$allocation_nodes" == "$REMAINING12_EXPECTED_NNODES"', text)
        self.assertIn("srun --exclusive --exact --nodes=1", text)
        self.assertIn("--mem=160G", text)


if __name__ == "__main__":
    unittest.main()
