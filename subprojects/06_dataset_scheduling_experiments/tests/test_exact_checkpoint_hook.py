from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exact_checkpoint_hook", ROOT / "training" / "exact_checkpoint_hook.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ExactCheckpointHookTests(unittest.TestCase):
    def test_parse_iterations_deduplicates_and_sorts(self) -> None:
        self.assertEqual(MODULE.parse_iterations("800,512,800,11911"), (512, 800, 11911))

    def test_parse_iterations_rejects_nonpositive(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            MODULE.parse_iterations("0,512")

    def test_empty_environment_value_disables_hook(self) -> None:
        self.assertEqual(MODULE.parse_iterations(""), ())


if __name__ == "__main__":
    unittest.main()
