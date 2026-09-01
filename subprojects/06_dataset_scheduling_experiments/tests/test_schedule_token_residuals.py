from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_five_schedules", ROOT / "dataset" / "build_five_schedules.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScheduleTokenResidualTests(unittest.TestCase):
    def test_hard_glossapi_first_quota_uses_prefix_and_closes(self) -> None:
        quotas = MODULE.hard_glossapi_token_quotas(
            [10, 20, 30], 25, glossapi_first=True
        )
        self.assertEqual(quotas, [10, 15, 0])
        self.assertEqual(sum(quotas), 25)

    def test_hard_glossapi_last_quota_uses_suffix_and_closes(self) -> None:
        quotas = MODULE.hard_glossapi_token_quotas(
            [10, 20, 30], 25, glossapi_first=False
        )
        self.assertEqual(quotas, [0, 0, 25])
        self.assertEqual(sum(quotas), 25)


if __name__ == "__main__":
    unittest.main()
