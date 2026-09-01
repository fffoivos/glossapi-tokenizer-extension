from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from checkpoint_eval_export_receipt import canonical_sha256  # noqa: E402


class CheckpointEvalExportReceiptTests(unittest.TestCase):
    def test_canonical_hash_is_mapping_order_independent(self) -> None:
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))


if __name__ == "__main__":
    unittest.main()
