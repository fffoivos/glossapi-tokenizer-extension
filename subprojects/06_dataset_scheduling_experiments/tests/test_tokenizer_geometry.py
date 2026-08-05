from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "initialization" / "tokenizer_geometry.py"
SPEC = importlib.util.spec_from_file_location("tokenizer_geometry", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class TokenizerGeometryTests(unittest.TestCase):
    def write_tokenizer(self, value: dict) -> Path:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "tokenizer.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_derives_recursive_base_leaves_without_decoding(self) -> None:
        path = self.write_tokenizer(
            {
                "model": {
                    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
                    "merges": ["irrelevant prefix", "a b", "ab c"],
                }
            }
        )
        self.assertEqual(
            MODULE.derive_added_token_base_ids(
                path, base_vocab_size=3, target_vocab_size=5
            ),
            {3: [0, 1], 4: [0, 1, 2]},
        )

    def test_rejects_non_live_operand(self) -> None:
        path = self.write_tokenizer(
            {
                "model": {
                    "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4},
                    "merges": ["irrelevant prefix", "a missing", "ab c"],
                }
            }
        )
        with self.assertRaisesRegex(ValueError, "operand is not live"):
            MODULE.derive_added_token_base_ids(
                path, base_vocab_size=3, target_vocab_size=5
            )


if __name__ == "__main__":
    unittest.main()
