from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "initialization" / "build_mini_tokenizer_overlay.py"
SPEC = importlib.util.spec_from_file_location("build_mini_tokenizer_overlay", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def tokenizer(vocab: dict[str, int], merges: list[str], added: list[dict]) -> dict:
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel"},
        "post_processor": None,
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": merges,
            "ignore_merges": True,
        },
    }


class MiniTokenizerOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mini = tokenizer(
            {"a": 0, "b": 1, "<mini>": 2},
            ["a b"],
            [{"id": 2, "content": "<mini>", "special": True}],
        )
        self.production = tokenizer(
            {"a": 0, "b": 1, "<production>": 2, "ab": 3, "aba": 4},
            ["a b", "a b", "ab a"],
            [{"id": 2, "content": "<production>", "special": True}],
        )

    def test_preserves_mini_base_and_appends_production_chain(self) -> None:
        overlay, summary = MODULE.build_overlay(
            self.mini, self.production, base_vocab_size=3, target_vocab_size=5
        )
        ids = MODULE.by_id(overlay["model"]["vocab"])
        self.assertEqual(ids, {0: "a", 1: "b", 2: "<mini>", 3: "ab", 4: "aba"})
        self.assertEqual(overlay["model"]["merges"], ["a b", "a b", "ab a"])
        self.assertEqual(overlay["added_tokens"], self.mini["added_tokens"])
        self.assertEqual(summary["base_id_mismatch_count_vs_production_tokenizer"], 1)
        self.assertTrue(summary["mini_base_ids_preserved"])
        self.assertTrue(summary["production_appended_ids_preserved"])

    def test_rejects_merge_prefix_drift(self) -> None:
        production = copy.deepcopy(self.production)
        production["model"]["merges"][0] = "b a"
        with self.assertRaisesRegex(ValueError, "exact prefix"):
            MODULE.build_overlay(
                self.mini, production, base_vocab_size=3, target_vocab_size=5
            )

    def test_rejects_orphan_appended_token(self) -> None:
        production = copy.deepcopy(self.production)
        del production["model"]["vocab"]["aba"]
        production["model"]["vocab"]["orphan"] = 4
        with self.assertRaisesRegex(ValueError, "token/merge result mismatch"):
            MODULE.build_overlay(
                self.mini, production, base_vocab_size=3, target_vocab_size=5
            )

    def test_reconciles_existing_mini_pad_token_without_id_change(self) -> None:
        tokenizer_config, special_map = MODULE.reconcile_pad_metadata(
            {"bos_token": "<s>"}, {"bos_token": {"content": "<s>"}}
        )
        self.assertEqual(tokenizer_config["pad_token"], "<pad>")
        self.assertEqual(special_map["pad_token"]["content"], "<pad>")
        self.assertEqual(MODULE.PAD_TOKEN_ID, 10)


if __name__ == "__main__":
    unittest.main()
