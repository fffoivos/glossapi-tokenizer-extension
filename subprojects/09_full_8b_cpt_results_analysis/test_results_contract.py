#!/usr/bin/env python3
"""Regression tests for the compact results package."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent


class ResultsContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full = json.loads(
            (ROOT / "presentations" / "FULL8_RESULTS.data.json").read_text(
                encoding="utf-8"
            )
        )
        cls.scale = json.loads(
            (ROOT / "presentations" / "FULL8_VS_0P5B.data.json").read_text(
                encoding="utf-8"
            )
        )

    def test_completed_terminal_evidence(self) -> None:
        self.assertEqual(self.full["completion"]["status"], "completed")
        self.assertEqual(self.full["meta"]["snapshot_iteration"], 18_284)
        self.assertEqual(len(self.full["new_greekmmlu"]), 19)
        self.assertEqual(
            sum(len(rows) for rows in self.full["per_document_validation"].values()),
            39,
        )

    def test_observed_greekmmlu_peak(self) -> None:
        best = max(
            self.full["new_greekmmlu"], key=lambda row: row["clean_accuracy"]
        )
        self.assertEqual(best["iteration"], 9_536)
        self.assertAlmostEqual(best["clean_accuracy"], 0.5681044619097716)

    def test_cooldown_improved_every_document_panel(self) -> None:
        for panel, rows in self.full["per_document_validation"].items():
            with self.subTest(panel=panel):
                self.assertLess(rows[-1]["bpb"], rows[-2]["bpb"])

    def test_scale_comparison_is_not_promoted_to_a_winner(self) -> None:
        self.assertFalse(self.scale["selection"]["winner_selected"])
        self.assertEqual(
            self.scale["selection"]["provisional_observed_leader"], "D0_mixed"
        )


if __name__ == "__main__":
    unittest.main()
