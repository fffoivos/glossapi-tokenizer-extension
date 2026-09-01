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
        cls.native = json.loads(
            (
                ROOT
                / "presentations"
                / "NATIVE_GREEK_3CP_BENCHMARKS.data.json"
            ).read_text(encoding="utf-8")
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

    def test_native_screen_checkpoint_contract(self) -> None:
        self.assertEqual(
            [row["iteration"] for row in self.native["checkpoints"]],
            [0, 9_536, 18_284],
        )
        self.assertEqual(len(self.native["benchmarks"]), 9)
        self.assertEqual(self.native["contamination"]["scored_examples"], 83_970)

    def test_native_screen_exclusion_accounting(self) -> None:
        filtered = [
            row
            for row in self.native["benchmarks"]
            if not row.get("already_decontaminated", False)
        ]
        self.assertEqual(
            sum(row["excluded_examples"] for row in filtered),
            self.native["contamination"]["excluded_scored_examples"],
        )
        self.assertEqual(
            sum(row["n_filtered"] for row in filtered),
            self.native["contamination"]["retained_scored_examples"],
        )


if __name__ == "__main__":
    unittest.main()
