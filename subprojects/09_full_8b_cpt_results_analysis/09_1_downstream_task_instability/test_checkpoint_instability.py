#!/usr/bin/env python3
"""Regression tests for arXiv:2510.04848 expressions (1) and (2)."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from analyze_predictions import analyze_prediction_files
from checkpoint_instability import (
    analyze_example_trajectories,
    instability_score,
    mean_total_variation,
)


class FormulaTest(unittest.TestCase):
    def test_expression_1_binary_flips(self) -> None:
        self.assertEqual(mean_total_variation([0, 1, 0, 1]), 1.0)
        self.assertAlmostEqual(mean_total_variation([0, 0, 1, 1]), 1.0 / 3.0)

    def test_expression_1_continuous_scores(self) -> None:
        self.assertAlmostEqual(mean_total_variation([0.2, 0.5, 0.4]), 0.2)

    def test_expression_2_exact_match(self) -> None:
        self.assertAlmostEqual(instability_score(["A", "A", "B", "C"]), 2.0 / 3.0)

    def test_expression_2_custom_similarity(self) -> None:
        similarity = lambda left, right: 1.0 - abs(left - right)
        self.assertAlmostEqual(instability_score([0.0, 0.25, 0.75], similarity), 0.375)

    def test_rejects_short_and_invalid_trajectories(self) -> None:
        with self.assertRaises(ValueError):
            mean_total_variation([1])
        with self.assertRaises(ValueError):
            instability_score(["A"])
        with self.assertRaises(ValueError):
            mean_total_variation([0.0, math.nan])
        with self.assertRaises(ValueError):
            instability_score(["A", "B"], lambda _left, _right: 1.1)


class AggregationTest(unittest.TestCase):
    def test_example_macro_aggregation(self) -> None:
        result = analyze_example_trajectories(
            {"a": [0, 1, 0], "b": [1, 1, 1]},
            {"a": [0, 1, 1], "b": [2, 2, 2]},
        )
        self.assertEqual(result["example_count"], 2)
        self.assertEqual(result["checkpoint_count"], 3)
        self.assertEqual(result["mean_example_mtv"], 0.5)
        self.assertEqual(result["mean_example_is"], 0.25)

    def test_rejects_example_set_and_checkpoint_drift(self) -> None:
        with self.assertRaises(ValueError):
            analyze_example_trajectories({"a": [0, 1]}, {"b": [0, 1]})
        with self.assertRaises(ValueError):
            analyze_example_trajectories(
                {"a": [0, 1], "b": [1, 0, 1]},
                {"a": [0, 1], "b": [1, 0, 1]},
            )


class PredictionAdapterTest(unittest.TestCase):
    def write_predictions(self, path: Path, rows: list[dict[str, object]]) -> None:
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_predictions_jsonl_mapping_and_cancellation_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_0 = root / "zero.jsonl"
            checkpoint_1 = root / "one.jsonl"
            self.write_predictions(
                checkpoint_0,
                [
                    {"example_id": "a", "correct": True, "pred_index": 0},
                    {"example_id": "b", "correct": False, "pred_index": 1},
                ],
            )
            self.write_predictions(
                checkpoint_1,
                [
                    {"example_id": "a", "correct": False, "pred_index": 1},
                    {"example_id": "b", "correct": True, "pred_index": 0},
                ],
            )

            result = analyze_prediction_files(
                [("iter0", checkpoint_0), ("iter1", checkpoint_1)]
            )

        self.assertEqual(result["expression_1"]["mean_example_mtv"], 1.0)
        self.assertEqual(result["expression_1"]["aggregate_score_trajectory_mtv"], 0.0)
        self.assertEqual(result["expression_2"]["mean_example_is"], 1.0)
        self.assertEqual(result["adjacent_transitions"][0]["score_change_rate"], 1.0)
        self.assertEqual(result["adjacent_transitions"][0]["output_change_rate"], 1.0)

    def test_selected_ids_and_alignment_are_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_0 = root / "zero.jsonl"
            checkpoint_1 = root / "one.jsonl"
            self.write_predictions(
                checkpoint_0,
                [{"example_id": "a", "correct": True, "pred_index": 0}],
            )
            self.write_predictions(
                checkpoint_1,
                [{"example_id": "b", "correct": True, "pred_index": 0}],
            )
            with self.assertRaises(ValueError):
                analyze_prediction_files(
                    [("iter0", checkpoint_0), ("iter1", checkpoint_1)]
                )
            with self.assertRaises(ValueError):
                analyze_prediction_files(
                    [("iter0", checkpoint_0), ("iter1", checkpoint_1)],
                    selected_ids={"a"},
                )


if __name__ == "__main__":
    unittest.main()
