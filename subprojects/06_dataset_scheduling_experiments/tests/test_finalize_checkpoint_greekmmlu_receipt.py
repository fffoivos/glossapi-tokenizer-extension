from __future__ import annotations

import ast
import importlib.util
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FinalizeCheckpointGreekMMLUReceiptTests(unittest.TestCase):
    def test_finalizer_requires_all_three_metrics(self) -> None:
        tree = ast.parse(
            (ROOT / "evaluation" / "finalize_checkpoint_greekmmlu_receipt.py").read_text()
        )
        constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)}
        self.assertTrue({"accuracy", "choice_nll", "correct_answer_bpb"}.issubset(constants))
        self.assertIn("float32", constants)
        self.assertIn("evaluation_namespace", constants)

    def test_decontaminated_subset_metrics_are_continuous(self) -> None:
        path = ROOT / "evaluation" / "finalize_checkpoint_greekmmlu_receipt.py"
        spec = importlib.util.spec_from_file_location("finalize_checkpoint_greekmmlu_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        rows = [
            {
                "correct": True,
                "answer_index": 0,
                "correct_answer_utf8_bytes": 4,
                "choice_scores": [
                    {"avg_logprob": -1.0, "sum_logprob": -2.0},
                    {"avg_logprob": -2.0, "sum_logprob": -4.0},
                ],
            },
            {
                "correct": False,
                "answer_index": 1,
                "correct_answer_utf8_bytes": 2,
                "choice_scores": [
                    {"avg_logprob": -0.5, "sum_logprob": -1.0},
                    {"avg_logprob": -1.5, "sum_logprob": -3.0},
                ],
            },
        ]
        metrics = module.summarize_prediction_rows(rows)
        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertGreater(metrics["choice_nll"], 0.0)
        self.assertAlmostEqual(metrics["correct_answer_bpb"], 5.0 / math.log(2) / 6.0)


if __name__ == "__main__":
    unittest.main()
