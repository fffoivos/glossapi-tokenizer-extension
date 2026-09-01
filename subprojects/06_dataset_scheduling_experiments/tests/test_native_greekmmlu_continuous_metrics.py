from __future__ import annotations

import importlib.util
import math
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUNDLED_RUNNER = ROOT / "native_greek_eval" / "run_native_greek_mcq_eval.py"
EXTERNAL_RUNNER = (
    Path(os.environ["NATIVE_GREEK_EVAL_ROOT"]) / "run_native_greek_mcq_eval.py"
    if os.environ.get("NATIVE_GREEK_EVAL_ROOT")
    else Path("/__missing_native_greek_eval_root__")
)
REPOSITORY_RUNNER = (
    ROOT.parent
    / "03_apertus_extension_and_embedding_adaptation"
    / "03_4_implementation_experiments"
    / "init_bakeoff"
    / "eval"
    / "run_native_greek_mcq_eval.py"
)
RUNNER = next(
    path
    for path in (BUNDLED_RUNNER, EXTERNAL_RUNNER, REPOSITORY_RUNNER)
    if path.is_file()
)
SPEC = importlib.util.spec_from_file_location("native_greek_mcq_runner", RUNNER)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
import sys

sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class NativeGreekMMLUContinuousMetricTests(unittest.TestCase):
    def test_choice_nll_and_correct_answer_bpb(self) -> None:
        rows = [
            {
                "benchmark": "greekmmlu",
                "subject": "history",
                "correct": True,
                "answer_index": 0,
                "correct_answer_utf8_bytes": 4,
                "choice_scores": [
                    {"avg_logprob": -1.0, "sum_logprob": -2.0, "num_tokens": 2},
                    {"avg_logprob": -2.0, "sum_logprob": -4.0, "num_tokens": 2},
                ],
            }
        ]
        summary, headline, diagnostics, aggregate = MODULE._summarize(rows)
        expected_nll = math.log(math.exp(-1.0) + math.exp(-2.0)) + 1.0
        expected_bpb = 2.0 / math.log(2.0) / 4.0
        all_row = next(row for row in summary if row["subject"] == "__all__")
        self.assertAlmostEqual(all_row["choice_nll"], expected_nll)
        self.assertAlmostEqual(all_row["correct_answer_bpb"], expected_bpb)
        self.assertEqual(diagnostics, [])
        self.assertEqual(headline[0]["benchmark"], "greekmmlu")
        self.assertAlmostEqual(aggregate["headline"]["micro_choice_nll"], expected_nll)
        self.assertAlmostEqual(
            aggregate["headline"]["micro_correct_answer_bpb"], expected_bpb
        )


if __name__ == "__main__":
    unittest.main()
