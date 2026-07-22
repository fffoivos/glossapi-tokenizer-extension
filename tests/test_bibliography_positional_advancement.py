from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.bibliography_positional_advancement import (  # noqa: E402
    paired_work_recall_interval,
    threshold_at_precision,
)


def test_threshold_at_precision_respects_ties() -> None:
    targets = np.asarray([1, 0, 1, 0, 1], dtype=np.int8)
    probability = np.asarray([0.9, 0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    selected = threshold_at_precision(targets, probability, 0.6)
    assert selected["threshold"] == float(probability[4])
    assert selected["precision"] >= 0.6
    assert selected["recall"] == 1.0


def test_paired_work_interval_detects_consistent_recall_gain() -> None:
    targets = np.tile(np.asarray([1, 0], dtype=np.int8), 20)
    baseline = np.tile(np.asarray([0.4, 0.1], dtype=np.float32), 20)
    candidate = np.tile(np.asarray([0.9, 0.1], dtype=np.float32), 20)
    work_codes = np.repeat(np.arange(20, dtype=np.int32), 2)
    interval = paired_work_recall_interval(
        targets, baseline, candidate, work_codes,
        baseline_threshold=0.5, candidate_threshold=0.5, replicates=500, seed=7,
    )
    assert interval["lower_95"] == interval["upper_95"] == 1.0
