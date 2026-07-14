from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models import bibliography_positional_models as models  # noqa: E402


class _FakeModel:
    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.n_iter_ = np.asarray([1])


class _FakeTransform:
    pass


def test_parallel_outer_folds_write_their_own_predictions() -> None:
    original_sklearn = models._sklearn
    original_fit = models._fit_linear
    original_predict = models._predict
    models._sklearn = lambda: {  # type: ignore[assignment]
        "average_precision": lambda truth, probability: float(np.mean(probability))
    }
    models._fit_linear = (  # type: ignore[assignment]
        lambda arm, counts, position, gaps, targets, indices, c_value, seed:
        (_FakeModel(float(seed)), _FakeTransform())
    )
    models._predict = (  # type: ignore[assignment]
        lambda model, transform, counts, position, gaps, indices:
        np.full(len(indices), model.offset, dtype=np.float64)
    )
    try:
        n, fold_count = 20, 5
        counts = np.zeros((n, 35), dtype=np.uint32)
        position = np.zeros((n, 35, 4), dtype=np.float32)
        gaps = np.zeros((n, 7), dtype=np.float32)
        targets = np.asarray([0, 1] * (n // 2), dtype=np.int8)
        folds = np.arange(n, dtype=np.uint8) % fold_count
        with tempfile.TemporaryDirectory() as directory:
            probability, reports = models._fit_nested_linear(
                "P0", counts, position, gaps, targets, folds,
                np.ones(n, dtype=bool), n_folds=fold_count, c_grid=(0.1,),
                seed=100, model_dir=Path(directory), parallel_folds=fold_count,
            )
        assert [row["outer_fold"] for row in reports] == list(range(fold_count))
        for fold in range(fold_count):
            assert np.all(probability[folds == fold] == 100 + fold)
    finally:
        models._sklearn = original_sklearn
        models._fit_linear = original_fit
        models._predict = original_predict
