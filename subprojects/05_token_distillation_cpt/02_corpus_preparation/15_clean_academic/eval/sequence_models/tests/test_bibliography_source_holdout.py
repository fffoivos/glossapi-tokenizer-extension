from types import SimpleNamespace

import numpy as np
import pytest

from sequence_models.bibliography_signal_source_holdout_eval import (
    validate_source_folds,
)
from sequence_models.bibliography_source_holdout_table import (
    SCHEMA_VERSION,
    assign_source_folds,
)


def _documents():
    return (
        {"source": "beta", "line_start": 0, "line_end": 2},
        {"source": "alpha", "line_start": 2, "line_end": 5},
        {"source": "beta", "line_start": 5, "line_end": 7},
    )


def test_source_fold_assignment_keeps_complete_sources_together():
    rows, folds, mapping = assign_source_folds(_documents(), line_count=7)
    assert mapping == {"alpha": 0, "beta": 1}
    assert [row["fold"] for row in rows] == [1, 0, 1]
    assert folds.tolist() == [1, 1, 0, 0, 0, 1, 1]
    table = SimpleNamespace(
        documents=rows,
        manifest={"n_folds": 2},
    )
    packet = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "leave_one_complete_source_out",
        "source_to_fold": mapping,
    }
    assert validate_source_folds(table, packet) == mapping


def test_source_fold_assignment_rejects_missing_source():
    with pytest.raises(ValueError, match="non-empty sources"):
        assign_source_folds(
            ({"source": "", "line_start": 0, "line_end": 1},), line_count=1
        )


def test_source_fold_validation_rejects_mixed_fold():
    rows, _folds, mapping = assign_source_folds(_documents(), line_count=7)
    rows[0]["fold"] = 0
    table = SimpleNamespace(documents=rows, manifest={"n_folds": 2})
    packet = {
        "schema_version": SCHEMA_VERSION,
        "strategy": "leave_one_complete_source_out",
        "source_to_fold": mapping,
    }
    with pytest.raises(ValueError, match="split across folds|mixes sources"):
        validate_source_folds(table, packet)
