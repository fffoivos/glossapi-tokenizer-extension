import json
from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_auxiliary_scope_veto import (
    has_auxiliary_scope,
    materialize_auxiliary_headings,
)


def test_auxiliary_scope_requires_an_exact_nearby_physical_heading() -> None:
    auxiliary = np.asarray([True, False, False, True, False])
    absolute = np.asarray([10, 11, 12, 20, 21])
    assert has_auxiliary_scope(auxiliary, absolute, 2, window=2)
    assert has_auxiliary_scope(auxiliary, absolute, 4, window=2)
    assert not has_auxiliary_scope(
        np.zeros_like(auxiliary), absolute, 2, window=2
    )


def test_auxiliary_scope_does_not_cross_a_physical_gap() -> None:
    auxiliary = np.asarray([True, False])
    absolute = np.asarray([10, 100])
    assert not has_auxiliary_scope(auxiliary, absolute, 1, window=2)


def test_atx_auxiliary_scope_ends_at_the_next_atx_heading(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "train",
                "document_id": "d1",
                "lines": [
                    {"text": "## ΣΧΕΤΙΖΟΜΕΝΑ ΧΝΑΡΙΑ"},
                    {"text": "citation-shaped item"},
                    {"text": "another item"},
                    {"text": "## ΕΠΟΜΕΝΟ ΚΕΦΑΛΑΙΟ"},
                    {"text": "body"},
                ],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    table = SimpleNamespace(
        targets=np.zeros(5, dtype=np.uint8),
        documents=({"document_id": "d1", "line_start": 0, "line_end": 5},),
    )
    headings, scope = materialize_auxiliary_headings(table, source)
    assert headings.tolist() == [True, False, False, False, False]
    assert scope.tolist() == [True, True, True, False, False]
