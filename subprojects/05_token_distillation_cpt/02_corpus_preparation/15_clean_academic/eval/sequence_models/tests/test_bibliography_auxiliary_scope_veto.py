import json
from types import SimpleNamespace

import numpy as np

from sequence_models.bibliography_auxiliary_scope_veto import (
    has_auxiliary_scope,
    is_exact_non_bibliography_scope_heading,
    materialize_auxiliary_headings,
    normalized_scope_heading_key,
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


def test_auxiliary_scope_inside_a_candidate_is_vetoed() -> None:
    auxiliary = np.asarray([False, False, True, True, False])
    absolute = np.arange(5)
    assert has_auxiliary_scope(auxiliary, absolute, 0, end=3, window=2)


def test_numbered_auxiliary_heading_keeps_its_controlled_subtitle() -> None:
    text = "## 3. Λίστα Επιλεγμένων Παραλλαγών: Στοιχεία Αρχείου"
    assert normalized_scope_heading_key(text).startswith(
        "λιστα επιλεγμενων παραλλαγων:"
    )
    assert is_exact_non_bibliography_scope_heading(text)


def test_why_and_ocr_examples_are_exact_body_scopes() -> None:
    assert is_exact_non_bibliography_scope_heading("## ΓΙΑΤΙ")
    assert is_exact_non_bibliography_scope_heading("ΠΑΡΑ∆ΕΙΓΜΑΤΑ")
    assert not is_exact_non_bibliography_scope_heading("## ΒΙΒΛΙΟΓΡΑΦΙΑ")


def test_abbreviations_is_a_line_role_but_not_a_section_veto() -> None:
    assert not is_exact_non_bibliography_scope_heading("## ΣΥΝΤΟΜΟΓΡΑΦΙΕΣ")
    assert not is_exact_non_bibliography_scope_heading("## List of Abbreviations")
    assert is_exact_non_bibliography_scope_heading("## RELATED MATERIAL")


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


def test_selected_variants_scope_persists_through_atu_subheading(tmp_path) -> None:
    source = tmp_path / "train.jsonl"
    source.write_text(
        json.dumps(
            {
                "split": "train",
                "document_id": "d1",
                "lines": [
                    {"text": "## 3. Λίστα Επιλεγμένων Παραλλαγών: Στοιχεία Αρχείου"},
                    {"text": "## ΑΤ/ATU 709: Χιονάτη"},
                    {"text": "1. archive record"},
                    {"text": "## 4. Επόμενη ενότητα"},
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


def test_requested_validation_split_is_materialized_without_train_rows(
    tmp_path,
) -> None:
    source = tmp_path / "joint.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "split": split,
                    "document_id": document_id,
                    "lines": [{"text": "## RELATED MATERIAL"}, {"text": "item"}],
                }
            )
            for split, document_id in (("train", "train-doc"), ("validation", "val-doc"))
        )
        + "\n",
        encoding="utf-8",
    )
    table = SimpleNamespace(
        targets=np.zeros(2, dtype=np.uint8),
        documents=({"document_id": "val-doc", "line_start": 0, "line_end": 2},),
    )
    headings, scope = materialize_auxiliary_headings(
        table, source, expected_split="validation"
    )
    assert headings.tolist() == [True, False]
    assert scope.tolist() == [True, True]
