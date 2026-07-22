from __future__ import annotations

from sequence_models.bibliography_entry_failure_review import line_status


def test_line_status_palette_is_exhaustive() -> None:
    assert line_status(predicted=True, silver_bib=True) == "agreement_bib"
    assert line_status(predicted=True, silver_bib=False) == "classifier_only"
    assert line_status(predicted=False, silver_bib=True) == "silver_only"
    assert line_status(predicted=False, silver_bib=False) == "agreement_non_bib"
