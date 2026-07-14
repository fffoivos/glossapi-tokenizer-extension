import numpy as np

from sequence_models.bibliography_entry_models import Table
from sequence_models.bibliography_unseen_block_review import (
    HTML,
    choose_review_documents,
    select_candidate_pool,
)


def _rows(count=4):
    return [
        {
            "document_id": f"{source}-{index}",
            "work_id": f"work-{source}-{index}",
            "source": source,
        }
        for source in ("greek_phd", "kallipos", "openarchives")
        for index in range(count)
    ]


def test_candidate_pool_is_balanced_deterministic_and_seeded():
    rows = _rows()
    first = select_candidate_pool(rows, per_source=2, seed="a")
    second = select_candidate_pool(list(reversed(rows)), per_source=2, seed="a")
    third = select_candidate_pool(rows, per_source=2, seed="b")
    assert [row["document_id"] for row in first] == [row["document_id"] for row in second]
    assert [row["document_id"] for row in first] != [row["document_id"] for row in third]
    assert {source: sum(row["source"] == source for row in first) for source in ("greek_phd", "kallipos", "openarchives")} == {
        "greek_phd": 2,
        "kallipos": 2,
        "openarchives": 2,
    }


def test_candidate_pool_excludes_prior_documents_and_works():
    rows = _rows(count=5)
    excluded_document = rows[0]
    excluded_work = rows[6]
    selected = select_candidate_pool(
        rows,
        per_source=2,
        seed="fresh",
        excluded_document_ids={excluded_document["document_id"]},
        excluded_work_ids={excluded_work["work_id"]},
    )
    assert excluded_document["document_id"] not in {row["document_id"] for row in selected}
    assert excluded_work["work_id"] not in {row["work_id"] for row in selected}
    assert len(selected) == 6


def test_review_selection_prefers_documents_with_predicted_blocks():
    documents = []
    cursor = 0
    for row in _rows(count=3):
        documents.append({**row, "line_start": cursor, "line_end": cursor + 2})
        cursor += 2
    table = Table(
        root=None,
        manifest={},
        documents=tuple(documents),
        counts=np.empty((cursor, 0)),
        targets=np.zeros(cursor),
        original_labels=np.zeros(cursor),
        header_kinds=np.zeros(cursor),
        abs_indices=np.arange(cursor),
        token_counts=np.zeros(cursor),
        char_lengths=np.zeros(cursor),
        block_indices=np.zeros(cursor),
        document_indices=np.zeros(cursor),
        folds=np.zeros(cursor),
    )
    prediction = np.zeros(cursor, dtype=bool)
    for source_offset in (0, 3, 6):
        prediction[(source_offset + 1) * 2] = True
    chosen, summary = choose_review_documents(_rows(count=3), table, prediction, per_source=2)
    assert chosen == [1, 0, 4, 3, 7, 6]
    assert all(summary[source]["selected_with_predicted_blocks"] == 1 for source in summary)


def test_reader_has_bidirectional_block_navigation():
    assert "↑ Previous BIB block" in HTML
    assert "↓ Next BIB block" in HTML
    assert "ArrowUp" in HTML and "ArrowDown" in HTML
    assert "window.scrollBy" in HTML
    assert "toggleWrong" in HTML
    assert "toggleBlockWrong" in HTML
    assert "Mark whole block WRONG" in HTML
    assert "toggleDocWeird" in HTML
    assert "bib-unseen-weird-docs-v1" in HTML
    assert "Mark document WEIRD" in HTML
    assert "bib-unseen-wrong-lines-v1" in HTML
    assert "bib-unseen-should-bib-lines-v1" in HTML
    assert "SHOULD BE BIB" in HTML
    assert "directionalBlock" in HTML
    assert "alignCurrentBlock" in HTML
    assert "content-visibility" not in HTML
    assert "Export review JSON" in HTML
    assert "bibliography-unseen-review-v1" in HTML
    assert "wrong_predicted_lines" in HTML
    assert "aria-pressed" in HTML
