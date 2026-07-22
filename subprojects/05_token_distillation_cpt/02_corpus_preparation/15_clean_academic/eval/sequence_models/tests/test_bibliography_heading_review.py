from __future__ import annotations

from sequence_models.bibliography_heading_review import (
    adjudicate, select_review_subset, validate_review,
)


def _provenance() -> dict:
    return {
        "cases": [
            {
                "candidate_id": "c1", "document_id": "d", "work_id": "w",
                "line_id": "l1", "abs_idx": 4, "text_sha256": "a" * 64,
                "original_region_label": "BIB", "existing_trusted_role": "ENTRY",
            },
            {
                "candidate_id": "c2", "document_id": "d", "work_id": "w",
                "line_id": "l2", "abs_idx": 9, "text_sha256": "b" * 64,
                "original_region_label": "O",
            },
        ]
    }


def _review(reviewer: str, labels: tuple[str, str]) -> dict:
    return {
        "schema_version": "bibliography-heading-review-v1", "reviewer": reviewer,
        "cases": [
            {"candidate_id": "c1", "label": labels[0], "confidence": .9, "reason": "context"},
            {"candidate_id": "c2", "label": labels[1], "confidence": .8, "reason": "context"},
        ],
    }


def test_heading_adjudication_trusts_only_exact_agreement() -> None:
    rows, report = adjudicate(
        _provenance(), _review("a", ("BIB_HEADER", "NON_BIB_HEADER")),
        _review("b", ("BIB_HEADER", "NOT_HEADER")),
    )
    assert rows[0]["role"] == "BIB_HEADER"
    assert rows[0]["boundary_flag"] == "HARD_STOP"
    assert rows[1]["role"] == "UNKNOWN"
    assert rows[1]["role_status"] == "UNRESOLVED"
    assert report["agreement_count"] == 1


def test_heading_review_validation_rejects_inventory_drift() -> None:
    expected = {"c1": {"candidate_id": "c1"}, "c2": {"candidate_id": "c2"}}
    normalized = validate_review(
        _review("a", ("BIB_SUBHEADER", "NOT_HEADER")), expected, "a"
    )
    assert [row["candidate_id"] for row in normalized["cases"]] == ["c1", "c2"]


def test_not_header_preserves_an_existing_trusted_nonheading_role() -> None:
    rows, _ = adjudicate(
        _provenance(), _review("a", ("NOT_HEADER", "NOT_HEADER")),
        _review("b", ("NOT_HEADER", "NOT_HEADER")),
    )
    assert rows[0]["role"] == "ENTRY"
    assert rows[1]["role"] == "OTHER"


def test_review_selection_is_stratified_and_keeps_mandatory_headings() -> None:
    cases = [
        {"candidate_id": f"c{index}", "context": [
            {"line_id": f"l{index}", "abs_idx": index, "text": text, "target": True},
        ]}
        for index, text in enumerate((
            "ΒΙΒΛΙΟΓΡΑΦΙΑ", "Chapter One", "2. Methods", "TITLE", "Another Title", "Final Title",
        ))
    ]
    packet = {
        "schema_version": "bibliography-heading-review-packet-v1",
        "blinding": {"old_labels_hidden": True}, "cases": cases,
    }
    provenance = {
        "schema_version": "bibliography-heading-review-provenance-v1",
        "cases": [
            {
                "candidate_id": f"c{index}", "text": case["context"][0]["text"],
                "source": "s", "fold": index % 2, "entry_probability": index / 10,
                "existing_trusted_role": "BIB_HEADER" if index == 0 else None,
            }
            for index, case in enumerate(cases)
        ],
    }
    selected_packet, _, receipt = select_review_subset(
        packet, provenance, per_source_quota=3, code_commit="abc", slurm_job_id="7",
    )
    selected_ids = {row["candidate_id"] for row in selected_packet["cases"]}
    assert "c0" in selected_ids
    assert len(selected_ids) >= 3
    assert receipt["selected_candidate_count"] == len(selected_ids)
