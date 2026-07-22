from __future__ import annotations

from sequence_models.bibliography_header_mask_audit import (
    audit_document_headers,
    sample_candidates,
)


def _document() -> dict:
    texts = [
        (0, "Προηγούμενο σώμα", "O"),
        (1, "ΒΙΒΛΙΟΓΡΑΦΙΑ", "BIB"),
        (2, "Παπαδόπουλος, Α. (2020). Τίτλος. Αθήνα: Εκδόσεις.", "BIB"),
        (3, "ΞΕΝΟΓΛΩΣΣΗ ΒΙΒΛΙΟΓΡΑΦΙΑ", "BIB"),
        (4, "λοιπά", "BIB"),
        (5, "Smith, J. (2021). Book. London: Press.", "BIB"),
        (6, "Jones, P. (2022). Another book. London: Press.", "BIB"),
        (7, "Επόμενο σώμα", "O"),
    ]
    return {
        "document_id": "doc-1",
        "work_id": "work-1",
        "source": "openarchives",
        "split": "train",
        "coverage": "full_document",
        "n_physical_lines": 8,
        "lines": [
            {"abs_idx": abs_idx, "text": text, "label": label}
            for abs_idx, text, label in texts
        ],
    }


def test_rules_nominate_but_do_not_change_silver_labels() -> None:
    result = audit_document_headers(_document(), context_radius=2)
    by_line = {row["abs_idx"]: row for row in result["candidates"]}
    assert by_line[1]["stratum"] == "exact_heading"
    assert by_line[3]["stratum"] == "exact_subheading"
    assert by_line[2]["stratum"] == "block_start_probe"
    assert by_line[4]["stratum"] == "internal_sparse_probe"
    assert all(row["silver_label"] == "BIB" for row in by_line.values())
    assert by_line[1]["context"][0]["label"] == "O"
    assert by_line[1]["context"][2]["label"] == "BIB"


def test_short_real_entry_is_only_a_probe_not_an_exclusion() -> None:
    result = audit_document_headers(_document())
    candidate = next(row for row in result["candidates"] if row["abs_idx"] == 2)
    assert candidate["stratum"] == "block_start_probe"
    assert "mask_decision" not in candidate
    assert candidate["silver_label"] == "BIB"


def test_sampling_is_deterministic_and_stratified() -> None:
    result = audit_document_headers(_document())
    candidates = result["candidates"] * 4
    first = sample_candidates(candidates, per_stratum=2, seed="fixed")
    second = sample_candidates(candidates, per_stratum=2, seed="fixed")
    assert first == second
    assert {row["stratum"] for row in first} == {
        "exact_heading",
        "exact_subheading",
        "block_start_probe",
        "internal_sparse_probe",
    }
