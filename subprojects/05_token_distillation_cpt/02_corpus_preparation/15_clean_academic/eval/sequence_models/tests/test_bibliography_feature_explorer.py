from __future__ import annotations

import json
from pathlib import Path

from sequence_models.bibliography_feature_explorer import (
    FEATURE_SPECS,
    build_page,
    build_payload,
    select_documents,
)


def _document(source: str, index: int) -> dict:
    document_id = f"{source}-{index}"
    return {
        "schema_version": "academic-structure-gold-v1",
        "document_id": document_id,
        "work_id": f"work-{document_id}",
        "source": source,
        "split": "train",
        "coverage": "full_document",
        "n_physical_lines": 120,
        "lines": [
            {
                "line_id": f"{document_id}-{line}",
                "abs_idx": line,
                "text": (
                    "Παπαδόπουλος, Α. (2006). Nat. Chem. Biol. 2, pp. 10–18. "
                    "doi:10.1000/example"
                    if line == 10
                    else f"Κανονική γραμμή κειμένου {line} για το έγγραφο."
                ),
                # Labels must never leak into the explorer.
                "label": "BIB" if line == 10 else "O",
            }
            for line in range(1, 111)
        ],
    }


def test_source_balanced_selection_and_payload_are_label_blind(tmp_path: Path) -> None:
    path = tmp_path / "documents.jsonl"
    rows = [
        _document(source, index)
        for source in ("greek_phd", "kallipos", "openarchives")
        for index in range(4)
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    selected, input_sha, eligible = select_documents(
        path,
        document_count=6,
        sources=("greek_phd", "kallipos", "openarchives"),
        split="train",
        coverage="full_document",
        seed="test-seed",
    )
    payload = build_payload(
        selected,
        input_path=str(path),
        input_sha256=input_sha,
        eligible_counts=eligible,
        split="train",
        coverage="full_document",
        seed="test-seed",
    )
    assert payload["selection"]["source_counts"] == {
        "greek_phd": 2,
        "kallipos": 2,
        "openarchives": 2,
    }
    assert len(payload["documents"]) == 6
    assert len(payload["lines"]) == 660
    assert len(payload["features"]) == len(FEATURE_SPECS) == 35
    assert all("label" not in line for line in payload["lines"])
    assert all("label" not in document for document in payload["documents"])
    citation = next(line for line in payload["lines"] if line["abs_idx"] == 10)
    assert citation["features"]["year_count"] == 1
    assert citation["features"]["doi_count"] == 1
    assert citation["features"]["page_range_count"] == 1
    assert citation["char_length"] == len(citation["text"])
    assert citation["matches"]["doi_count"] == [
        [citation["text"].index("doi:"), len(citation["text"])]
    ]
    assert set(citation["matches"]) == set(citation["features"])


def test_page_has_live_feature_filters_and_unit_scoring() -> None:
    documents = [_document("greek_phd", 1)]
    payload = build_payload(
        documents,
        input_path="/remote/documents.jsonl",
        input_sha256="a" * 64,
        eligible_counts={"greek_phd": 1},
        split="train",
        coverage="full_document",
        seed="test-seed",
    )
    page = build_page(payload)
    assert "Feature menu" in page
    assert "data-feature=" in page
    assert "one point for each enabled feature" in page
    assert "Rank: matches / 100 chars" in page
    assert "class=\"charhit\"" in page
    assert "line.matches" in page
    assert "b.value-a.value||b.m.points-a.m.points" in page
    assert "data-spotlight=" in page
    assert "function spotlight(key)" in page
    assert "function clearSpotlight()" in page
    assert "Hover a coloured badge or sidebar feature label" in page
    assert "raw count is non-zero" not in page  # Python docstring is not rendered.
    assert "weighted_score_used" in page
