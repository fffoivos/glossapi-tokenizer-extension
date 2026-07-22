from __future__ import annotations

import gzip
import json
from pathlib import Path

from sequence_models.bibliography_one_block_reader import (
    _build_document_packet,
    _reader_page,
    run_build,
)


def _document(document_id: str, labels: list[str]) -> dict:
    return {
        "document_id": document_id,
        "work_id": f"work-{document_id}",
        "source": "openarchives",
        "split": "train",
        "coverage": "full_document",
        "historical_mode": "whole",
        "n_physical_lines": len(labels),
        "lines": [
            {
                "abs_idx": index,
                "text": (
                    "Παπαδόπουλος, Α. (2006). Αθήνα: Εκδόσεις Δοκιμή, pp. 10-18."
                    if label == "BIB"
                    else f"Κανονική γραμμή κειμένου {index}."
                ),
                "label": label,
            }
            for index, label in enumerate(labels)
        ],
    }


def test_document_packet_requires_exactly_one_bib_block() -> None:
    assert _build_document_packet(_document("zero", ["O", "O"]), 64) is None
    assert (
        _build_document_packet(_document("two", ["BIB", "O", "BIB"]), 64)
        is None
    )
    result = _build_document_packet(
        _document("one", ["O", "BIB", "BIB", "O"]), 64
    )
    assert result is not None
    metadata, compressed, histogram = result
    payload = json.loads(gzip.decompress(compressed))
    assert metadata["bib_start_abs_idx"] == 1
    assert metadata["bib_end_abs_idx"] == 2
    assert metadata["bib_line_count"] == 2
    assert payload["lines"][1][3] == 1
    assert payload["lines"][1][2] > 0
    assert payload["lines"][1][4]
    assert sum(histogram.values()) == 4


def test_page_has_document_menu_scores_matches_and_silver_underline() -> None:
    page = _reader_page()
    assert 'id="documentSelect"' in page
    assert 'id="jumpBib"' in page
    assert 'class="score"' in page
    assert 'class="feature-rail"' in page
    assert "feature-label" in page
    assert "match-box" in page
    assert ".silver-bib .line-text" in page
    assert "red underline = existing silver BIB label" in page
    assert "DecompressionStream" in page
    assert "renderDocument(payload,token,true)" in page


def test_build_preserves_all_one_block_documents(tmp_path: Path) -> None:
    input_path = tmp_path / "silver.jsonl"
    rows = [
        _document("one-a", ["O", "BIB", "BIB", "O"]),
        _document("zero", ["O", "TOC", "O"]),
        _document("two", ["BIB", "O", "BIB"]),
        _document("one-b", ["BIB", "BIB", "O"]),
    ]
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    args = type(
        "Args",
        (),
        {
            "input": str(input_path),
            "output_dir": str(tmp_path / "reader"),
            "workers": 1,
            "max_physical_gap": 64,
            "code_commit": "test-commit",
            "slurm_job_id": "test-job",
        },
    )()
    receipt = run_build(args)
    output = tmp_path / "reader"
    manifest = json.loads((output / "manifest.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["selected_document_count"] == 2
    assert manifest["selection"]["input_document_count"] == 4
    assert [row["document_id"] for row in manifest["documents"]] == [
        "one-a",
        "one-b",
    ]
    assert len(list((output / "documents").glob("*.json.gz"))) == 2

