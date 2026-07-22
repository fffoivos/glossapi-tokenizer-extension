from __future__ import annotations

import json
from pathlib import Path

from sequence_models.bibliography_block_audit import audit_document, run, summarize


def _document(document_id: str, labels: list[tuple[int, str]]) -> dict:
    return {
        "document_id": document_id,
        "work_id": f"work-{document_id}",
        "source": "openarchives",
        "split": "train",
        "coverage": "full_document",
        "n_physical_lines": labels[-1][0] + 1,
        "lines": [
            {"abs_idx": abs_idx, "text": f"line {abs_idx}", "label": label}
            for abs_idx, label in labels
        ],
    }


def test_counts_continuous_bib_runs_and_preserves_spans() -> None:
    audited = audit_document(
        _document(
            "doc-1",
            [(0, "O"), (1, "BIB"), (2, "BIB"), (4, "O"), (5, "BIB")],
        )
    )
    assert audited["bib_block_count"] == 2
    assert audited["bib_line_count"] == 3
    assert audited["blocks"] == [
        {
            "block_index": 0,
            "start_abs_idx": 1,
            "end_abs_idx": 2,
            "present_line_count": 2,
            "physical_span_line_count": 2,
        },
        {
            "block_index": 1,
            "start_abs_idx": 5,
            "end_abs_idx": 5,
            "present_line_count": 1,
            "physical_span_line_count": 1,
        },
    ]


def test_large_coordinate_gap_splits_window_seam() -> None:
    audited = audit_document(
        _document("doc-gap", [(1, "BIB"), (2, "BIB"), (1000, "BIB")]),
        max_physical_gap=64,
    )
    assert audited["bib_block_count"] == 2
    assert audited["coverage_gap_splits"] == 1


def test_legacy_numeric_labels_are_supported() -> None:
    audited = audit_document(
        {
            "doc_id": "legacy",
            "source": "greek_phd",
            "n_lines": 4,
            "lines": [[0, "body", 0], [1, "reference", 1], [2, "contents", 2]],
        }
    )
    assert audited["bib_block_count"] == 1
    assert audited["label_counts"] == {"BIB": 1, "O": 1, "TOC": 1}


def test_summary_includes_zero_documents_and_ranked_tail() -> None:
    rows = [
        audit_document(_document("doc-1", [(0, "BIB"), (1, "O"), (2, "BIB")])),
        audit_document(_document("doc-2", [(0, "BIB"), (1, "O"), (2, "BIB")])),
        audit_document(_document("doc-3", [(0, "O")])),
    ]
    summary = summarize(rows, top_n=2)
    assert summary["document_count"] == 3
    assert summary["mean"] == 1.333333
    assert summary["median"] == 2
    assert summary["maximum"] == 2
    assert summary["documents_with_no_bib_block"] == 1
    assert [row["document_id"] for row in summary["top_documents"]] == [
        "doc-1",
        "doc-2",
    ]


def test_run_preserves_every_document_and_writes_plot(tmp_path: Path) -> None:
    input_path = tmp_path / "silver.jsonl"
    documents = [
        _document("doc-1", [(0, "O"), (1, "BIB")]),
        _document("doc-2", [(0, "O")]),
    ]
    input_path.write_text(
        "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
    )
    args = type(
        "Args",
        (),
        {
            "input": str(input_path),
            "output_dir": str(tmp_path / "output"),
            "top_n": 25,
            "max_physical_gap": 64,
            "code_commit": "abc123",
            "slurm_job_id": "test-job",
        },
    )()
    receipt = run(args)
    output = tmp_path / "output"
    assert receipt["status"] == "passed"
    assert receipt["document_count"] == 2
    assert len((output / "documents.jsonl").read_text().splitlines()) == 2
    assert "<svg" in (output / "distribution.svg").read_text()
    assert "Highest-scoring documents" in (output / "distribution.svg").read_text()
