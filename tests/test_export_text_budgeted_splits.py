from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "subprojects"
    / "01_2_training_dataset_mix"
    / "scripts"
    / "export_text_budgeted_splits.py"
)


def write_rows(root: Path, rows: list[dict[str, object]]) -> None:
    data_root = root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, data_root / "rows.parquet")


def run_export(input_root: Path, output_root: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input-root",
            str(input_root),
            "--output-root",
            str(output_root),
            "--threads",
            "1",
            "--train-chars",
            "1000",
            "--val-chars",
            "0",
            "--test-chars",
            "0",
            "--row-group-size",
            "2",
            "--seed-salt",
            "test",
            *extra,
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_export_badness_filter_rejects_null_empty_and_high_scores(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    write_rows(
        input_root,
        [
            {
                "source_dataset": "valid",
                "source_doc_id": "valid",
                "text": "έγκυρο κείμενο",
                "greek_badness_score": "10",
                "mojibake_badness_score": "0.0",
                "needs_ocr": False,
                "ocr_success": True,
            },
            {
                "source_dataset": "null",
                "source_doc_id": "null",
                "text": "λείπει score",
                "greek_badness_score": None,
                "mojibake_badness_score": "0.0",
                "needs_ocr": False,
                "ocr_success": True,
            },
            {
                "source_dataset": "empty",
                "source_doc_id": "empty",
                "text": "άδειο score",
                "greek_badness_score": "",
                "mojibake_badness_score": "0.0",
                "needs_ocr": False,
                "ocr_success": True,
            },
            {
                "source_dataset": "high",
                "source_doc_id": "high",
                "text": "υψηλό score",
                "greek_badness_score": "70",
                "mojibake_badness_score": "0.0",
                "needs_ocr": False,
                "ocr_success": True,
            },
        ],
    )

    result = run_export(input_root, output_root)

    assert result.returncode == 0, result.stderr + result.stdout
    exported = pq.read_table(output_root / "exports" / "train.parquet").to_pylist()
    assert exported == [{"text": "έγκυρο κείμενο"}]


def test_export_badness_filter_requires_score_columns_by_default(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    write_rows(
        input_root,
        [
            {
                "source_dataset": "missing",
                "source_doc_id": "missing",
                "text": "χωρίς στήλες",
                "needs_ocr": False,
                "ocr_success": True,
            }
        ],
    )

    result = run_export(input_root, output_root)

    assert result.returncode != 0
    assert "missing greek_badness_score" in result.stderr


def test_export_duplicate_source_doc_ids_do_not_cross_join(tmp_path: Path) -> None:
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    write_rows(
        input_root,
        [
            {
                "source_dataset": "dup",
                "source_doc_id": "same-id",
                "text": "πρώτη γραμμή",
                "greek_badness_score": "10",
                "mojibake_badness_score": "0.0",
                "needs_ocr": False,
                "ocr_success": True,
            },
            {
                "source_dataset": "dup",
                "source_doc_id": "same-id",
                "text": "δεύτερη γραμμή",
                "greek_badness_score": "10",
                "mojibake_badness_score": "0.0",
                "needs_ocr": False,
                "ocr_success": True,
            },
        ],
    )

    result = run_export(input_root, output_root)

    assert result.returncode == 0, result.stderr + result.stdout
    exported = pq.read_table(output_root / "exports" / "train.parquet").to_pylist()
    assert sorted(row["text"] for row in exported) == ["δεύτερη γραμμή", "πρώτη γραμμή"]
    summary = (output_root / "summary.json").read_text(encoding="utf-8")
    assert '"rows": 2' in summary
