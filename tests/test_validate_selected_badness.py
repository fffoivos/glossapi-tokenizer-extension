from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "subprojects"
    / "05_token_distillation_cpt"
    / "03_training_experiments"
    / "dataset_build"
    / "validate_selected_badness.py"
)


def write_selected(path: Path, rows: list[dict[str, object]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows), path)


def run_validator(selected: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--selected", str(selected), *extra],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_validate_selected_badness_accepts_clean_selected_pool(tmp_path: Path) -> None:
    selected = tmp_path / "selected.parquet"
    report = tmp_path / "report.json"
    write_selected(
        selected,
        [
            {
                "source_dataset": "HPLT/ell_Grek_ge8_no_mt_clean60",
                "source_doc_id": "hplt-1",
                "greek_badness_score": 59.99,
                "text": "clean hplt",
            },
            {
                "source_dataset": "openarchives.gr/example",
                "source_doc_id": "oa-1",
                "greek_badness_score": 0.0,
                "text": "clean oa",
            },
        ],
    )

    result = run_validator(selected, "--output-json", str(report))

    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["rows_scanned"] == 2
    assert payload["policy"]["greek_badness_score"] == "< 60.0"


def test_validate_selected_badness_rejects_equal_threshold_by_default(tmp_path: Path) -> None:
    selected = tmp_path / "selected.parquet"
    report = tmp_path / "report.json"
    write_selected(
        selected,
        [
            {
                "source_dataset": "openarchives.gr/example",
                "source_doc_id": "oa-60",
                "greek_badness_score": 60.0,
                "text": "boundary score",
            }
        ],
    )

    result = run_validator(selected, "--output-json", str(report))

    assert result.returncode != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["ok"] is False
    assert payload["violations"]["greek_badness_score_out_of_policy"] == 1
    assert payload["samples"][0]["reason"] == "greek_badness_score_out_of_policy"


def test_validate_selected_badness_rejects_non_clean60_hplt_label(tmp_path: Path) -> None:
    selected = tmp_path / "selected.parquet"
    report = tmp_path / "report.json"
    write_selected(
        selected,
        [
            {
                "source_dataset": "HPLT/ell_Grek_ge8_no_mt",
                "source_doc_id": "hplt-old",
                "greek_badness_score": 1.0,
                "text": "old hplt label",
            }
        ],
    )

    result = run_validator(selected, "--output-json", str(report))

    assert result.returncode != 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["violations"]["hplt_source_dataset_not_clean60"] == 1


def test_validate_selected_badness_requires_current_report(tmp_path: Path) -> None:
    selected = tmp_path / "selected.parquet"
    report = tmp_path / "report.json"
    write_selected(
        selected,
        [
            {
                "source_dataset": "openarchives.gr/example",
                "source_doc_id": "oa-1",
                "greek_badness_score": 1.0,
                "text": "clean",
            }
        ],
    )
    first = run_validator(selected, "--output-json", str(report))
    assert first.returncode == 0, first.stderr + first.stdout

    current = run_validator(selected, "--require-current-report", str(report))
    assert current.returncode == 0, current.stderr + current.stdout

    subset_report = tmp_path / "subset_report.json"
    subset = run_validator(
        selected,
        "--source-regex",
        "^openarchives\\.gr",
        "--output-json",
        str(subset_report),
    )
    assert subset.returncode == 0, subset.stderr + subset.stdout
    policy_mismatch = run_validator(selected, "--require-current-report", str(subset_report))
    assert policy_mismatch.returncode != 0
    assert "policy mismatch" in policy_mismatch.stderr

    write_selected(
        selected,
        [
            {
                "source_dataset": "openarchives.gr/example",
                "source_doc_id": "oa-1",
                "greek_badness_score": 1.0,
                "text": "clean",
            },
            {
                "source_dataset": "openarchives.gr/example",
                "source_doc_id": "oa-2",
                "greek_badness_score": 2.0,
                "text": "new row",
            },
        ],
    )
    stale = run_validator(selected, "--require-current-report", str(report))
    assert stale.returncode != 0
    assert "stale" in stale.stderr
