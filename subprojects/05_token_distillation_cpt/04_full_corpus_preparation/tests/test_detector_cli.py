from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
BINARY = (
    HERE.parent
    / "02_corpus_preparation"
    / "15_clean_academic"
    / "reference_detector"
    / "target"
    / "debug"
    / "reference_detect"
)


@pytest.mark.skipif(not BINARY.is_file(), reason="run cargo build before the Phase-04 tests")
def test_partial_detector_error_is_fatal_and_valid_rows_have_provenance(tmp_path: Path) -> None:
    spans = tmp_path / "spans.jsonl"
    counters = tmp_path / "counters.jsonl"
    text = "## ΠΕΡΙΕΧΟΜΕΝΑ\n1. Εισαγωγή .... 1\n2. Μέθοδος .... 4"
    result = subprocess.run(
        [
            str(BINARY),
            "--mode",
            "structure-spans",
            "--source",
            "fixture",
            "--input",
            "-",
            "--out-spans",
            str(spans),
            "--out-counters",
            str(counters),
        ],
        input=json.dumps({"id": "good", "text": text}) + "\n" + json.dumps({"id": "bad"}) + "\n",
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    rows = [json.loads(line) for line in counters.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    good = rows[0]
    assert good["source"] == "fixture"
    assert good["row_uid"] == hashlib.sha256(b"fixture\0good").hexdigest()
    assert good["original_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert "error" in rows[1]
