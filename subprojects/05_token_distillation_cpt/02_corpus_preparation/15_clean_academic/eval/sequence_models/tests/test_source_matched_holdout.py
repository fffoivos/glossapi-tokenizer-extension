from __future__ import annotations

import json
import re
import shutil
import subprocess

import pytest

from sequence_models.build_holdout_review_site import build_site
from sequence_models.source_matched_holdout import (
    SketchIndex,
    _parse_quotas,
    _strip_numbered_text,
    bottom_k_word_shingles,
    load_historical_manifest,
)


def test_bottom_k_sketch_detects_identical_and_rejects_unrelated() -> None:
    repeated = " ".join(f"λέξη{i % 31}" for i in range(1000))
    unrelated = " ".join(f"άλλο{i % 29}" for i in range(1000))
    left = bottom_k_word_shingles(repeated)
    right = bottom_k_word_shingles(repeated)
    other = bottom_k_word_shingles(unrelated)
    index = SketchIndex()
    index.add("old", "greek_phd", left)
    assert index.closest("greek_phd", right) == ("old", 1.0)
    assert index.closest("greek_phd", other)[1] < 0.2
    assert index.closest("openarchives", right) == (None, 0.0)


def test_strip_numbered_text() -> None:
    assert _strip_numbered_text("L00001: α\nL00007: β") == "α\nβ"


def test_quota_parser_requires_all_sources() -> None:
    assert _parse_quotas([]) == {
        "greek_phd": 150,
        "kallipos": 150,
        "openarchives": 200,
    }
    with pytest.raises(ValueError):
        _parse_quotas(["greek_phd=1"])


def test_historical_manifest_enforces_full_2000(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps({"source": "greek_phd", "doc_id": "x"}) + "\n")
    with pytest.raises(ValueError, match="expected 2000"):
        load_historical_manifest(path)


def test_review_site_joins_private_key_and_codex_response(tmp_path) -> None:
    request_id = "a" * 64
    request = {
        "schema_version": "academic-structure-codex56-audit-request-v1",
        "request_id": request_id,
        "request_sha256": "b" * 64,
        "source": "kallipos",
        "target_abs_idx": 4,
        "lines": [{"abs_idx": 4, "line_id": "line", "text": "ΒΙΒΛΙΟΓΡΑΦΙΑ"}],
    }
    key = {
        "request_id": request_id,
        "source": "kallipos",
        "document_id": "doc",
        "source_doc_id": "paper_A_9999",
        "work_id": "paper_A_9999",
        "n_physical_lines": 10,
        "stratum": "bib_high_risk",
        "candidate_prediction": "BIB",
        "candidate_spans": [],
    }
    response = {
        "request_id": request_id,
        "label": "BIB",
        "confidence": 0.99,
        "start_abs_idx": 4,
        "end_abs_idx": 4,
        "structural_cues": ["heading"],
    }
    requests = tmp_path / "requests.jsonl"
    keys = tmp_path / "key.jsonl"
    responses = tmp_path / "responses.jsonl"
    requests.write_text(json.dumps(request) + "\n")
    keys.write_text(json.dumps(key) + "\n")
    responses.write_text(json.dumps(response) + "\n")
    output = tmp_path / "index.html"
    receipt = build_site(
        requests=requests,
        key=keys,
        responses=responses,
        output=output,
    )
    assert receipt["case_count"] == 1
    page = output.read_text()
    assert "paper_A_9999" in page
    assert "ToC &amp; bibliography holdout review" in page
    assert "One highlighted target line per case" in page
    assert "ArrowLeft:'O'" in page
    assert "ArrowUp:'TOC'" in page
    assert "ArrowDown:'BIB'" in page
    assert "ArrowRight:'UNKNOWN'" in page
    assert "ALL 3 AGREE" in page
    assert "C2 + CODEX AGREE" in page
    assert "Resume undecided" in page
    assert "position unavailable" in page
    assert "legacy_packet_sha256" in page
    node = shutil.which("node")
    if node:
        script = re.search(r"<script>(.*)</script>", page, re.DOTALL)
        assert script is not None
        script_path = tmp_path / "review-site.js"
        script_path.write_text(script.group(1))
        subprocess.run([node, "--check", script_path], check=True)
