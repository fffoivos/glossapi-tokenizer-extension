from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_models.build_sealed_ab_comparison_site import build_documents, build_site


def _fixtures() -> tuple[list[dict], list[dict], dict, dict]:
    documents = [
        {
            "document_id": "doc-good",
            "source": "kallipos",
            "source_doc_id": "paper-1",
            "source_row_id": "rows:1",
            "lines": [
                {"line_id": "g1", "abs_idx": 0, "text": "Body"},
                {"line_id": "g2", "abs_idx": 1, "text": "Reference"},
            ],
        },
        {
            "document_id": "doc-bad",
            "source": "openarchives",
            "source_doc_id": "work-2",
            "source_row_id": "rows:2",
            "lines": [
                {"line_id": "b1", "abs_idx": 4, "text": "Weak filler"},
                {"line_id": "b2", "abs_idx": 5, "text": "Unknown"},
            ],
        },
    ]
    keys = [
        {"document_id": "doc-good", "line_id": "g1", "line_alias": "a1"},
        {"document_id": "doc-good", "line_id": "g2", "line_alias": "a2"},
        {"document_id": "doc-bad", "line_id": "b1", "line_alias": "a3"},
        {"document_id": "doc-bad", "line_id": "b2", "line_alias": "a4"},
    ]
    pass_a = {
        "reviewer": "reviewer-a",
        "lines": [
            {"line_alias": "a1", "role": "OTHER"},
            {"line_alias": "a2", "role": "ENTRY"},
            {"line_alias": "a3", "role": "FILLER"},
            {"line_alias": "a4", "role": "UNKNOWN"},
        ],
    }
    pass_b = {
        "reviewer": "reviewer-b",
        "lines": [
            {"line_alias": "a1", "role": "OTHER"},
            {"line_alias": "a2", "role": "CONTINUATION"},
            {"line_alias": "a3", "role": "OTHER"},
            {"line_alias": "a4", "role": "UNKNOWN"},
        ],
    }
    return documents, keys, pass_a, pass_b


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_build_documents_sorts_worst_first_and_matches_gate_binary_semantics() -> None:
    documents, keys, pass_a, pass_b = _fixtures()
    manifest, payloads = build_documents(documents, keys, pass_a, pass_b)
    assert [row["document_id"] for row in manifest] == ["doc-bad", "doc-good"]
    assert manifest[0]["binary_disagreements"] == 2
    assert manifest[0]["binary_agreement"] == 0.0
    assert manifest[1]["binary_disagreements"] == 0
    assert manifest[1]["exact_role_agreement"] == 0.5
    assert payloads[0]["lines"][0]["binary_agree"] is False
    assert payloads[1]["lines"][1]["binary_agree"] is True
    assert payloads[1]["lines"][1]["exact_agree"] is False


def test_build_site_writes_static_reader_and_receipt(tmp_path: Path) -> None:
    documents, keys, pass_a, pass_b = _fixtures()
    documents_path, key_path = tmp_path / "documents.jsonl", tmp_path / "keys.jsonl"
    pass_a_path, pass_b_path = tmp_path / "a.json", tmp_path / "b.json"
    _write_jsonl(documents_path, documents)
    _write_jsonl(key_path, keys)
    _write_json(pass_a_path, pass_a)
    _write_json(pass_b_path, pass_b)
    output = tmp_path / "site"
    receipt = build_site(
        documents_path=documents_path,
        line_key_path=key_path,
        pass_a_path=pass_a_path,
        pass_b_path=pass_b_path,
        output_dir=output,
    )
    assert receipt["document_count"] == 2
    assert receipt["binary_disagreement_count"] == 2
    assert (output / "index.html").is_file()
    assert "Previous disagreement" in (output / "index.html").read_text()
    assert "binary-disagree" in (output / "styles.css").read_text()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["documents"][0]["document_id"] == "doc-bad"
    assert len(list((output / "data").glob("*.json"))) == 2


def test_build_documents_fails_closed_on_missing_pass_line() -> None:
    documents, keys, pass_a, pass_b = _fixtures()
    pass_b["lines"].pop()
    with pytest.raises(ValueError, match="cover exactly"):
        build_documents(documents, keys, pass_a, pass_b)
