from __future__ import annotations

import json
from pathlib import Path

import pytest

from sequence_models.build_sealed_ab_comparison_site import (
    build_documents,
    build_site,
    build_task_agreement,
    load_annotation_provenance,
)


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
        {"document_id": "doc-good", "document_alias": "dg", "line_id": "g1", "line_alias": "a1"},
        {"document_id": "doc-good", "document_alias": "dg", "line_id": "g2", "line_alias": "a2"},
        {"document_id": "doc-bad", "document_alias": "db", "line_id": "b1", "line_alias": "a3"},
        {"document_id": "doc-bad", "document_alias": "db", "line_id": "b2", "line_alias": "a4"},
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


def _write_provenance(
    tmp_path: Path,
    name: str,
    models: tuple[str, str],
) -> tuple[Path, Path]:
    packet_path = tmp_path / f"{name}.packet.jsonl"
    chunks = [
        {
            "chunk_id": f"{name}-good",
            "document_alias": "dg",
            "owned_start_offset": 0,
            "owned_end_offset_exclusive": 2,
            "lines": [
                {"offset": 0, "line_alias": "a1"},
                {"offset": 1, "line_alias": "a2"},
            ],
        },
        {
            "chunk_id": f"{name}-bad",
            "document_alias": "db",
            "owned_start_offset": 0,
            "owned_end_offset_exclusive": 2,
            "lines": [
                {"offset": 0, "line_alias": "a3"},
                {"offset": 1, "line_alias": "a4"},
            ],
        },
    ]
    _write_jsonl(packet_path, chunks)
    response_dir = tmp_path / f"{name}-responses"
    response_dir.mkdir()
    for index, (chunk, model) in enumerate(zip(chunks, models, strict=True)):
        _write_json(
            response_dir / f"{index}.json",
            {
                "annotation_runtime": {"model": model},
                "review": {"chunks": [{"chunk_id": chunk["chunk_id"]}]},
            },
        )
    return packet_path, response_dir


def test_provenance_resolves_owned_lines_and_document_models(tmp_path: Path) -> None:
    packet, responses = _write_provenance(
        tmp_path, "a", ("gpt-5.6-sol", "gpt-5.6-terra")
    )
    provenance = load_annotation_provenance(packet, responses)
    assert provenance["line_models"] == {
        "a1": "Sol",
        "a2": "Sol",
        "a3": "Terra",
        "a4": "Terra",
    }
    assert provenance["documents"]["dg"]["model"] == "Sol"
    assert provenance["documents"]["db"]["chunk_counts"] == {"Terra": 1}


def test_task_agreement_uses_downstream_recodings() -> None:
    lines = [
        {"pass_a_role": "ENTRY", "pass_b_role": "CONTINUATION"},
        {"pass_a_role": "BIB_HEADER", "pass_b_role": "BIB_SUBHEADER"},
        {"pass_a_role": "NON_BIB_HEADER", "pass_b_role": "ENTRY"},
        {"pass_a_role": "FILLER", "pass_b_role": "CONTINUATION"},
    ]
    result = build_task_agreement(lines)
    assert result["bibliography_membership"]["exact_agreement"] == 0.75
    heading = result["heading_types"]
    assert heading["candidate_union_including_missed_headings"]["line_count"] == 2
    assert heading["both_identified_a_heading"]["line_count"] == 1
    gap = result["gap_line_types"]
    assert gap["candidate_union_including_missed_gap_lines"]["line_count"] == 2
    assert gap["both_identified_a_gap_line"]["line_count"] == 1


def test_build_site_labels_actual_models(tmp_path: Path) -> None:
    documents, keys, pass_a, pass_b = _fixtures()
    documents_path, key_path = tmp_path / "documents.jsonl", tmp_path / "keys.jsonl"
    pass_a_path, pass_b_path = tmp_path / "a.json", tmp_path / "b.json"
    _write_jsonl(documents_path, documents)
    _write_jsonl(key_path, keys)
    _write_json(pass_a_path, pass_a)
    _write_json(pass_b_path, pass_b)
    packet_a, responses_a = _write_provenance(
        tmp_path, "pa", ("gpt-5.6-sol", "gpt-5.6-terra")
    )
    packet_b, responses_b = _write_provenance(
        tmp_path, "pb", ("gpt-5.6-terra", "gpt-5.6-sol")
    )
    output = tmp_path / "site-with-models"
    build_site(
        documents_path=documents_path,
        line_key_path=key_path,
        pass_a_path=pass_a_path,
        pass_b_path=pass_b_path,
        pass_a_packet_path=packet_a,
        pass_b_packet_path=packet_b,
        pass_a_response_dir=responses_a,
        pass_b_response_dir=responses_b,
        output_dir=output,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    good = next(row for row in manifest["documents"] if row["document_id"] == "doc-good")
    assert good["pass_a_model"] == "Sol"
    assert good["pass_b_model"] == "Terra"
    payload = json.loads((output / "data" / "doc-good.json").read_text())
    assert payload["lines"][0]["annotator_pair"] == "Sol->Terra"
    assert "task_agreement_by_annotator_pair" in manifest
