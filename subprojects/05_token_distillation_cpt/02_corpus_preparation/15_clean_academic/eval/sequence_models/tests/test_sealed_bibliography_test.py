from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import pytest

from sequence_models.sealed_bibliography_test import (
    CHUNK_SCHEMA,
    LINE_KEY_SCHEMA,
    MERGED_SCHEMA,
    PASS_SCHEMA,
    PRIVATE_DOCUMENT_SCHEMA,
    QUALITY_PACKET_SCHEMA,
    QUALITY_RESPONSE_SCHEMA,
    ROLE_RESPONSE_SCHEMA,
    SELECTION_CANDIDATE_SCHEMA,
    SELECTION_RECEIPT_SCHEMA,
    QUALITY_CONSENSUS_SCHEMA,
    Exclusions,
    GlobalSketchIndex,
    _dedup_candidate,
    _row_identity_hashes,
    _ownership,
    _line_id,
    bottom_k_word_shingles,
    build_quality_adjudication_packet,
    chunk_ranges,
    merge_labels,
    merge_quality,
    finalize_selection,
    ingest_batch,
    prepare_run,
    prepare_annotation,
    validate_role_response,
)


def _dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def _dump_rows(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_global_sketch_index_detects_cross_source_copy() -> None:
    copied = " ".join(f"λέξη{i % 37}" for i in range(1200))
    other = " ".join(f"άλλο{i % 41}" for i in range(1200))
    index = GlobalSketchIndex()
    index.add("greek-phd-old", bottom_k_word_shingles(copied))
    assert index.closest(bottom_k_word_shingles(copied)) == ("greek-phd-old", 1.0)
    assert index.closest(bottom_k_word_shingles(other))[1] < 0.2


def test_historical_aliases_and_prior_pool_dedup_are_global() -> None:
    historical = set(_row_identity_hashes("greek_phd", {"doc_id": "same-work"}))
    fresh = set(
        _row_identity_hashes(
            "greek_phd", {"source_doc_id": "same-work", "work_id": "same-work"}
        )
    )
    assert historical & fresh

    copied = " ".join(f"κείμενο{i % 43}" for i in range(1000))
    prior_index = GlobalSketchIndex()
    prior_index.add("prior500:openarchives", bottom_k_word_shingles(copied))
    exclusions = Exclusions(set(), set(), set(), prior_index, [])
    accepted = GlobalSketchIndex()
    reason, _, _, _ = _dedup_candidate(
        text=copied, normalized_hash="1" * 64, exclusions=exclusions,
        accepted_index=accepted, accepted_exact_hashes=set(), threshold=0.8,
    )
    assert reason == "excluded_global_prior_near_duplicate"

    novel = " ".join(f"νέο{i % 47}" for i in range(1000))
    reason, _, signature, exact_key = _dedup_candidate(
        text=novel, normalized_hash="2" * 64, exclusions=exclusions,
        accepted_index=accepted, accepted_exact_hashes=set(), threshold=0.8,
    )
    assert reason is None
    accepted.add("fresh:greek_phd", signature)
    reason, _, _, _ = _dedup_candidate(
        text=novel, normalized_hash="3" * 64, exclusions=exclusions,
        accepted_index=accepted, accepted_exact_hashes={exact_key}, threshold=0.8,
    )
    assert reason == "excluded_global_selected_near_duplicate"


def test_staggered_chunks_have_one_core_owner_and_different_boundaries() -> None:
    lines = [
        {"line_id": f"l{index}", "text": f"line {index} " + "x" * 20}
        for index in range(930)
    ]
    first = chunk_ranges(lines, pass_id="pass-a")
    second = chunk_ranges(lines, pass_id="pass-b")
    assert first != second
    assert first[0][0] == 0 and first[-1][1] == len(lines)
    assert any(start == 0 for start, _ in second)
    assert any(end == len(lines) for _, end in second)
    for ranges in (first, second):
        ownership = _ownership(ranges, len(lines))
        claimed = []
        for (start, _), (owned_start, owned_end) in zip(ranges, ownership, strict=True):
            claimed.extend(range(start + owned_start, start + owned_end))
        assert sorted(claimed) == list(range(len(lines)))
        assert len(claimed) == len(set(claimed))
        assert all(end - start <= 400 for start, end in ranges)


def _chunk() -> dict:
    return {
        "schema_version": CHUNK_SCHEMA,
        "kind": "full_document_role_pass",
        "pass_id": "pass-a",
        "chunk_id": "ch_test",
        "document_alias": "doc_test",
        "source": "greek_phd",
        "presentation_index": 0,
        "start_present_position": 0,
        "end_present_position_exclusive": 4,
        "owned_start_offset": 0,
        "owned_end_offset_exclusive": 4,
        "target_offsets": [],
        "n_physical_lines": 4,
        "n_present_lines": 4,
        "lines": [
            {
                "offset": index,
                "line_alias": f"ln_{index}",
                "abs_idx": index,
                "document_position_percent": index * 25.0,
                "text": f"line {index}",
            }
            for index in range(4)
        ],
    }


def test_rle_validation_requires_exact_complete_coverage() -> None:
    payload = {
        "schema_version": ROLE_RESPONSE_SCHEMA,
        "reviewer": "sol-a",
        "chunks": [
            {
                "chunk_id": "ch_test",
                "runs": [
                    {"start_offset": 0, "end_offset": 1, "role": "OTHER", "confidence": 0.9},
                    {"start_offset": 2, "end_offset": 3, "role": "ENTRY", "confidence": 0.8},
                ],
                "notes": "",
            }
        ],
    }
    assert validate_role_response([_chunk()], payload, "sol-a")["reviewer"] == "sol-a"
    payload["chunks"][0]["runs"][1]["start_offset"] = 3
    with pytest.raises(ValueError, match="exactly and contiguously"):
        validate_role_response([_chunk()], payload, "sol-a")


def test_prepare_annotation_stable_aliases_and_complete_core_coverage(tmp_path: Path) -> None:
    document_id = "a" * 64
    lines = [
        {
            "line_id": _line_id(document_id, index, f"text {index} " + "α" * 30),
            "abs_idx": index,
            "text": f"text {index} " + "α" * 30,
        }
        for index in range(510)
    ]
    documents = tmp_path / "documents.jsonl"
    _dump_rows(
        documents,
        [
            {
                "schema_version": PRIVATE_DOCUMENT_SCHEMA,
                "document_id": document_id,
                "source": "greek_phd",
                "n_physical_lines": len(lines),
                "lines": lines,
            }
        ],
    )
    selection = tmp_path / "selection.json"
    _dump(selection, {"sealed_outputs": {"documents_sha256": _hash(documents)}})
    secret = tmp_path / "alias.key"
    secret.write_bytes(b"s" * 32)
    secret.chmod(0o600)
    pass_a = tmp_path / "a.jsonl"
    pass_b = tmp_path / "b.jsonl"
    key = tmp_path / "key.jsonl"
    receipt = tmp_path / "receipt.json"
    result = prepare_annotation(
        argparse.Namespace(
            documents=str(documents), selection_receipt=str(selection), alias_secret=str(secret),
            max_lines=400, max_chars=80_000, overlap=15, pass_a_out=str(pass_a),
            pass_b_out=str(pass_b), line_key_out=str(key), receipt_out=str(receipt),
        )
    )
    assert result["line_count"] == 510
    key_rows = [json.loads(line) for line in key.read_text().splitlines()]
    assert len(key_rows) == 510
    assert len({row["line_alias"] for row in key_rows}) == 510
    a_rows = [json.loads(line) for line in pass_a.read_text().splitlines()]
    b_rows = [json.loads(line) for line in pass_b.read_text().splitlines()]
    assert [(row["start_present_position"], row["end_present_position_exclusive"]) for row in a_rows] != [
        (row["start_present_position"], row["end_present_position_exclusive"]) for row in b_rows
    ]
    for packet in (a_rows, b_rows):
        owners = []
        for row in packet:
            owners.extend(
                line["line_alias"]
                for line in row["lines"][
                    row["owned_start_offset"] : row["owned_end_offset_exclusive"]
                ]
            )
        assert set(owners) == {row["line_alias"] for row in key_rows}
        assert len(owners) == len(set(owners))


def _quality_response(reviewer: str, rows: list[tuple[str, str]]) -> dict:
    return {
        "schema_version": QUALITY_RESPONSE_SCHEMA,
        "reviewer": reviewer,
        "documents": [
            {"document_alias": alias, "decision": decision, "confidence": 0.9, "reasons": ["review"]}
            for alias, decision in rows
        ],
    }


def test_quality_third_packet_is_blind_disagreement_subset_and_merges(tmp_path: Path) -> None:
    packet = tmp_path / "quality.jsonl"
    packet_rows = [
        {"schema_version": QUALITY_PACKET_SCHEMA, "document_alias": "q_a", "sample_lines": []},
        {"schema_version": QUALITY_PACKET_SCHEMA, "document_alias": "q_b", "sample_lines": []},
    ]
    _dump_rows(packet, packet_rows)
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _dump(a, _quality_response("quality-a", [("q_a", "KEEP"), ("q_b", "KEEP")]))
    _dump(b, _quality_response("quality-b", [("q_a", "UNUSABLE"), ("q_b", "KEEP")]))
    third_packet = tmp_path / "third.jsonl"
    third_receipt = tmp_path / "third.receipt.json"
    build_quality_adjudication_packet(
        argparse.Namespace(
            packet=str(packet), response_a=str(a), reviewer_a="quality-a",
            response_b=str(b), reviewer_b="quality-b", output=str(third_packet),
            receipt_out=str(third_receipt),
        )
    )
    third_rows = [json.loads(line) for line in third_packet.read_text().splitlines()]
    assert [row["document_alias"] for row in third_rows] == ["q_a"]
    assert "decision" not in third_rows[0]
    _dump(c, _quality_response("quality-c", [("q_a", "KEEP")]))
    output = tmp_path / "consensus.json"
    merged = merge_quality(
        argparse.Namespace(
            packet=str(packet), response=[str(a), str(b), str(c)],
            reviewer_id=["quality-a", "quality-b", "quality-c"], output=str(output),
        )
    )
    assert merged["status"] == "passed"
    assert {row["document_alias"]: row["decision"] for row in merged["documents"]} == {
        "q_a": "KEEP", "q_b": "KEEP"
    }


def test_finalize_selection_emits_private_provenance_but_public_hashes_only(tmp_path: Path) -> None:
    candidates = []
    for source_index, source in enumerate(("greek_phd", "kallipos", "openarchives")):
        for index in range(50):
            number = source_index * 50 + index + 1
            candidates.append(
                {
                    "schema_version": SELECTION_CANDIDATE_SCHEMA,
                    "document_id": f"{number:064x}", "source": source,
                    "source_dataset": source, "source_doc_id": f"source-{number}",
                    "work_id": f"work-{number}", "work_key": f"work-{number}",
                    "stable_uid": f"stable-{number}", "source_repo_id": "repo",
                    "source_revision": "rev", "source_artifact_path": "private/path",
                    "source_row_id": str(number), "representation_generation": "canonical",
                    "original_text_sha256": f"{number + 1000:064x}",
                    "normalized_text_sha256": f"{number + 2000:064x}",
                    "materialized_text_sha256": f"{number + 3000:064x}",
                    "selection_rank": f"{index:064x}", "n_physical_lines": 1,
                    "n_present_lines": 1, "text_characters": 4,
                    "quality": {"flagged_for_dual_sol": False},
                    "near_duplicate_audit": {"threshold": 0.8},
                    "lines": [{"line_id": f"{number + 4000:064x}", "abs_idx": 0, "text": "text"}],
                }
            )
    candidate_path = tmp_path / "candidates.jsonl"
    packet_path = tmp_path / "quality.jsonl"
    _dump_rows(candidate_path, candidates)
    packet_path.write_text("", encoding="utf-8")
    candidate_receipt = tmp_path / "candidate.receipt.json"
    _dump(
        candidate_receipt,
        {
            "schema_version": SELECTION_RECEIPT_SCHEMA,
            "quotas": {source: 50 for source in ("greek_phd", "kallipos", "openarchives")},
            "outputs": {
                "candidates_sha256": _hash(candidate_path),
                "quality_packet_sha256": _hash(packet_path),
            },
        },
    )
    quality = tmp_path / "quality.consensus.json"
    _dump(
        quality,
        {
            "schema_version": QUALITY_CONSENSUS_SCHEMA, "status": "passed",
            "packet_sha256": _hash(packet_path), "documents": [],
        },
    )
    documents = tmp_path / "documents.private.jsonl"
    public = tmp_path / "exclusions.public.json"
    receipt = tmp_path / "selection.final.json"
    result = finalize_selection(
        argparse.Namespace(
            candidates=str(candidate_path), candidate_receipt=str(candidate_receipt),
            quality_consensus=str(quality), documents_out=str(documents),
            public_exclusions_out=str(public), receipt_out=str(receipt),
        )
    )
    assert result["source_counts"] == {"greek_phd": 50, "kallipos": 50, "openarchives": 50}
    private_row = json.loads(documents.read_text().splitlines()[0])
    assert private_row["source_artifact_path"] == "private/path"
    public_value = json.loads(public.read_text())
    assert public_value["document_count"] == 150
    assert all(
        set(row) == {
            "document_id", "source_identity_sha256", "source_doc_identity_sha256",
            "work_identity_sha256", "stable_identity_sha256", "normalized_text_sha256",
            "materialized_text_sha256",
        }
        for row in public_value["documents"]
    )
    assert "private/path" not in public.read_text()


def test_immutable_remote_ingest_rejects_changed_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    packet = tmp_path / "packet.jsonl"
    _dump_rows(packet, [_chunk()])
    prompt = tmp_path / "prompt.md"
    schema = tmp_path / "schema.json"
    prompt.write_text("prompt", encoding="utf-8")
    schema.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    prepare_run(
        argparse.Namespace(
            packet=str(packet), pass_id="pass-a", reviewer_id="sol-a",
            model="gpt-5.6-sol", reasoning_effort="high", prompt=str(prompt),
            output_schema=str(schema), batch_size=1, run_dir=str(run_dir),
        )
    )
    payload = {
        "schema_version": ROLE_RESPONSE_SCHEMA, "reviewer": "sol-a",
        "chunks": [{
            "chunk_id": "ch_test", "notes": "",
            "runs": [{"start_offset": 0, "end_offset": 3, "role": "OTHER", "confidence": 0.9}],
        }],
    }
    arguments = argparse.Namespace(packet=str(packet), run_dir=str(run_dir), batch_index=0)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert ingest_batch(arguments)["status"] == "accepted"
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    assert ingest_batch(arguments)["status"] == "accepted"
    payload["chunks"][0]["runs"][0]["confidence"] = 0.8
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload)))
    with pytest.raises(ValueError, match="immutable artifact differs"):
        ingest_batch(arguments)


def _pass(path: Path, pass_id: str, reviewer: str, roles: dict[str, str]) -> None:
    _dump(
        path,
        {
            "schema_version": PASS_SCHEMA,
            "pass_id": pass_id,
            "reviewer": reviewer,
            "lines": [
                {
                    "line_alias": alias, "document_alias": "doc", "source": source,
                    "role": role, "confidence": 0.9,
                }
                for alias, (source, role) in sorted(roles.items())
            ],
        },
    )


def test_role_merge_uses_third_vote_and_enforces_source_gates(tmp_path: Path) -> None:
    keys = []
    roles_a: dict[str, tuple[str, str]] = {}
    roles_b: dict[str, tuple[str, str]] = {}
    for source_index, source in enumerate(("greek_phd", "kallipos", "openarchives")):
        for index in range(100):
            alias = f"ln_{source_index}_{index}"
            keys.append(
                {
                    "schema_version": LINE_KEY_SCHEMA, "line_alias": alias,
                    "document_alias": f"doc_{source}", "document_id": f"d{source_index}",
                    "line_id": f"l{source_index}_{index}", "abs_idx": index, "source": source,
                }
            )
            roles_a[alias] = (source, "ENTRY" if index >= 50 else "OTHER")
            roles_b[alias] = roles_a[alias]
    disagree = "ln_0_70"
    roles_b[disagree] = ("greek_phd", "OTHER")
    key_path = tmp_path / "key.jsonl"
    _dump_rows(key_path, keys)
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    c = tmp_path / "c.json"
    _pass(a, "pass-a", "sol-a", roles_a)
    _pass(b, "pass-b", "sol-b", roles_b)
    _pass(c, "adjudication", "sol-c", {disagree: ("greek_phd", "ENTRY")})
    output = tmp_path / "labels.jsonl"
    receipt = tmp_path / "receipt.json"
    result = merge_labels(
        argparse.Namespace(
            line_key=str(key_path), pass_a=str(a), pass_b=str(b), adjudication=str(c),
            output=str(output), receipt_out=str(receipt),
        )
    )
    assert result["status"] == "passed"
    assert result["a_b_binary_agreement_overall"] == pytest.approx(299 / 300)
    labels = [json.loads(line) for line in output.read_text().splitlines()]
    assert next(row for row in labels if row["line_alias"] == disagree)["role"] == "ENTRY"
    assert all(row["schema_version"] == MERGED_SCHEMA for row in labels)
