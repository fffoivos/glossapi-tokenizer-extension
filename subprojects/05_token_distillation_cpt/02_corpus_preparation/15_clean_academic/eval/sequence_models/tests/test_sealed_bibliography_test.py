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
    QUALITY_BATCH_MAX_CHARS,
    QUALITY_DOCUMENT_MAX_CHARS,
    CONSENSUS_SCHEMA,
    Exclusions,
    GlobalSketchIndex,
    _dedup_candidate,
    _directory_inventory_sha256,
    _document_id,
    _load_chunks,
    _public_exclusion_manifest,
    _merge_shard_inventory,
    _quality_batches,
    _quality_sample,
    _require_expected_sha256,
    _validate_quality_packet_row,
    _verify_consumed_shards,
    _row_identity_hashes,
    _ownership,
    _line_id,
    bottom_k_word_shingles,
    build_quality_adjudication_packet,
    chunk_ranges,
    merge_labels,
    merge_quality,
    finalize_selection,
    finalize_pass,
    freeze,
    export_quality_batch,
    ingest_batch,
    load_public_exclusions,
    load_exclusions,
    prepare_run,
    prepare_annotation,
    prepare_quality_run,
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
        accepted_index=accepted, accepted_normalized_hashes=set(),
        accepted_materialized_hashes=set(), threshold=0.8,
    )
    assert reason == "excluded_global_prior_near_duplicate"

    novel = " ".join(f"νέο{i % 47}" for i in range(1000))
    reason, _, signature, materialized_hash = _dedup_candidate(
        text=novel, normalized_hash="2" * 64, exclusions=exclusions,
        accepted_index=accepted, accepted_normalized_hashes=set(),
        accepted_materialized_hashes=set(), threshold=0.8,
    )
    assert reason is None
    accepted.add("fresh:greek_phd", signature)
    reason, _, _, _ = _dedup_candidate(
        text=novel, normalized_hash="3" * 64, exclusions=exclusions,
        accepted_index=accepted, accepted_normalized_hashes={"2" * 64},
        accepted_materialized_hashes={materialized_hash}, threshold=0.8,
    )
    assert reason == "excluded_exact_selected_materialized_text"
    reason, _, _, _ = _dedup_candidate(
        text="prefix " + novel, normalized_hash="4" * 64, exclusions=exclusions,
        accepted_index=accepted, accepted_normalized_hashes={"2" * 64},
        accepted_materialized_hashes={materialized_hash}, threshold=0.8,
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


def test_staggered_prefix_respects_character_cap_with_pathologically_long_lines() -> None:
    lines = [
        {"line_id": f"l{index}", "text": f"{index}:" + "x" * 29_995}
        for index in range(40)
    ]
    ranges = chunk_ranges(
        lines, pass_id="pass-b", max_lines=400, max_chars=80_000, overlap=15
    )
    assert ranges
    assert all(
        sum(len(row["text"]) for row in lines[start:end]) + max(end - start - 1, 0)
        <= 80_000
        for start, end in ranges
    )
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
        "max_lines": 400,
        "max_characters": 80_000,
        "max_display_characters_per_line": 20_000,
        "n_physical_lines": 4,
        "n_present_lines": 4,
        "lines": [
            {
                "offset": index,
                "line_alias": f"ln_{index}",
                "abs_idx": index,
                "document_position_percent": index * 25.0,
                "text": f"line {index}",
                "display_truncated": False,
                "original_character_count": len(f"line {index}"),
                "display_character_count": len(f"line {index}"),
                "original_text_sha256": hashlib.sha256(
                    f"line {index}".encode("utf-8")
                ).hexdigest(),
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


def test_packet_loader_rechecks_declared_line_and_character_caps(tmp_path: Path) -> None:
    chunk = _chunk()
    chunk["max_characters"] = 20
    chunk["max_display_characters_per_line"] = 20
    packet = tmp_path / "over-cap.jsonl"
    _dump_rows(packet, [chunk])
    with pytest.raises(ValueError, match="characters, cap is"):
        _load_chunks(packet)


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


def test_role_packet_bounds_overlong_line_but_key_and_sealed_text_remain_full(tmp_path: Path) -> None:
    document_id = "b" * 64
    original = "Αρχή " + "β" * 200_000 + " τέλος"
    line_id = _line_id(document_id, 0, original)
    documents = tmp_path / "documents.jsonl"
    _dump_rows(
        documents,
        [{
            "schema_version": PRIVATE_DOCUMENT_SCHEMA, "document_id": document_id,
            "source": "openarchives", "n_physical_lines": 1, "n_present_lines": 1,
            "lines": [{"line_id": line_id, "abs_idx": 0, "text": original}],
        }],
    )
    selection = tmp_path / "selection.json"
    _dump(selection, {"sealed_outputs": {"documents_sha256": _hash(documents)}})
    secret = tmp_path / "alias.key"
    secret.write_bytes(b"k" * 32)
    secret.chmod(0o600)
    pass_a, pass_b, key, receipt = (
        tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "key.jsonl",
        tmp_path / "receipt.json",
    )
    prepare_annotation(
        argparse.Namespace(
            documents=str(documents), selection_receipt=str(selection), alias_secret=str(secret),
            max_lines=400, max_chars=80_000, overlap=15, pass_a_out=str(pass_a),
            pass_b_out=str(pass_b), line_key_out=str(key), receipt_out=str(receipt),
        )
    )
    packet = json.loads(pass_a.read_text().splitlines()[0])
    display = packet["lines"][0]
    assert display["display_truncated"] is True
    assert display["original_character_count"] == len(original)
    assert display["display_character_count"] <= 20_000
    assert "⟦DISPLAY TRUNCATED⟧" in display["text"]
    assert json.loads(key.read_text().splitlines()[0])["line_id"] == line_id
    assert json.loads(documents.read_text().splitlines()[0])["lines"][0]["text"] == original
    assert all(
        sum(len(line["text"]) for line in row["lines"]) + max(len(row["lines"]) - 1, 0)
        <= 80_000
        for row in (json.loads(value) for value in pass_b.read_text().splitlines())
    )


def _quality_response(reviewer: str, rows: list[tuple[str, str]]) -> dict:
    return {
        "schema_version": QUALITY_RESPONSE_SCHEMA,
        "reviewer": reviewer,
        "documents": [
            {"document_alias": alias, "decision": decision, "confidence": 0.9, "reasons": ["review"]}
            for alias, decision in rows
        ],
    }


def _quality_packet_row(alias: str, text: str = "sample") -> dict:
    line = {"line_id": hashlib.sha256(alias.encode()).hexdigest(), "abs_idx": 0, "text": text}
    sample, policy = _quality_sample([line])
    return {
        "schema_version": QUALITY_PACKET_SCHEMA,
        "document_alias": alias,
        "sample_policy": policy,
        "sample_lines": sample,
    }


def test_quality_sampling_bounds_overlong_display_without_mutating_source() -> None:
    original = "α" * 500_000
    row = _quality_packet_row("q_long", original)
    sample = row["sample_lines"][0]
    assert sample["display_truncated"] is True
    assert sample["original_character_count"] == len(original)
    assert "⟦DISPLAY TRUNCATED⟧" in sample["text"]
    assert original == "α" * 500_000
    serialized = _validate_quality_packet_row(
        row, max_document_characters=QUALITY_DOCUMENT_MAX_CHARS
    )
    assert serialized <= QUALITY_DOCUMENT_MAX_CHARS
    batches = _quality_batches(
        [row, _quality_packet_row("q_short")], batch_size=2,
        max_document_characters=QUALITY_DOCUMENT_MAX_CHARS,
        max_batch_characters=QUALITY_BATCH_MAX_CHARS,
    )
    assert all(len(batch) <= 2 for batch in batches)
    assert all(
        sum(len(json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))) for item in batch)
        + max(len(batch) - 1, 0)
        <= QUALITY_BATCH_MAX_CHARS
        for batch in batches
    )


def test_quality_run_binds_caps_and_export_revalidates_bounded_envelope(tmp_path: Path) -> None:
    packet = tmp_path / "quality.jsonl"
    _dump_rows(packet, [_quality_packet_row("q_long", "α" * 500_000), _quality_packet_row("q_2")])
    prompt = tmp_path / "prompt.md"
    schema = tmp_path / "schema.json"
    prompt.write_text("prompt", encoding="utf-8")
    schema.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    contract = prepare_quality_run(
        argparse.Namespace(
            packet=str(packet), pass_id="quality-a", reviewer_id="quality-sol-a",
            model="gpt-5.6-sol", reasoning_effort="high", prompt=str(prompt),
            output_schema=str(schema), batch_size=2,
            max_document_characters=QUALITY_DOCUMENT_MAX_CHARS,
            max_batch_characters=QUALITY_BATCH_MAX_CHARS, run_dir=str(run_dir),
        )
    )
    assert contract["quality_character_caps"]["serialized_per_document"] == QUALITY_DOCUMENT_MAX_CHARS
    envelope = export_quality_batch(
        argparse.Namespace(packet=str(packet), run_dir=str(run_dir), batch_index=0)
    )
    assert envelope["status"] == "pending"
    assert envelope["envelope_character_count"] <= QUALITY_BATCH_MAX_CHARS
    assert envelope["envelope_character_count"] == len(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def test_quality_third_packet_is_blind_disagreement_subset_and_merges(tmp_path: Path) -> None:
    packet = tmp_path / "quality.jsonl"
    packet_rows = [
        _quality_packet_row("q_a"),
        _quality_packet_row("q_b"),
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
            "document_id", "canonical_identity_sha256s", "normalized_text_sha256",
            "materialized_text_sha256",
        }
        for row in public_value["documents"]
    )
    assert "private/path" not in public.read_text()


def test_public_exclusion_round_trip_rejects_cross_field_alias_and_exact_copies(
    tmp_path: Path,
) -> None:
    shared_alias = "source-doc-exposed-later-as-work-id"
    row = {
        "document_id": "1" * 64,
        "source": "greek_phd",
        "source_doc_id": shared_alias,
        "work_id": "work-id",
        "work_key": "work-key",
        "stable_uid": "stable-id",
        "normalized_text_sha256": "2" * 64,
        "materialized_text_sha256": "3" * 64,
    }
    manifest_path = tmp_path / "public.json"
    _dump(manifest_path, _public_exclusion_manifest([row]))

    loaded = load_public_exclusions(manifest_path)

    assert loaded.rejects("greek_phd", {"work_id": shared_alias})
    assert not loaded.rejects("kallipos", {"work_id": shared_alias})
    assert loaded.rejects(
        "kallipos",
        {"work_id": "new-work", "normalized_text_sha256": "2" * 64},
    )
    assert loaded.rejects(
        "openarchives",
        {"stable_uid": "new-stable", "materialized_text_sha256": "3" * 64},
    )
    assert shared_alias not in manifest_path.read_text(encoding="utf-8")


def _descriptor(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _hash(path)}


def _pinned_route(
    tmp_path: Path, *, receipt_count: int = 1
) -> tuple[list[Path], list[Path], Path, Path]:
    root = (tmp_path / "normalization").resolve()
    canonical = root / "canonical"
    source_receipt = canonical / ".receipts" / "sources" / "route.json"
    source_receipt.parent.mkdir(parents=True)
    _dump(source_receipt, {"schema_version": "source-receipt-v1"})
    shards: list[Path] = []
    receipts: list[Path] = []
    shard_descriptors = []
    receipt_descriptors = []
    for index in range(receipt_count):
        shard = canonical / "route" / f"part-{index:05d}.parquet"
        shard.parent.mkdir(parents=True, exist_ok=True)
        shard.write_bytes(f"PAR1-shard-{index}".encode())
        shard_descriptor = _descriptor(shard)
        receipt = canonical / ".receipts" / "files" / "route" / f"file-{index:05d}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        _dump(
            receipt,
            {
                "exact_source_dataset_counts": {"dataset": 1},
                "shards": [{"output": shard_descriptor}],
            },
        )
        shards.append(shard)
        receipts.append(receipt)
        shard_descriptors.append(shard_descriptor)
        receipt_descriptors.append(_descriptor(receipt))
    manifest = root / "normalization_manifest.json"
    _dump(
        manifest,
        {
            "schema_version": "full_cpt_normalization_manifest_v1",
            "output": str(canonical),
            "sources": [
                {
                    "source_id": "route",
                    "receipt": _descriptor(source_receipt),
                    "files": receipt_descriptors,
                    "shards": shard_descriptors,
                }
            ],
        },
    )
    return shards, receipts, manifest, root


@pytest.mark.parametrize("replacement", [b"PAR1-changed", b"longer-than-the-original"])
def test_consumed_shard_verification_rejects_hash_or_size_drift(
    tmp_path: Path, replacement: bytes
) -> None:
    shards, receipts, manifest, root = _pinned_route(tmp_path)
    shards[0].write_bytes(replacement)
    with pytest.raises(ValueError, match="bytes differ from the normalization manifest"):
        _verify_consumed_shards(
            shards,
            receipts,
            normalization_manifest=manifest,
            normalization_root=root,
            receipt_source="route",
            source_dataset="dataset",
        )


def test_consumed_shard_verification_rejects_symlink_and_duplicate_paths(
    tmp_path: Path,
) -> None:
    shards, receipts, manifest, root = _pinned_route(tmp_path)
    original = shards[0].read_bytes()
    target = tmp_path / "symlink-target.parquet"
    target.write_bytes(original)
    shards[0].unlink()
    shards[0].symlink_to(target)
    with pytest.raises(ValueError, match="linked/non-canonical"):
        _verify_consumed_shards(
            shards,
            receipts,
            normalization_manifest=manifest,
            normalization_root=root,
            receipt_source="route",
            source_dataset="dataset",
        )

    shards, receipts, manifest, root = _pinned_route(tmp_path / "duplicate")
    with pytest.raises(ValueError, match="duplicated"):
        _verify_consumed_shards(
            [shards[0], shards[0]],
            receipts,
            normalization_manifest=manifest,
            normalization_root=root,
            receipt_source="route",
            source_dataset="dataset",
        )


def test_consumed_shards_require_every_pinned_relevant_receipt(tmp_path: Path) -> None:
    shards, receipts, manifest, root = _pinned_route(tmp_path, receipt_count=2)
    receipts[1].unlink()
    with pytest.raises(ValueError, match="normalization file receipt is absent"):
        _verify_consumed_shards(
            shards[:1],
            receipts[:1],
            normalization_manifest=manifest,
            normalization_root=root,
            receipt_source="route",
            source_dataset="dataset",
        )


def test_identical_shared_route_shards_are_deduplicated_in_global_inventory(
    tmp_path: Path,
) -> None:
    shards, receipts, manifest, root = _pinned_route(tmp_path)
    inventory = _verify_consumed_shards(
        shards,
        receipts,
        normalization_manifest=manifest,
        normalization_root=root,
        receipt_source="route",
        source_dataset="dataset",
    )
    combined: dict[str, dict] = {}
    _merge_shard_inventory(combined, inventory)
    _merge_shard_inventory(combined, inventory)
    assert list(combined.values()) == inventory


def test_historical_text_inventory_must_match_expected_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sequence_models import source_matched_holdout

    historical_manifest = tmp_path / "historical.jsonl"
    previous = tmp_path / "previous.jsonl"
    historical_manifest.write_text("", encoding="utf-8")
    previous.write_text("", encoding="utf-8")
    historical_rows = [{"source": "greek_phd", "doc_id": "historical"}]

    def fake_manifest(_path: Path) -> tuple[list[dict], dict]:
        return historical_rows, {}

    def fake_texts(_rows: list[dict], _root: Path) -> list[dict]:
        fake_texts.inventory_sha256 = "a" * 64
        return [{"source": "greek_phd", "doc_id": "historical", "text": "text"}]

    monkeypatch.setattr(source_matched_holdout, "load_historical_manifest", fake_manifest)
    monkeypatch.setattr(source_matched_holdout, "load_historical_texts", fake_texts)
    with pytest.raises(ValueError, match="historical text inventory SHA256 drift"):
        load_exclusions(
            historical_manifest=historical_manifest,
            historical_root=tmp_path,
            previous_documents=previous,
            expected_historical_inventory_sha256="b" * 64,
            expected_historical=1,
            expected_previous=0,
        )


def test_rust_package_inventory_ignores_mutable_python_cache(tmp_path: Path) -> None:
    package = tmp_path / "glossapi_rs_noise"
    package.mkdir()
    (package / "__init__.py").write_text("version = 1\n", encoding="utf-8")
    (package / "glossapi_rs_noise.abi3.so").write_bytes(b"rust-extension")
    before = _directory_inventory_sha256(package)
    cache = package / "__pycache__"
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"mutable cache")
    assert _directory_inventory_sha256(package) == before
    with pytest.raises(ValueError, match="GlossAPI Rust package inventory SHA256 drift"):
        _require_expected_sha256(
            before, "0" * 64, label="GlossAPI Rust package inventory"
        )


def _freeze_inputs(tmp_path: Path) -> tuple[argparse.Namespace, Path]:
    documents = []
    labels = []
    for source_index, source in enumerate(("greek_phd", "kallipos", "openarchives")):
        for index in range(50):
            number = source_index * 50 + index
            stable_uid = f"stable-{number}"
            materialized_hash = hashlib.sha256(f"materialized-{number}".encode()).hexdigest()
            document_id = _document_id(source, stable_uid, materialized_hash)
            text = f"line {number}"
            line_id = _line_id(document_id, 0, text)
            documents.append(
                {
                    "schema_version": PRIVATE_DOCUMENT_SCHEMA,
                    "document_id": document_id,
                    "source": source,
                    "source_doc_id": f"source-{number}",
                    "work_id": f"work-{number}",
                    "work_key": f"work-{number}",
                    "stable_uid": stable_uid,
                    "normalized_text_sha256": hashlib.sha256(
                        f"normalized-{number}".encode()
                    ).hexdigest(),
                    "materialized_text_sha256": materialized_hash,
                    "n_present_lines": 1,
                    "lines": [{"line_id": line_id, "abs_idx": 0, "text": text}],
                }
            )
            labels.append(
                {
                    "schema_version": MERGED_SCHEMA,
                    "document_id": document_id,
                    "line_id": line_id,
                    "line_alias": f"line-{number}",
                    "abs_idx": 0,
                    "source": source,
                    "role": "OTHER",
                    "binary_label": "NON_BIB",
                    "votes": ["OTHER", "OTHER"],
                    "consensus_count": 2,
                    "label_origin": "dual_sol",
                }
            )

    documents_path = tmp_path / "documents.jsonl"
    labels_path = tmp_path / "labels.jsonl"
    public_path = tmp_path / "public.json"
    consensus_path = tmp_path / "consensus.json"
    _dump_rows(documents_path, documents)
    _dump_rows(labels_path, labels)
    _dump(public_path, _public_exclusion_manifest(documents))
    _dump(
        consensus_path,
        {
            "schema_version": CONSENSUS_SCHEMA,
            "status": "passed",
            "label_semantics": "dual-Sol LLM-silver; not human gold",
            "line_count": len(labels),
            "a_b_binary_agreement_overall": 1.0,
            "a_b_binary_agreement_by_source": {
                "greek_phd": 1.0,
                "kallipos": 1.0,
                "openarchives": 1.0,
            },
            "unresolved_count": 0,
            "labels_sha256": _hash(labels_path),
            "unresolved_fraction": 0.0,
            "gates": {
                "complete_coverage": True,
                "binary_agreement_overall_gte_0_98": True,
                "binary_agreement_each_source_gte_0_95": True,
                "unresolved_fraction_lte_0_005": True,
            },
            "inputs": {
                "line_key_sha256": "1" * 64,
                "pass_a_sha256": "2" * 64,
                "pass_b_sha256": "3" * 64,
                "adjudication_sha256": None,
            },
        },
    )
    return (
        argparse.Namespace(
            documents=str(documents_path),
            public_exclusions=str(public_path),
            labels=str(labels_path),
            consensus_receipt=str(consensus_path),
            output=str(tmp_path / "freeze.json"),
            lock_inputs=False,
        ),
        public_path,
    )


@pytest.mark.parametrize("corruption", ["identity_hash", "source_counts"])
def test_freeze_rejects_structurally_valid_public_manifest_drift(
    tmp_path: Path, corruption: str
) -> None:
    args, public_path = _freeze_inputs(tmp_path)
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if corruption == "identity_hash":
        identities = set(public["documents"][0]["canonical_identity_sha256s"])
        identities.remove(next(iter(identities)))
        identities.add(hashlib.sha256(b"corrupted public identity").hexdigest())
        public["documents"][0]["canonical_identity_sha256s"] = sorted(identities)
    else:
        public["source_counts"]["greek_phd"] -= 1
        public["source_counts"]["kallipos"] += 1
    _dump(public_path, public)
    load_public_exclusions(public_path)

    with pytest.raises(ValueError, match="differ from the sealed private documents"):
        freeze(args)


def test_freeze_rejects_missing_or_false_terminal_gate(tmp_path: Path) -> None:
    args, _ = _freeze_inputs(tmp_path)
    consensus_path = Path(args.consensus_receipt)
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    consensus["gates"]["complete_coverage"] = False
    _dump(consensus_path, consensus)
    with pytest.raises(ValueError, match="every passed terminal gate"):
        freeze(args)


def test_freeze_recomputes_role_from_votes(tmp_path: Path) -> None:
    args, _ = _freeze_inputs(tmp_path)
    labels_path = Path(args.labels)
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    labels[0]["role"] = "UNKNOWN"
    labels[0]["binary_label"] = None
    labels[0]["consensus_count"] = 0
    _dump_rows(labels_path, labels)
    consensus_path = Path(args.consensus_receipt)
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    consensus["labels_sha256"] = _hash(labels_path)
    _dump(consensus_path, consensus)
    with pytest.raises(ValueError, match="role/consensus differs from its votes"):
        freeze(args)


def test_freeze_recomputes_binary_agreement_from_votes(tmp_path: Path) -> None:
    args, _ = _freeze_inputs(tmp_path)
    consensus_path = Path(args.consensus_receipt)
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    consensus["a_b_binary_agreement_overall"] = 0.99
    _dump(consensus_path, consensus)
    with pytest.raises(ValueError, match="metrics differ from sealed votes"):
        freeze(args)


def test_freeze_recomputes_unknown_count_and_fraction(tmp_path: Path) -> None:
    args, _ = _freeze_inputs(tmp_path)
    labels_path = Path(args.labels)
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    labels[0].update(
        {
            "role": "UNKNOWN",
            "binary_label": None,
            "votes": ["UNKNOWN", "ENTRY", "FILLER"],
            "consensus_count": 0,
            "label_origin": "dual_sol_plus_de_novo_third",
        }
    )
    _dump_rows(labels_path, labels)
    consensus_path = Path(args.consensus_receipt)
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    consensus["labels_sha256"] = _hash(labels_path)
    consensus["a_b_binary_agreement_overall"] = 149 / 150
    consensus["a_b_binary_agreement_by_source"]["greek_phd"] = 49 / 50
    consensus["inputs"]["adjudication_sha256"] = "4" * 64
    _dump(consensus_path, consensus)
    with pytest.raises(ValueError, match="unresolved count/fraction differs"):
        freeze(args)


def test_freeze_rejects_impossible_adjudication_structure(tmp_path: Path) -> None:
    args, _ = _freeze_inputs(tmp_path)
    labels_path = Path(args.labels)
    labels = [json.loads(line) for line in labels_path.read_text().splitlines()]
    labels[0].update(
        {
            "votes": ["OTHER", "OTHER", "ENTRY"],
            "label_origin": "dual_sol_plus_de_novo_third",
        }
    )
    _dump_rows(labels_path, labels)
    consensus_path = Path(args.consensus_receipt)
    consensus = json.loads(consensus_path.read_text(encoding="utf-8"))
    consensus["labels_sha256"] = _hash(labels_path)
    consensus["inputs"]["adjudication_sha256"] = "4" * 64
    _dump(consensus_path, consensus)
    with pytest.raises(ValueError, match="impossible A/B/adjudication"):
        freeze(args)


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


def test_finalize_pass_rejects_packet_drift_before_aggregation(tmp_path: Path) -> None:
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
    changed = _chunk()
    changed["source"] = "openarchives"
    packet.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from the immutable run contract"):
        finalize_pass(
            argparse.Namespace(
                packet=str(packet), run_dir=str(run_dir), output=str(tmp_path / "pass.json")
            )
        )


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
