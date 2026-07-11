from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    scripts = str(path.parent)
    sys.path.insert(0, scripts)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts)
    return module


PRODUCTION = load_module(
    "phase04_structural_span_production",
    HERE / "scripts" / "structural_span_production.py",
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rust_offsets_are_verified_as_python_codepoints() -> None:
    text = "😀 πρόλογος\nἙλληνικὴ βιβλιογραφία 📚\nκανονικό κείμενο"
    selected = "Ἑλληνικὴ βιβλιογραφία 📚"
    start = text.index(selected)
    end = start + len(selected)
    span = {
        "char_start": start,
        "char_end": end,
        "original_chars": len(text),
        "trigger": selected[:40],
    }
    assert PRODUCTION._verify_python_codepoint_span(span, text) == (start, end)

    # UTF-8 byte offsets differ because of the astral and Greek characters and
    # must fail even though the numbers remain ordinary integers.
    byte_start = len(text[:start].encode("utf-8"))
    with pytest.raises(ValueError, match="offsets|boundaries|trigger"):
        PRODUCTION._verify_python_codepoint_span(
            {**span, "char_start": byte_start, "char_end": byte_start + len(selected)},
            text,
        )


def test_source_balanced_selection_preserves_per_source_risk_order(
    tmp_path: Path,
) -> None:
    database = tmp_path / "candidates.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE candidates (prediction_id TEXT, source_dataset TEXT, risk_score REAL)"
    )
    for source in ("a", "b", "c"):
        for rank in range(5):
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?)",
                (f"{source}-{rank}", source, 10.0 - rank),
            )
    connection.commit()
    connection.close()

    selected = PRODUCTION._balanced_select(database, 9)
    counts: dict[str, int] = {}
    by_source: dict[str, list[float]] = {}
    for row in selected:
        counts[row["source_dataset"]] = counts.get(row["source_dataset"], 0) + 1
        by_source.setdefault(row["source_dataset"], []).append(row["risk_score"])
    assert counts == {"a": 3, "b": 3, "c": 3}
    assert all(values == sorted(values, reverse=True) for values in by_source.values())


def test_production_packet_requires_fifty_cases_per_head(tmp_path: Path) -> None:
    database = tmp_path / "two-head.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE candidates (prediction_id TEXT, source_dataset TEXT, kind TEXT, risk_score REAL)"
    )
    for kind in ("bib_span", "toc_span"):
        for index in range(50):
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?)",
                (f"{kind}-{index}", f"source-{index % 5}", kind, 100.0 - index),
            )
    connection.commit()
    connection.close()
    selected = PRODUCTION._balanced_select(database, 100, require_balanced_heads=True)
    assert sum(row["kind"] == "bib_span" for row in selected) == 50
    assert sum(row["kind"] == "toc_span" for row in selected) == 50

    connection = sqlite3.connect(database)
    connection.execute(
        "DELETE FROM candidates WHERE kind='toc_span' AND prediction_id='toc_span-49'"
    )
    connection.commit()
    connection.close()
    with pytest.raises(ValueError, match="50 are required"):
        PRODUCTION._balanced_select(database, 100, require_balanced_heads=True)


def test_joint_llm_silver_is_required_without_claiming_human_gold(
    tmp_path: Path,
) -> None:
    split = tmp_path / "split.json"
    inventory = "1" * 64
    write_json(
        split,
        {
            "schema_version": "academic-structure-split-v1",
            "inventory_sha256": inventory,
            "assignments": {},
        },
    )
    receipt = tmp_path / "silver.json"
    base = {
        "schema_version": "academic-structure-silver-contract-receipt-v1",
        "status": "pass",
        "evidence_tier": "LLM_silver",
        "production_eligible": False,
        "inventory_sha256": inventory,
        "split_manifest_sha256": sha(split),
    }
    write_json(
        receipt, {**base, "task_scope_counts": {"bibliography_binary_windows": 1738}}
    )
    with pytest.raises(ValueError, match=r"STRUCT_2K|joint bibliography\+ToC"):
        PRODUCTION._validate_silver_evidence(receipt, split)

    write_json(receipt, {**base, "task_scope_counts": {"bibliography_toc_windows": 1}})
    assert (
        PRODUCTION._validate_silver_evidence(receipt, split)["evidence_tier"]
        == "LLM_silver"
    )


def test_detection_routes_only_explicit_apply_after_review_rows(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizers = pytest.importorskip("tokenizers")
    tokenizer = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    corpus = tmp_path / "stage50" / "corpus"
    corpus.mkdir(parents=True)
    parquet = corpus / "part.parquet"
    rows = [
        {
            "stable_uid": "1" * 64,
            "source_dataset": "academic-apply",
            "text": "Βιβλιογραφία 📚\nSmith 2020",
            "cleaning_profile": "academic_ocr",
            "structural_policy": "apply_after_review",
            "eligible_for_training": True,
        },
        {
            "stable_uid": "2" * 64,
            "source_dataset": "academic-shadow",
            "text": "Βιβλιογραφία\nShadow 2020",
            "cleaning_profile": "academic_ocr",
            "structural_policy": "shadow",
            "eligible_for_training": True,
        },
        {
            "stable_uid": "3" * 64,
            "source_dataset": "web",
            "text": "References\nWeb 2020",
            "cleaning_profile": "web_articles",
            "structural_policy": "apply_after_review",
            "eligible_for_training": True,
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    stage50 = tmp_path / "stage50" / "cleaning_manifest.json"
    write_json(
        stage50,
        {
            "schema_version": "full_cpt_cleaning_manifest_v1",
            "status": "completed",
            "cleaning_pass": "post_source_post_pii",
            "structural_applied": False,
            "output": str(corpus),
            "tokenizer_sha256": sha(tokenizer_path),
            "counts": {
                "tokens_final": sum(
                    len(item.ids)
                    for item in tokenizer.encode_batch(
                        [row["text"] for row in rows], add_special_tokens=False
                    )
                )
            },
            "files": [
                {
                    "relative_path": "part.parquet",
                    "kept_rows": 3,
                    "output": PRODUCTION.file_receipt(parquet, relative_to=corpus),
                }
            ],
        },
    )
    policy = tmp_path / "policy.json"
    write_json(
        policy,
        {
            "schema_version": "full_cpt_cleaning_policy_v1",
            "structural": {
                "allowed_apply_profiles": ["academic_ocr", "academic_sectioned"]
            },
        },
    )

    binary = tmp_path / "reference_detect"
    binary.write_text(
        """#!/usr/bin/env python3
import hashlib, json, sys
def arg(name): return sys.argv[sys.argv.index(name) + 1]
source = arg('--source')
spans = open(arg('--out-spans'), 'w', encoding='utf-8')
counters = open(arg('--out-counters'), 'w', encoding='utf-8')
for line in sys.stdin:
    row = json.loads(line); uid = row['id']; text = row['text']
    digest = hashlib.sha256(text.encode()).hexdigest()
    row_uid = hashlib.sha256((source + '\\0' + uid).encode()).hexdigest()
    end = text.find('\\n'); end = len(text) if end < 0 else end
    counter = {'doc_id': uid, 'source': source, 'row_uid': row_uid,
      'original_sha256': digest, 'original_chars': len(text),
      'bib_model_id': 'bib-v1', 'bib_decoder_id': 'bib-dec-v1',
      'toc_model_id': 'toc-v1', 'toc_decoder_id': 'toc-dec-v1',
      'bib_spans': 1, 'bib_lines': 1, 'toc_spans': 0, 'toc_lines': 0,
      'overlap_pairs': 0, 'overlap_chars': 0, 'overlap_lines': 0}
    counters.write(json.dumps(counter, ensure_ascii=False) + '\\n')
    span = {'doc_id': uid, 'source': source, 'row_uid': row_uid,
      'original_sha256': digest, 'original_chars': len(text), 'kind': 'bib_span',
      'char_start': 0, 'char_end': end, 'line_start': 0, 'line_end': 0,
      'trigger': text[:end][:40], 'gated_by': 'span_lr p=0.900',
      'model_id': 'bib-v1:bib-dec-v1'}
    spans.write(json.dumps(span, ensure_ascii=False) + '\\n')
spans.close(); counters.close()
""",
        encoding="utf-8",
    )
    os.chmod(binary, 0o755)
    commit = "a" * 40
    build = tmp_path / "build.json"
    write_json(
        build,
        {
            "schema_version": "full_cpt_detector_build_receipt_v1",
            "status": "passed",
            "code_commit": commit,
            "binary": {
                "path": str(binary.resolve()),
                "size": binary.stat().st_size,
                "sha256": sha(binary),
            },
        },
    )
    bib, toc, smoother, config = (
        tmp_path / "bib.json",
        tmp_path / "toc.json",
        tmp_path / "smoother.json",
        tmp_path / "config.json",
    )
    for path, value in (
        (bib, {"bib": 1}),
        (toc, {"toc": 1}),
        (smoother, {"smooth": 1}),
        (config, {"config": 1}),
    ):
        write_json(path, value)
    code = tmp_path / "model.rs"
    code.write_text("// exact model code\n", encoding="utf-8")
    parity = tmp_path / "parity.json"
    write_json(
        parity,
        {
            "schema_version": "struct_rust_parity_receipt_v1",
            "status": "passed",
            "evidence_status": "LLM_silver",
            "binary_sha256": sha(binary),
            "model_sha256": {
                "bib": sha(bib),
                "toc": sha(toc),
                "smoother": sha(smoother),
            },
        },
    )
    raw_root, raw_manifest = tmp_path / "raw", tmp_path / "raw-manifest.json"
    args = argparse.Namespace(
        stage50_cleaning_manifest=stage50,
        cleaning_policy=policy,
        detector_binary=binary,
        detector_build_receipt=build,
        parity_receipt=None,
        model_code=[code],
        sequence_config=config,
        smoother=smoother,
        bib_model=bib,
        toc_model=toc,
        code_commit=commit,
        output_dir=raw_root,
        manifest=raw_manifest,
        batch_rows=2,
        rayon_threads=2,
    )
    assert PRODUCTION.detect(args) == 0
    manifest = PRODUCTION.validate_raw_manifest(raw_manifest)
    assert manifest["counts"]["stage50_rows_scanned"] == 3
    assert manifest["counts"]["documents"] == 1
    assert manifest["counts"]["excluded_nonacademic_rows"] == 2
    assert manifest["counts"]["prediction_rows"] == 1
    assert manifest["detector"]["parity_status"] == "unavailable"
    assert manifest["eligible_structural_policy"] == "apply_after_review"
    index_path = Path(manifest["files"][0]["index"]["path"])
    indexed = list(PRODUCTION._iter_jsonl(index_path))
    assert [row["source_dataset"] for row in indexed] == ["academic-apply"]
    assert indexed[0]["structural_policy"] == "apply_after_review"
    assert PRODUCTION.detect(args) == 0, (
        "a completed run must validate and resume without rescoring"
    )

    loss_report = tmp_path / "token-loss.json"
    per_document = tmp_path / "token-loss.parquet"
    assert (
        PRODUCTION.token_loss(
            argparse.Namespace(
                raw_manifest=raw_manifest,
                tokenizer_json=tokenizer_path,
                per_document=per_document,
                report=loss_report,
                work_dir=tmp_path / "loss-work",
                batch_rows=2,
                tokenizer_batch_docs=2,
            )
        )
        == 0
    )
    loss = json.loads(loss_report.read_text(encoding="utf-8"))
    assert loss["all_routed"]["documents"] == 1
    assert loss["all_routed"]["tokens_removed_bibliography"] > 0
    assert loss["all_routed"]["tokens_removed_toc"] == 0
    assert loss["training_eligible"]["tokens_removed_union"] > 0
    assert loss["per_document"]["sha256"] == sha(per_document)


def _review_cases() -> list[dict]:
    cases = []
    for index in range(100):
        case_id = f"{index:064x}"
        cases.append(
            {
                "case_id": case_id,
                "review_context_sha256": hashlib.sha256(
                    f"context-{index}".encode()
                ).hexdigest(),
                "predicted_deletion_chars": 20,
                "document_chars": 1000,
                "stable_uid": hashlib.sha256(f"document-{index}".encode()).hexdigest(),
            }
        )
    return cases


def test_manual_audit_is_exact_and_never_auto_adjudicated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet_manifest = tmp_path / "packet.json"
    write_json(packet_manifest, {"fixture": True})
    cases = _review_cases()
    monkeypatch.setattr(
        PRODUCTION,
        "_load_packet",
        lambda _path: ({"raw_manifest_sha256": "a" * 64}, cases),
    )
    annotations = tmp_path / "annotations.jsonl"
    write_jsonl(
        annotations,
        [
            {
                "case_id": case["case_id"],
                "review_context_sha256": case["review_context_sha256"],
                "decision": "structural_only",
                "running_prose_chars_removed": 0,
                "main_text_chars_removed": 0,
                "catastrophic_document_deletion": False,
                "reviewer_notes": "",
            }
            for case in cases
        ],
    )
    receipt = tmp_path / "manual.json"
    write_json(
        receipt,
        {
            "schema_version": "academic_structural_manual_audit_receipt_v1",
            "status": "completed",
            "annotation_method": "manual",
            "automatic_adjudication_used": False,
            "reviewer_id": "fixture-reviewer",
            "packet_manifest_sha256": sha(packet_manifest),
            "annotations": {**PRODUCTION.file_receipt(annotations), "rows": 100},
        },
    )
    policy = tmp_path / "policy.json"
    write_json(
        policy,
        {
            "schema_version": "full_cpt_cleaning_policy_v1",
            "structural": {
                "application_gates": {
                    "minimum_reviewed_deletions": 100,
                    "maximum_running_prose_deletion_rate": 0.001,
                    "minimum_main_text_retention_rate": 0.999,
                    "maximum_catastrophic_document_deletion_rate": 0.0,
                }
            },
        },
    )
    output = tmp_path / "validation.json"
    result = PRODUCTION.validate_audit(
        argparse.Namespace(
            packet_manifest=packet_manifest,
            manual_receipt=receipt,
            annotations=annotations,
            cleaning_policy=policy,
            output=output,
        )
    )
    assert result == 0
    validation = json.loads(output.read_text(encoding="utf-8"))
    assert validation["status"] == "passed"
    assert validation["metrics"] == {
        "catastrophic_document_deletion_rate": 0.0,
        "main_text_retention_rate": 1.0,
        "running_prose_deletion_rate": 0.0,
    }

    output.unlink()
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["automatic_adjudication_used"] = True
    write_json(receipt, value)
    with pytest.raises(ValueError, match="automatic adjudication"):
        PRODUCTION.validate_audit(
            argparse.Namespace(
                packet_manifest=packet_manifest,
                manual_receipt=receipt,
                annotations=annotations,
                cleaning_policy=policy,
                output=output,
            )
        )


def test_final_spans_bind_only_after_final_receipt_and_overlap_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uid, text_hash = "2" * 64, "3" * 64
    raw_spans = tmp_path / "raw-spans.jsonl"
    raw_span = {
        "doc_id": uid,
        "original_sha256": text_hash,
        "kind": "bib_span",
        "char_start": 10,
        "char_end": 20,
        "model_id": "bib:model",
    }
    write_jsonl(raw_spans, [raw_span])
    raw_manifest_path = tmp_path / "raw.json"
    write_json(raw_manifest_path, {"raw": True})
    model_receipt = tmp_path / "model.json"
    write_json(model_receipt, {"receipt": "final-before-span-conversion"})
    raw = {
        "output_root": str(tmp_path),
        "stage50_cleaning_manifest_sha256": "4" * 64,
        "conflicts": {"overlap_pairs": 0},
        "counts": {"prediction_rows": 1},
        "files": [{"spans": PRODUCTION.file_receipt(raw_spans)}],
    }
    monkeypatch.setattr(PRODUCTION, "validate_raw_manifest", lambda _path: raw)
    monkeypatch.setattr(
        PRODUCTION,
        "validate_model_receipt",
        lambda _path, raw_manifest=None: {"status": "passed"},
    )
    output = tmp_path / "final-spans.jsonl"
    manifest = tmp_path / "final-spans-manifest.json"
    assert (
        PRODUCTION.rebind(
            argparse.Namespace(
                raw_manifest=raw_manifest_path,
                model_receipt=model_receipt,
                output=output,
                manifest=manifest,
                work_dir=tmp_path / "work",
            )
        )
        == 0
    )
    final_row = json.loads(output.read_text(encoding="utf-8"))
    assert final_row["model_receipt_sha256"] == sha(model_receipt)
    assert final_row["input_text_sha256"] == text_hash
    assert "structural_spans" not in json.loads(
        model_receipt.read_text(encoding="utf-8")
    )
    assert (
        json.loads(manifest.read_text(encoding="utf-8"))["receipt_cycle_avoided"]
        is True
    )

    output.unlink()
    manifest.unlink()
    raw["conflicts"]["overlap_pairs"] = 1
    with pytest.raises(ValueError, match="overlap"):
        PRODUCTION.rebind(
            argparse.Namespace(
                raw_manifest=raw_manifest_path,
                model_receipt=model_receipt,
                output=output,
                manifest=manifest,
                work_dir=tmp_path / "work",
            )
        )


def test_new_slurm_stages_are_cpu_only_and_stop_at_manual_boundary() -> None:
    clariden = HERE / "clariden"
    detect = (clariden / "52_detect_stage50_structural_spans.sbatch").read_text(
        encoding="utf-8"
    )
    packet = (clariden / "53_build_structural_review_packet.sbatch").read_text(
        encoding="utf-8"
    )
    promote = (clariden / "54_promote_structural_spans.sbatch").read_text(
        encoding="utf-8"
    )
    for body in (detect, packet, promote):
        assert "phase04_require_cpu_request" in body
        assert "--gres" not in body and "--gpus" not in body
    assert "AUTOMATIC_ADJUDICATION_FORBIDDEN=1" in packet
    assert "STRUCTURAL_MANUAL_AUDIT_RECEIPT" in promote
    assert "build-model-receipt" in promote
    assert " rebind " in promote
