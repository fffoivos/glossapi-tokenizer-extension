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


def test_strict_parity_requires_imported_validation_and_zero_historical_test_access(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "detector"
    bib = tmp_path / "bib.json"
    toc = tmp_path / "toc.json"
    smoother = tmp_path / "smooth.json"
    for path in (binary, bib, toc, smoother):
        path.write_bytes(path.name.encode())
    source_receipt = tmp_path / "source.json"
    write_json(source_receipt, {"fixture": True})
    source_split = tmp_path / "split.json"
    write_json(source_split, {"fixture": True})
    corpus_sha = "c" * 64
    parity = tmp_path / "parity.json"
    value = {
        "schema_version": "struct_rust_parity_receipt_v1",
        "status": "passed",
        "evidence_status": "LLM_silver",
        "input_snapshot_method": (
            "private_job_local_o_nofollow_copy_rehash_before_publish"
        ),
        "inputs_rehashed_before_publication": True,
        "binary_sha256": sha(binary),
        "corpus_sha256": corpus_sha,
        "source_receipt_sha256": sha(source_receipt),
        "source_split_manifest_sha256": sha(source_split),
        "evaluation_partition": "validation",
        "partition_semantics": (
            "derived_historical_train_validation_runtime_parity_not_quality_holdout"
        ),
        "historical_test_documents_loaded": 0,
        "heldout_documents": 2,
        "positive_document_counts": {"bib": 1, "toc": 1},
        "tolerance": 0.001,
        "heads": {
            "bib": {
                "documents": 2,
                "span_mismatches": 0,
                "max_probability_difference": 0.0,
            },
            "toc": {
                "documents": 2,
                "span_mismatches": 0,
                "max_probability_difference": 0.0,
            },
        },
        "model_sha256": {
            "bib": sha(bib),
            "toc": sha(toc),
            "smoother": sha(smoother),
        },
    }
    write_json(parity, value)
    policy = {
        "validation": {
            "required_parity_documents": 2,
            "maximum_probability_delta": 0.001,
        }
    }
    assert (
        PRODUCTION._validate_parity(
            parity,
            detector_binary=binary,
            bib_model=bib,
            toc_model=toc,
            smoother=smoother,
            silver_receipt={
                "silver_sha256": corpus_sha,
                "split_counts": {"validation": 2},
            },
            silver_receipt_sha256=sha(source_receipt),
            silver_split_manifest_sha256=sha(source_split),
            cleaning_policy=policy,
        )["evaluation_partition"]
        == "validation"
    )
    value["historical_test_documents_loaded"] = 1
    write_json(parity, value)
    with pytest.raises(ValueError, match="validation-only joint source"):
        PRODUCTION._validate_parity(
            parity,
            detector_binary=binary,
            bib_model=bib,
            toc_model=toc,
            smoother=smoother,
            silver_receipt={
                "silver_sha256": corpus_sha,
                "split_counts": {"validation": 2},
            },
            silver_receipt_sha256=sha(source_receipt),
            silver_split_manifest_sha256=sha(source_split),
            cleaning_policy=policy,
        )


def _strict_parity_fixture(tmp_path: Path) -> dict[str, object]:
    binary = tmp_path / "detector"
    bib = tmp_path / "bib.json"
    toc = tmp_path / "toc.json"
    smoother = tmp_path / "smooth.json"
    for path in (binary, bib, toc, smoother):
        path.write_bytes(path.name.encode())
    source_receipt = tmp_path / "source.json"
    source_split = tmp_path / "split.json"
    write_json(source_receipt, {"fixture": True})
    write_json(source_split, {"fixture": True})
    corpus_sha = "c" * 64
    parity = tmp_path / "parity.json"
    value = {
        "schema_version": "struct_rust_parity_receipt_v1",
        "status": "passed",
        "evidence_status": "LLM_silver",
        "input_snapshot_method": (
            "private_job_local_o_nofollow_copy_rehash_before_publish"
        ),
        "inputs_rehashed_before_publication": True,
        "binary_sha256": sha(binary),
        "corpus_sha256": corpus_sha,
        "source_receipt_sha256": sha(source_receipt),
        "source_split_manifest_sha256": sha(source_split),
        "evaluation_partition": "validation",
        "partition_semantics": (
            "derived_historical_train_validation_runtime_parity_not_quality_holdout"
        ),
        "historical_test_documents_loaded": 0,
        "heldout_documents": 2,
        "positive_document_counts": {"bib": 1, "toc": 1},
        "tolerance": 0.001,
        "heads": {
            "bib": {
                "documents": 2,
                "span_mismatches": 0,
                "max_probability_difference": 0.0,
            },
            "toc": {
                "documents": 2,
                "span_mismatches": 0,
                "max_probability_difference": 0.0,
            },
        },
        "model_sha256": {
            "bib": sha(bib),
            "toc": sha(toc),
            "smoother": sha(smoother),
        },
    }
    policy = {
        "validation": {
            "required_parity_documents": 608,
            "maximum_probability_delta": 0.001,
            "structural_parity_corpus_sha256": None,
        }
    }
    return {
        "binary": binary,
        "bib": bib,
        "toc": toc,
        "smoother": smoother,
        "source_receipt": source_receipt,
        "source_split": source_split,
        "corpus_sha": corpus_sha,
        "parity": parity,
        "value": value,
        "policy": policy,
    }


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("tolerance", True, "finite non-negative"),
        ("tolerance", float("nan"), "finite non-negative"),
        ("tolerance", float("inf"), "finite non-negative"),
        ("tolerance", -0.1, "finite non-negative"),
        ("tolerance", 0.002, "looser than cleaning policy"),
        ("bib_delta", True, "finite non-negative"),
        ("bib_delta", float("nan"), "finite non-negative"),
        ("bib_delta", float("inf"), "finite non-negative"),
        ("bib_delta", -0.1, "finite non-negative"),
        ("bib_delta", 0.002, "exceeds its tolerance"),
        ("bib_documents", True, "positive integer"),
        ("bib_documents", 1, "differs from top level"),
        ("bib_documents_missing", None, "positive integer"),
        ("policy_max", True, "finite non-negative"),
        ("policy_max", float("nan"), "finite non-negative"),
        ("policy_max", -0.1, "finite non-negative"),
    ],
)
def test_strict_parity_rejects_adversarial_numeric_and_head_coverage(
    tmp_path: Path, target: str, value: object, message: str
) -> None:
    fixture = _strict_parity_fixture(tmp_path)
    receipt = fixture["value"]
    assert isinstance(receipt, dict)
    if target == "tolerance":
        receipt["tolerance"] = value
    elif target == "bib_delta":
        receipt["heads"]["bib"]["max_probability_difference"] = value
    elif target == "bib_documents":
        receipt["heads"]["bib"]["documents"] = value
    elif target == "bib_documents_missing":
        receipt["heads"]["bib"].pop("documents")
    else:
        fixture["policy"]["validation"]["maximum_probability_delta"] = value
    write_json(fixture["parity"], receipt)
    with pytest.raises(ValueError, match=message):
        PRODUCTION._validate_parity(
            fixture["parity"],
            detector_binary=fixture["binary"],
            bib_model=fixture["bib"],
            toc_model=fixture["toc"],
            smoother=fixture["smoother"],
            silver_receipt={
                "silver_sha256": fixture["corpus_sha"],
                "split_counts": {"validation": 2},
            },
            silver_receipt_sha256=sha(fixture["source_receipt"]),
            silver_split_manifest_sha256=sha(fixture["source_split"]),
            cleaning_policy=fixture["policy"],
            enforce_policy_document_count=False,
        )


def test_stage52_uses_exact_source_validation_count_but_promotion_keeps_policy_gate(
    tmp_path: Path,
) -> None:
    fixture = _strict_parity_fixture(tmp_path)
    write_json(fixture["parity"], fixture["value"])
    kwargs = {
        "detector_binary": fixture["binary"],
        "bib_model": fixture["bib"],
        "toc_model": fixture["toc"],
        "smoother": fixture["smoother"],
        "silver_receipt": {
            "silver_sha256": fixture["corpus_sha"],
            "split_counts": {"validation": 2},
        },
        "silver_receipt_sha256": sha(fixture["source_receipt"]),
        "silver_split_manifest_sha256": sha(fixture["source_split"]),
        "cleaning_policy": fixture["policy"],
    }
    assert (
        PRODUCTION._validate_parity(
            fixture["parity"], **kwargs, enforce_policy_document_count=False
        )["heldout_documents"]
        == 2
    )
    with pytest.raises(ValueError, match="apply policy requires 608"):
        PRODUCTION._validate_parity(
            fixture["parity"], **kwargs, enforce_policy_document_count=True
        )

    mismatched_source = dict(kwargs)
    mismatched_source["silver_receipt"] = {
        "silver_sha256": fixture["corpus_sha"],
        "split_counts": {"validation": 3},
    }
    with pytest.raises(ValueError, match="split_counts.validation"):
        PRODUCTION._validate_parity(
            fixture["parity"],
            **mismatched_source,
            enforce_policy_document_count=False,
        )


def test_promotion_requires_exact_detection_time_file_receipt(tmp_path: Path) -> None:
    original = tmp_path / "original.json"
    copy = tmp_path / "copy.json"
    write_json(original, {"same": "bytes"})
    copy.write_bytes(original.read_bytes())
    embedded = PRODUCTION.file_receipt(original)
    assert (
        PRODUCTION._require_exact_file_receipt(embedded, original, label="fixture")
        == original.resolve()
    )
    with pytest.raises(ValueError, match="detection time"):
        PRODUCTION._require_exact_file_receipt(embedded, copy, label="fixture")


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("classifier_selection_receipt", "promotion classifier selection"),
        ("silver_receipt", "promotion joint source receipt"),
        ("silver_split_manifest", "promotion joint source split manifest"),
        ("parity_receipt", "promotion parity receipt"),
    ],
)
def test_stage54_rejects_post_hoc_same_byte_qualification_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    message: str,
) -> None:
    originals = {}
    substitutes = {}
    for name in (
        "classifier_selection_receipt",
        "silver_receipt",
        "silver_split_manifest",
        "parity_receipt",
    ):
        original = tmp_path / f"{name}.json"
        substitute = tmp_path / f"{name}.copy.json"
        write_json(original, {"name": name})
        substitute.write_bytes(original.read_bytes())
        originals[name] = original
        substitutes[name] = substitute
    raw = {
        "conflicts": {"overlap_pairs": 0},
        "qualification": {
            "classifier_selection_receipt": PRODUCTION.file_receipt(
                originals["classifier_selection_receipt"]
            ),
            "joint_source_receipt": PRODUCTION.file_receipt(
                originals["silver_receipt"]
            ),
            "joint_source_split_manifest": PRODUCTION.file_receipt(
                originals["silver_split_manifest"]
            ),
        },
        "detector": {
            "parity_receipt": PRODUCTION.file_receipt(originals["parity_receipt"])
        },
    }
    raw_manifest = tmp_path / "raw.json"
    write_json(raw_manifest, {"fixture": True})
    monkeypatch.setattr(PRODUCTION, "validate_raw_manifest", lambda _path: raw)
    supplied = dict(originals)
    supplied[target] = substitutes[target]
    with pytest.raises(ValueError, match=message):
        PRODUCTION.build_model_receipt(
            argparse.Namespace(
                raw_manifest=raw_manifest,
                audit_validation=tmp_path / "unused-audit.json",
                output=tmp_path / "model.json",
                **supplied,
            )
        )


def test_raw_identity_binds_selection_source_split_and_parity() -> None:
    kwargs = {
        "stage50_manifest_sha256": "1" * 64,
        "detector_binary_sha256": "2" * 64,
        "detector_build_receipt_sha256": "3" * 64,
        "classifier_selection_receipt_sha256": "4" * 64,
        "joint_source_receipt_sha256": "5" * 64,
        "joint_source_split_manifest_sha256": "6" * 64,
        "parity_receipt_sha256": "7" * 64,
        "cleaning_policy_sha256": "8" * 64,
        "allowed_apply_profiles": ["academic_ocr"],
        "eligible_structural_policy": "apply_after_review",
        "artifacts": {"fixture": {"sha256": "9" * 64}},
    }
    baseline = PRODUCTION._raw_identity(**kwargs)
    for field in (
        "classifier_selection_receipt_sha256",
        "joint_source_receipt_sha256",
        "joint_source_split_manifest_sha256",
        "parity_receipt_sha256",
    ):
        changed = dict(kwargs)
        changed[field] = "a" * 64
        assert PRODUCTION._raw_identity(**changed) != baseline


def test_detection_routes_only_explicit_apply_after_review_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
    silver_receipt = tmp_path / "silver-receipt.json"
    silver_split = tmp_path / "silver-split.json"
    classifier_selection = tmp_path / "classifier-selection.json"
    for path, value in (
        (silver_receipt, {"fixture": "source"}),
        (silver_split, {"fixture": "split"}),
        (classifier_selection, {"fixture": "selection"}),
    ):
        write_json(path, value)
    monkeypatch.setattr(
        PRODUCTION,
        "_validate_detection_qualification",
        lambda **_kwargs: (
            {"inventory_sha256": "1" * 64},
            {
                "selected_architecture": "c0-rust-lr-hysteresis",
                "joint_ladder": {"run_receipt_sha256": "2" * 64},
            },
            {"evaluation_partition": "validation", "heldout_documents": 2},
        ),
    )
    raw_root, raw_manifest = tmp_path / "raw", tmp_path / "raw-manifest.json"
    args = argparse.Namespace(
        stage50_cleaning_manifest=stage50,
        cleaning_policy=policy,
        detector_binary=binary,
        detector_build_receipt=build,
        classifier_selection_receipt=classifier_selection,
        silver_receipt=silver_receipt,
        silver_split_manifest=silver_split,
        parity_receipt=parity,
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
    assert manifest["detector"]["parity_status"] == "passed"
    assert manifest["qualification"]["classifier_selection_receipt"]["sha256"] == sha(
        classifier_selection
    )
    assert manifest["qualification"]["joint_source_receipt"]["sha256"] == sha(
        silver_receipt
    )
    assert manifest["qualification"]["joint_source_split_manifest"]["sha256"] == sha(
        silver_split
    )
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
    bridge = (
        HERE.parent
        / "02_corpus_preparation"
        / "15_clean_academic"
        / "eval"
        / "sequence_models"
        / "clariden"
        / "build_joint_c0_bridge.sbatch"
    ).read_text(encoding="utf-8")
    for body in (detect, packet, promote, bridge):
        assert "phase04_require_cpu_request" in body
        assert "--gres" not in body and "--gpus" not in body
    assert "STRUCTURAL_CLASSIFIER_SELECTION_RECEIPT" in detect
    assert "STRUCTURAL_PRODUCTION_SEQUENCE_CONFIG" in detect
    assert "${STRUCTURAL_PARITY_RECEIPT:?" in detect
    assert "--classifier-selection-receipt" in detect
    assert "--silver-receipt" in detect
    assert "--silver-split-manifest" in detect
    assert "--parity-receipt" in detect
    assert "parity_args" not in detect
    assert "AUTOMATIC_ADJUDICATION_FORBIDDEN=1" in packet
    assert "STRUCTURAL_MANUAL_AUDIT_RECEIPT" in promote
    assert "--classifier-selection-receipt" in promote
    assert "build-model-receipt" in promote
    assert " rebind " in promote
    assert "c0-rust-lr-hysteresis" in bridge
    assert "rust_parity_struct.py" in bridge
