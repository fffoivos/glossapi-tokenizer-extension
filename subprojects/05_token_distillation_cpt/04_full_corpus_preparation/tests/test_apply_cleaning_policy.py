from __future__ import annotations

import importlib.util
import json
import subprocess
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


PII = load_module("phase04_greek_pii", HERE / "scripts" / "greek_pii.py")
CLEAN = load_module("phase04_clean_policy", HERE / "scripts" / "apply_cleaning_policy.py")


def test_greek_pii_masks_only_validated_or_labelled_identifiers() -> None:
    valid_afm = next(f"{number:09d}" for number in range(1, 100_000_000) if PII.afm_valid(f"{number:09d}"))
    text = f"ΑΦΜ: {valid_afm}, τηλ. +30 2101234567 και αριθμός 123456789."
    masked, counts = PII.mask_greek_identifiers(text)
    assert "<afm-pii>" in masked
    assert "<phone-pii>" in masked
    assert "123456789" in masked
    assert counts == {"afm": 1, "phone": 1}


def test_structural_spans_require_post_pii_hash() -> None:
    text = "Κύριο κείμενο\n\nΒιβλιογραφία\nΑναφορά"
    start = text.index("Βιβλιογραφία")
    span = {
        "char_start": start,
        "char_end": len(text),
        "input_text_sha256": CLEAN.sha256_text(text),
        "rule_id": "academic.bibliography.v2",
    }
    output, reasons = CLEAN.apply_spans(text, [span])
    assert output == "Κύριο κείμενο"
    assert reasons == ["academic.bibliography.v2"]
    span["input_text_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="post-PII"):
        CLEAN.apply_spans(text, [span])


def canonical_row(**updates):
    base = {
        "source_id": "demo",
        "source_dataset": "demo",
        "source_doc_id": "doc-1",
        "text": "<p>Καλημέρα</p> email@example.gr +30 2101234567",
        "title": None,
        "author": None,
        "greek_badness_score": 0.0,
        "mojibake_badness_score": 0.0,
        "needs_ocr": False,
        "is_empty": False,
        "ocr_success": True,
        "is_historical_or_polytonic": False,
        "source_family_id": "demo",
        "acquisition_source_id": "demo",
        "source_repo_id": "owner/demo",
        "source_revision": "rev",
        "source_artifact_path": "data.parquet",
        "source_row_id": "data.parquet:0:0",
        "source_text_field": "content",
        "original_text_sha256": "a" * 64,
        "normalized_text_sha256": "b" * 64,
        "stable_uid": "c" * 64,
        "work_key": "d" * 64,
        "work_id": "doc-1",
        "representation_generation": "new_family",
        "lineage_alias_id": "e" * 64,
        "source_metadata_json": "{}",
        "cleaning_profile": "web_articles",
        "structural_policy": "disabled",
        "training_eligibility": "eligible_open",
        "source_role": "additive_candidate",
    }
    base.update(updates)
    return base


def test_cleaning_cli_masks_pii_and_drops_private_diavgeia(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    tokenizers = pytest.importorskip("tokenizers")

    input_root = tmp_path / "normalized"
    input_path = input_root / "part.parquet"
    input_root.mkdir()
    private_row = canonical_row(
        source_dataset="diavgeia",
        source_doc_id="private",
        text="Ιδιωτική απόφαση",
        acquisition_source_id="diavgeia",
        source_family_id="diavgeia",
        source_repo_id="glossAPI/diavgeia",
        stable_uid="f" * 64,
        cleaning_profile="diavgeia",
        source_metadata_json=json.dumps({"metadata_json": json.dumps({"privateData": True})}),
    )
    pq.write_table(pa.Table.from_pylist([canonical_row(), private_row], schema=CLEAN.canonical_schema()), input_path)

    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    admission = {
        "schema_version": "source_quality_review_admission_v1",
        "pending_adjudications": 0,
        "sources": [
            {"source_dataset": "demo", "decision": "include"},
            {"source_dataset": "diavgeia", "decision": "include"},
        ],
    }
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(json.dumps(admission))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({"status": "audit_only", "structural": {}}))
    manifest = tmp_path / "cleaning.json"
    subprocess.run(
        [
            sys.executable,
            str(HERE / "scripts" / "apply_cleaning_policy.py"),
            "--input",
            str(input_root),
            "--output",
            str(tmp_path / "cleaned"),
            "--quarantine",
            str(tmp_path / "quarantine"),
            "--ledger",
            str(tmp_path / "ledger"),
            "--manifest",
            str(manifest),
            "--source-admission",
            str(admission_path),
            "--cleaning-policy",
            str(policy_path),
            "--tokenizer-json",
            str(tokenizer_path),
        ],
        check=True,
    )
    cleaned = pq.read_table(tmp_path / "cleaned" / "part.parquet").to_pylist()
    assert len(cleaned) == 1
    assert "<email-pii>" in cleaned[0]["text"]
    assert "<phone-pii>" in cleaned[0]["text"]
    assert "<p>" not in cleaned[0]["text"]
    assert cleaned[0]["eligible_for_training"] is True
    ledger = pq.read_table(tmp_path / "ledger" / "part.parquet").to_pylist()
    by_id = {row["source_doc_id"]: row for row in ledger}
    assert by_id["doc-1"]["action"] == "keep"
    assert by_id["private"]["action"] == "drop"
    assert "diavgeia_privateData_true" in by_id["private"]["reasons_json"]
    assert json.loads(manifest.read_text())["structural_applied"] is False


def test_preclean_include_after_cleaning_is_retained_but_not_promoted(tmp_path: Path) -> None:
    policy = {
        "schema_version": "source_quality_review_admission_v1",
        "pending_adjudications": 0,
        "sources": [{"source_dataset": "exact-name", "decision": "include_after_cleaning"}],
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(policy))
    status, admissions = CLEAN.load_admission(path)
    assert status == "approved"
    assert CLEAN.admission_for("repo-route", "exact-name", "additive_candidate", admissions)[
        "decision"
    ] == "include_after_cleaning"


def test_pending_review_admission_fails_closed(tmp_path: Path) -> None:
    policy = {
        "schema_version": "source_quality_review_admission_v1",
        "pending_adjudications": 1,
        "sources": [{"source_dataset": "demo", "decision": "pending_adjudication"}],
    }
    path = tmp_path / "admission.json"
    path.write_text(json.dumps(policy))
    status, _ = CLEAN.load_admission(path)
    assert status == "pending"
