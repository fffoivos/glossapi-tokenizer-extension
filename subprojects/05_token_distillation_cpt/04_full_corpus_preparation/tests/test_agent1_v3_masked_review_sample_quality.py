from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPT = HERE / "scripts" / "agent1_v3_masked_review_sample_quality.py"


def load_module():
    spec = importlib.util.spec_from_file_location("agent1_v3_masked_review_sample_quality_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


QUALITY = load_module()
EVIDENCE = QUALITY.evidence


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    text = "Μασκαρισμένο δείγμα [EMAIL_0001]."
    sample = tmp_path / "sample.jsonl"
    row = {
        "schema_version": EVIDENCE.MASKED_SAMPLE_SCHEMA,
        "sample_id": digest("sample"),
        "source_id": "source-a",
        "source_dataset": "dataset-a",
        "source_revision": "revision-a",
        "source_route": "pdf_ocr",
        "sampling_stratum": "risk",
        "original_text_sha256": digest("original"),
        "review_copy_sha256": digest(text),
        "review_request_sha256": digest("request"),
        "text_variant": QUALITY.TEXT_VARIANT,
        "review_copy": text,
    }
    sample.write_text(EVIDENCE.canonical_json(row) + "\n", encoding="utf-8")
    inventory = [
        {
            "sample_id": row["sample_id"],
            "source_id": row["source_id"],
            "sampling_stratum": row["sampling_stratum"],
            "original_text_sha256": row["original_text_sha256"],
            "review_copy_sha256": row["review_copy_sha256"],
            "review_request_sha256": row["review_request_sha256"],
        }
    ]
    receipt = {
        "schema_version": EVIDENCE.MASKED_SAMPLE_RECEIPT_SCHEMA,
        "status": "passed",
        "text_variant": QUALITY.TEXT_VARIANT,
        "raw_corpus_included": False,
        "output": {**EVIDENCE.file_binding(sample), "rows": 1},
        "primary_sample_count": 1,
        "primary_sample_inventory": inventory,
        "primary_sample_inventory_sha256": EVIDENCE.sha256_json(inventory),
    }
    receipt["receipt_sha256"] = EVIDENCE.receipt_digest(receipt)
    receipt_path = tmp_path / "sample-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
    return sample, receipt_path


def test_v3_masked_sample_adapter_preserves_hash_boundary_without_legacy_schema(tmp_path: Path) -> None:
    sample, receipt = write_fixture(tmp_path)
    rows, loaded_receipt = QUALITY._read_adapter_rows(sample, receipt)
    assert len(rows) == 1
    assert loaded_receipt["raw_corpus_included"] is False
    adapted = QUALITY._quality_rows(rows, sample)
    assert adapted[0]["stable_uid"] == rows[0]["sample_id"]
    assert adapted[0]["normalized_text_sha256"] == rows[0]["original_text_sha256"]
    assert adapted[0]["profile_text_sha256"] == rows[0]["review_copy_sha256"]
    assert adapted[0]["profile_text_variant"] == QUALITY.TEXT_VARIANT
    assert adapted[0]["source_repo_id"] == "source-a"


def test_v3_masked_sample_adapter_rejects_review_copy_hash_drift(tmp_path: Path) -> None:
    sample, receipt = write_fixture(tmp_path)
    row = json.loads(sample.read_text(encoding="utf-8"))
    row["review_copy"] = "αλλοιωμένο"
    sample.write_text(EVIDENCE.canonical_json(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="byte/hash binding drift|review-copy hash drift"):
        QUALITY._read_adapter_rows(sample, receipt)
