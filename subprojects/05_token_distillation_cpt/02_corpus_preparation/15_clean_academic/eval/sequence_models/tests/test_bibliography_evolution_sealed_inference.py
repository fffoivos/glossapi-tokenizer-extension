from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sequence_models.bibliography_evolution_contract import ContractError, sha256_file
from sequence_models.bibliography_evolution_sealed_inference import (
    EXPECTED_SOURCES,
    _verified_annotation_inputs,
    materialize_unlabelled_table,
)
from sequence_models.bibliography_entry_models import load_table


def _documents(path: Path) -> list[dict]:
    sources = ("greek_phd", "kallipos", "openarchives")
    rows = []
    for index in range(150):
        document_id = f"{index + 1:064x}"
        text = f"Reference {index}. Example, A. 2020. pp. 1-2."
        line_id = hashlib.sha256(
            f"bibliography-sealed-line-v1\0{document_id}\00\0{text}".encode()
        ).hexdigest()
        rows.append(
            {
                "document_id": document_id,
                "source": sources[index // 50],
                "work_id": f"work-{index}",
                "n_physical_lines": 1,
                "lines": [{"line_id": line_id, "abs_idx": 0, "text": text}],
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return rows


def _frozen(path: Path, documents: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "bibliography-sealed-freeze-v1",
                "status": "frozen_prediction_blind_test_set",
                "document_count": 150,
                "line_count": 150,
                "source_document_counts": EXPECTED_SOURCES,
                "sealed_hashes": {
                    "documents_sha256": sha256_file(documents),
                    # Deliberately no files with either of these hashes exist.
                    # Prediction-only verification must not discover/open them.
                    "labels_sha256": "a" * 64,
                    "consensus_receipt_sha256": "b" * 64,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_prediction_visible_freeze_verification_never_needs_labels(tmp_path: Path) -> None:
    documents = tmp_path / "documents.private.jsonl"
    expected = _documents(documents)
    frozen = tmp_path / "FROZEN.receipt.json"
    _frozen(frozen, documents)
    observed, receipt = _verified_annotation_inputs(
        documents,
        frozen,
        expected_documents_sha256=sha256_file(documents),
        expected_frozen_sha256=sha256_file(frozen),
    )
    assert observed == expected
    assert receipt["sealed_hashes"]["labels_sha256"] == "a" * 64


def test_unlabelled_feature_table_is_exactly_source_and_line_aligned(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.private.jsonl"
    documents = _documents(documents_path)
    table_root = tmp_path / "table"
    receipt = materialize_unlabelled_table(documents, documents_path, table_root)
    table = load_table(table_root, expected_split="sealed_unlabelled")
    assert receipt["document_count"] == 150
    assert receipt["line_count"] == 150
    assert receipt["source_document_counts"] == EXPECTED_SOURCES
    assert table.counts.shape[0] == 150
    assert np.count_nonzero(table.targets) == 0
    assert np.count_nonzero(table.original_labels) == 0
    assert "not labels" in table.manifest["target_semantics"]


def test_prediction_visible_freeze_rejects_unfrozen_or_drifted_bytes(tmp_path: Path) -> None:
    documents = tmp_path / "documents.private.jsonl"
    _documents(documents)
    frozen = tmp_path / "FROZEN.receipt.json"
    with pytest.raises(ContractError, match="FROZEN"):
        _verified_annotation_inputs(
            documents,
            frozen,
            expected_documents_sha256=sha256_file(documents),
            expected_frozen_sha256="0" * 64,
        )
    _frozen(frozen, documents)
    documents_link = tmp_path / "documents.link.jsonl"
    documents_link.symlink_to(documents)
    with pytest.raises(ContractError, match="symlink"):
        _verified_annotation_inputs(
            documents_link,
            frozen,
            expected_documents_sha256=sha256_file(documents),
            expected_frozen_sha256=sha256_file(frozen),
        )
    with pytest.raises(ContractError, match="document bytes"):
        _verified_annotation_inputs(
            documents,
            frozen,
            expected_documents_sha256="0" * 64,
            expected_frozen_sha256=sha256_file(frozen),
        )
