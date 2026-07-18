from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sequence_models.bibliography_evolution_contract import ContractError, sha256_file
from sequence_models.bibliography_evolution_sealed_inference import (
    EXPECTED_SOURCES,
    _composition_parent_ids,
    _verified_annotation_inputs,
    materialize_unlabelled_table,
    prepare_inference,
    runtime_code_inventory,
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


def test_runtime_inventory_covers_every_package_source_and_clean_commit() -> None:
    inventory = runtime_code_inventory()
    root = Path(__file__).resolve().parents[1]
    expected = sorted(path.name for path in root.glob("*.py"))
    observed = [row["path"] for row in inventory["files"]]
    assert observed == expected
    assert "bibliography_feature_explorer.py" in observed
    assert "bibliography_evolution_metrics.py" in observed
    assert inventory["git_clean"] is True
    assert len(inventory["git_commit"]) == 40


def test_g5_orientation_comes_from_owned_runner_artifacts(tmp_path: Path) -> None:
    left_prediction = tmp_path / "left.prediction.npy"
    left_barrier = tmp_path / "left.barrier.npz"
    right_prediction = tmp_path / "right.prediction.npy"
    right_barrier = tmp_path / "right.barrier.npz"
    for path in (left_prediction, left_barrier, right_prediction, right_barrier):
        path.write_bytes(path.name.encode("utf-8"))
    spec = {
        # Deliberately reverse lineage order. Runner ownership remains the
        # semantic authority for non-commutative left-minus-right.
        "parent_candidate_ids": ["right", "left"],
        "runner": {
            "argv": [
                "--left-prediction", str(left_prediction),
                "--right-prediction", str(right_prediction),
                "--left-barrier-artifact", str(left_barrier),
                "--right-barrier-artifact", str(right_barrier),
            ]
        },
        "input_receipts": {
            "lp": {"path": str(left_prediction), "parent_candidate_id": "left"},
            "lb": {"path": str(left_barrier), "parent_candidate_id": "left"},
            "rp": {"path": str(right_prediction), "parent_candidate_id": "right"},
            "rb": {"path": str(right_barrier), "parent_candidate_id": "right"},
        },
    }
    assert _composition_parent_ids(spec) == ("left", "right")
    spec["input_receipts"]["rb"]["parent_candidate_id"] = "left"
    with pytest.raises(Exception, match="orientation differs"):
        _composition_parent_ids(spec)


def test_prepare_inference_rejects_manifest_symlink_before_resolve(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "manifest.link.json"
    link.symlink_to(manifest)
    with pytest.raises(Exception, match="symlink"):
        prepare_inference(
            link,
            tmp_path / "documents.jsonl",
            tmp_path / "FROZEN.json",
            tmp_path / "output",
        )
