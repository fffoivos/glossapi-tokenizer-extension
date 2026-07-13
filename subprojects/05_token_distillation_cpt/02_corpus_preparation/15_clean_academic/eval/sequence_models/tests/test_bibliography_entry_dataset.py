from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sequence_models.bibliography_entry_dataset import (
    FEATURE_NAMES,
    HEADER_EXACT,
    SUBHEADER_EXACT,
    TARGET_ENTRY,
    TARGET_MASK,
    TARGET_NEGATIVE,
    assign_grouped_folds,
    materialize_document,
    run,
    sha256_file,
)


def _document(document_id: str = "doc-1", work_id: str = "work-1") -> dict:
    lines = [
        (0, "Κανονικό σώμα.", "O"),
        (1, "ΒΙΒΛΙΟΓΡΑΦΙΑ", "BIB"),
        (2, "Ελληνική βιβλιογραφία", "BIB"),
        (3, "Smith, J. (2020). A title. London: Press.", "BIB"),
        (4, "Περιεχόμενα", "TOC"),
        (5, "άγνωστο", "UNKNOWN"),
    ]
    return {
        "schema_version": "academic-structure-gold-v1",
        "document_id": document_id,
        "work_id": work_id,
        "source": "openarchives",
        "split": "train",
        "coverage": "full_document",
        "n_physical_lines": 6,
        "n_present_lines": 6,
        "lines": [
            {
                "line_id": f"{document_id}:{abs_idx}",
                "abs_idx": abs_idx,
                "text": text,
                "label": label,
                "token_count": len(text.split()),
                "is_running_prose": None,
            }
            for abs_idx, text, label in lines
        ],
    }


def test_exact_headers_are_masked_but_entries_and_region_labels_survive() -> None:
    result = materialize_document(_document())
    assert result["targets"].tolist() == [
        TARGET_NEGATIVE,
        TARGET_MASK,
        TARGET_MASK,
        TARGET_ENTRY,
        TARGET_NEGATIVE,
        TARGET_MASK,
    ]
    assert result["header_kinds"].tolist() == [
        0,
        HEADER_EXACT,
        SUBHEADER_EXACT,
        0,
        0,
        0,
    ]
    assert result["original_labels"].tolist() == [0, 1, 1, 1, 2, 3]
    assert result["block_indices"].tolist() == [-1, 0, 0, 0, -1, -1]
    assert result["counts"].shape == (6, len(FEATURE_NAMES))
    assert result["counts"][3].sum() > 0


def test_grouped_folds_never_split_a_work_id() -> None:
    documents = []
    for index in range(15):
        documents.append(
            {
                "document_id": f"doc-{index}",
                "work_id": f"work-{index // 2}",
                "source": ("greek_phd", "kallipos", "openarchives")[index % 3],
                "line_count": 10 + index,
                "block_count": index % 3,
            }
        )
    first, assignments = assign_grouped_folds(documents, n_folds=5, seed="fixed")
    second, _ = assign_grouped_folds(documents, n_folds=5, seed="fixed")
    assert first == second
    for document, fold in zip(documents, first, strict=True):
        assert assignments[document["work_id"]] == fold
    assert set(first) == set(range(5))


def test_run_writes_lossless_arrays_and_receipt(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    rows = [_document("doc-1", "shared"), _document("doc-2", "shared")]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "output"
    args = argparse.Namespace(
        input=str(source),
        output_dir=str(output),
        split="train",
        workers=1,
        n_folds=5,
        fold_seed="fixed",
        expected_input_sha256=sha256_file(source),
        code_commit="test-commit",
        slurm_job_id="test-job",
    )
    receipt = run(args)
    assert receipt["status"] == "passed_exact_header_mask_only"
    assert receipt["document_count"] == 2
    assert receipt["work_count"] == 1
    assert receipt["line_count"] == 12
    assert np.load(output / "counts.npy", allow_pickle=False).shape == (
        12,
        len(FEATURE_NAMES),
    )
    assert np.load(output / "targets.npy", allow_pickle=False).tolist().count(TARGET_MASK) == 6
    documents = [json.loads(line) for line in (output / "documents.jsonl").read_text().splitlines()]
    assert documents[0]["fold"] == documents[1]["fold"]
    assert json.loads((output / "folds.json").read_text())["group_key"] == "work_id"
