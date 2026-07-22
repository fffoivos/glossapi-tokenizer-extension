from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sequence_models.bibliography_entry_dataset import run as run_dataset
from sequence_models.bibliography_entry_dataset import sha256_file
from sequence_models.bibliography_entry_models import (
    ALL_ARMS,
    Transform,
    binary_metrics,
    load_table,
    run,
)


def _document(index: int) -> dict:
    document_id = f"doc-{index}"
    texts = (
        ("Ordinary running prose without citation evidence.", "O"),
        ("ΒΙΒΛΙΟΓΡΑΦΙΑ", "BIB"),
        (f"[{index + 1}] Smith, J. (2020). Title. London: Press, pp. 1-9.", "BIB"),
        ("Another ordinary sentence in the body.", "O"),
    )
    return {
        "schema_version": "academic-structure-gold-v1",
        "document_id": document_id,
        "work_id": f"work-{index}",
        "source": ("greek_phd", "kallipos", "openarchives")[index % 3],
        "split": "train",
        "coverage": "full_document" if index % 2 else "annotated_windows",
        "n_physical_lines": len(texts),
        "n_present_lines": len(texts),
        "lines": [
            {
                "line_id": f"{document_id}:{line_index}",
                "abs_idx": line_index,
                "text": text,
                "label": label,
                "token_count": len(text.split()),
                "is_running_prose": None,
            }
            for line_index, (text, label) in enumerate(texts)
        ],
    }


def _table(tmp_path: Path) -> Path:
    source = tmp_path / "source.jsonl"
    source.write_text(
        "".join(json.dumps(_document(index)) + "\n" for index in range(15)),
        encoding="utf-8",
    )
    table = tmp_path / "table"
    run_dataset(
        argparse.Namespace(
            input=str(source),
            output_dir=str(table),
            split="train",
            workers=1,
            n_folds=5,
            fold_seed="fixed",
            expected_input_sha256=sha256_file(source),
            code_commit="test",
            slurm_job_id="test",
        )
    )
    return table


def test_transforms_preserve_declared_feature_interfaces() -> None:
    counts = np.asarray([[0, 1], [2, 4], [1, 0]], dtype=np.uint32)
    assert Transform.fit("L1", counts).apply(counts).shape == (3, 2)
    assert Transform.fit("L2", counts).apply(counts).shape == (3, 2)
    assert Transform.fit("L3", counts).apply(counts).shape == (3, 2)
    assert Transform.fit("L4", counts).apply(counts).shape == (3, 4)
    assert Transform.fit("D1", counts).apply(counts).shape == (3, 4)


def test_binary_metrics_are_exact() -> None:
    metrics = binary_metrics(
        np.asarray([1, 1, 0, 0]),
        np.asarray([0.9, 0.4, 0.6, 0.1]),
        threshold=0.5,
    )
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == 0.5
    assert metrics["tp"] == metrics["fp"] == metrics["fn"] == 1


def test_oof_ladder_writes_every_arm_without_opening_validation(tmp_path: Path) -> None:
    table_dir = _table(tmp_path)
    table = load_table(table_dir)
    assert table.manifest["split"] == "train"
    output = tmp_path / "models"
    receipt = run(
        argparse.Namespace(
            table_dir=str(table_dir),
            output_dir=str(output),
            c_grid=(0.5,),
            seed=7,
            code_commit="test",
            slurm_job_id="test",
        )
    )
    assert receipt["status"] == "passed_train_oof_only_validation_unopened"
    assert receipt["validation_opened"] is False
    assert set(receipt["arms"]) == set(ALL_ARMS)
    for arm in ALL_ARMS:
        probability = np.load(output / f"{arm}.oof_probability.npy", allow_pickle=False)
        labelled = table.targets >= 0
        assert np.isfinite(probability[labelled]).all()
        assert ((0 <= probability[labelled]) & (probability[labelled] <= 1)).all()
    assert len(list((output / "models").glob("L1.fold*.pkl"))) == 5
    assert len(list((output / "models").glob("D1.fold*.pkl"))) == 5
