from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sequence_models.bibliography_entry_dataset import FEATURE_NAMES
from sequence_models.bibliography_entry_models import Table
from sequence_models.bibliography_evolution_contract import ContractError
from sequence_models.bibliography_evolution_headers import ROLE_TO_ID
from sequence_models.bibliography_heading_deployment import (
    predict_documents,
    probabilities_to_role_ids,
)


def _table(tmp_path: Path, *, work_id: str = "unseen-work") -> Table:
    size = 3
    return Table(
        root=tmp_path,
        manifest={"split": "validation"},
        documents=(
            {
                "document_id": "document-1",
                "work_id": work_id,
                "source": "greek_phd",
                "line_start": 0,
                "line_end": size,
                "line_count": size,
            },
        ),
        counts=np.zeros((size, len(FEATURE_NAMES)), dtype=np.uint32),
        targets=np.zeros(size, dtype=np.int8),
        original_labels=np.zeros(size, dtype=np.uint8),
        header_kinds=np.zeros(size, dtype=np.uint8),
        abs_indices=np.arange(size, dtype=np.uint32),
        token_counts=np.ones(size, dtype=np.uint32),
        char_lengths=np.ones(size, dtype=np.uint32),
        block_indices=np.full(size, -1, dtype=np.int32),
        document_indices=np.zeros(size, dtype=np.uint32),
        folds=np.zeros(size, dtype=np.uint8),
    )


def _documents() -> list[dict]:
    return [
        {
            "document_id": "document-1",
            "work_id": "unseen-work",
            "source": "greek_phd",
            "split": "validation",
            "lines": [
                {"abs_idx": 0, "text": "This is ordinary prose."},
                {"abs_idx": 1, "text": "REFERENCES"},
                {"abs_idx": 2, "text": "Smith, A. 2020. Example. pp. 1-2."},
            ],
        }
    ]


class _ConstantBundle:
    def predict(self, texts: list[str], numeric: np.ndarray) -> np.ndarray:
        assert numeric.shape[0] == len(texts)
        return np.tile(
            np.asarray([0.8, 0.1, 0.2, 0.5], dtype=np.float32),
            (len(texts), 1),
        )


def test_role_assignment_requires_candidate_and_any_heading_threshold() -> None:
    probability = np.asarray(
        [
            [0.9, 0.8, 0.1, 0.0],
            [0.4, 0.0, 0.1, 0.3],
            [0.9, 0.0, 0.1, 0.8],
        ],
        dtype=np.float32,
    )
    roles = probabilities_to_role_ids(
        probability, np.asarray([True, True, False]), threshold=0.5
    )
    assert roles.tolist() == [ROLE_TO_ID["BIB_HEADER"], ROLE_TO_ID["NONE"], ROLE_TO_ID["NONE"]]


def test_unseen_mode_uses_five_fold_mean_and_scored_candidates_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "training_work_ids.json").write_text(
        json.dumps({"work_ids": ["train-work"]}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "sequence_models.bibliography_heading_deployment.verify_deployment",
        lambda _root: {},
    )
    calls: list[str] = []

    def load(path: Path) -> _ConstantBundle:
        calls.append(path.name)
        return _ConstantBundle()

    monkeypatch.setattr(
        "sequence_models.bibliography_heading_deployment._load_bundle", load
    )
    roles, probability, candidates = predict_documents(
        deployment,
        _table(tmp_path),
        _documents(),
        np.asarray([0.0, 0.0, 0.9], dtype=np.float32),
    )
    assert calls == [f"fold{fold}.pkl" for fold in range(5)]
    assert candidates.tolist() == [False, True, False]
    assert np.allclose(probability[1], [0.8, 0.1, 0.2, 0.5])
    assert not np.count_nonzero(probability[[0, 2]])
    assert roles.tolist() == [0, ROLE_TO_ID["NON_BIB_HEADER"], 0]


def test_unseen_mode_rejects_training_work_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    deployment = tmp_path / "deployment"
    deployment.mkdir()
    (deployment / "training_work_ids.json").write_text(
        json.dumps({"work_ids": ["train-work"]}) + "\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        "sequence_models.bibliography_heading_deployment.verify_deployment",
        lambda _root: {},
    )
    with pytest.raises(ContractError, match="overlaps a training work"):
        predict_documents(
            deployment,
            _table(tmp_path, work_id="train-work"),
            [
                {
                    **_documents()[0],
                    "work_id": "train-work",
                }
            ],
            np.asarray([0.0, 0.0, 0.9], dtype=np.float32),
        )
