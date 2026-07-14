import json

import pytest

from sequence_models.bibliography_signal_validation import (
    BLOCK_SCHEMA,
    _validation_quality_exclusions,
    select_train_recall_candidate,
)


def test_recall_candidate_is_selected_only_from_train_oof_precision_floor():
    report = {
        "schema_version": BLOCK_SCHEMA,
        "validation_opened": False,
        "candidates": [
            {"metrics": {"line_precision": 0.89, "line_recall": 0.99, "token_recall": 0.99}},
            {"metrics": {"line_precision": 0.91, "line_recall": 0.96, "token_recall": 0.97}},
            {"metrics": {"line_precision": 0.95, "line_recall": 0.94, "token_recall": 0.98}},
        ],
    }
    assert select_train_recall_candidate(report) is report["candidates"][1]


def test_quality_exclusions_preserve_blind_and_outcome_directed_populations(tmp_path):
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "bibliography-validation-quality-decisions-v1",
                "prediction_blind_excluded_document_ids": ["blind"],
                "outcome_directed_additional_excluded_document_ids": ["followup"],
                "documents": [
                    {"document_id": "blind", "decision": "exclude"},
                    {"document_id": "followup", "decision": "exclude"},
                    {"document_id": "keep", "decision": "keep"},
                ],
            }
        ),
        encoding="utf-8",
    )
    excluded, blind, followup, _packet = _validation_quality_exclusions(path)
    assert excluded == {"blind", "followup"}
    assert blind == {"blind"}
    assert followup == {"followup"}


def test_quality_exclusions_reject_inconsistent_provenance(tmp_path):
    path = tmp_path / "quality.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "bibliography-validation-quality-decisions-v1",
                "prediction_blind_excluded_document_ids": ["blind"],
                "outcome_directed_additional_excluded_document_ids": [],
                "documents": [
                    {"document_id": "blind", "decision": "exclude"},
                    {"document_id": "untracked", "decision": "exclude"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="provenance"):
        _validation_quality_exclusions(path)
