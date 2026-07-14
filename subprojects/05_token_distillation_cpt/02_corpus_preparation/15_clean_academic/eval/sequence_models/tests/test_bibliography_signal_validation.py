from sequence_models.bibliography_signal_validation import (
    BLOCK_SCHEMA,
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
