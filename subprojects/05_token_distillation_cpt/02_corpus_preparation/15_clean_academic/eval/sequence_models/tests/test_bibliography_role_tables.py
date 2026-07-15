from __future__ import annotations

from sequence_models.bibliography_role_tables import _connector_supervision
from sequence_models.bibliography_role_v2 import ROLE_TO_ID


def test_exact_continuation_supplies_connector_and_subtype_targets() -> None:
    result = _connector_supervision(
        {"role": "CONTINUATION", "role_status": "AGREED_REVIEW"}
    )
    assert result["role"] == ROLE_TO_ID["CONTINUATION"]
    assert result["connector_target"] == 1
    assert result["connector_trusted"] == 1
    assert result["subtype_target"] == 1
    assert result["subtype_trusted"] == 1


def test_continuation_filler_disagreement_preserves_only_connector_truth() -> None:
    result = _connector_supervision(
        {
            "role": "UNKNOWN", "role_status": "UNRESOLVED",
            "raw_role_votes": {"a": ["CONTINUATION"], "b": ["FILLER"]},
        }
    )
    assert result["connector_target"] == 1
    assert result["connector_trusted"] == 1
    assert result["subtype_trusted"] == 0
    assert result["other_trusted"] == 0


def test_non_connector_disagreement_remains_fully_masked() -> None:
    result = _connector_supervision(
        {
            "role": "UNKNOWN", "role_status": "UNRESOLVED",
            "raw_role_votes": {"a": ["FILLER"], "b": ["OTHER"]},
        }
    )
    assert result["connector_trusted"] == 0
