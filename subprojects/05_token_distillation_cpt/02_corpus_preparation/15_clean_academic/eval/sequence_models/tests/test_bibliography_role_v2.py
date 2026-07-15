from __future__ import annotations

from pathlib import Path

from sequence_models.bibliography_role_dataset import (
    TARGET_ENTRY_ANCHOR,
    TARGET_MASK,
    TARGET_NEGATIVE,
    entry_anchor_target,
    load_role_contract,
)
from sequence_models.bibliography_role_v2 import apply_heading_overrides, migrate_row


ROOT = Path(__file__).resolve().parents[1]


def _row(role: str = "HEADER") -> dict:
    return {
        "schema_version": "bibliography-role-overlay-v2",
        "document_id": "d",
        "line_id": "l",
        "role": role,
        "role_status": "AGREED_REVIEW",
        "role_confidence": 0.9,
        "reviewers": ["a", "b"],
        "raw_role_votes": {"a": [role], "b": [role]},
    }


def test_contract_v2_accounts_for_every_operational_role() -> None:
    contract = load_role_contract(ROOT / "bibliography_role_contract_v2.json")
    assert entry_anchor_target(
        role="ENTRY", role_status="AGREED_REVIEW", contract=contract
    ) == TARGET_ENTRY_ANCHOR
    assert entry_anchor_target(
        role="BIB_HEADER", role_status="AGREED_REVIEW", contract=contract
    ) == TARGET_NEGATIVE
    assert entry_anchor_target(
        role="UNKNOWN", role_status="AGREED_REVIEW", contract=contract
    ) == TARGET_MASK


def test_migration_preserves_provenance_and_maps_raw_votes() -> None:
    migrated = migrate_row(_row())
    assert migrated["role"] == "BIB_HEADER"
    assert migrated["migrated_from_role"] == "HEADER"
    assert migrated["raw_role_votes"] == {"a": ["BIB_HEADER"], "b": ["BIB_HEADER"]}


def test_heading_override_cannot_silently_drop_absent_lines() -> None:
    migrated = migrate_row(_row("NON_BIB"))
    result = apply_heading_overrides(
        [migrated],
        {
            ("d", "l"): {
                "role": "NON_BIB_HEADER",
                "role_status": "AGREED_REVIEW",
                "role_confidence": 1.0,
                "reviewers": ["x", "y"],
                "raw_role_votes": {"x": ["NON_BIB_HEADER"], "y": ["NON_BIB_HEADER"]},
                "label_origin": "dual_heading_review",
            }
        },
    )
    assert result[0]["role"] == "NON_BIB_HEADER"
    assert result[0]["heading_override_applied"] is True
