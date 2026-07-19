from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sequence_models.repair_contextual_bibliography_roles import repair_pass, run


def _fixture() -> tuple[dict, list[dict]]:
    roles = [
        "OTHER",
        "FILLER",
        "FILLER",
        "OTHER",
        "CONTINUATION",
        "ENTRY",
        "FILLER",
        "OTHER",
        "BIB_HEADER",
        "FILLER",
        "OTHER",
    ]
    role_pass = {
        "schema_version": "raw-v1",
        "pass_id": "pass-a",
        "reviewer": "annotator-a",
        "lines": [
            {
                "line_alias": f"a{index}",
                "document_alias": "da",
                "source": "greek_phd",
                "role": role,
                "confidence": 0.9,
            }
            for index, role in enumerate(roles)
        ],
    }
    keys = [
        {
            "line_alias": f"a{index}",
            "document_alias": "da",
            "document_id": "document-a",
            "line_id": f"line-{index}",
            "source": "greek_phd",
            "abs_idx": 2 * index,
        }
        for index in range(len(roles))
    ]
    return role_pass, keys


def test_repair_changes_only_context_roles_in_unanchored_components() -> None:
    role_pass, keys = _fixture()
    original = copy.deepcopy(role_pass)
    derived, audit, summary = repair_pass(role_pass, keys)
    assert role_pass == original
    roles = [row["role"] for row in derived["lines"]]
    assert roles == [
        "OTHER",
        "OTHER",
        "OTHER",
        "OTHER",
        "CONTINUATION",
        "ENTRY",
        "FILLER",
        "OTHER",
        "BIB_HEADER",
        "OTHER",
        "OTHER",
    ]
    assert [row["old_role"] for row in audit] == ["FILLER", "FILLER", "FILLER"]
    assert all(row["entry_anchor_count"] == 0 for row in audit)
    assert summary["changed_line_count"] == 3
    assert summary["changed_document_count"] == 1
    assert summary["changed_by_source_and_old_role"] == {"greek_phd:FILLER": 3}
    assert summary["anchored_context_tail_profile"]["CONTINUATION"][
        "maximum_nearest_entry_distance"
    ] == 1


def test_run_preserves_input_and_fails_closed_on_existing_output(tmp_path: Path) -> None:
    role_pass, keys = _fixture()
    pass_path = tmp_path / "pass.json"
    key_path = tmp_path / "keys.jsonl"
    pass_text = json.dumps(role_pass) + "\n"
    pass_path.write_text(pass_text, encoding="utf-8")
    key_path.write_text(
        "".join(json.dumps(row) + "\n" for row in keys), encoding="utf-8"
    )
    output = tmp_path / "repair"
    receipt = run(pass_path=pass_path, line_key_path=key_path, output_dir=output)
    assert receipt["original_data_mutated"] is False
    assert pass_path.read_text(encoding="utf-8") == pass_text
    assert receipt["changed_line_count"] == 3
    assert (output / "changes.audit.jsonl").is_file()
    with pytest.raises(FileExistsError):
        run(pass_path=pass_path, line_key_path=key_path, output_dir=output)
