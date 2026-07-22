from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sequence_models.repair_non_markdown_header_roles import repair_pass, run


def _fixture() -> tuple[dict, list[dict], list[dict]]:
    roles_and_texts = [
        ("BIB_HEADER", "# References"),
        ("BIB_SUBHEADER", "  ## Greek sources"),
        ("NON_BIB_HEADER", "   #### Appendix"),
        ("NON_BIB_HEADER", "Chapter 4"),
        ("BIB_HEADER", "##References"),
        ("BIB_SUBHEADER", "####### Too deep"),
        ("ENTRY", "# Citation-looking content"),
        ("OTHER", "Ordinary prose"),
    ]
    role_pass = {
        "schema_version": "context-repaired-v1",
        "pass_id": "pass-a",
        "reviewer": "annotator-a+context-repair-v1",
        "lines": [
            {
                "line_alias": f"a{index}",
                "document_alias": "da",
                "source": "greek_phd",
                "role": role,
                "confidence": 0.9,
            }
            for index, (role, _) in enumerate(roles_and_texts)
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
        for index in range(len(roles_and_texts))
    ]
    documents = [
        {
            "document_id": "document-a",
            "lines": [
                {"line_id": f"line-{index}", "abs_idx": 2 * index, "text": text}
                for index, (_, text) in enumerate(roles_and_texts)
            ],
        }
    ]
    return role_pass, keys, documents


def test_repair_changes_only_non_markdown_header_roles() -> None:
    role_pass, keys, documents = _fixture()
    original = copy.deepcopy(role_pass)
    derived, audit, summary = repair_pass(role_pass, keys, documents)
    assert role_pass == original
    assert [row["role"] for row in derived["lines"]] == [
        "BIB_HEADER",
        "BIB_SUBHEADER",
        "NON_BIB_HEADER",
        "OTHER",
        "OTHER",
        "OTHER",
        "ENTRY",
        "OTHER",
    ]
    assert [row["old_role"] for row in audit] == [
        "NON_BIB_HEADER",
        "BIB_HEADER",
        "BIB_SUBHEADER",
    ]
    assert all(row["new_role"] == "OTHER" for row in audit)
    assert summary["changed_line_count"] == 3
    assert summary["retained_markdown_headers_by_source_and_role"] == {
        "greek_phd:BIB_HEADER": 1,
        "greek_phd:BIB_SUBHEADER": 1,
        "greek_phd:NON_BIB_HEADER": 1,
    }


def test_repair_requires_documents_to_cover_keyed_lines() -> None:
    role_pass, keys, documents = _fixture()
    documents[0]["lines"].pop()
    with pytest.raises(ValueError, match="cover every keyed line"):
        repair_pass(role_pass, keys, documents)


def test_run_preserves_input_and_fails_closed_on_existing_output(tmp_path: Path) -> None:
    role_pass, keys, documents = _fixture()
    pass_path = tmp_path / "pass.json"
    key_path = tmp_path / "keys.jsonl"
    documents_path = tmp_path / "documents.jsonl"
    pass_text = json.dumps(role_pass) + "\n"
    pass_path.write_text(pass_text, encoding="utf-8")
    key_path.write_text(
        "".join(json.dumps(row) + "\n" for row in keys), encoding="utf-8"
    )
    documents_path.write_text(
        "".join(json.dumps(row) + "\n" for row in documents), encoding="utf-8"
    )
    output = tmp_path / "repair"
    receipt = run(
        pass_path=pass_path,
        line_key_path=key_path,
        documents_path=documents_path,
        output_dir=output,
    )
    assert receipt["original_data_mutated"] is False
    assert pass_path.read_text(encoding="utf-8") == pass_text
    assert receipt["changed_line_count"] == 3
    assert (output / "changes.audit.jsonl").is_file()
    with pytest.raises(FileExistsError):
        run(
            pass_path=pass_path,
            line_key_path=key_path,
            documents_path=documents_path,
            output_dir=output,
        )
