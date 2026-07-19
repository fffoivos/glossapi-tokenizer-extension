from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.materialize_consensus_silver import materialize, task_consensus  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def test_task_consensus_preserves_only_task_specific_agreement() -> None:
    decisions = task_consensus("CONTINUATION", "FILLER")
    assert decisions["bibliography_membership"] == {"label": "BIB", "trusted": True}
    assert decisions["entry_seed"] == {"label": "NOT_ENTRY", "trusted": True}
    assert decisions["heading_type"] == {"label": "NOT_HEADER", "trusted": True}
    assert decisions["context_role"] == {"label": "UNKNOWN", "trusted": False}
    assert decisions["fine_role"] == {"label": "UNKNOWN", "trusted": False}

    unknown = task_consensus("UNKNOWN", "OTHER")
    assert all(not decision["trusted"] for decision in unknown.values())


def test_materialize_excludes_documents_and_masks_disagreements(tmp_path: Path) -> None:
    documents_path, key_path = tmp_path / "documents.jsonl", tmp_path / "key.jsonl"
    pass_a_path, pass_b_path = tmp_path / "a.json", tmp_path / "b.json"
    documents = [
        {
            "document_id": "keep", "work_id": "work-keep", "source": "kallipos",
            "source_doc_id": "source-keep",
            "lines": [
                {"line_id": "line-1", "abs_idx": 0, "text": "## Βιβλιογραφία"},
                {"line_id": "line-2", "abs_idx": 1, "text": "A. Author. 2020."},
            ],
        },
        {
            "document_id": "drop", "work_id": "work-drop", "source": "greek_phd",
            "source_doc_id": "source-drop",
            "lines": [{"line_id": "line-3", "abs_idx": 0, "text": "footnote"}],
        },
    ]
    keys = [
        {"document_id": "keep", "document_alias": "doc-1", "line_id": "line-1", "line_alias": "a1", "abs_idx": 0, "source": "kallipos"},
        {"document_id": "keep", "document_alias": "doc-1", "line_id": "line-2", "line_alias": "a2", "abs_idx": 1, "source": "kallipos"},
        {"document_id": "drop", "document_alias": "doc-2", "line_id": "line-3", "line_alias": "a3", "abs_idx": 0, "source": "greek_phd"},
    ]
    rows_a = [
        {"document_alias": "doc-1", "line_alias": "a1", "source": "kallipos", "role": "BIB_HEADER", "confidence": 0.9},
        {"document_alias": "doc-1", "line_alias": "a2", "source": "kallipos", "role": "CONTINUATION", "confidence": 0.8},
        {"document_alias": "doc-2", "line_alias": "a3", "source": "greek_phd", "role": "ENTRY", "confidence": 0.7},
    ]
    rows_b = [
        {"document_alias": "doc-1", "line_alias": "a1", "source": "kallipos", "role": "BIB_HEADER", "confidence": 0.85},
        {"document_alias": "doc-1", "line_alias": "a2", "source": "kallipos", "role": "FILLER", "confidence": 0.75},
        {"document_alias": "doc-2", "line_alias": "a3", "source": "greek_phd", "role": "OTHER", "confidence": 0.7},
    ]
    _write_jsonl(documents_path, documents)
    _write_jsonl(key_path, keys)
    _write_json(pass_a_path, {"reviewer": "A", "lines": rows_a})
    _write_json(pass_b_path, {"reviewer": "B", "lines": rows_b})

    output = tmp_path / "output"
    receipt = materialize(
        documents_path=documents_path,
        line_key_path=key_path,
        pass_a_path=pass_a_path,
        pass_b_path=pass_b_path,
        excluded_document_ids=["drop"],
        output_dir=output,
        code_commit="test-commit",
        slurm_job_id="123",
    )
    labels = [json.loads(line) for line in (output / "labels.task-consensus.jsonl").read_text().splitlines()]
    overlays = [json.loads(line) for line in (output / "fine-role.overlay-v3.jsonl").read_text().splitlines()]
    filtered_docs = [json.loads(line) for line in (output / "documents.consensus-silver.jsonl").read_text().splitlines()]

    assert receipt["document_count"] == 1
    assert receipt["line_count"] == 2
    assert receipt["binary_disagreement_count"] == 0
    assert receipt["binary_agreed_line_count"] == 2
    assert [row["document_id"] for row in filtered_docs] == ["keep"]
    assert labels[1]["tasks"]["bibliography_membership"] == {"label": "BIB", "trusted": True}
    assert labels[1]["tasks"]["context_role"] == {"label": "UNKNOWN", "trusted": False}
    assert overlays[1]["role"] == "UNKNOWN"
    assert overlays[1]["role_status"] == "UNRESOLVED"
    assert json.loads((output / "exclusions.json").read_text())["documents"][0]["document_id"] == "drop"

    with pytest.raises(FileExistsError):
        materialize(
            documents_path=documents_path,
            line_key_path=key_path,
            pass_a_path=pass_a_path,
            pass_b_path=pass_b_path,
            excluded_document_ids=["drop"],
            output_dir=output,
            code_commit="test-commit",
            slurm_job_id="123",
        )
