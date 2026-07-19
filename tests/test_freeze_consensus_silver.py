from __future__ import annotations

import json
import stat
import sys
from pathlib import Path

import pytest

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.audit_consensus_silver import audit  # noqa: E402
from sequence_models.contract import sha256_file  # noqa: E402
from sequence_models.freeze_consensus_silver import (  # noqa: E402
    FREEZE_STATUS,
    _task_metrics,
    freeze_consensus_silver,
)
from sequence_models.materialize_consensus_silver import materialize  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    documents_path = tmp_path / "documents.jsonl"
    key_path = tmp_path / "key.jsonl"
    pass_a_path = tmp_path / "a.json"
    pass_b_path = tmp_path / "b.json"
    sources = ["greek_phd", "kallipos", "openarchives"] + ["greek_phd"] * 147
    document_ids = ["g-keep", "k-keep", "o-keep"] + [
        f"g-drop-{index}" for index in range(147)
    ]
    documents = []
    keys = []
    rows_a = []
    rows_b = []
    for index, (source, document_id) in enumerate(zip(sources, document_ids, strict=True)):
        line_id = f"line-{index}"
        alias = f"alias-{index}"
        document_alias = f"doc-{index}"
        documents.append(
            {
                "document_id": document_id,
                "work_id": f"work-{index}",
                "source_doc_id": f"source-{index}",
                "source": source,
                "lines": [{"line_id": line_id, "abs_idx": 0, "text": f"text {index}"}],
            }
        )
        keys.append(
            {
                "document_id": document_id,
                "document_alias": document_alias,
                "line_id": line_id,
                "line_alias": alias,
                "abs_idx": 0,
                "source": source,
            }
        )
        role_a, role_b = ("CONTINUATION", "FILLER") if source == "kallipos" else ("OTHER", "OTHER")
        rows_a.append(
            {
                "document_alias": document_alias,
                "line_alias": alias,
                "source": source,
                "role": role_a,
                "confidence": 0.9,
            }
        )
        rows_b.append(
            {
                "document_alias": document_alias,
                "line_alias": alias,
                "source": source,
                "role": role_b,
                "confidence": 0.8,
            }
        )
    _write_jsonl(documents_path, documents)
    _write_jsonl(key_path, keys)
    _write_json(pass_a_path, {"reviewer": "A", "lines": rows_a})
    _write_json(pass_b_path, {"reviewer": "B", "lines": rows_b})
    artifact = tmp_path / "artifact"
    materialize(
        documents_path=documents_path,
        line_key_path=key_path,
        pass_a_path=pass_a_path,
        pass_b_path=pass_b_path,
        excluded_document_ids=document_ids[3:],
        output_dir=artifact,
        code_commit="fixture",
        slurm_job_id="1",
    )
    audit_path = tmp_path / "audit.json"
    audit(
        artifact_dir=artifact,
        original_documents_path=documents_path,
        original_line_key_path=key_path,
        pass_a_path=pass_a_path,
        pass_b_path=pass_b_path,
        output_path=audit_path,
        expected_document_count=3,
        expected_line_count=3,
    )
    original = tmp_path / "original-blocked.json"
    _write_json(
        original,
        {
            "status": "blocked",
            "a_b_binary_agreement_overall": 0.977,
            "gates": {"binary_agreement_overall_gte_0_98": False},
        },
    )
    return artifact, audit_path, original


def test_freeze_preserves_original_failure_and_locks_consensus(tmp_path: Path) -> None:
    artifact, audit_path, original = _fixture(tmp_path)
    output = artifact / "FROZEN.consensus-silver.receipt.json"
    result = freeze_consensus_silver(
        artifact_dir=artifact,
        audit_path=audit_path,
        original_blocked_receipt_path=original,
        output_path=output,
        expected_source_document_counts={"greek_phd": 1, "kallipos": 1, "openarchives": 1},
        expected_document_count=3,
        expected_line_count=3,
        code_commit="test",
        lock_inputs=True,
    )

    assert result["status"] == FREEZE_STATUS
    assert all(result["gates"].values())
    assert result["task_metrics"]["bibliography_membership"]["agreement_rate_on_comparable"] == 1.0
    assert result["task_metrics"]["fine_role"]["disagreement_count"] == 1
    assert result["original_150_document_attempt"]["status"] == "blocked"
    assert result["sealed_hashes"]["independent_audit_sha256"] == sha256_file(audit_path)
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    assert stat.S_IMODE((artifact / "labels.task-consensus.jsonl").stat().st_mode) == 0o440
    assert stat.S_IMODE(audit_path.stat().st_mode) == 0o440


def test_task_metrics_do_not_count_unknown_votes_as_disagreement() -> None:
    metrics = _task_metrics({"agreement": 98, "disagreement": 2, "unavailable": 5})
    assert metrics["agreement_rate_on_comparable"] == 0.98
    assert metrics["trusted_coverage_fraction"] == 98 / 105
    assert metrics["unresolved_fraction"] == 7 / 105


def test_freeze_rejects_a_nonblocked_original_attempt(tmp_path: Path) -> None:
    artifact, audit_path, original = _fixture(tmp_path)
    _write_json(original, {"status": "passed", "gates": {"binary_agreement_overall_gte_0_98": True}})
    with pytest.raises(ValueError, match="not the preserved blocked attempt"):
        freeze_consensus_silver(
            artifact_dir=artifact,
            audit_path=audit_path,
            original_blocked_receipt_path=original,
            output_path=artifact / "FROZEN.consensus-silver.receipt.json",
            expected_source_document_counts={"greek_phd": 1, "kallipos": 1, "openarchives": 1},
            expected_document_count=3,
            expected_line_count=3,
            code_commit="test",
        )
