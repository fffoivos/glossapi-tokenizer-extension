from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

EVAL_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.codex56_audit import (  # noqa: E402
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    build_audit,
    preflight_silver_splits,
    summarize_findings,
    validate_responses,
)
from sequence_models.contract import (  # noqa: E402
    GoldDocument,
    GoldLine,
    canonical_json_sha256,
)


def _document(document_id: str, offset: int = 0) -> GoldDocument:
    lines = []
    for index in range(24):
        if index % 4 == 0:
            text, label, prose = (
                f"{index + 1}. Ενότητα ........ {index + 1}",
                "TOC",
                False,
            )
        elif index % 4 == 1:
            text, label, prose = (
                f"Smith, J. ({2000 + index}). Title. London: Press.",
                "BIB",
                False,
            )
        elif index % 4 == 2:
            text, label, prose = (
                "Όπως υποστηρίζει ο Smith (2020), αυτό είναι κανονικό σώμα κειμένου.",
                "O",
                True,
            )
        else:
            text, label, prose = "| Έτος | 2024 |", "O", False
        lines.append(
            GoldLine(f"{document_id}-l{index}", offset + index, text, label, 12, prose)
        )
    return GoldDocument(
        document_id=document_id,
        work_id=f"w-{document_id}",
        representation_id=f"r-{document_id}",
        source="fixture",
        split="validation",
        coverage="full",
        n_physical_lines=offset + len(lines),
        n_present_lines=len(lines),
        annotation_status="LLM_silver",
        annotator_ids=("gpt-5.5",),
        adjudicator_id=None,
        tokenizer_id="fixture",
        tokenizer_revision="1",
        lines=tuple(lines),
        annotation_engine="gpt-5.5",
        task_scope="bibliography_toc_windows",
    )


def test_builder_is_deterministic_blinded_and_stratified() -> None:
    documents = [_document(f"d{index}") for index in range(4)]
    baseline = {doc.document_id: ["O"] * len(doc.lines) for doc in documents}
    candidate = {
        doc.document_id: [
            line.label if line.label != "O" else ("TOC" if i % 4 == 3 else "O")
            for i, line in enumerate(doc.lines)
        ]
        for doc in documents
    }
    left = build_audit(documents, baseline, candidate, per_stratum=3, context_radius=2)
    right = build_audit(documents, baseline, candidate, per_stratum=3, context_radius=2)
    assert left == right
    requests, keys, manifest = left
    assert manifest["counts"] == {stratum: 3 for stratum in manifest["counts"]}
    assert len(requests) == len(keys) == 12
    serialized = str(requests)
    assert "stratum" not in serialized
    assert "gold_label" not in serialized
    assert "candidate_prediction" not in serialized
    assert all("request_sha256" in row for row in requests)


def test_builder_round_robins_sources_within_each_stratum() -> None:
    documents = []
    for source in ("a", "b", "c"):
        for index in range(3):
            documents.append(replace(_document(f"{source}-{index}"), source=source))
    baseline = {doc.document_id: ["O"] * len(doc.lines) for doc in documents}
    candidate = {
        doc.document_id: [line.label for line in doc.lines] for doc in documents
    }

    _requests, _keys, manifest = build_audit(
        documents,
        baseline,
        candidate,
        per_stratum=6,
        require_full=False,
    )

    for counts in manifest["source_counts"].values():
        if counts:
            assert max(counts.values()) - min(counts.values()) <= 1


def test_annotated_window_context_never_crosses_an_unknown_physical_gap() -> None:
    base = _document("windowed")
    lines = tuple(
        replace(line, abs_idx=index)
        for line, index in zip(base.lines[:8], [0, 1, 100, 101, 102, 103, 104, 105])
    )
    document = replace(
        base,
        coverage="annotated_windows",
        n_physical_lines=200,
        n_present_lines=100,
        lines=lines,
    )
    baseline = {document.document_id: ["O"] * len(lines)}
    candidate = {document.document_id: [line.label for line in lines]}

    requests, _keys, _manifest = build_audit(
        [document],
        baseline,
        candidate,
        per_stratum=2,
        require_full=False,
    )

    assert requests
    for request in requests:
        indices = [line["abs_idx"] for line in request["lines"]]
        assert all(right == left + 1 for left, right in zip(indices, indices[1:]))
        assert request["context_coverage"] == "contiguous_observed_window_only"
        assert request["crosses_unrepresented_interval"] is False


def test_full_document_context_never_crosses_more_than_two_known_blanks() -> None:
    base = _document("full-gap")
    lines = tuple(
        replace(line, abs_idx=index)
        for line, index in zip(base.lines[:8], [0, 1, 100, 101, 102, 103, 104, 105])
    )
    document = replace(
        base,
        coverage="full_document",
        n_physical_lines=200,
        n_present_lines=8,
        lines=lines,
    )
    baseline = {document.document_id: ["O"] * len(lines)}
    candidate = {document.document_id: [line.label for line in lines]}

    requests, _keys, _manifest = build_audit(
        [document],
        baseline,
        candidate,
        per_stratum=2,
        require_full=False,
    )

    assert requests
    for request in requests:
        indices = [line["abs_idx"] for line in request["lines"]]
        assert all(right - left <= 3 for left, right in zip(indices, indices[1:]))
        assert (
            request["context_coverage"]
            == "full_document_with_at_most_two_known_blank_lines_per_gap"
        )


def test_builder_fails_closed_on_stratum_shortfall() -> None:
    document = _document("small")
    baseline = {document.document_id: ["O"] * len(document.lines)}
    candidate = {document.document_id: ["O"] * len(document.lines)}
    with pytest.raises(ValueError, match="cannot reach"):
        build_audit([document], baseline, candidate, per_stratum=2)


def test_builder_rejects_sealed_or_mixed_splits_before_selection() -> None:
    sealed = _document("sealed")
    sealed = replace(sealed, split="test")
    predictions = {sealed.document_id: ["O"] * len(sealed.lines)}
    with pytest.raises(ValueError, match="sealed split"):
        build_audit(
            [sealed],
            predictions,
            predictions,
            per_stratum=1,
            require_full=False,
        )


def test_file_preflight_rejects_sealed_split_before_lines_are_parsed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sealed.jsonl"
    path.write_text(
        '{"schema_version":"academic-structure-gold-v1","split":"test",'
        '"lines":"intentionally invalid"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sealed split"):
        preflight_silver_splits(path, allowed_split="validation")

    train = _document("train")
    train = replace(train, split="train")
    predictions = {train.document_id: ["O"] * len(train.lines)}
    with pytest.raises(ValueError, match="does not match allowed split"):
        build_audit(
            [train],
            predictions,
            predictions,
            per_stratum=1,
            require_full=False,
        )


def test_response_validation_binds_model_hash_span_and_evidence() -> None:
    documents = [_document(f"d{index}") for index in range(4)]
    baseline = {doc.document_id: ["O"] * len(doc.lines) for doc in documents}
    candidate = {
        doc.document_id: [line.label for line in doc.lines] for doc in documents
    }
    requests, _keys, _manifest = build_audit(
        documents, baseline, candidate, per_stratum=1, require_full=False
    )
    responses = []
    for request in requests:
        responses.append(
            {
                "schema_version": RESPONSE_SCHEMA,
                "request_id": request["request_id"],
                "request_sha256": request["request_sha256"],
                "reviewer_model": "gpt-5.6-luna",
                "label": "OTHER",
                "start_abs_idx": None,
                "end_abs_idx": None,
                "should_remove": False,
                "confidence": 0.9,
                "structural_cues": ["running prose"],
                "evidence_abs_indices": [request["target_abs_idx"]],
            }
        )
    receipt = validate_responses(requests, responses, expected_model="gpt-5.6-luna")
    assert receipt["status"] == "passed"
    assert receipt["response_count"] == len(requests)

    bad = [dict(row) for row in responses]
    bad[0]["reviewer_model"] = "fallback"
    with pytest.raises(ValueError, match="model mismatch"):
        validate_responses(requests, bad, expected_model="gpt-5.6-luna")

    bad = [dict(row) for row in responses]
    bad[0].update(label="BIB", should_remove=True)
    with pytest.raises(ValueError, match="requires a span"):
        validate_responses(requests, bad, expected_model="gpt-5.6-luna")

    action_request = requests[0]
    context_indices = [row["abs_idx"] for row in action_request["lines"]]
    outside_target = next(
        index for index in context_indices if index != action_request["target_abs_idx"]
    )
    bad = [dict(row) for row in responses]
    bad[0].update(
        label="BIB",
        should_remove=True,
        start_abs_idx=outside_target,
        end_abs_idx=outside_target,
    )
    with pytest.raises(ValueError, match="cover the target"):
        validate_responses(requests, bad, expected_model="gpt-5.6-luna")

    tampered = [dict(row) for row in requests]
    tampered[0] = dict(tampered[0])
    tampered[0]["lines"] = [dict(row) for row in tampered[0]["lines"]]
    tampered[0]["lines"][0]["text"] = "tampered context"
    with pytest.raises(ValueError, match="content hash mismatch"):
        validate_responses(tampered, responses, expected_model="gpt-5.6-luna")


def test_findings_apply_staged_expansion_policy() -> None:
    keys = []
    requests = []
    responses = []
    for index in range(20):
        request_id = f"{index:064x}"
        request = {
            "schema_version": REQUEST_SCHEMA,
            "request_id": request_id,
            "prompt_version": "p1",
            "source": "source-a",
            "opaque_document_id": f"{index + 100:064x}",
            "target_abs_idx": index,
            "context_start_abs_idx": index,
            "context_end_abs_idx": index,
            "lines": [{"line_id": f"l{index}", "abs_idx": index, "text": "κείμενο"}],
        }
        request["request_sha256"] = canonical_json_sha256(request)
        requests.append(request)
        keys.append(
            {
                "request_id": request_id,
                "request_sha256": request["request_sha256"],
                "source": "source-a",
                "stratum": "toc_high_risk",
                "gold_label": "O",
            }
        )
        responses.append(
            {
                "schema_version": RESPONSE_SCHEMA,
                "request_id": request_id,
                "request_sha256": request["request_sha256"],
                "reviewer_model": "gpt-5.6-luna",
                "label": "TOC" if index < 6 else "OTHER",
                "start_abs_idx": index if index < 6 else None,
                "end_abs_idx": index if index < 6 else None,
                "should_remove": index < 6,
                "confidence": 0.95,
                "structural_cues": ["fixture"],
                "evidence_abs_indices": [index],
            }
        )
    findings = summarize_findings(
        keys, requests, responses, expected_model="gpt-5.6-luna"
    )
    assert findings["affected_sources"] == ["source-a"]
    assert findings["slices"][0]["expand_slice"] is True
    assert findings["recommend_full_1392_reaudit"] is True
    assert "disagreement_rate" not in findings
    assert findings["audit_sample_disagreement_rate"] == pytest.approx(0.3)
