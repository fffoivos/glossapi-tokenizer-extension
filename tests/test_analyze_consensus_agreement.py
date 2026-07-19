from __future__ import annotations

import json
import sys
from pathlib import Path

EVAL_DIR = (
    Path(__file__).resolve().parents[1]
    / "subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval"
)
sys.path.insert(0, str(EVAL_DIR))

from sequence_models.analyze_consensus_agreement import (  # noqa: E402
    _detection_summary,
    _task_summary,
    analyze,
)
from sequence_models.materialize_consensus_silver import LABEL_SCHEMA, task_consensus  # noqa: E402


def test_summaries_keep_denominators_separate() -> None:
    task = _task_summary({"agreement": 90, "disagreement": 10, "unavailable": 5})
    assert task["agreement_rate_on_comparable"] == 0.9
    assert task["trusted_coverage_fraction"] == 90 / 105
    detected = _detection_summary({"union": 10, "both": 7, "one": 3, "exact_subtype": 6})
    assert detected["detection_agreement_on_union"] == 0.7
    assert detected["subtype_agreement_conditional_on_both"] == 6 / 7


def test_analysis_reports_detection_and_subtype_separately(tmp_path: Path) -> None:
    votes = [
        ("BIB_HEADER", "BIB_HEADER"),
        ("NON_BIB_HEADER", "OTHER"),
        ("CONTINUATION", "FILLER"),
        ("UNKNOWN", "OTHER"),
    ]
    rows = []
    for index, (left, right) in enumerate(votes):
        rows.append(
            {
                "schema_version": LABEL_SCHEMA,
                "document_id": "doc",
                "line_id": f"line-{index}",
                "source": "greek_phd",
                "pass_a_role": left,
                "pass_b_role": right,
                "tasks": task_consensus(left, right),
            }
        )
    labels = tmp_path / "labels.jsonl"
    labels.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = analyze(
        labels_path=labels,
        output_path=tmp_path / "analysis.json",
        code_commit="test",
    )
    assert result["category_metrics"]["heading"]["union_detected_count"] == 2
    assert result["category_metrics"]["heading"]["both_detected_count"] == 1
    assert result["category_metrics"]["heading"]["detection_agreement_on_union"] == 0.5
    assert result["category_metrics"]["context_line"]["both_detected_count"] == 1
    assert result["category_metrics"]["context_line"]["subtype_agreement_conditional_on_both"] == 0
