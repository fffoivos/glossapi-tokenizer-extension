from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module():
    path = HERE / "scripts" / "run_codex_source_reviews.py"
    scripts = str(path.parent)
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("phase04_codex_reviews", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


REVIEW = load_module()


def request(review_id: str, slot: str, source: str = "demo") -> dict:
    return {
        "review_id": review_id,
        "sample_id": "a" * 64,
        "reviewer_slot": slot,
        "source_dataset": source,
    }


def response_schema() -> dict:
    return json.loads((HERE / "schemas" / "source_review_response.schema.json").read_text())


def test_batch_schema_keeps_response_refs_resolvable() -> None:
    schema = REVIEW.make_batch_schema(response_schema(), 3)
    assert schema["properties"]["responses"]["minItems"] == 3
    assert schema["$defs"]["response"]["properties"]["defects"]["properties"][
        "pii"
    ]["$ref"] == "#/$defs/severity"
    assert "severity" in schema["$defs"]
    assert schema["$defs"]["response"]["properties"]["schema_version"]["type"] == "string"
    assert schema["$defs"]["severity"]["type"] == "string"


def test_batch_plan_never_mixes_reviewer_slots() -> None:
    rows = [
        request("p1", "primary"),
        request("s1", "secondary"),
        request("p2", "primary"),
        request("p3", "primary", "other"),
    ]
    batches = REVIEW.batch_plan(rows, 6)
    assert sorted(len(batch) for batch in batches) == [1, 1, 2]
    for batch in batches:
        assert len({row["reviewer_slot"] for row in batch}) == 1
        assert len({row["source_dataset"] for row in batch}) == 1


def test_response_identity_drift_is_rejected() -> None:
    req = request("review-1", "primary")
    response = {
        "schema_version": "source_quality_review_response_v1",
        **req,
        "substantive_training_value": "high",
        "quality_score": 4,
        "language_register": "modern_greek",
        "defects": {
            key: "none"
            for key in (
                "html_or_markup",
                "boilerplate",
                "ocr_corruption",
                "mojibake",
                "fragmentation",
                "tables_or_loops",
                "pii",
                "non_greek_drift",
            )
        },
        "variability": {"template_similarity": "low", "substantive_variation": "high"},
        "action": "include",
        "defects_deterministically_repairable": True,
        "safety_or_license_blocker": False,
        "confidence": "high",
        "evidence": "Clean substantive Greek prose.",
    }
    assert REVIEW.response_for_request(response, req)["review_id"] == "review-1"
    response["sample_id"] = "b" * 64
    with pytest.raises(ValueError, match="identity drift"):
        REVIEW.response_for_request(response, req)
