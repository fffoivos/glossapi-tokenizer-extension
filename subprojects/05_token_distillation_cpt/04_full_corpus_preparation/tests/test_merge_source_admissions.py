from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module():
    path = HERE / "scripts" / "merge_source_admissions.py"
    spec = importlib.util.spec_from_file_location("phase04_merge_admissions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MERGE = load_module()


def report(phase: str, rows: list[tuple[str, str]]) -> dict:
    return {
        "schema_version": "source_quality_review_admission_v1",
        "review_phase": phase,
        "pending_adjudications": 0,
        "sources": [
            {
                "source_dataset": name,
                "decision": decision,
                "reasons": [f"{phase}:{decision}"],
                "post_clean_review_required": decision == "include_after_cleaning",
            }
            for name, decision in rows
        ],
    }


def test_merge_overlays_only_required_postclean_sources() -> None:
    pre = report("pre_clean", [("clean", "include"), ("repair", "include_after_cleaning")])
    post = report("post_clean", [("repair", "include")])
    result = MERGE.merge(pre, post)
    by_name = {row["source_dataset"]: row for row in result["sources"]}
    assert by_name["clean"]["decision"] == "include"
    assert by_name["repair"]["decision"] == "include"
    assert by_name["repair"]["preclean_decision"] == "include_after_cleaning"
    assert by_name["repair"]["post_clean_review_required"] is False


def test_merge_fails_if_postclean_coverage_is_partial_or_expanded() -> None:
    pre = report("pre_clean", [("a", "include_after_cleaning"), ("b", "include")])
    with pytest.raises(ValueError, match=r"missing=\['a'\]"):
        MERGE.merge(pre, report("post_clean", []))
    with pytest.raises(ValueError, match=r"unexpected=\['b'\]"):
        MERGE.merge(pre, report("post_clean", [("a", "include"), ("b", "include")]))
