from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]
SCRIPTS = HERE / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module():
    path = SCRIPTS / "source_license.py"
    spec = importlib.util.spec_from_file_location("phase04_source_license", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LICENSE = load_module()
SOURCES_PATH = HERE / "configs" / "sources.json"
ADJUDICATION_PATH = HERE / "configs" / "source_license_adjudication.json"


def tracked() -> tuple[dict, dict]:
    return (
        json.loads(SOURCES_PATH.read_text(encoding="utf-8")),
        json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8")),
    )


def test_tracked_adjudication_is_complete_default_deny_and_receipt_bound() -> None:
    sources, adjudication = tracked()
    assert LICENSE.validate_adjudication(
        adjudication, sources, registry_path=SOURCES_PATH
    ) == []
    assert adjudication["status"] == "technical_audit_complete"
    assert adjudication["audit_provenance"]["review_class"] == (
        "technical_evidence_review_not_legal_advice"
    )
    assert adjudication["default_policy"] == "deny_training_and_redistribution"
    assert adjudication["summary"]["candidate_sources"] == 23
    assert adjudication["summary"]["candidate_local_training_allowed_sources"] == 7
    assert adjudication["summary"]["candidate_redistribution_allowed_sources"] == 4
    assert adjudication["summary"]["base_local_training_allowed"] is True


def test_summary_ids_are_derived_exactly_and_base_is_not_counted_as_candidate() -> None:
    sources, adjudication = tracked()
    local = sorted(
        row["source_id"] for row in adjudication["sources"] if row["local_training"]["eligible"]
    )
    public = sorted(
        row["source_id"] for row in adjudication["sources"] if row["redistribution"]["eligible"]
    )
    assert adjudication["summary"]["candidate_local_training_allowed_source_ids"] == local
    assert adjudication["summary"]["candidate_redistribution_allowed_source_ids"] == public
    assert "nanochat_base" not in local

    mutated = copy.deepcopy(adjudication)
    mutated["summary"]["candidate_local_training_allowed_sources"] += 1
    mutated["summary"]["candidate_redistribution_allowed_source_ids"] = []
    errors = LICENSE.validate_adjudication(mutated, sources, registry_path=SOURCES_PATH)
    assert any("candidate_local_training_allowed_sources" in error for error in errors)
    assert any("candidate_redistribution_allowed_source_ids" in error for error in errors)


def test_no_nd_source_is_trainable_and_no_nc_source_is_public() -> None:
    _, adjudication = tracked()
    for row in adjudication["sources"]:
        declared = row["declared_license"] or ""
        if "-nd" in declared:
            assert row["local_training"]["eligible"] is False
        if "-nc" in declared:
            assert row["redistribution"]["eligible"] is False


def test_registry_revision_category_and_coverage_drift_fail_closed() -> None:
    sources, adjudication = tracked()
    mutated = copy.deepcopy(adjudication)
    mutated["sources"][0]["revision"] = "0" * 40
    mutated["sources"][1]["registry_training_eligibility"] = "eligible_open"
    mutated["sources"].pop()
    errors = LICENSE.validate_adjudication(mutated, sources, registry_path=SOURCES_PATH)
    assert any("revision: does not match" in error for error in errors)
    assert any("registry_training_eligibility" in error for error in errors)
    assert any("candidate coverage differs" in error for error in errors)


def test_runtime_loader_and_lookup_are_source_specific() -> None:
    decisions = LICENSE.load_adjudication(
        ADJUDICATION_PATH, source_registry_path=SOURCES_PATH
    )
    assert decisions["diavgeia"]["training_eligible"] is True
    assert decisions["diavgeia"]["redistribution_eligible"] is True
    assert decisions["elocus"]["training_eligible"] is True
    assert decisions["elocus"]["redistribution_eligible"] is False
    assert decisions["amna_press"]["training_eligible"] is False
    assert decisions["nanochat_base"]["redistribution_eligible"] is False
    assert LICENSE.decision_for("diavgeia", "eligible_open", decisions) is decisions["diavgeia"]
    with pytest.raises(ValueError, match="category drift"):
        LICENSE.decision_for("diavgeia", "policy_review", decisions)
    with pytest.raises(ValueError, match="no license adjudication"):
        LICENSE.decision_for("unknown", "eligible_open", decisions)


def test_json_schema_accepts_the_tracked_matrix() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(
        (HERE / "schemas" / "source_license_adjudication.schema.json").read_text(
            encoding="utf-8"
        )
    )
    _, adjudication = tracked()
    jsonschema.Draft202012Validator(schema).validate(adjudication)
