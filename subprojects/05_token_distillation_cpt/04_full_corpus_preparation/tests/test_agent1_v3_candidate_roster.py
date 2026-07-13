from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parents[1]


def load_module():
    path = HERE / "scripts" / "validate_agent1_v3_candidate_roster.py"
    spec = importlib.util.spec_from_file_location("agent1_v3_candidate_roster_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ROSTER_VALIDATOR = load_module()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def production_roster() -> dict:
    return load_json(HERE / "configs" / "agent1_v3_candidate_roster.json")


def production_sources() -> dict:
    return load_json(HERE / "configs" / "sources.json")


def test_production_roster_has_complete_logical_first_route_basis() -> None:
    roster = production_roster()
    report = ROSTER_VALIDATOR.validate_roster(
        roster, source_registry=production_sources()
    )

    assert report["candidate_count"] == 26
    assert report["source_registry_coverage_verified"] is True
    assert report["logical_source_priority"] == "logical_source_then_observed_extraction"
    assert report["logical_error_modes"] == {
        source_id: roster["logical_error_modes"][source_id]
        for source_id in sorted(roster["candidate_source_ids"])
    }
    assert set(report["sources"]) == set(roster["candidate_source_ids"])
    for source_id, entry in report["sources"].items():
        assert entry["logical_acquisition_type"] == roster["source_routes"][source_id]
        assert entry["logical_error_modes"] == roster["logical_error_modes"][source_id]
        assert roster["review_routes"][source_id] == entry["logical_acquisition_type"]
        assert roster["extraction_routes"][source_id] in entry[
            "allowed_observed_extraction_routes"
        ]
        if source_id == "diavgeia":
            assert entry["secondary_exception_routes"] == []
        else:
            assert entry["secondary_exception_routes"]


def test_documented_observed_exception_is_allowed_but_secondary() -> None:
    report = ROSTER_VALIDATOR.validate_roster(
        production_roster(), source_registry=production_sources()
    )

    # Psepheda's default source-level fallback is PDF/OCR, but an individual
    # repository-page representation may be HTML-derived.  The exception must
    # be allowed without permitting it to override the logical PDF/OCR route.
    assert report["sources"]["psepheda"]["declared_extraction_route_fallback"] == "pdf_ocr"
    observed = ROSTER_VALIDATOR.validate_observed_extraction_route(
        report, source_id="psepheda", observed_extraction_route="html_web"
    )
    assert observed == {
        "source_id": "psepheda",
        "logical_acquisition_type": "pdf_ocr",
        "observed_extraction_route": "html_web",
        "observed_route_priority": "secondary_exception_only",
    }
    with pytest.raises(ValueError, match="documented secondary exception"):
        ROSTER_VALIDATOR.validate_observed_extraction_route(
            report, source_id="psepheda", observed_extraction_route="structured"
        )

    assert report["sources"]["diavgeia"]["allowed_observed_extraction_routes"] == [
        "pdf_ocr"
    ]
    with pytest.raises(ValueError, match="documented secondary exception"):
        ROSTER_VALIDATOR.validate_observed_extraction_route(
            report, source_id="diavgeia", observed_extraction_route="html_web"
        )

    assert report["sources"]["opengov_deliberations_v2"][
        "allowed_observed_extraction_routes"
    ] == ["html_web", "mixed", "structured"]
    with pytest.raises(ValueError, match="documented secondary exception"):
        ROSTER_VALIDATOR.validate_observed_extraction_route(
            report,
            source_id="opengov_deliberations_v2",
            observed_extraction_route="pdf_ocr",
        )


def test_validator_rejects_route_basis_coverage_or_priority_drift() -> None:
    roster = production_roster()
    missing_basis = copy.deepcopy(roster)
    del missing_basis["route_basis"]["sources"]["ert_press"]
    with pytest.raises(ValueError, match="route_basis.sources coverage mismatch"):
        ROSTER_VALIDATOR.validate_roster(
            missing_basis, source_registry=production_sources()
        )

    priority_drift = copy.deepcopy(roster)
    priority_drift["route_basis"]["priority"] = "observed_extraction_then_logical_source"
    with pytest.raises(ValueError, match="route_basis.priority"):
        ROSTER_VALIDATOR.validate_roster(
            priority_drift, source_registry=production_sources()
        )


def test_validator_rejects_logical_route_or_registry_coverage_drift() -> None:
    roster = production_roster()
    logical_drift = copy.deepcopy(roster)
    logical_drift["source_routes"]["amna_press"] = "pdf_ocr"
    with pytest.raises(ValueError, match="source_routes must equal logical_acquisition_type"):
        ROSTER_VALIDATOR.validate_roster(
            logical_drift, source_registry=production_sources()
        )

    undocumented_fallback = copy.deepcopy(roster)
    undocumented_fallback["extraction_routes"]["psepheda"] = "structured"
    with pytest.raises(ValueError, match="documented secondary observed exception"):
        ROSTER_VALIDATOR.validate_roster(
            undocumented_fallback, source_registry=production_sources()
        )

    opengov_pdf_drift = copy.deepcopy(roster)
    opengov_pdf_drift["logical_error_modes"]["opengov_deliberations_v2"] = [
        "html_web",
        "pdf_ocr",
        "structured",
    ]
    with pytest.raises(ValueError, match="frozen logical acquisition modes"):
        ROSTER_VALIDATOR.validate_roster(
            opengov_pdf_drift, source_registry=production_sources()
        )

    non_mixed_drift = copy.deepcopy(roster)
    non_mixed_drift["logical_error_modes"]["psepheda"] = [
        "html_web",
        "pdf_ocr",
    ]
    with pytest.raises(ValueError, match="non-mixed logical_error_modes"):
        ROSTER_VALIDATOR.validate_roster(
            non_mixed_drift, source_registry=production_sources()
        )

    incomplete_registry = copy.deepcopy(production_sources())
    incomplete_registry["sources"] = [
        source
        for source in incomplete_registry["sources"]
        if source["source_id"] != "ert_press"
    ]
    with pytest.raises(ValueError, match="coverage mismatch"):
        ROSTER_VALIDATOR.validate_roster(
            roster, source_registry=incomplete_registry
        )
