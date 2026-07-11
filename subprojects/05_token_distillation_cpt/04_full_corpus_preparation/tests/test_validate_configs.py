from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


VALIDATE = load_module("phase04_validate_backlog", HERE / "scripts" / "validate_configs.py")


def tracked_configs():
    sources = VALIDATE.load_json(HERE / "configs" / "sources.json")
    backlog = VALIDATE.load_json(HERE / "configs" / "source_backlog.json")
    return sources, backlog


def test_tracked_source_backlog_is_non_acquiring_and_valid() -> None:
    sources, backlog = tracked_configs()
    assert VALIDATE.validate_backlog(backlog, sources) == []
    assert backlog["entries"]
    assert all(entry["acquisition_eligible"] is False for entry in backlog["entries"])


def test_backlog_rejects_acquisition_enablement() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["acquisition_eligible"] = True
    assert any(
        "acquisition_eligible must remain false" in error
        for error in VALIDATE.validate_backlog(mutated, sources)
    )


def test_backlog_rejects_source_registry_collision() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["repo_id"] = sources["sources"][0]["repo_id"]
    assert any(
        "repo is already present in sources.json" in error
        for error in VALIDATE.validate_backlog(mutated, sources)
    )


def test_backlog_rejects_unpinned_revision_and_untyped_metrics() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["revision"] = "main"
    mutated["entries"][0]["known_metrics"]["repository_rows"] = "1016"
    errors = VALIDATE.validate_backlog(mutated, sources)
    assert any("revision must be a lowercase 40-hex" in error for error in errors)
    assert any("known_metrics.repository_rows" in error for error in errors)


def test_backlog_repository_bytes_cover_every_candidate_artifact() -> None:
    sources, backlog = tracked_configs()
    mutated = copy.deepcopy(backlog)
    mutated["entries"][0]["known_metrics"]["repository_bytes"] -= 1
    errors = VALIDATE.validate_backlog(mutated, sources)
    assert any("must equal candidate file bytes" in error for error in errors)
