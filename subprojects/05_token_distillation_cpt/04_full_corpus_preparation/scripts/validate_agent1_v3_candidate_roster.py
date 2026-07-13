#!/usr/bin/env python3
"""Validate Agent1 v3's immutable logical-first source route declaration.

The candidate roster is bound byte-for-byte into the run contract.  This
validator adds the semantic checks that JSON shape alone cannot express:
every active source has one documented logical acquisition route, a rationale,
and an explicit finite set of secondary observed-extraction exceptions.

``extraction_routes`` is a declared source-level fallback, not a claim that
every document has that observed route.  A document may use one of the
documented exceptions, but its observed route remains secondary to the
logical source route used for review/error-model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


ROSTER_SCHEMA = "agent1_full_corpus_v3_candidate_roster_v1"
ROUTE_BASIS_SCHEMA = "agent1_v3_source_route_basis_v1"
VALIDATION_SCHEMA = "agent1_v3_candidate_roster_route_basis_validation_v1"
LOGICAL_SOURCE_PRIORITY = "logical_source_then_observed_extraction"
ALLOWED_ROUTES = frozenset({"html_web", "pdf_ocr", "mixed", "structured"})
LOGICAL_ERROR_MODE_ROUTES = frozenset({"html_web", "pdf_ocr", "structured"})
LOGICAL_ERROR_MODE_ORDER = ("html_web", "pdf_ocr", "structured")
EXPECTED_MIXED_LOGICAL_ERROR_MODES = {
    "opengov_deliberations_v2": ("html_web", "structured"),
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return value


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nonempty_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _candidate_ids(roster: Mapping[str, Any]) -> list[str]:
    candidates = roster.get("candidate_source_ids")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidate_source_ids must be a non-empty list")
    result = [
        _nonempty_string(source_id, label=f"candidate_source_ids[{index}]")
        for index, source_id in enumerate(candidates)
    ]
    if len(result) != len(set(result)):
        raise ValueError("candidate_source_ids must be unique")
    return result


def _route_map(
    roster: Mapping[str, Any], *, field: str, candidates: Sequence[str]
) -> dict[str, str]:
    value = roster.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    candidate_set = set(candidates)
    value_keys = set(value)
    missing = sorted(candidate_set - value_keys)
    extra = sorted(value_keys - candidate_set)
    if missing or extra:
        raise ValueError(
            f"{field} coverage mismatch; missing={missing}, extra={extra}"
        )
    result: dict[str, str] = {}
    for source_id in candidates:
        route = value[source_id]
        if not isinstance(route, str) or route not in ALLOWED_ROUTES:
            raise ValueError(f"{field}[{source_id!r}] has unsupported route: {route!r}")
        result[source_id] = route
    return result


def _logical_error_mode_map(
    roster: Mapping[str, Any], *, candidates: Sequence[str], source_routes: Mapping[str, str]
) -> dict[str, list[str]]:
    """Validate the explicit, source-logical primary diagnostic closure."""

    value = roster.get("logical_error_modes")
    if not isinstance(value, Mapping):
        raise ValueError("logical_error_modes must be an object")
    candidate_set = set(candidates)
    missing = sorted(candidate_set - set(value))
    extra = sorted(set(value) - candidate_set)
    if missing or extra:
        raise ValueError(
            f"logical_error_modes coverage mismatch; missing={missing}, extra={extra}"
        )
    result: dict[str, list[str]] = {}
    for source_id in candidates:
        modes = value[source_id]
        if not isinstance(modes, list) or not modes:
            raise ValueError(f"logical_error_modes[{source_id!r}] must be a non-empty list")
        if (
            any(not isinstance(mode, str) or mode not in LOGICAL_ERROR_MODE_ROUTES for mode in modes)
            or len(modes) != len(set(modes))
        ):
            raise ValueError(
                f"logical_error_modes[{source_id!r}] has unsupported or duplicate modes"
            )
        canonical = [mode for mode in LOGICAL_ERROR_MODE_ORDER if mode in modes]
        if modes != canonical:
            raise ValueError(f"logical_error_modes[{source_id!r}] must use canonical mode order")
        source_route = source_routes[source_id]
        if source_route != "mixed":
            if modes != [source_route]:
                raise ValueError(
                    f"{source_id}: non-mixed logical_error_modes must exactly equal source_route"
                )
        elif source_id in EXPECTED_MIXED_LOGICAL_ERROR_MODES:
            expected = list(EXPECTED_MIXED_LOGICAL_ERROR_MODES[source_id])
            if modes != expected:
                raise ValueError(
                    f"{source_id}: logical_error_modes differs from frozen logical acquisition modes"
                )
        elif len(modes) < 2:
            raise ValueError(
                f"{source_id}: mixed logical_error_modes must name at least two primary modes"
            )
        result[source_id] = list(modes)
    for source_id in EXPECTED_MIXED_LOGICAL_ERROR_MODES:
        if source_id in candidate_set and source_routes[source_id] != "mixed":
            raise ValueError(
                f"{source_id}: source_route must remain mixed for frozen logical_error_modes"
            )
    return result


def _source_registry_ids(source_registry: Mapping[str, Any]) -> list[str]:
    rows = source_registry.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("sources.json must contain a non-empty sources list")
    ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"sources.json sources[{index}] must be an object")
        ids.append(_nonempty_string(row.get("source_id"), label=f"sources[{index}].source_id"))
    if len(ids) != len(set(ids)):
        raise ValueError("sources.json contains duplicate source_id values")
    return ids


def validate_roster(
    roster: Mapping[str, Any], *, source_registry: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Return a deterministic route-basis validation report or raise.

    The report deliberately exposes the allowed observed routes per source so a
    per-document derivation can validate a representation-specific exception
    without forcing it to equal ``extraction_routes[source_id]``.
    """

    if roster.get("schema_version") != ROSTER_SCHEMA:
        raise ValueError(f"unsupported candidate roster schema: {roster.get('schema_version')!r}")
    candidates = _candidate_ids(roster)
    source_routes = _route_map(roster, field="source_routes", candidates=candidates)
    review_routes = _route_map(roster, field="review_routes", candidates=candidates)
    extraction_routes = _route_map(roster, field="extraction_routes", candidates=candidates)

    route_policy = roster.get("route_policy")
    if not isinstance(route_policy, Mapping):
        raise ValueError("route_policy must be an object")
    if route_policy.get("priority") != LOGICAL_SOURCE_PRIORITY:
        raise ValueError("route_policy.priority must preserve logical-source-first routing")

    route_basis = roster.get("route_basis")
    if not isinstance(route_basis, Mapping):
        raise ValueError("route_basis must be an object")
    if route_basis.get("schema_version") != ROUTE_BASIS_SCHEMA:
        raise ValueError("route_basis.schema_version is unsupported")
    if route_basis.get("priority") != LOGICAL_SOURCE_PRIORITY:
        raise ValueError("route_basis.priority must preserve logical-source-first routing")
    _nonempty_string(route_basis.get("semantics"), label="route_basis.semantics")
    source_basis = route_basis.get("sources")
    if not isinstance(source_basis, Mapping):
        raise ValueError("route_basis.sources must be an object")
    basis_keys = set(source_basis)
    candidate_set = set(candidates)
    missing = sorted(candidate_set - basis_keys)
    extra = sorted(basis_keys - candidate_set)
    if missing or extra:
        raise ValueError(
            f"route_basis.sources coverage mismatch; missing={missing}, extra={extra}"
        )

    report_sources: dict[str, dict[str, Any]] = {}
    for source_id in sorted(candidates):
        entry = source_basis[source_id]
        if not isinstance(entry, Mapping):
            raise ValueError(f"route_basis.sources[{source_id!r}] must be an object")
        logical_route = entry.get("logical_acquisition_type")
        if not isinstance(logical_route, str) or logical_route not in ALLOWED_ROUTES:
            raise ValueError(
                f"route_basis.sources[{source_id!r}].logical_acquisition_type is unsupported"
            )
        _nonempty_string(
            entry.get("rationale"), label=f"route_basis.sources[{source_id!r}].rationale"
        )
        if source_routes[source_id] != logical_route:
            raise ValueError(
                f"{source_id}: source_routes must equal logical_acquisition_type "
                f"({source_routes[source_id]!r} != {logical_route!r})"
            )
        if review_routes[source_id] != logical_route:
            raise ValueError(
                f"{source_id}: review_routes must equal logical_acquisition_type "
                f"({review_routes[source_id]!r} != {logical_route!r})"
            )

        exceptions = entry.get("expected_observed_extraction_exceptions")
        if not isinstance(exceptions, list):
            raise ValueError(
                f"route_basis.sources[{source_id!r}].expected_observed_extraction_exceptions "
                "must be a list"
            )
        exception_routes: set[str] = set()
        for index, exception in enumerate(exceptions):
            label = (
                f"route_basis.sources[{source_id!r}]"
                f".expected_observed_extraction_exceptions[{index}]"
            )
            if not isinstance(exception, Mapping):
                raise ValueError(f"{label} must be an object")
            observed_route = exception.get("observed_extraction_route")
            if not isinstance(observed_route, str) or observed_route not in ALLOWED_ROUTES:
                raise ValueError(f"{label}.observed_extraction_route is unsupported")
            if observed_route == logical_route:
                raise ValueError(
                    f"{label}.observed_extraction_route must be an exception to the logical route"
                )
            if observed_route in exception_routes:
                raise ValueError(f"{source_id}: duplicate documented observed route {observed_route!r}")
            exception_routes.add(observed_route)
            _nonempty_string(exception.get("rationale"), label=f"{label}.rationale")
            if exception.get("secondary_only") is not True:
                raise ValueError(f"{label}.secondary_only must be true")

        allowed_observed_routes = sorted({logical_route, *exception_routes})
        declared_extraction_route = extraction_routes[source_id]
        if declared_extraction_route not in allowed_observed_routes:
            raise ValueError(
                f"{source_id}: extraction_routes value {declared_extraction_route!r} is neither "
                "the logical route nor a documented secondary observed exception"
            )
        report_sources[source_id] = {
            "logical_acquisition_type": logical_route,
            "declared_extraction_route_fallback": declared_extraction_route,
            "allowed_observed_extraction_routes": allowed_observed_routes,
            "secondary_exception_routes": sorted(exception_routes),
        }

    logical_error_modes = _logical_error_mode_map(
        roster, candidates=candidates, source_routes=source_routes
    )
    for source_id in candidates:
        report_sources[source_id]["logical_error_modes"] = logical_error_modes[source_id]

    source_registry_coverage_verified = False
    if source_registry is not None:
        registry_ids = _source_registry_ids(source_registry)
        registry_set = set(registry_ids)
        missing = sorted(registry_set - candidate_set)
        extra = sorted(candidate_set - registry_set)
        if missing or extra:
            raise ValueError(
                "candidate roster/source registry coverage mismatch; "
                f"missing_from_roster={missing}, unknown_to_registry={extra}"
            )
        source_registry_coverage_verified = True

    return {
        "schema_version": VALIDATION_SCHEMA,
        "candidate_count": len(candidates),
        "candidate_source_ids": sorted(candidates),
        "logical_source_priority": LOGICAL_SOURCE_PRIORITY,
        "logical_error_modes": {
            source_id: logical_error_modes[source_id] for source_id in sorted(candidates)
        },
        "source_registry_coverage_verified": source_registry_coverage_verified,
        "sources": report_sources,
    }


def validate_observed_extraction_route(
    route_report: Mapping[str, Any], *, source_id: str, observed_extraction_route: str
) -> dict[str, str]:
    """Validate one representation-specific route against the frozen basis.

    This intentionally accepts a documented exception even when it differs
    from the source-level ``declared_extraction_route_fallback``.  It returns
    whether that route is the logical primary or a secondary-only exception,
    making accidental route-priority inversion visible to callers.
    """

    if route_report.get("schema_version") != VALIDATION_SCHEMA:
        raise ValueError("unsupported route-basis validation report")
    sources = route_report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("route-basis validation report has no source map")
    entry = sources.get(source_id)
    if not isinstance(entry, Mapping):
        raise ValueError(f"unknown source_id for observed extraction route: {source_id!r}")
    logical_route = entry.get("logical_acquisition_type")
    allowed_routes = entry.get("allowed_observed_extraction_routes")
    secondary_routes = entry.get("secondary_exception_routes")
    if (
        not isinstance(logical_route, str)
        or not isinstance(allowed_routes, list)
        or not all(isinstance(route, str) for route in allowed_routes)
        or not isinstance(secondary_routes, list)
        or not all(isinstance(route, str) for route in secondary_routes)
    ):
        raise ValueError(f"{source_id}: malformed route-basis validation entry")
    if observed_extraction_route not in allowed_routes:
        raise ValueError(
            f"{source_id}: observed extraction route {observed_extraction_route!r} is not "
            "the logical route or a documented secondary exception"
        )
    return {
        "source_id": source_id,
        "logical_acquisition_type": logical_route,
        "observed_extraction_route": observed_extraction_route,
        "observed_route_priority": (
            "logical_primary"
            if observed_extraction_route == logical_route
            else "secondary_exception_only"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-roster", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    args = parser.parse_args()

    roster = read_json(args.candidate_roster)
    source_registry = read_json(args.sources)
    report = validate_roster(roster, source_registry=source_registry)
    report["candidate_roster_sha256"] = sha256_file(args.candidate_roster)
    report["sources_sha256"] = sha256_file(args.sources)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
