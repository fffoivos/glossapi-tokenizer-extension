#!/usr/bin/env python3
"""Review-only primitives for Agent 1's isolated full-corpus v3 lane.

The module deliberately has no dependency on the v2 source-review scripts.
It owns the small, deterministic boundary between the full-scan quality rows
and the compact Codex-review artifacts:

* high-precision review-copy masking which preserves Unicode-code-point
  positions and line endings;
* candidate-roster / review-route closure;
* 60/20/20 (or 100/50/50) disjoint deterministic sampling;
* immutable primary/secondary request identities and strict 1--5 responses;
* a conservative adjudication manifest for low-confidence or materially
  divergent reviews.

It intentionally does *not* read corpus text at scale, invoke Codex, make an
admission decision, or mutate canonical corpus records.  Bulk callers should
run the sampling side on a Clariden CPU node and use the returned selected IDs
to materialize compact review copies in their own attempt directory.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ROSTER_SCHEMA = "agent1_full_corpus_v3_candidate_roster_v1"
POLICY_SCHEMA = "agent1_full_corpus_v3_policy_v1"
ROUTE_VALIDATION_SCHEMA = "agent1_v3_candidate_roster_route_validation_v1"
ROUTE_POLICY_PRIORITY = "logical_source_then_observed_extraction"
SAMPLE_MANIFEST_SCHEMA = "agent1_v3_review_sample_manifest_v1"
REQUEST_SCHEMA = "agent1_v3_review_request_v1"
RESPONSE_SCHEMA = "agent1_v3_review_response_v1"
ADJUDICATION_MANIFEST_SCHEMA = "agent1_v3_review_adjudication_manifest_v1"

REQUIRED_REVIEW_MODEL = "gpt-5.6-luna"
ALLOWED_ROUTES = frozenset({"html_web", "pdf_ocr", "mixed", "structured"})
OBSERVED_EXTRACTION_ROUTE_BASES = frozenset(
    {
        "explicit_row_route",
        "row_representation_metadata",
        "declared_extraction_route_fallback",
        "unavailable",
        # Quality evidence created from legacy canonical shards may not carry
        # the three native fields.  Phase 2 writes this explicit fallback
        # rather than silently pretending it observed a per-document route.
        "legacy_canonical_without_observed_route",
    }
)
MAX_OBSERVED_SECONDARY_RISK_BONUS = 2.0
OBSERVED_SECONDARY_ROUTE_WEIGHT = 0.5
STRATA = ("random", "risk", "cluster")
DEFAULT_QUOTAS = {"random": 60, "risk": 20, "cluster": 20}
LARGE_QUOTAS = {"random": 100, "risk": 50, "cluster": 50}
MINIMUM_ELIGIBLE_DOCUMENTS = 100
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

ISSUE_CODES = frozenset(
    {
        "html_or_markup",
        "navigation_or_boilerplate",
        "ocr_corruption",
        "mojibake",
        "fragmentation",
        "broken_words_or_hyphenation",
        "page_furniture",
        "tables_or_formulas",
        "template_replay",
        "structured_incomplete",
        "pii",
        "non_greek_drift",
        "placeholder_or_empty",
        "copyright_or_license",
        "other",
    }
)
RECOMMENDATIONS = frozenset(
    {"include", "include_after_cleaning", "low_weight", "exclude", "uncertain"}
)


# These patterns are intentionally high precision.  In particular, a bare
# nine- or eleven-digit number is not treated as a Greek identifier: label and
# checksum/length evidence are required.  Review masking is not the later
# corpus anonymization stage.
EMAIL_RE = re.compile(
    r"(?<![\w.+-])[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+"
    r"[A-ZΑ-Ω]{2,63}(?![\w.-])",
    re.IGNORECASE,
)
IPV4_CANDIDATE_RE = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
# Empty hextets are permitted so compressed values such as ``fe80::1`` and
# ``::1`` reach the stdlib validator; the latter is what supplies precision.
IPV6_CANDIDATE_RE = re.compile(
    r"(?<![\w:])(?:[0-9A-F]{0,4}:){2,7}[0-9A-F:]{0,4}(?![\w:])", re.IGNORECASE
)
IBAN_RE = re.compile(
    r"(?<![A-Z0-9])GR[ -]?\d{2}(?:[ -]?[A-Z0-9]){23}(?![A-Z0-9])", re.IGNORECASE
)
AFM_RE = re.compile(
    r"(?<![\w])(?:Α\.?\s*Φ\.?\s*Μ\.?|ΑΦΜ|AFM)\s*[:#-]?\s*"
    r"(?P<value>\d(?:[ .-]?\d){8})(?!\d)",
    re.IGNORECASE,
)
AMKA_RE = re.compile(
    r"(?<![\w])(?:Α\.?\s*Μ\.?\s*Κ\.?\s*Α\.?|ΑΜΚΑ|AMKA)\s*[:#-]?\s*"
    r"(?P<value>\d(?:[ .-]?\d){10})(?!\d)",
    re.IGNORECASE,
)
IDENTITY_OR_PASSPORT_RE = re.compile(
    r"(?<![\w])(?:Α\.?\s*Δ\.?\s*Τ\.?|ΑΔΤ|"
    r"Δελτίο\s+(?:Αστυνομικής\s+)?Ταυτότητας|"
    r"Αριθ(?:μός|μ\.)\s*Ταυτότητας|"
    r"identity\s*(?:card|number|no\.?)?|"
    r"passport\s*(?:number|no\.?)?|διαβατηρ(?:ίου|ιο))"
    r"\s*[:#№-]?\s*"
    r"(?P<value>(?:[A-ZΑ-Ω]{1,3}\s*[-.]?\s*\d{5,9})|(?:[A-Z0-9Α-Ω]{5,12}))(?![\w-])",
    re.IGNORECASE,
)
PHONE_RE = re.compile(
    r"(?<![\w+])(?:\+?30[ .()\-]*)?"
    r"(?:2(?:[ .()\-]*\d){9}|69(?:[ .()\-]*\d){8})(?!\w)"
)


def canonical_json(value: Any) -> str:
    """Return the canonical JSON representation used for deterministic IDs."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(seed: str, namespace: str, stable_uid: str) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{seed}\0{namespace}\0{stable_uid}".encode("utf-8")).digest(),
        "big",
    )


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_int_1_to_5(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 5:
        raise ValueError(f"{name} must be an integer in [1, 5]")
    return value


def _digits(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def _valid_greek_afm(value: str) -> bool:
    digits = _digits(value)
    if len(digits) != 9:
        return False
    check = sum(int(digit) * (2 ** (8 - index)) for index, digit in enumerate(digits[:8]))
    return (check % 11) % 10 == int(digits[8])


def _valid_iban(value: str) -> bool:
    compact = re.sub(r"[ -]", "", value).upper()
    if len(compact) != 27 or not compact.startswith("GR") or not compact.isalnum():
        return False
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
    return int(numeric) % 97 == 1


@dataclass(frozen=True)
class RedactionSpan:
    """A review-only direct-identifier span in Unicode code-point offsets."""

    kind: str
    char_start: int
    char_end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "char_start": self.char_start,
            "char_end": self.char_end,
        }


def _valid_ipv4(match: re.Match[str]) -> bool:
    try:
        return str(ipaddress.ip_address(match.group(0))) == match.group(0)
    except ValueError:
        return False


def _valid_ipv6(match: re.Match[str]) -> bool:
    candidate = match.group(0)
    # A colon-delimited time or prose fragment can match the loose candidate
    # expression; ipaddress performs the actual syntax validation.
    try:
        return isinstance(ipaddress.ip_address(candidate), ipaddress.IPv6Address)
    except ValueError:
        return False


def _identifier_spans(text: str) -> list[RedactionSpan]:
    """Find verified direct identifiers without retaining identifier values."""

    spans: list[RedactionSpan] = []

    def append_matches(
        kind: str,
        pattern: re.Pattern[str],
        *,
        value_group: str | None = None,
        validator: Any | None = None,
    ) -> None:
        for match in pattern.finditer(text):
            value = match.group(value_group) if value_group else match.group(0)
            if validator is not None and not validator(value if value_group else match):
                continue
            start, end = match.span(value_group) if value_group else match.span()
            if start == end or "\n" in text[start:end] or "\r" in text[start:end]:
                continue
            spans.append(RedactionSpan(kind=kind, char_start=start, char_end=end))

    append_matches("email", EMAIL_RE)
    append_matches("ip", IPV4_CANDIDATE_RE, validator=_valid_ipv4)
    append_matches("ip", IPV6_CANDIDATE_RE, validator=_valid_ipv6)
    append_matches("iban", IBAN_RE, validator=lambda match: _valid_iban(match.group(0)))
    append_matches("afm", AFM_RE, value_group="value", validator=_valid_greek_afm)
    append_matches("amka", AMKA_RE, value_group="value", validator=lambda value: len(_digits(value)) == 11)
    append_matches("identity_or_passport", IDENTITY_OR_PASSPORT_RE, value_group="value")
    append_matches("phone", PHONE_RE)

    # Resolve rare overlaps deterministically.  The longer match wins, then
    # the category order below.  This avoids producing a visible fragment when
    # a value could superficially look like more than one identifier.
    priority = {
        "iban": 0,
        "email": 1,
        "identity_or_passport": 2,
        "afm": 3,
        "amka": 4,
        "phone": 5,
        "ip": 6,
    }
    accepted: list[RedactionSpan] = []
    for span in sorted(
        spans,
        key=lambda item: (item.char_start, -(item.char_end - item.char_start), priority[item.kind]),
    ):
        if any(
            span.char_start < existing.char_end and existing.char_start < span.char_end
            for existing in accepted
        ):
            continue
        accepted.append(span)
    return sorted(accepted, key=lambda item: (item.char_start, item.char_end, item.kind))


def redact_review_copy(text: str, *, mask_character: str = "█") -> tuple[str, dict[str, Any]]:
    """Mask high-confidence direct identifiers without shifting text offsets.

    Offsets are Unicode code-point offsets (the indexing unit used by Python
    strings and the corpus span contracts).  Every replacement is exactly one
    code point per original code point, and the matcher rejects multiline
    matches, so ``len()``, every newline position, and all unaffected spans are
    unchanged.  The report includes only type/count/offset evidence, never a
    copied identifier value.
    """

    if not isinstance(text, str):
        raise TypeError("review-copy input text must be a string")
    if not isinstance(mask_character, str) or len(mask_character) != 1 or mask_character in "\r\n":
        raise ValueError("mask_character must be exactly one non-newline Unicode code point")
    spans = _identifier_spans(text)
    masked = list(text)
    for span in spans:
        for index in range(span.char_start, span.char_end):
            masked[index] = mask_character
    review_copy = "".join(masked)
    if len(review_copy) != len(text) or [i for i, char in enumerate(text) if char in "\r\n"] != [
        i for i, char in enumerate(review_copy) if char in "\r\n"
    ]:
        raise AssertionError("review-copy masking changed positional structure")
    counts = dict(sorted(Counter(span.kind for span in spans).items()))
    report = {
        "schema_version": "agent1_v3_review_copy_redactions_v1",
        "offset_unit": "unicode_codepoints",
        "high_precision_identifier_patterns_masked": True,
        "original_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "review_copy_sha256": hashlib.sha256(review_copy.encode("utf-8")).hexdigest(),
        "original_characters": len(text),
        "review_copy_characters": len(review_copy),
        "redaction_counts": counts,
        "redaction_spans": [span.as_dict() for span in spans],
    }
    return review_copy, report


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Write a compact artifact atomically; stage contracts own no-overwrite policy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    """Read JSONL/JSON rows, loading PyArrow only for an explicit Parquet input."""

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on runner image
            raise RuntimeError("reading Parquet quality rows requires pyarrow") from exc
        for row in pq.read_table(path).to_pylist():
            if not isinstance(row, dict):
                raise ValueError(f"{path}: non-object Parquet row")
            yield row
        return
    if suffix == ".json":
        payload = load_json(path)
        if isinstance(payload, dict):
            payload = payload.get("rows")
        if not isinstance(payload, list):
            raise ValueError(f"{path}: JSON metric input must be a list or an object with rows")
        for index, row in enumerate(payload, 1):
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index}: metric row must be an object")
            yield row
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: metric row must be an object")
            yield row


def _roster_hash(roster: Mapping[str, Any]) -> str:
    return sha256_json(dict(roster))


def _validated_route_map(
    roster: Mapping[str, Any],
    *,
    field: str,
    candidates: Sequence[str],
    fallback: Mapping[str, str] | None = None,
) -> dict[str, str]:
    value = roster.get(field)
    if value is None:
        if fallback is None:
            raise ValueError(f"{field} must be an object")
        return dict(fallback)
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    candidate_set = set(candidates)
    route_set = set(value)
    missing = sorted(candidate_set - route_set)
    extra = sorted(route_set - candidate_set)
    if missing or extra:
        raise ValueError(f"candidate roster {field} coverage mismatch; missing={missing}, extra={extra}")
    invalid = sorted(
        source
        for source, route in value.items()
        if not isinstance(route, str) or route not in ALLOWED_ROUTES
    )
    if invalid:
        raise ValueError(f"candidate roster has unsupported/missing {field}: {invalid}")
    return {source: str(value[source]) for source in candidates}


def _validated_observed_route_allowances(
    roster: Mapping[str, Any],
    *,
    candidates: Sequence[str],
    source_routes: Mapping[str, str],
    extraction_routes: Mapping[str, str],
) -> dict[str, list[str]]:
    """Validate logical-primary observed-route exceptions when declared.

    The route-basis block is the immutable documentation for cases such as a
    PDF-derived work that visibly retains HTML template residue.  No raw row
    can turn such an exception into the source's logical error model.
    """

    route_basis = roster.get("route_basis")
    if route_basis is None:
        # Kept only for legacy test/evidence fixtures.  The declared fallback
        # is finite, never an implicit allowance for every route.
        return {
            source: sorted({source_routes[source], extraction_routes[source]})
            for source in candidates
        }
    if (
        not isinstance(route_basis, Mapping)
        or route_basis.get("schema_version") != "agent1_v3_source_route_basis_v1"
        or route_basis.get("priority") != ROUTE_POLICY_PRIORITY
        or not isinstance(route_basis.get("sources"), Mapping)
    ):
        raise ValueError("candidate roster route_basis must document logical-primary observed exceptions")
    sources = route_basis["sources"]
    if set(sources) != set(candidates):
        raise ValueError("candidate roster route_basis source coverage mismatch")
    result: dict[str, list[str]] = {}
    for source in candidates:
        entry = sources[source]
        if not isinstance(entry, Mapping):
            raise ValueError(f"candidate roster route_basis[{source!r}] must be an object")
        logical = entry.get("logical_acquisition_type")
        if logical != source_routes[source]:
            raise ValueError(f"candidate roster route_basis logical route drift for {source!r}")
        exceptions = entry.get("expected_observed_extraction_exceptions")
        if not isinstance(exceptions, list) or not exceptions:
            raise ValueError(f"candidate roster route_basis exceptions missing for {source!r}")
        exception_routes: set[str] = set()
        for index, exception in enumerate(exceptions):
            if not isinstance(exception, Mapping):
                raise ValueError(f"candidate roster route_basis exception {source!r}[{index}] must be an object")
            route = exception.get("observed_extraction_route")
            if not isinstance(route, str) or route not in ALLOWED_ROUTES or route == logical:
                raise ValueError(f"candidate roster route_basis exception {source!r}[{index}] is unsupported")
            if exception.get("secondary_only") is not True:
                raise ValueError(f"candidate roster route_basis exception {source!r}[{index}] must be secondary_only")
            if not isinstance(exception.get("rationale"), str) or not exception["rationale"].strip():
                raise ValueError(f"candidate roster route_basis exception {source!r}[{index}] lacks rationale")
            if route in exception_routes:
                raise ValueError(f"candidate roster route_basis repeats exception {route!r} for {source!r}")
            exception_routes.add(route)
        allowed = {logical, *exception_routes}
        if extraction_routes[source] not in allowed:
            raise ValueError(
                f"candidate roster extraction fallback is not logical/documented exception for {source!r}"
            )
        result[source] = sorted(allowed)
    return result


def validate_candidate_roster_routes(roster: Mapping[str, Any]) -> dict[str, Any]:
    """Validate logical-first source, review, and extraction provenance.

    ``source_routes`` is the primary error model.  The independently bound
    extraction route records the representation that surfaced in a corpus;
    it can make a secondary defect more likely but never overrides the source
    route selected for review sampling.
    """

    if roster.get("schema_version") != ROSTER_SCHEMA:
        raise ValueError(f"unsupported candidate roster schema: {roster.get('schema_version')!r}")
    candidates = roster.get("candidate_source_ids")
    if (
        not isinstance(candidates, list)
        or not candidates
        or any(not isinstance(value, str) or not value.strip() for value in candidates)
        or len(candidates) != len(set(candidates))
    ):
        raise ValueError("candidate_source_ids must be a non-empty list of unique strings")
    review_routes = _validated_route_map(
        roster, field="review_routes", candidates=candidates
    )
    source_routes = _validated_route_map(
        roster,
        field="source_routes",
        candidates=candidates,
        fallback=review_routes,
    )
    extraction_routes = _validated_route_map(
        roster,
        field="extraction_routes",
        candidates=candidates,
        fallback=review_routes,
    )
    allowed_observed_routes = _validated_observed_route_allowances(
        roster,
        candidates=candidates,
        source_routes=source_routes,
        extraction_routes=extraction_routes,
    )
    route_policy = roster.get("route_policy")
    if route_policy is not None and (
        not isinstance(route_policy, Mapping)
        or route_policy.get("priority") != ROUTE_POLICY_PRIORITY
    ):
        raise ValueError("candidate roster route_policy must prioritize logical source provenance")
    excluded = roster.get("inventory_only_exclusions", [])
    if not isinstance(excluded, list):
        raise ValueError("inventory_only_exclusions must be a list when present")
    for index, entry in enumerate(excluded):
        if not isinstance(entry, Mapping) or not isinstance(entry.get("reason"), str) or not entry["reason"]:
            raise ValueError(f"inventory_only_exclusions[{index}] needs evidence reason")
    return {
        "schema_version": ROUTE_VALIDATION_SCHEMA,
        "roster_sha256": _roster_hash(roster),
        "candidate_count": len(candidates),
        "candidate_source_ids": sorted(candidates),
        "logical_source_priority": ROUTE_POLICY_PRIORITY,
        "source_routes": {source: source_routes[source] for source in sorted(candidates)},
        "review_routes": {source: review_routes[source] for source in sorted(candidates)},
        "extraction_routes": {source: extraction_routes[source] for source in sorted(candidates)},
        "allowed_observed_extraction_routes": {
            source: allowed_observed_routes[source] for source in sorted(candidates)
        },
        "inventory_only_exclusion_count": len(excluded),
    }


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _first_string(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _source_id_from_row(row: Mapping[str, Any], routes: Mapping[str, str]) -> str:
    # source_id is authoritative; source_dataset remains preserved independently
    # because several representations can legitimately share an upstream name.
    for field in ("source_id", "source_dataset"):
        value = row.get(field)
        if isinstance(value, str) and value in routes:
            return value
    provided = {field: row.get(field) for field in ("source_id", "source_dataset")}
    raise ValueError(f"metric row is not bound to a candidate roster source: {provided!r}")


def _cluster_id_from_row(row: Mapping[str, Any], stable_uid: str) -> str:
    for field in (
        "review_cluster_id",
        "minhash_cluster_id",
        "template_cluster_id",
        "structural_template_id",
        "near_duplicate_cluster_id",
        "cluster_id",
    ):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return f"{field}:{value}"
    # An unclustered document is distinct evidence, not a shared empty cluster.
    return f"singleton:{stable_uid}"


def risk_score_from_metrics(
    row: Mapping[str, Any],
    *,
    source_route: str | None = None,
    extraction_route: str | None = None,
    observed_extraction_route: str | None = None,
) -> float:
    """Return a deterministic, source-route-aware quality-risk score.

    A quality runtime may provide ``review_risk_score``/``risk_score`` as a
    base ordering signal.  It never bypasses the route-aware diagnostics:
    logical source provenance must still determine which documented errors get
    primary weight.  This is deliberately an ordering aid, never an admission
    score.
    """

    explicit_score: float | None = None
    for field in ("review_risk_score", "risk_score", "quality_risk_score"):
        value = _finite_number(row.get(field))
        if value is not None:
            explicit_score = value
            break

    def number(*names: str) -> float:
        for name in names:
            value = _finite_number(row.get(name))
            if value is not None:
                return value
        return 0.0

    def truthy(*names: str) -> float:
        return float(any(row.get(name) is True for name in names))

    route = source_route or _first_string(row, ("source_route", "review_route")) or "mixed"
    if route not in ALLOWED_ROUTES:
        raise ValueError(f"unsupported source route for risk selection: {route!r}")
    # ``extraction_route`` is retained as a backwards-compatible source-level
    # declared fallback argument.  Native v3 rows carry the observed field
    # separately, which must win whenever it is available.
    observed_extraction = (
        observed_extraction_route
        or _first_string(row, ("observed_extraction_route",))
        or extraction_route
        or _first_string(row, ("extraction_route",))
    )
    if observed_extraction is not None and observed_extraction not in ALLOWED_ROUTES:
        raise ValueError(f"unsupported extraction route for risk selection: {observed_extraction!r}")

    html_risk = min(
        4.0,
        number("raw_html_tags_per_1000_chars", "html_tags_per_1000_chars", "markup_rate")
        + number("entity_rate", "html_entity_rate")
        + number("script_style_rate", "navigation_boilerplate_rate"),
    )
    ocr_risk = (
        min(
            4.0,
            number("raw_mojibake_per_1000_chars", "mojibake_per_1000_chars")
            + number("raw_replacement_per_1000_chars", "replacement_per_1000_chars"),
        )
        + min(3.0, number("raw_control_per_1000_chars", "control_per_1000_chars"))
        + min(3.0, 4.0 * number("raw_repeated_line_fraction", "repeated_line_fraction"))
        + min(3.0, 4.0 * number("raw_one_token_line_fraction", "one_token_line_fraction"))
        + min(3.0, 3.0 * number("cleaner_removed_character_fraction"))
        + min(3.0, number("cleaner_badness_score", "rust_noise_badness_score"))
        + 1.0 * truthy("toc_header_detected", "bibliography_header_detected")
    )
    structured_risk = min(
        4.0,
        number("structured_missing_field_count", "missing_required_field_count", "field_flattening_failures"),
    ) + min(
        3.0,
        4.0
        * number(
            "repeated_parent_context_fraction",
            "parent_context_template_fraction",
            "structured_template_replay_fraction",
        ),
    )
    completeness = _finite_number(row.get("schema_content_completeness"))
    if completeness is not None:
        structured_risk += min(3.0, 3.0 * max(0.0, 1.0 - completeness))

    # Logical source provenance is always weighted most heavily.  All
    # non-primary modes retain a small baseline because strange republishing
    # paths exist.  Observed extraction evidence may add a *bounded* bonus,
    # never enough to overwrite the source-route error model.
    weights = {"html_web": 0.25, "pdf_ocr": 0.25, "structured": 0.25}
    if route == "mixed":
        weights.update({"html_web": 1.0, "pdf_ocr": 1.0, "structured": 0.5})
    else:
        weights[route] = 1.0
    route_risks = {
        "html_web": html_risk,
        "pdf_ocr": ocr_risk,
        "structured": structured_risk,
    }
    score = (
        weights["html_web"] * html_risk
        + weights["pdf_ocr"] * ocr_risk
        + weights["structured"] * structured_risk
    )
    if route != "mixed" and observed_extraction is not None and observed_extraction != route:
        secondary_routes = (
            ("html_web", "pdf_ocr")
            if observed_extraction == "mixed"
            else (observed_extraction,)
        )
        observed_bonus = sum(
            max(0.0, OBSERVED_SECONDARY_ROUTE_WEIGHT - weights[secondary_route])
            * route_risks[secondary_route]
            for secondary_route in secondary_routes
            if secondary_route in route_risks and secondary_route != route
        )
        score += min(MAX_OBSERVED_SECONDARY_RISK_BONUS, observed_bonus)
    score += 1.0 * truthy("digital_governance_footer_detected")
    score += 2.0 * truthy("private_data_true", "personnel_cue_detected")
    score += min(2.0, number("direct_identifier_match_count"))
    if number("original_characters", "characters") < 200:
        score += 1.0
    return round((explicit_score or 0.0) + score, 9)


@dataclass(frozen=True)
class MetricRow:
    source_id: str
    source_dataset: str
    source_revision: str
    stable_uid: str
    source_route: str
    review_route: str
    extraction_route: str
    observed_extraction_route: str
    observed_extraction_route_basis: str
    observed_extraction_route_evidence: str
    observed_extraction_route_priority: str
    cluster_id: str
    risk_score: float

    def identity(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "source_dataset": self.source_dataset,
            "source_revision": self.source_revision,
            "stable_uid": self.stable_uid,
        }


def _observed_extraction_from_metric(
    row: Mapping[str, Any],
    *,
    source_id: str,
    logical_source_route: str,
    declared_extraction_route: str,
    allowed_observed_routes: Sequence[str],
    row_number: int,
) -> tuple[str, str, str, str]:
    """Return per-document observed route or the immutable declared fallback."""

    provided = row.get("observed_extraction_route")
    if provided is None:
        if declared_extraction_route not in allowed_observed_routes:
            raise ValueError(
                f"metric row {row_number}: declared extraction fallback is not documented for {source_id}"
            )
        return (
            declared_extraction_route,
            "declared_extraction_route_fallback",
            "roster:extraction_route",
            (
                "logical_primary"
                if declared_extraction_route == logical_source_route
                else "secondary_exception_only"
            ),
        )
    if not isinstance(provided, str) or provided not in ALLOWED_ROUTES:
        raise ValueError(
            f"metric row {row_number}: observed_extraction_route must be a supported route"
        )
    if provided not in allowed_observed_routes:
        raise ValueError(
            f"metric row {row_number}: observed_extraction_route {provided!r} is not "
            f"the logical route or a documented secondary exception for {source_id}"
        )
    basis = row.get("observed_extraction_route_basis")
    evidence = row.get("observed_extraction_route_evidence")
    if not isinstance(basis, str) or basis not in OBSERVED_EXTRACTION_ROUTE_BASES:
        raise ValueError(
            f"metric row {row_number}: observed_extraction_route_basis is unsupported"
        )
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 200:
        raise ValueError(
            f"metric row {row_number}: observed_extraction_route_evidence must be a non-empty <=200-char code"
        )
    if basis == "unavailable":
        raise ValueError(
            f"metric row {row_number}: unavailable observed extraction cannot carry a candidate route"
        )
    expected_priority = (
        "logical_primary"
        if provided == logical_source_route
        else "secondary_exception_only"
    )
    supplied_priority = row.get("observed_extraction_route_priority")
    if supplied_priority is not None and supplied_priority != expected_priority:
        raise ValueError(
            f"metric row {row_number}: observed_extraction_route_priority reverses logical-source priority"
        )
    return (
        provided,
        basis,
        evidence,
        expected_priority,
    )


def normalize_metric_rows(rows: Iterable[Mapping[str, Any]], roster: Mapping[str, Any]) -> list[MetricRow]:
    route_report = validate_candidate_roster_routes(roster)
    source_routes = route_report["source_routes"]
    review_routes = route_report["review_routes"]
    extraction_routes = route_report["extraction_routes"]
    allowed_observed_routes = route_report["allowed_observed_extraction_routes"]
    normalized: list[MetricRow] = []
    seen_uids: set[str] = set()
    for row_number, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ValueError(f"metric row {row_number} must be an object")
        source_id = _source_id_from_row(row, source_routes)
        source_dataset = _first_string(row, ("source_dataset", "source_id"))
        source_revision = _first_string(row, ("source_revision", "revision"))
        stable_uid = _first_string(row, ("stable_uid", "sample_id", "document_id"))
        if source_dataset is None or source_revision is None or stable_uid is None:
            raise ValueError(
                f"metric row {row_number} needs source_dataset/source_revision/stable_uid (or document_id)"
            )
        # Stable UIDs are a primary evidence key.  A non-hash arbitrary name is
        # unsafe because request/response manifests must be portable and exact.
        _require_sha256(f"metric row {row_number}.stable_uid", stable_uid)
        if stable_uid in seen_uids:
            raise ValueError(f"metric row {row_number}: duplicate stable_uid {stable_uid}")
        seen_uids.add(stable_uid)
        for field, declared, expected in (
            ("source_route", row.get("source_route"), source_routes[source_id]),
            ("review_route", row.get("review_route"), review_routes[source_id]),
            ("extraction_route", row.get("extraction_route"), extraction_routes[source_id]),
        ):
            if declared is not None and declared != expected:
                raise ValueError(
                    f"metric row {row_number}: {field} drift for {source_id}: "
                    f"{declared!r} != {expected!r}"
                )
        (
            observed_extraction_route,
            observed_extraction_route_basis,
            observed_extraction_route_evidence,
            observed_extraction_route_priority,
        ) = _observed_extraction_from_metric(
            row,
            source_id=source_id,
            logical_source_route=source_routes[source_id],
            declared_extraction_route=extraction_routes[source_id],
            allowed_observed_routes=allowed_observed_routes[source_id],
            row_number=row_number,
        )
        normalized.append(
            MetricRow(
                source_id=source_id,
                source_dataset=source_dataset,
                source_revision=source_revision,
                stable_uid=stable_uid,
                source_route=source_routes[source_id],
                review_route=review_routes[source_id],
                extraction_route=extraction_routes[source_id],
                observed_extraction_route=observed_extraction_route,
                observed_extraction_route_basis=observed_extraction_route_basis,
                observed_extraction_route_evidence=observed_extraction_route_evidence,
                observed_extraction_route_priority=observed_extraction_route_priority,
                cluster_id=_cluster_id_from_row(row, stable_uid),
                risk_score=risk_score_from_metrics(
                    row,
                    source_route=source_routes[source_id],
                    extraction_route=extraction_routes[source_id],
                    observed_extraction_route=observed_extraction_route,
                ),
            )
        )
    if not normalized:
        raise ValueError("full-scan metric input contains no candidate documents")
    return sorted(normalized, key=lambda item: (item.source_id, item.stable_uid))


def _scaled_quotas(base: Mapping[str, int], target: int) -> dict[str, int]:
    base_total = sum(base.values())
    if target < 0 or target > base_total:
        raise ValueError("sample target must be in the range [0, configured quota total]")
    if target == base_total:
        return {stratum: int(base[stratum]) for stratum in STRATA}
    raw = {stratum: target * int(base[stratum]) / base_total for stratum in STRATA}
    result = {stratum: int(math.floor(raw[stratum])) for stratum in STRATA}
    remaining = target - sum(result.values())
    # Stable stratum order breaks equal fractional remainders.
    for stratum in sorted(STRATA, key=lambda name: (-(raw[name] - result[name]), STRATA.index(name)))[:remaining]:
        result[stratum] += 1
    return result


def _source_target(
    source_id: str, eligible_count: int, large_or_heterogeneous_sources: set[str]
) -> tuple[dict[str, int], int, bool]:
    base = LARGE_QUOTAS if source_id in large_or_heterogeneous_sources else DEFAULT_QUOTAS
    configured_target = sum(base.values())
    target = min(eligible_count, configured_target)
    return _scaled_quotas(base, target), configured_target, eligible_count < configured_target


def _cluster_selection(
    rows: Sequence[MetricRow], *, count: int, selected: set[str], seed: str
) -> list[MetricRow]:
    if count == 0:
        return []
    groups: dict[str, list[MetricRow]] = defaultdict(list)
    for row in rows:
        if row.stable_uid not in selected:
            groups[row.cluster_id].append(row)
    group_order = sorted(
        groups,
        key=lambda cluster: (-len(groups[cluster]), _rank(seed, "cluster-group", cluster), cluster),
    )
    chosen: list[MetricRow] = []
    # First cover as many distinct high-mass clusters as possible.
    for cluster in group_order:
        available = groups[cluster]
        if not available or len(chosen) >= count:
            break
        chosen.append(
            min(
                available,
                key=lambda row: (_rank(seed, "cluster-representative", row.stable_uid), row.stable_uid),
            )
        )
    # If one/few clusters dominate, fill the quota with further representatives
    # ordered by cluster mass rather than input order.
    if len(chosen) < count:
        chosen_uids = {row.stable_uid for row in chosen}
        remaining = [
            row
            for cluster in group_order
            for row in groups[cluster]
            if row.stable_uid not in chosen_uids
        ]
        remaining.sort(
            key=lambda row: (
                -len(groups[row.cluster_id]),
                _rank(seed, "cluster-fill", row.stable_uid),
                row.stable_uid,
            )
        )
        chosen.extend(remaining[: count - len(chosen)])
    if len(chosen) != count:
        raise AssertionError("cluster sampling could not satisfy an available quota")
    return chosen


def _select_source_rows(
    source_rows: Sequence[MetricRow], *, quotas: Mapping[str, int], seed: str
) -> list[dict[str, Any]]:
    """Select risk first, then cluster representatives, then random rows.

    Risk receives precedence so the risk stratum is literally the highest-risk
    available documents.  The later strata are selected from the remaining
    pool, making the three reported strata disjoint by construction.
    """

    selected: set[str] = set()
    chosen: list[tuple[str, MetricRow]] = []

    risk_rows = sorted(
        source_rows,
        key=lambda row: (-row.risk_score, _rank(seed, "risk", row.stable_uid), row.stable_uid),
    )[: int(quotas["risk"])]
    for row in risk_rows:
        selected.add(row.stable_uid)
        chosen.append(("risk", row))

    cluster_rows = _cluster_selection(
        source_rows, count=int(quotas["cluster"]), selected=selected, seed=seed
    )
    for row in cluster_rows:
        if row.stable_uid in selected:
            raise AssertionError("cluster selection reused a risk document")
        selected.add(row.stable_uid)
        chosen.append(("cluster", row))

    random_rows = sorted(
        (row for row in source_rows if row.stable_uid not in selected),
        key=lambda row: (_rank(seed, "random", row.stable_uid), row.stable_uid),
    )[: int(quotas["random"])]
    if len(random_rows) != int(quotas["random"]):
        raise AssertionError("random sampling could not satisfy an available quota")
    for row in random_rows:
        selected.add(row.stable_uid)
        chosen.append(("random", row))

    if len(selected) != sum(int(quotas[stratum]) for stratum in STRATA):
        raise AssertionError("selection does not close against its stratum quotas")

    cluster_sizes = Counter(row.cluster_id for row in source_rows)
    records: list[dict[str, Any]] = []
    for stratum, row in chosen:
        records.append(
            {
                **row.identity(),
                "source_route": row.source_route,
                "review_route": row.review_route,
                "extraction_route": row.extraction_route,
                "observed_extraction_route": row.observed_extraction_route,
                "observed_extraction_route_basis": row.observed_extraction_route_basis,
                "observed_extraction_route_evidence": row.observed_extraction_route_evidence,
                "observed_extraction_route_priority": row.observed_extraction_route_priority,
                "sampling_stratum": stratum,
                "risk_score": row.risk_score,
                "review_cluster_id": row.cluster_id,
                "review_cluster_size": cluster_sizes[row.cluster_id],
                "selection_rank": _rank(seed, f"{stratum}-selection", row.stable_uid),
            }
        )
    return sorted(
        records,
        key=lambda item: (
            item["source_id"],
            STRATA.index(str(item["sampling_stratum"])),
            int(item["selection_rank"]),
            str(item["stable_uid"]),
        ),
    )


def build_sample_manifest(
    rows: Iterable[Mapping[str, Any]],
    roster: Mapping[str, Any],
    *,
    seed: str,
    large_or_heterogeneous_sources: Iterable[str] = (),
    require_all_candidate_sources: bool = True,
    full_scan_metrics_sha256: str | None = None,
) -> dict[str, Any]:
    """Produce an immutable source-by-source sample selection manifest.

    ``rows`` must be the complete candidate full-scan metric inventory, not an
    already sampled subset.  The manifest makes a small source exhaustive when
    it has fewer eligible documents than its target, and explicitly records a
    failed 100-document denominator rather than inventing review rows.
    """

    _require_nonempty_string("selection seed", seed)
    route_report = validate_candidate_roster_routes(roster)
    normalized = normalize_metric_rows(rows, roster)
    by_source: dict[str, list[MetricRow]] = defaultdict(list)
    for row in normalized:
        by_source[row.source_id].append(row)
    candidates = set(route_report["candidate_source_ids"])
    missing_sources = sorted(candidates - set(by_source))
    if missing_sources and require_all_candidate_sources:
        raise ValueError(
            "full-scan metric input is missing candidate roster sources; do not silently omit: "
            f"{missing_sources}"
        )
    unknown_large = sorted(set(large_or_heterogeneous_sources) - candidates)
    if unknown_large:
        raise ValueError(f"large/heterogeneous sources are not roster candidates: {unknown_large}")
    large_sources = set(large_or_heterogeneous_sources)
    if full_scan_metrics_sha256 is not None:
        _require_sha256("full_scan_metrics_sha256", full_scan_metrics_sha256)

    selected_records: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    for source_id in sorted(by_source):
        source_rows = sorted(by_source[source_id], key=lambda item: item.stable_uid)
        revisions = {row.source_revision for row in source_rows}
        datasets = {row.source_dataset for row in source_rows}
        if len(revisions) != 1:
            raise ValueError(f"{source_id}: full-scan metrics mix source revisions: {sorted(revisions)}")
        if len(datasets) != 1:
            raise ValueError(f"{source_id}: full-scan metrics mix source_dataset identities: {sorted(datasets)}")
        quotas, configured_target, target_unavailable = _source_target(
            source_id, len(source_rows), large_sources
        )
        records = _select_source_rows(source_rows, quotas=quotas, seed=seed)
        if len(records) != min(len(source_rows), configured_target):
            raise AssertionError("sample count drift")
        strata_counts = Counter(str(row["sampling_stratum"]) for row in records)
        if dict(strata_counts) != {stratum: quotas[stratum] for stratum in STRATA if quotas[stratum]}:
            raise AssertionError("sample strata do not close against selected quotas")
        eligible_identity = [row.identity() for row in source_rows]
        selected_identity = [
            {
                "stable_uid": row["stable_uid"],
                "sampling_stratum": row["sampling_stratum"],
            }
            for row in records
        ]
        exhaustive = len(records) == len(source_rows)
        minimum_unattainable = len(source_rows) < MINIMUM_ELIGIBLE_DOCUMENTS
        denominator = {
            "eligible_document_count": len(source_rows),
            "minimum_required_documents": MINIMUM_ELIGIBLE_DOCUMENTS,
            "configured_review_target": configured_target,
            "selected_unique_documents": len(records),
            "selection_is_exhaustive": exhaustive,
            "minimum_requirement_status": (
                "unattainable_exhaustive" if minimum_unattainable else "met"
            ),
            "target_status": "unattainable_exhaustive" if target_unavailable else "met",
            "denominator_exception": (
                "eligible_inventory_below_100_all_documents_selected"
                if minimum_unattainable
                else None
            ),
        }
        source_summaries.append(
            {
                "source_id": source_id,
                "source_dataset": next(iter(datasets)),
                "source_revision": next(iter(revisions)),
                "source_route": source_rows[0].source_route,
                "review_route": source_rows[0].review_route,
                "extraction_route": source_rows[0].extraction_route,
                "allowed_observed_extraction_routes": list(
                    route_report["allowed_observed_extraction_routes"][source_id]
                ),
                "observed_extraction_route_counts": dict(
                    sorted(
                        Counter(
                            row.observed_extraction_route for row in source_rows
                        ).items()
                    )
                ),
                "observed_extraction_route_basis_counts": dict(
                    sorted(
                        Counter(
                            row.observed_extraction_route_basis for row in source_rows
                        ).items()
                    )
                ),
                "observed_extraction_route_priority_counts": dict(
                    sorted(
                        Counter(
                            row.observed_extraction_route_priority for row in source_rows
                        ).items()
                    )
                ),
                "large_or_heterogeneous": source_id in large_sources,
                "review_denominator": denominator,
                "requested_strata": {stratum: quotas[stratum] for stratum in STRATA},
                "actual_strata": {stratum: int(strata_counts.get(stratum, 0)) for stratum in STRATA},
                "eligible_inventory_sha256": sha256_json(eligible_identity),
                "selected_inventory_sha256": sha256_json(selected_identity),
            }
        )
        selected_records.extend(records)

    if len({str(row["stable_uid"]) for row in selected_records}) != len(selected_records):
        raise AssertionError("a document was selected more than once across sources")
    payload: dict[str, Any] = {
        "schema_version": SAMPLE_MANIFEST_SCHEMA,
        "seed": seed,
        "candidate_roster_sha256": route_report["roster_sha256"],
        "full_scan_metrics_sha256": full_scan_metrics_sha256,
        "route_validation": route_report,
        "sources": source_summaries,
        "selected_documents": sorted(
            selected_records,
            key=lambda row: (
                str(row["source_id"]),
                STRATA.index(str(row["sampling_stratum"])),
                int(row["selection_rank"]),
                str(row["stable_uid"]),
            ),
        ),
        "selected_document_count": len(selected_records),
        "missing_candidate_sources": missing_sources,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def select_secondary_samples(
    selected_documents: Iterable[Mapping[str, Any]], *, seed: str, fraction: float = 0.1
) -> list[dict[str, Any]]:
    """Select a deterministic secondary-review subset within every source/stratum.

    Selecting within each stratum keeps the 10% audit from becoming a random
    slice of only the large random stratum.  Rounding is explicit: each
    non-empty source/stratum gets ``ceil(n * fraction)`` requests.
    """

    _require_nonempty_string("secondary selection seed", seed)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 <= fraction <= 1:
        raise ValueError("secondary fraction must be in [0, 1]")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in selected_documents:
        source_id = _require_nonempty_string("selected source_id", row.get("source_id"))
        stable_uid = _require_sha256("selected stable_uid", row.get("stable_uid"))
        stratum = row.get("sampling_stratum")
        if stratum not in STRATA:
            raise ValueError(f"selected {stable_uid}: unsupported sampling stratum {stratum!r}")
        if stable_uid in seen:
            raise ValueError(f"selected documents repeat stable_uid {stable_uid}")
        seen.add(stable_uid)
        groups[(source_id, str(stratum))].append(dict(row))
    result: list[dict[str, Any]] = []
    for (source_id, stratum), group in sorted(groups.items()):
        count = int(math.ceil(len(group) * float(fraction)))
        if count == 0:
            continue
        ranked = sorted(
            group,
            key=lambda row: (
                _rank(seed, f"secondary:{source_id}:{stratum}", str(row["stable_uid"])),
                str(row["stable_uid"]),
            ),
        )
        for row in ranked[:count]:
            result.append(
                {
                    **row,
                    "reviewer_slot": "secondary",
                    "secondary_selection_rank": _rank(
                        seed, f"secondary:{source_id}:{stratum}", str(row["stable_uid"])
                    ),
                }
            )
    return sorted(
        result,
        key=lambda row: (
            str(row["source_id"]),
            STRATA.index(str(row["sampling_stratum"])),
            int(row["secondary_selection_rank"]),
            str(row["stable_uid"]),
        ),
    )


def validate_review_model(model: str, *, required_model: str = REQUIRED_REVIEW_MODEL) -> None:
    if model != required_model:
        raise ValueError(
            f"review model must be exactly {required_model!r}; no model fallback is permitted"
        )


def _request_hash(request: Mapping[str, Any]) -> str:
    payload = dict(request)
    payload.pop("request_sha256", None)
    return sha256_json(payload)


def _request_observed_route_fields(
    sample: Mapping[str, Any], *, source_route: str
) -> dict[str, str]:
    """Validate compact observed-route context without changing logical route.

    Native v3 selection records always provide all four fields.  The explicit
    fallback keeps synthetic preflight and historical test packets executable,
    while making their absence visible rather than silently claiming a
    document-level observation.
    """

    observed = sample.get("observed_extraction_route")
    if observed is None:
        observed = sample.get("extraction_route", source_route)
    if not isinstance(observed, str) or observed not in ALLOWED_ROUTES:
        raise ValueError("sample.observed_extraction_route must be a supported route")
    basis = sample.get("observed_extraction_route_basis", "unavailable")
    if not isinstance(basis, str) or basis not in OBSERVED_EXTRACTION_ROUTE_BASES:
        raise ValueError("sample.observed_extraction_route_basis is unsupported")
    evidence = sample.get("observed_extraction_route_evidence", "request:source_route_fallback")
    if (
        not isinstance(evidence, str)
        or not evidence
        or len(evidence) > 256
        or any(character.isspace() or ord(character) < 0x20 for character in evidence)
    ):
        raise ValueError("sample.observed_extraction_route_evidence must be a bounded non-whitespace code")
    expected_priority = (
        "logical_primary"
        if observed == source_route
        else "secondary_exception_only"
    )
    priority = sample.get("observed_extraction_route_priority", expected_priority)
    if priority != expected_priority:
        raise ValueError(
            "sample.observed_extraction_route_priority must preserve logical-source priority"
        )
    return {
        "observed_extraction_route": observed,
        "observed_extraction_route_basis": basis,
        "observed_extraction_route_evidence": evidence,
        "observed_extraction_route_priority": expected_priority,
    }


def make_review_request(
    sample: Mapping[str, Any],
    *,
    reviewer_slot: str,
    original_text_sha256: str,
    review_copy_sha256: str,
    prompt_sha256: str,
    response_schema_sha256: str,
    model: str,
    code_commit: str,
    attempt: int = 1,
    review_copy: str | None = None,
    comparison_bundle: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind one compact review request to immutable selection and prompt inputs."""

    if reviewer_slot not in {"primary", "secondary", "adjudicator"}:
        raise ValueError("reviewer_slot must be primary, secondary, or adjudicator")
    validate_review_model(model)
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    source_id = _require_nonempty_string("sample.source_id", sample.get("source_id"))
    source_dataset = _require_nonempty_string("sample.source_dataset", sample.get("source_dataset"))
    source_revision = _require_nonempty_string("sample.source_revision", sample.get("source_revision"))
    source_route = sample.get("source_route")
    if source_route not in ALLOWED_ROUTES:
        raise ValueError("sample.source_route must be a supported review route")
    observed_route_fields = _request_observed_route_fields(
        sample, source_route=str(source_route)
    )
    stable_uid = _require_sha256("sample.stable_uid", sample.get("stable_uid"))
    stratum = sample.get("sampling_stratum")
    if stratum not in STRATA:
        raise ValueError("sample.sampling_stratum must be a supported stratum")
    for name, value in (
        ("original_text_sha256", original_text_sha256),
        ("review_copy_sha256", review_copy_sha256),
        ("prompt_sha256", prompt_sha256),
        ("response_schema_sha256", response_schema_sha256),
    ):
        _require_sha256(name, value)
    _require_nonempty_string("code_commit", code_commit)
    if review_copy is not None and hashlib.sha256(review_copy.encode("utf-8")).hexdigest() != review_copy_sha256:
        raise ValueError("review_copy content does not match review_copy_sha256")
    identity = {
        "schema_version": REQUEST_SCHEMA,
        "sample_id": stable_uid,
        "reviewer_slot": reviewer_slot,
        "source_id": source_id,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_route": source_route,
        **observed_route_fields,
        "sampling_stratum": stratum,
        "original_text_sha256": original_text_sha256,
        "review_copy_sha256": review_copy_sha256,
        "prompt_sha256": prompt_sha256,
        "response_schema_sha256": response_schema_sha256,
        "model": model,
        "code_commit": code_commit,
        "attempt": attempt,
    }
    review_id = sha256_json({"kind": "agent1_v3_review_id", **identity})
    request = {"review_id": review_id, **identity}
    if review_copy is not None:
        request["review_copy"] = review_copy
    if comparison_bundle is not None:
        request["comparison_bundle"] = [dict(item) for item in comparison_bundle]
    request["request_sha256"] = _request_hash(request)
    return request


def _validate_request_binding(request: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("unsupported request schema_version")
    for field in (
        "review_id",
        "sample_id",
        "original_text_sha256",
        "review_copy_sha256",
        "prompt_sha256",
        "response_schema_sha256",
    ):
        try:
            _require_sha256(f"request.{field}", request.get(field))
        except ValueError as exc:
            errors.append(str(exc))
    for field in ("source_id", "source_dataset", "source_revision", "code_commit", "model"):
        try:
            _require_nonempty_string(f"request.{field}", request.get(field))
        except ValueError as exc:
            errors.append(str(exc))
    if request.get("source_route") not in ALLOWED_ROUTES:
        errors.append("request.source_route is unsupported")
    observed_route = request.get("observed_extraction_route")
    if observed_route not in ALLOWED_ROUTES:
        errors.append("request.observed_extraction_route is unsupported")
    observed_basis = request.get("observed_extraction_route_basis")
    if observed_basis not in OBSERVED_EXTRACTION_ROUTE_BASES:
        errors.append("request.observed_extraction_route_basis is unsupported")
    observed_evidence = request.get("observed_extraction_route_evidence")
    if (
        not isinstance(observed_evidence, str)
        or not observed_evidence
        or len(observed_evidence) > 256
        or any(character.isspace() or ord(character) < 0x20 for character in observed_evidence)
    ):
        errors.append("request.observed_extraction_route_evidence is unsupported")
    expected_priority = (
        "logical_primary"
        if observed_route == request.get("source_route")
        else "secondary_exception_only"
    )
    if request.get("observed_extraction_route_priority") != expected_priority:
        errors.append("request.observed_extraction_route_priority reverses logical-source priority")
    if request.get("sampling_stratum") not in STRATA:
        errors.append("request.sampling_stratum is unsupported")
    if request.get("reviewer_slot") not in {"primary", "secondary", "adjudicator"}:
        errors.append("request.reviewer_slot is unsupported")
    if not isinstance(request.get("attempt"), int) or isinstance(request.get("attempt"), bool) or int(request["attempt"]) < 1:
        errors.append("request.attempt must be a positive integer")
    if request.get("request_sha256") != _request_hash(request):
        errors.append("request_sha256 drift")
    return errors


RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "review_id",
        "sample_id",
        "reviewer_slot",
        "source_id",
        "source_dataset",
        "source_revision",
        "source_route",
        "observed_extraction_route",
        "observed_extraction_route_basis",
        "observed_extraction_route_evidence",
        "observed_extraction_route_priority",
        "sampling_stratum",
        "original_text_sha256",
        "review_copy_sha256",
        "prompt_sha256",
        "response_schema_sha256",
        "model",
        "code_commit",
        "attempt",
        "cleanliness_score",
        "quality_score",
        "diversity_contribution_score",
        "issues",
        "recommendation",
        "confidence_score",
        "evidence",
    }
)


def validate_review_response(
    response: Mapping[str, Any], request: Mapping[str, Any] | None = None
) -> list[str]:
    """Return strict v3 response errors; an empty list denotes a valid result."""

    if not isinstance(response, Mapping):
        return ["review response must be an object"]
    errors: list[str] = []
    missing = sorted(RESPONSE_FIELDS - set(response))
    extra = sorted(set(response) - RESPONSE_FIELDS)
    if missing:
        errors.append(f"missing required fields: {missing}")
    if extra:
        errors.append(f"unexpected fields: {extra}")
    if response.get("schema_version") != RESPONSE_SCHEMA:
        errors.append("unsupported response schema_version")
    for field in (
        "review_id",
        "sample_id",
        "original_text_sha256",
        "review_copy_sha256",
        "prompt_sha256",
        "response_schema_sha256",
    ):
        try:
            _require_sha256(field, response.get(field))
        except ValueError as exc:
            errors.append(str(exc))
    for field in ("source_id", "source_dataset", "source_revision", "model", "code_commit"):
        try:
            _require_nonempty_string(field, response.get(field))
        except ValueError as exc:
            errors.append(str(exc))
    if response.get("source_route") not in ALLOWED_ROUTES:
        errors.append("source_route is unsupported")
    observed_route = response.get("observed_extraction_route")
    if observed_route not in ALLOWED_ROUTES:
        errors.append("observed_extraction_route is unsupported")
    observed_basis = response.get("observed_extraction_route_basis")
    if observed_basis not in OBSERVED_EXTRACTION_ROUTE_BASES:
        errors.append("observed_extraction_route_basis is unsupported")
    observed_evidence = response.get("observed_extraction_route_evidence")
    if (
        not isinstance(observed_evidence, str)
        or not observed_evidence
        or len(observed_evidence) > 256
        or any(character.isspace() or ord(character) < 0x20 for character in observed_evidence)
    ):
        errors.append("observed_extraction_route_evidence is unsupported")
    expected_priority = (
        "logical_primary"
        if observed_route == response.get("source_route")
        else "secondary_exception_only"
    )
    if response.get("observed_extraction_route_priority") != expected_priority:
        errors.append("observed_extraction_route_priority reverses logical-source priority")
    if response.get("sampling_stratum") not in STRATA:
        errors.append("sampling_stratum is unsupported")
    if response.get("reviewer_slot") not in {"primary", "secondary", "adjudicator"}:
        errors.append("reviewer_slot is unsupported")
    if not isinstance(response.get("attempt"), int) or isinstance(response.get("attempt"), bool) or int(response.get("attempt", 0)) < 1:
        errors.append("attempt must be a positive integer")
    for field in (
        "cleanliness_score",
        "quality_score",
        "diversity_contribution_score",
        "confidence_score",
    ):
        try:
            _require_int_1_to_5(field, response.get(field))
        except ValueError as exc:
            errors.append(str(exc))
    if response.get("recommendation") not in RECOMMENDATIONS:
        errors.append("recommendation is unsupported")
    evidence = response.get("evidence")
    if not isinstance(evidence, str) or not evidence.strip() or len(evidence) > 1000:
        errors.append("evidence must be non-empty and at most 1000 characters")
    issues = response.get("issues")
    if not isinstance(issues, list):
        errors.append("issues must be a list")
    else:
        seen_codes: set[str] = set()
        for index, issue in enumerate(issues):
            if not isinstance(issue, Mapping) or set(issue) != {"code", "severity_score"}:
                errors.append(f"issues[{index}] must contain exactly code and severity_score")
                continue
            code = issue.get("code")
            if code not in ISSUE_CODES:
                errors.append(f"issues[{index}].code is unsupported")
            elif code in seen_codes:
                errors.append(f"issues[{index}].code is duplicated")
            seen_codes.add(str(code))
            try:
                _require_int_1_to_5(f"issues[{index}].severity_score", issue.get("severity_score"))
            except ValueError as exc:
                errors.append(str(exc))
    if request is not None:
        errors.extend(_validate_request_binding(request))
        for field in (
            "review_id",
            "sample_id",
            "reviewer_slot",
            "source_id",
            "source_dataset",
            "source_revision",
            "source_route",
            "observed_extraction_route",
            "observed_extraction_route_basis",
            "observed_extraction_route_evidence",
            "observed_extraction_route_priority",
            "sampling_stratum",
            "original_text_sha256",
            "review_copy_sha256",
            "prompt_sha256",
            "response_schema_sha256",
            "model",
            "code_commit",
            "attempt",
        ):
            if response.get(field) != request.get(field):
                errors.append(f"response/request identity drift: {field}")
    return errors


def assert_valid_review_response(response: Mapping[str, Any], request: Mapping[str, Any]) -> None:
    errors = validate_review_response(response, request)
    if errors:
        raise ValueError("; ".join(errors))


def _severity_by_code(response: Mapping[str, Any]) -> dict[str, int]:
    return {
        str(item["code"]): int(item["severity_score"])
        for item in response["issues"]
        if isinstance(item, Mapping)
    }


def material_disagreement(primary: Mapping[str, Any], secondary: Mapping[str, Any]) -> list[str]:
    """Return only disagreements large enough to require adjudication."""

    reasons: list[str] = []
    if primary["recommendation"] != secondary["recommendation"]:
        reasons.append("recommendation")
    for field in ("cleanliness_score", "quality_score", "diversity_contribution_score"):
        if abs(int(primary[field]) - int(secondary[field])) >= 2:
            reasons.append(field)
    first = _severity_by_code(primary)
    second = _severity_by_code(secondary)
    for code in sorted(set(first) | set(second)):
        if abs(first.get(code, 0) - second.get(code, 0)) >= 2:
            reasons.append(f"issue_severity:{code}")
    return reasons


def _low_confidence_reasons(response: Mapping[str, Any], prefix: str) -> list[str]:
    reasons: list[str] = []
    if int(response["confidence_score"]) <= 2:
        reasons.append(f"{prefix}_low_confidence")
    if response["recommendation"] == "uncertain":
        reasons.append(f"{prefix}_uncertain_recommendation")
    return reasons


def make_adjudication_request(
    primary_request: Mapping[str, Any],
    primary_response: Mapping[str, Any],
    secondary_response: Mapping[str, Any] | None,
    *,
    reasons: Sequence[str],
) -> dict[str, Any]:
    """Create a deterministic third-slot request without rerolling either review."""

    request_errors = _validate_request_binding(primary_request)
    if request_errors:
        raise ValueError("invalid primary request: " + "; ".join(request_errors))
    assert_valid_review_response(primary_response, primary_request)
    if secondary_response is not None:
        # The caller validates the secondary response against its own request.
        errors = validate_review_response(secondary_response)
        if errors:
            raise ValueError("invalid secondary response: " + "; ".join(errors))
    request = dict(primary_request)
    request["reviewer_slot"] = "adjudicator"
    request["review_id"] = sha256_json(
        {
            "kind": "agent1_v3_adjudication_review_id",
            "sample_id": primary_request["sample_id"],
            "primary_review_id": primary_response["review_id"],
            "secondary_review_id": secondary_response["review_id"] if secondary_response else None,
            "reasons": sorted(set(reasons)),
            "prompt_sha256": primary_request["prompt_sha256"],
            "response_schema_sha256": primary_request["response_schema_sha256"],
            "model": primary_request["model"],
        }
    )
    request["adjudication_reasons"] = sorted(set(reasons))
    request["adjudication_context"] = {
        "primary_response": dict(primary_response),
        "secondary_response": dict(secondary_response) if secondary_response else None,
    }
    request["request_sha256"] = _request_hash(request)
    return request


def build_adjudication_manifest(
    requests: Iterable[Mapping[str, Any]], responses: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Report every review that must be retried or adjudicated before admission.

    Primary and secondary responses are never overwritten or sampled again.  A
    schema-valid adjudicator response closes only its own deterministic case;
    an incomplete primary/secondary response remains explicitly pending.
    """

    request_by_id: dict[str, dict[str, Any]] = {}
    requests_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for index, raw in enumerate(requests, 1):
        request = dict(raw)
        errors = _validate_request_binding(request)
        if errors:
            raise ValueError(f"request {index}: {'; '.join(errors)}")
        review_id = str(request["review_id"])
        slot = str(request["reviewer_slot"])
        if slot not in {"primary", "secondary"}:
            raise ValueError("initial requests may contain only primary/secondary slots")
        if review_id in request_by_id or slot in requests_by_sample[str(request["sample_id"])]:
            raise ValueError(f"request {index}: duplicate review identity")
        request_by_id[review_id] = request
        requests_by_sample[str(request["sample_id"])][slot] = request

    response_by_id: dict[str, dict[str, Any]] = {}
    unbound_adjudicator: list[dict[str, Any]] = []
    for index, raw in enumerate(responses, 1):
        response = dict(raw)
        review_id = response.get("review_id")
        if not isinstance(review_id, str) or review_id in response_by_id:
            raise ValueError(f"response {index}: duplicate or invalid review_id")
        if response.get("reviewer_slot") == "adjudicator":
            # Its request is derived after detecting the case below.
            errors = validate_review_response(response)
            if errors:
                raise ValueError(f"response {index}: {'; '.join(errors)}")
            unbound_adjudicator.append(response)
            response_by_id[review_id] = response
            continue
        request = request_by_id.get(review_id)
        if request is None:
            raise ValueError(f"response {index}: response has no requested review_id")
        assert_valid_review_response(response, request)
        response_by_id[review_id] = response

    adjudicator_by_id = {str(item["review_id"]): item for item in unbound_adjudicator}
    cases: list[dict[str, Any]] = []
    pending_count = 0
    for sample_id in sorted(requests_by_sample):
        slots = requests_by_sample[sample_id]
        primary_request = slots.get("primary")
        if primary_request is None:
            raise ValueError(f"{sample_id}: primary request is required")
        primary = response_by_id.get(str(primary_request["review_id"]))
        secondary_request = slots.get("secondary")
        secondary = (
            response_by_id.get(str(secondary_request["review_id"])) if secondary_request is not None else None
        )
        reasons: list[str] = []
        if primary is None:
            reasons.append("missing_primary_response")
        if secondary_request is not None and secondary is None:
            reasons.append("missing_secondary_response")
        if primary is not None:
            reasons.extend(_low_confidence_reasons(primary, "primary"))
        if secondary is not None:
            reasons.extend(_low_confidence_reasons(secondary, "secondary"))
        if primary is not None and secondary is not None:
            reasons.extend(f"material_disagreement:{item}" for item in material_disagreement(primary, secondary))
        if not reasons:
            continue

        retry_reasons = [reason for reason in reasons if reason.startswith("missing_")]
        case: dict[str, Any] = {
            "sample_id": sample_id,
            "source_id": primary_request["source_id"],
            "source_dataset": primary_request["source_dataset"],
            "source_revision": primary_request["source_revision"],
            "reasons": sorted(set(reasons)),
            "primary_review_id": primary_request["review_id"],
            "secondary_review_id": secondary_request["review_id"] if secondary_request else None,
            "status": "pending_retry" if retry_reasons else "pending_adjudication",
        }
        if retry_reasons:
            pending_count += 1
        else:
            assert primary is not None
            adjudication_request = make_adjudication_request(
                primary_request, primary, secondary, reasons=reasons
            )
            case["adjudication_request"] = adjudication_request
            adjudicator = adjudicator_by_id.get(str(adjudication_request["review_id"]))
            if adjudicator is None:
                pending_count += 1
            else:
                assert_valid_review_response(adjudicator, adjudication_request)
                case["status"] = "adjudicated"
                case["adjudicator_review_id"] = adjudicator["review_id"]
        cases.append(case)

    # A response cannot be silently ignored: every supplied adjudicator result
    # must close a case constructed from the immutable first-pass outcomes.
    referenced_adjudicators = {
        str(case.get("adjudicator_review_id")) for case in cases if case.get("adjudicator_review_id")
    }
    unexpected_adjudicators = sorted(set(adjudicator_by_id) - referenced_adjudicators)
    if unexpected_adjudicators:
        raise ValueError(f"unexpected adjudicator responses: {unexpected_adjudicators}")
    payload: dict[str, Any] = {
        "schema_version": ADJUDICATION_MANIFEST_SCHEMA,
        "request_count": len(request_by_id),
        "response_count": len(response_by_id),
        "case_count": len(cases),
        "pending_count": pending_count,
        "status": "complete" if pending_count == 0 else "pending_adjudication",
        "cases": cases,
    }
    payload["manifest_sha256"] = sha256_json(payload)
    return payload


def assert_adjudication_closed(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != ADJUDICATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported adjudication manifest")
    if manifest.get("pending_count") != 0 or manifest.get("status") != "complete":
        raise ValueError("adjudication remains pending; source admission is blocked")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(iter_rows(path))


def _cmd_validate_roster(args: argparse.Namespace) -> int:
    report = validate_candidate_roster_routes(load_json(args.roster))
    write_json_atomic(args.output, report)
    print(json.dumps({"ok": True, "output": str(args.output), "roster_sha256": report["roster_sha256"]}))
    return 0


def _cmd_sample(args: argparse.Namespace) -> int:
    roster = load_json(args.roster)
    manifest = build_sample_manifest(
        iter_rows(args.metrics),
        roster,
        seed=args.seed,
        large_or_heterogeneous_sources=args.large_or_heterogeneous_source,
        require_all_candidate_sources=not args.allow_missing_candidate_sources,
        full_scan_metrics_sha256=sha256_file(args.metrics),
    )
    write_json_atomic(args.output, manifest)
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "selected_document_count": manifest["selected_document_count"],
                "manifest_sha256": manifest["manifest_sha256"],
            }
        )
    )
    return 0


def _cmd_adjudication(args: argparse.Namespace) -> int:
    manifest = build_adjudication_manifest(_load_jsonl(args.requests), _load_jsonl(args.responses))
    write_json_atomic(args.output, manifest)
    print(json.dumps({"ok": True, "output": str(args.output), "pending_count": manifest["pending_count"]}))
    return 0 if manifest["pending_count"] == 0 else 2


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-roster", help="validate candidate review-route closure")
    validate.add_argument("--roster", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.set_defaults(func=_cmd_validate_roster)

    sample = sub.add_parser("sample", help="select deterministic review documents from full-scan rows")
    sample.add_argument("--roster", type=Path, required=True)
    sample.add_argument("--metrics", type=Path, required=True)
    sample.add_argument("--seed", required=True)
    sample.add_argument("--large-or-heterogeneous-source", action="append", default=[])
    sample.add_argument("--allow-missing-candidate-sources", action="store_true")
    sample.add_argument("--output", type=Path, required=True)
    sample.set_defaults(func=_cmd_sample)

    adjudication = sub.add_parser("adjudication", help="build the pending review/adjudication manifest")
    adjudication.add_argument("--requests", type=Path, required=True)
    adjudication.add_argument("--responses", type=Path, required=True)
    adjudication.add_argument("--output", type=Path, required=True)
    adjudication.set_defaults(func=_cmd_adjudication)
    return command


def main() -> int:
    args = parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
