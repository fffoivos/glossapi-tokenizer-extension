#!/usr/bin/env python3
"""Build deterministic, risk-stratified Luna reviews for HTML-to-GFM changes."""

from __future__ import annotations

import argparse
import html
import hashlib
import importlib.util
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from build_source_review_packet import redact_direct_identifiers
from source_lineage import canonical_json


SCHEMA = "gfm_transformation_review_packet_v1"
REQUEST_SCHEMA = "gfm_transformation_review_request_v1"
DEFAULT_TARGET_REGIONS = 100
MAX_EXCERPT_CHARACTERS = 12_000


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected an object")
    return value


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_json(row) + "\n")
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def cards_by_opaque_id(site_dir: Path) -> dict[str, dict[str, object]]:
    index = read_json(site_dir / "data/index.json")
    result: dict[str, dict[str, object]] = {}
    for relative in dict(index["source_files"]).values():
        payload = read_json(site_dir / str(relative))
        for card in payload["cards"]:
            if not isinstance(card, dict):
                raise ValueError("source card must be an object")
            opaque_id = str(card["opaque_id"])
            if opaque_id in result:
                raise ValueError(f"duplicate opaque id: {opaque_id}")
            result[opaque_id] = card
    return result


def clipped(value: str, maximum: int = MAX_EXCERPT_CHARACTERS) -> str:
    if len(value) <= maximum:
        return value
    segment = maximum // 3
    middle = max(0, (len(value) - segment) // 2)
    return (
        value[:segment]
        + "\n\n[… excerpt middle …]\n\n"
        + value[middle : middle + segment]
        + "\n\n[… excerpt end …]\n\n"
        + value[-segment:]
    )


def around(value: str, start: int, end: int, maximum: int = MAX_EXCERPT_CHARACTERS) -> str:
    start = max(0, min(start, len(value)))
    end = max(start, min(end, len(value)))
    if end - start >= maximum // 2:
        edge = maximum // 4
        prefix_end = min(end, start + edge)
        prefix_line_end = value.rfind("\n", start, prefix_end)
        if prefix_line_end > start + edge // 2:
            prefix_end = prefix_line_end + 1
        suffix_start = max(start, end - edge)
        suffix_line_start = value.find("\n", suffix_start, end)
        if 0 <= suffix_line_start < end - edge // 2:
            suffix_start = suffix_line_start + 1
        body = (
            value[start:prefix_end]
            + "\n[… focus span middle omitted from review excerpt …]\n"
            + value[suffix_start:end]
        )
    else:
        body = value[start:end]
    remaining = max(0, maximum - len(body))
    before = min(start, remaining // 2)
    after = min(len(value) - end, remaining - before)
    return value[start - before : start] + body + value[end : end + after]


def line_column_offset(value: str, line: int, column: int) -> int:
    """Translate HTMLParser's one-based line/zero-based column into an offset."""

    if line < 1 or column < 0:
        return 0
    starts = [0]
    starts.extend(match.end() for match in re.finditer("\n", value))
    if line > len(starts):
        return len(value)
    return min(len(value), starts[line - 1] + column)


def table_fallback_focus(
    cleaned: str,
    normalized: str,
    fallback: Mapping[str, object],
) -> tuple[int, int, str]:
    """Center fallback evidence on its source table and emitted readable block."""

    source_line = int(fallback.get("source_line") or 1)
    source_column = int(fallback.get("source_column") or 0)
    before_position = line_column_offset(cleaned, source_line, source_column)
    preview = str(fallback.get("plain_text_preview", ""))
    after_position = normalized.find(preview) if preview else -1
    if after_position < 0:
        after_position = 0

    before_excerpt = around(cleaned, before_position, before_position, maximum=2_000)
    after_excerpt = around(normalized, after_position, after_position + len(preview), maximum=2_000)
    for value in fallback.get("anchor_candidates", []):
        anchor = str(value)
        if anchor and before_excerpt.count(anchor) == 1 and after_excerpt.count(anchor) == 1:
            return before_position, after_position, anchor
    return before_position, after_position, ""


def repetition_boundary_excerpt(value: str, start: int, end: int, *, context: int = 900) -> str:
    """Show equal outside-boundary context plus bounded evidence from a removed span."""

    start = max(0, min(start, len(value)))
    end = max(start, min(end, len(value)))
    prefix = value[max(0, start - context) : start]
    suffix = value[end : min(len(value), end + context)]
    span = value[start:end]
    if len(span) > 800:
        span = span[:400] + "\n[… detected repetition middle …]\n" + span[-400:]
    return prefix + span + suffix


def unique_context_anchor(value: str, start: int, end: int) -> str:
    """Return the smallest bounded exact context that uniquely identifies a target."""

    start = max(0, min(start, len(value)))
    end = max(start, min(end, len(value)))
    for radius in (40, 80, 140, 240, 400):
        candidate = value[max(0, start - radius) : min(len(value), end + radius)]
        if candidate and value.count(candidate) == 1:
            return candidate
    return value[max(0, start - 400) : min(len(value), end + 400)]


def _rule_target_pattern(rule: str) -> re.Pattern[str] | None:
    if rule == "gfm_autolinks_preserved":
        return re.compile(r"<(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\s]*|[^<>\s]+@[^<>\s]+)>")
    if rule == "rowspan_cells_expanded":
        return re.compile(r"<(?:td|th)\b[^>]*\browspan\s*=", re.IGNORECASE)
    if rule == "colspan_cells_expanded":
        return re.compile(r"<(?:td|th)\b[^>]*\bcolspan\s*=", re.IGNORECASE)
    if rule == "table_comments_relocated_after_table":
        return re.compile(r"<!--\s*repeating-text-removed\s*-->", re.IGNORECASE)
    if rule.startswith("existing_gfm_table"):
        return re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
    for name in ("sup", "sub", "u"):
        if rule.startswith(name + "_"):
            return re.compile(rf"<{name}\b", re.IGNORECASE)
    if rule == "html_breaks_converted":
        return re.compile(r"<br\b", re.IGNORECASE)
    if "list" in rule:
        return re.compile(r"<(?:ul|ol|li)\b", re.IGNORECASE)
    if "table" in rule or "rowspan" in rule or "colspan" in rule:
        return re.compile(r"<table\b", re.IGNORECASE)
    return re.compile(r"<[A-Za-z][^>]*>")


def _candidate_visible_anchors(fragment: str) -> list[str]:
    candidates: list[str] = []
    for line in fragment.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        for cell in stripped[1:-1].split("|"):
            plain_cell = cell.replace("\\|", "|").strip()
            if len(plain_cell) >= 20 and not re.fullmatch(r":?-{3,}:?", plain_cell):
                candidates.append(plain_cell)
    for cell in re.finditer(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)\s*>", fragment, re.IGNORECASE | re.DOTALL):
        plain = re.sub(r"<[^>]+>", " ", cell.group(1))
        plain = re.sub(r"\s+", " ", html.unescape(plain)).strip()
        if len(plain) >= 20:
            candidates.append(plain)
    plain_fragment = re.sub(r"<!--.*?-->|<[^>]+>", " ", fragment, flags=re.DOTALL)
    plain_fragment = re.sub(r"\s+", " ", html.unescape(plain_fragment)).strip()
    words = plain_fragment.split()
    for width in (18, 12, 8, 5):
        for offset in range(0, max(1, len(words) - width + 1), max(1, width // 2)):
            candidate = " ".join(words[offset : offset + width])
            if len(candidate) >= 24:
                candidates.append(candidate)
    return sorted(set(candidates), key=lambda value: (-len(value), value))


def aligned_rule_focus(rule: str, before: str, after: str) -> tuple[int, int, str]:
    """Localize a rule to common visible text instead of unrelated document thirds."""

    pattern = _rule_target_pattern(rule)
    target = pattern.search(before) if pattern is not None else None
    before_target = target.start() if target is not None else 0
    target_text = target.group(0) if target is not None else ""
    if target_text and before.count(target_text) == 1 and after.count(target_text) == 1:
        return before_target, after.find(target_text), target_text
    fragment = before[max(0, before_target - 1_500) : min(len(before), before_target + 3_000)]
    for anchor in _candidate_visible_anchors(fragment):
        before_position = before.find(anchor, max(0, before_target - 1_500))
        after_position = after.find(anchor)
        if before.count(anchor) == 1 and after.count(anchor) == 1 and before_position >= 0 and after_position >= 0:
            return before_position, after_position, anchor
    return before_target, 0, ""


def region_id(parts: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json(parts).encode("utf-8")).hexdigest()


def make_region(
    *,
    row: Mapping[str, object],
    card: Mapping[str, object],
    family: str,
    rules: Sequence[str],
    risk_tier: str,
    ordinal: int,
    before: str,
    after: str,
    expected: str,
    focus_anchor: str = "",
) -> dict[str, object]:
    identity = {
        "opaque_id": row["opaque_id"],
        "family": family,
        "rules": list(rules),
        "risk_tier": risk_tier,
        "ordinal": ordinal,
        "before_sha256": sha256_text(before),
        "after_sha256": sha256_text(after),
    }
    return {
        "region_id": region_id(identity),
        "opaque_id": row["opaque_id"],
        "source_id": row["source_id"],
        "source_dataset": card.get("source_dataset", row["source_id"]),
        "source_doc_id": row["source_doc_id"],
        "document_path": card.get("document_path", f"documents/{row['source_id']}/{row['source_doc_id']}.txt"),
        "risk_tier": risk_tier,
        "transformation_family": family,
        "rule_ids": list(rules),
        "ordinal": ordinal,
        "focus_anchor": focus_anchor,
        "expected_behavior": expected,
        "before_sha256": identity["before_sha256"],
        "after_sha256": identity["after_sha256"],
        "before_text": clipped(before),
        "after_text": clipped(after),
    }


def routine_family(rule: str) -> str:
    if "table" in rule or rule in {"rowspan_cells_expanded", "colspan_cells_expanded"}:
        return "valid_table_conversion"
    if "image" in rule:
        return "generated_image_cleanup"
    if any(token in rule for token in ("heading", "list", "blockquote", "break", "paragraph", "block")):
        return "block_structure"
    if any(token in rule for token in ("bold", "strong", "italic", "emphasis", "link", "math", "code", "style")):
        return "inline_structure"
    return "other_markup"


def load_normalizer(path: Path) -> object:
    specification = importlib.util.spec_from_file_location("gfm_luna_packet_normalizer", path.resolve())
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import normalizer: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    if not callable(getattr(module, "clean_generated_image_artifacts", None)):
        raise RuntimeError("normalizer lacks clean_generated_image_artifacts")
    return module


def apply_recorded_repetition_pass(text: str, pass_record: Mapping[str, object]) -> str:
    """Replay one audited repetition pass without rerunning the detector."""

    pieces: list[str] = []
    cursor = 0
    for span_value in pass_record.get("spans", []):
        span = dict(span_value)
        start = int(span["start_index"])
        end = int(span["end_index"])
        if start < cursor or end <= start or end > len(text):
            raise ValueError(f"invalid recorded repetition span {start}:{end} for {len(text)} characters")
        pieces.extend((text[cursor:start], "<!-- repeating-text-removed -->"))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces)


def image_cleanup_input(raw: str, details: Sequence[Mapping[str, object]]) -> str:
    """Recover the exact text immediately before generated-image cleanup."""

    current = raw
    passes = sorted(
        (
            record
            for record in details
            if str(record["cleaning_stage"]) == "before_generated_image_cleanup"
        ),
        key=lambda value: int(value["pass_index"]),
    )
    for pass_record in passes:
        current = apply_recorded_repetition_pass(current, pass_record)
    return current


def nth_index(value: str, needle: str, ordinal: int) -> int:
    """Find a deterministic occurrence without confusing repeated captions."""

    if not needle or ordinal < 0:
        return -1
    position = -1
    for _ in range(ordinal + 1):
        position = value.find(needle, position + 1)
        if position < 0:
            return -1
    return position


def repetition_pass_inputs(
    raw: str,
    details: Sequence[Mapping[str, object]],
    *,
    normalizer: object,
) -> dict[tuple[str, int], str]:
    """Recover the exact text coordinate space used by each recorded pass."""

    by_stage: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for pass_record in details:
        by_stage[str(pass_record["cleaning_stage"])].append(pass_record)

    result: dict[tuple[str, int], str] = {}
    current = raw
    for pass_record in sorted(
        by_stage["before_generated_image_cleanup"], key=lambda value: int(value["pass_index"])
    ):
        pass_index = int(pass_record["pass_index"])
        result[("before_generated_image_cleanup", pass_index)] = current
        current = apply_recorded_repetition_pass(current, pass_record)

    current = normalizer.clean_generated_image_artifacts(current)
    for pass_record in sorted(
        by_stage["after_generated_image_cleanup"], key=lambda value: int(value["pass_index"])
    ):
        pass_index = int(pass_record["pass_index"])
        result[("after_generated_image_cleanup", pass_index)] = current
        current = apply_recorded_repetition_pass(current, pass_record)
    return result


def build_candidates(
    site_dir: Path, audit: Mapping[str, object], *, normalizer: object
) -> list[dict[str, object]]:
    cards = cards_by_opaque_id(site_dir)
    candidates: list[dict[str, object]] = []
    for row_value in audit["documents"]:
        row = dict(row_value)
        opaque_id = str(row["opaque_id"])
        card = cards[opaque_id]
        raw = str(read_json(site_dir / "data/documents" / f"{opaque_id}.json")["text"])
        normalized_payload = read_json(site_dir / "data/gfm/documents" / f"{opaque_id}.json") if row["changed"] else {}
        cleaned = str(normalized_payload.get("cleaned_text", raw))
        normalized = str(normalized_payload.get("normalized_markdown", raw))

        repetition_ordinal = 0
        marker_positions = [match.start() for match in re.finditer(re.escape("<!-- repeating-text-removed -->"), normalized)]
        repetition_details = [dict(value) for value in row.get("repetition_details", [])]
        pass_inputs = repetition_pass_inputs(raw, repetition_details, normalizer=normalizer)
        image_input = image_cleanup_input(raw, repetition_details)
        for pass_record in repetition_details:
            stage = str(pass_record["cleaning_stage"])
            pass_index = int(pass_record["pass_index"])
            stage_input = pass_inputs[(stage, pass_index)]
            for span in pass_record.get("spans", []):
                start = int(span.get("start_index", 0))
                end = int(span.get("end_index", start))
                rules = [str(rule) for rule in span.get("rules", ["complex_repetition"])]
                marker = marker_positions[min(repetition_ordinal, max(0, len(marker_positions) - 1))] if marker_positions else 0
                candidates.append(
                    make_region(
                        row=row,
                        card=card,
                        family="complex_repetition_removal",
                        rules=rules,
                        risk_tier="high",
                        ordinal=repetition_ordinal,
                        before=repetition_boundary_excerpt(stage_input, start, end),
                        after=around(
                            normalized,
                            marker,
                            marker + len("<!-- repeating-text-removed -->"),
                            maximum=1_831,
                        ),
                        focus_anchor=unique_context_anchor(
                            normalized,
                            marker,
                            marker + len("<!-- repeating-text-removed -->"),
                        ),
                        expected="This region audits one detector span: remove only that runaway block or numeric sequence, preserve both boundaries, and include its repeating-text-removed comment. Other comments in the excerpt may represent separately audited spans.",
                    )
                )
                repetition_ordinal += 1

        transformations = {str(key): int(value) for key, value in dict(row.get("transformations", {})).items()}
        for ordinal, fallback_value in enumerate(row.get("table_fallback_events", [])):
            fallback = dict(fallback_value)
            before_position, after_position, anchor = table_fallback_focus(cleaned, normalized, fallback)
            preview_length = len(str(fallback.get("plain_text_preview", "")))
            candidates.append(
                make_region(
                    row=row,
                    card=card,
                    family="table_readable_fallback",
                    rules=["table_fallback_reason_" + str(fallback.get("reason", "unknown"))],
                    risk_tier="high",
                    ordinal=ordinal,
                    before=around(cleaned, before_position, before_position + len(anchor), maximum=2_000),
                    after=around(
                        normalized,
                        after_position,
                        after_position + max(len(anchor), preview_length),
                        maximum=2_000,
                    ),
                    focus_anchor=anchor,
                    expected="This region audits only the damaged or nested table containing focus_anchor: that table must not remain a malformed pipe table, and every readable cell must remain in source order as lines separated by blank source rows. Ignore other valid tables in nearby context.",
                )
            )

        for rule in ("orphan_table_cells_flattened", "table_comments_relocated_after_table"):
            for ordinal in range(transformations.get(rule, 0)):
                before_position, after_position, anchor = aligned_rule_focus(rule, cleaned, normalized)
                candidates.append(
                    make_region(
                        row=row,
                        card=card,
                        family="table_structural_recovery",
                        rules=[rule],
                        risk_tier="high",
                        ordinal=ordinal,
                        before=around(cleaned, before_position, before_position + len(anchor), maximum=2_600),
                        after=around(normalized, after_position, after_position + len(anchor), maximum=2_600),
                        focus_anchor=anchor,
                        expected="Preserve readable text and approved removal comments while emitting valid non-HTML Markdown.",
                    )
                )

        before_occurrences: Counter[str] = Counter()
        after_occurrences: Counter[str] = Counter()
        for ordinal, event_value in enumerate(row.get("generated_image_events", [])):
            event = dict(event_value)
            original = str(event.get("original", ""))
            replacement = str(event.get("replacement", ""))
            if not replacement:
                # A deletion has no reliable post-normalization anchor.  Its
                # exact artifact-only behavior is closed by deterministic
                # audit; Luna samples only events it can compare locally.
                continue
            before_position = nth_index(image_input, original, before_occurrences[original])
            before_occurrences[original] += 1
            after_position = nth_index(normalized, replacement, after_occurrences[replacement])
            after_occurrences[replacement] += 1
            is_description = bool(event.get("image_description_commented"))
            if is_description and after_position < 0:
                # A separately audited follow-up repetition span can remove a
                # complete repeated provenance comment.  Do not mis-localize
                # that image event to the beginning of the document.
                continue
            before_position = max(0, before_position)
            after_position = max(0, after_position)
            focus_end = min(len(normalized), after_position + max(1, min(len(replacement), 400)))
            anchor = unique_context_anchor(normalized, after_position, focus_end) if replacement else ""
            candidates.append(
                make_region(
                    row=row,
                    card=card,
                    family=(
                        "removed_image_description_provenance"
                        if is_description
                        else "generated_image_cleanup"
                    ),
                    rules=[str(event["rule"])],
                    risk_tier="routine",
                    ordinal=ordinal,
                    before=around(
                        image_input,
                        before_position,
                        before_position + len(original),
                        maximum=2_600,
                    ),
                    after=around(
                        normalized,
                        after_position,
                        after_position + len(replacement),
                        maximum=2_600,
                    ),
                    focus_anchor=anchor,
                    expected=(
                        "Remove only the generated image filename and wrapper; preserve the complete existing description inside valid description-of-removed-image comments, keep embedded pipeline markers unnested, and add no narration."
                        if is_description
                        else "Remove the generated image target and wrapper while preserving any ordinary link label and adding no narration."
                    ),
                )
            )

        for ordinal, rule in enumerate(sorted(key for key, value in transformations.items() if value)):
            if rule.startswith("table_fallback_reason_") or rule in {
                "html_tables_fallback_to_text",
                "html_table_fallback_cells_preserved",
                "orphan_table_cells_flattened",
                "table_comments_relocated_after_table",
            }:
                continue
            before_position, after_position, anchor = aligned_rule_focus(rule, cleaned, normalized)
            candidates.append(
                make_region(
                    row=row,
                    card=card,
                    family=routine_family(rule),
                    rules=[rule],
                    risk_tier="routine",
                    ordinal=ordinal,
                    before=around(cleaned, before_position, before_position + len(anchor), maximum=2_600),
                    after=around(normalized, after_position, after_position + len(anchor), maximum=2_600),
                    focus_anchor=anchor,
                    expected="Retain readable text and existing Markdown, remove HTML syntax and attributes, and express only supported structure as valid GFM.",
                )
            )
    unique: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        unique[str(candidate["region_id"])] = candidate
    return list(unique.values())


def stable_rank(candidate: Mapping[str, object]) -> str:
    return sha256_text("gfm-luna-v1\0" + str(candidate["region_id"]))


def logical_region_identity(region: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(region["opaque_id"]),
        str(region["transformation_family"]),
        tuple(str(rule) for rule in region["rule_ids"]),
        str(region["risk_tier"]),
        int(region["ordinal"]),
    )


def select_revalidation_regions(
    candidates: Sequence[dict[str, object]], baseline: Mapping[str, object]
) -> tuple[list[dict[str, object]], int]:
    current: dict[tuple[object, ...], dict[str, object]] = {}
    for candidate in candidates:
        identity = logical_region_identity(candidate)
        if identity in current:
            raise ValueError(f"duplicate current logical region: {identity}")
        current[identity] = candidate
    selected: list[dict[str, object]] = []
    reused = 0
    for prior_value in baseline["regions"]:
        prior = dict(prior_value)
        identity = logical_region_identity(prior)
        candidate = current.get(identity)
        if candidate is None:
            raise ValueError(f"baseline logical region is absent from current candidates: {identity}")
        unchanged = (
            candidate["before_sha256"] == prior["before_sha256"]
            and candidate["after_sha256"] == prior["after_sha256"]
        )
        if bool(prior.get("validated")) and unchanged:
            reused += 1
        else:
            selected.append(candidate)
    return sorted(selected, key=lambda row: (row["risk_tier"] != "high", str(row["source_id"]), str(row["region_id"]))), reused


def select_regions(candidates: Sequence[dict[str, object]], target: int) -> list[dict[str, object]]:
    selected: dict[str, dict[str, object]] = {
        str(row["region_id"]): row for row in candidates if row["risk_tier"] == "high"
    }
    remaining = sorted(
        (row for row in candidates if str(row["region_id"]) not in selected),
        key=stable_rank,
    )

    covered_rules = {rule for row in selected.values() for rule in row["rule_ids"]}
    for row in remaining:
        if len(selected) >= target:
            break
        if any(rule not in covered_rules for rule in row["rule_ids"]):
            selected[str(row["region_id"])] = row
            covered_rules.update(row["rule_ids"])

    covered_sources = {str(row["source_id"]) for row in selected.values()}
    for row in remaining:
        if len(selected) >= target:
            break
        if str(row["region_id"]) not in selected and str(row["source_id"]) not in covered_sources:
            selected[str(row["region_id"])] = row
            covered_sources.add(str(row["source_id"]))

    by_family: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in remaining:
        if str(row["region_id"]) not in selected:
            by_family[str(row["transformation_family"])].append(row)
    families = sorted(by_family)
    while len(selected) < target and any(by_family.values()):
        for family in families:
            if len(selected) >= target:
                break
            if by_family[family]:
                row = by_family[family].pop(0)
                selected[str(row["region_id"])] = row
    return sorted(selected.values(), key=lambda row: (row["risk_tier"] != "high", str(row["source_id"]), str(row["region_id"])))


def make_request(region: Mapping[str, object], slot: str, normalizer_sha256: str) -> dict[str, object]:
    before, before_redactions = redact_direct_identifiers(str(region["before_text"]))
    after, after_redactions = redact_direct_identifiers(str(region["after_text"]))
    focus_anchor, focus_redactions = redact_direct_identifiers(str(region.get("focus_anchor", "")))
    review_id = sha256_text(
        canonical_json(
            {
                "region_id": region["region_id"],
                "reviewer_slot": slot,
                "normalizer_sha256": normalizer_sha256,
                "before_sha256": region["before_sha256"],
                "after_sha256": region["after_sha256"],
            }
        )
    )
    return {
        "schema_version": REQUEST_SCHEMA,
        "review_id": review_id,
        "region_id": region["region_id"],
        "reviewer_slot": slot,
        "normalizer_sha256": normalizer_sha256,
        "opaque_id": region["opaque_id"],
        "source_dataset": region["source_dataset"],
        "source_doc_id": region["source_doc_id"],
        "document_path": region["document_path"],
        "risk_tier": region["risk_tier"],
        "transformation_family": region["transformation_family"],
        "rule_ids": region["rule_ids"],
        "expected_behavior": region["expected_behavior"],
        "focus_anchor": focus_anchor,
        "before_sha256": region["before_sha256"],
        "after_sha256": region["after_sha256"],
        "before_text": before,
        "after_text": after,
        "redaction_counts": {
            "before": before_redactions,
            "after": after_redactions,
            "focus_anchor": focus_redactions,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--target-regions", type=int, default=DEFAULT_TARGET_REGIONS)
    parser.add_argument("--revalidate-from", type=Path)
    args = parser.parse_args(argv)

    site_dir = args.site_dir.resolve()
    audit = read_json(site_dir / "data/gfm_normalization_audit.json")
    if audit.get("status") != "passed":
        raise ValueError("normalization audit is not passed")
    normalizer_sha256 = hashlib.sha256(args.normalizer.read_bytes()).hexdigest()
    normalizer = load_normalizer(args.normalizer)
    candidates = build_candidates(site_dir, audit, normalizer=normalizer)
    reused_validated_regions = 0
    if args.revalidate_from is not None:
        selected, reused_validated_regions = select_revalidation_regions(
            candidates, read_json(args.revalidate_from)
        )
    else:
        selected = select_regions(candidates, args.target_regions)
    requests: list[dict[str, object]] = []
    for region in selected:
        requests.append(make_request(region, "primary", normalizer_sha256))
        if region["risk_tier"] == "high":
            requests.append(make_request(region, "secondary", normalizer_sha256))
    write_json_atomic(args.regions, {"schema_version": SCHEMA, "regions": selected})
    write_jsonl_atomic(args.requests, requests)
    summary = {
        "schema_version": SCHEMA,
        "status": "passed",
        "candidate_regions": len(candidates),
        "selected_regions": len(selected),
        "high_risk_regions": sum(row["risk_tier"] == "high" for row in selected),
        "review_requests": len(requests),
        "normalizer_sha256": normalizer_sha256,
        "families": dict(sorted(Counter(str(row["transformation_family"]) for row in selected).items())),
        "sources": dict(sorted(Counter(str(row["source_id"]) for row in selected).items())),
        "requests_sha256": hashlib.sha256(args.requests.read_bytes()).hexdigest(),
        "revalidation_mode": args.revalidate_from is not None,
        "reused_validated_regions": reused_validated_regions,
    }
    write_json_atomic(args.summary, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
