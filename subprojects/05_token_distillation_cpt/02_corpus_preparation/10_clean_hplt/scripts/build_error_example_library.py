#!/usr/bin/env python3
"""Build a browsable library of reviewed HPLT cleaning error examples.

The output is presentation-oriented:

error_examples/
  README.md
  E001_replacement_control_chars/
    ERROR.md
    01_example.txt

The source HPLT rows are not modified. The generated text files contain the
original document text plus lightweight marker tags around reviewed spans or
markers when those boundaries are known.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DEFAULT_OUTPUT = ROOT / "error_examples"


CONFIDENT_ERROR_IDS = (
    "E001",
    "E002",
    "E003",
    "E013",
    "E018",
)


CONFIDENT_ALLOWED_ACTIONS = {
    "E001": {"normalize_or_trim_span"},
    "E002": {"normalize_or_trim_span"},
    "E003": {"normalize_or_trim_span", "trim_span", "trim_prefix", "trim_suffix"},
    "E013": {"trim_span"},
    "E018": {"trim_span", "trim_suffix"},
}


PRESENTATION_CATEGORIES = [
    {
        "id": "C1",
        "name": "Detected Residue Pattern Removal",
        "description": "Remove a concrete pattern that visibly does not belong in the document text, such as `NEWLINENEWLINE`, U+FFFD, script/style markup, or a duplicate extraction span.",
        "error_ids": ("E002", "E001", "E003", "E013", "E018"),
    },
]


PRESENTATION_RANKING = {
    "E002": {
        "category": "C1",
        "rank": 1,
        "transformation": "Exact placeholder normalization.",
        "fp_confidence": "very high",
        "fn_confidence": "low",
        "confidence_note": "A literal reviewed placeholder such as `NEWLINENEWLINE` is extremely unlikely to be valid prose, but singleton evidence means this catches only known placeholder forms.",
    },
    "E001": {
        "category": "C1",
        "rank": 2,
        "transformation": "Exact bad-codepoint normalization/removal.",
        "fp_confidence": "very high",
        "fn_confidence": "medium",
        "confidence_note": "Literal U+FFFD/control/private-use characters are precise targets; the rule does not cover broader encoding or OCR corruption.",
    },
    "E003": {
        "category": "C1",
        "rank": 3,
        "transformation": "Exact markup/script/style/CMS residue trim.",
        "fp_confidence": "high",
        "fn_confidence": "medium",
        "confidence_note": "Literal HTML/CSS/JS residue is usually mechanical extraction dirt, but code examples and technology articles are real controls.",
    },
    "E013": {
        "category": "C1",
        "rank": 4,
        "transformation": "Exact duplicate-span removal.",
        "fp_confidence": "medium-high",
        "fn_confidence": "medium",
        "confidence_note": "Exact repeated spans are strong evidence, but deliberate repetition and formulaic text remain false-positive controls.",
    },
    "E018": {
        "category": "C1",
        "rank": 5,
        "transformation": "Duplicated body/tail removal while preserving one copy.",
        "fp_confidence": "medium-high",
        "fn_confidence": "medium-low",
        "confidence_note": "Very safe when the duplicate body is exact, less safe when boilerplate separates copies or choosing the best copy needs judgment.",
    },
}


ERRORS = OrderedDict(
    [
        (
            "E001",
            {
                "slug": "replacement_control_chars",
                "name": "Replacement/control/private-use characters",
                "definition": "Literal replacement, control, or private-use characters left inside otherwise readable text.",
                "scope": "Mostly localized normalization/span work; not a whole-document class unless the residue dominates.",
                "fp_boundary": "A document is not invalid merely because it contains a rare U+FFFD character or other isolated codepoint residue.",
            },
        ),
        (
            "E002",
            {
                "slug": "escaped_placeholder_residue",
                "name": "Escaped Unicode, HTML entities, percent-encoding residue",
                "definition": "Escaped placeholders or encoded residue that should be deterministically normalized when the mapping is exact.",
                "scope": "Currently singleton evidence; keep this narrow until more reviewed rows exist.",
                "fp_boundary": "Do not invent semantic repairs when the encoded form is ambiguous.",
            },
        ),
        (
            "E003",
            {
                "slug": "markup_code_remnants",
                "name": "CSS/JS/HTML/XML remnants",
                "definition": "Extraction residue from scripts, styles, XML, HTML, CMS shortcodes, or layout tags.",
                "scope": "Only small removable residue spans are active here; pure extraction failures are outside the C1 presentation.",
                "fp_boundary": "Code examples, legal markup references, and recoverable article bodies must be preserved.",
            },
        ),
        (
            "E013",
            {
                "slug": "internal_repetition_loops",
                "name": "Internal paragraph or sentence repetition loops",
                "definition": "Duplicated sentence, paragraph, note, or block repeated inside the same row by extraction failure.",
                "scope": "Localized trim class: preserve one good copy.",
                "fp_boundary": "Poems, songs, legal formulas, rhetorical prose, and deliberate repetition are controls.",
            },
        ),
        (
            "E018",
            {
                "slug": "duplicated_body_boilerplate",
                "name": "Duplicated main body separated by boilerplate",
                "definition": "A main body or tail block repeated with boilerplate or metadata between copies.",
                "scope": "Document-structure repetition; preserve one good copy and remove duplicate residue.",
                "fp_boundary": "Do not remove deliberate refrains, citations, or repeated headings that carry structure.",
            },
        ),
    ]
)


STRICT_POLICY = {
    "E001": {
        "tier": "strict_action_candidate",
        "role": "Exact bad-codepoint normalization.",
        "allowed_under_strict_principles": "Automatic candidate only for literal U+FFFD/control/private-use characters with exact character positions.",
    },
    "E002": {
        "tier": "strict_action_candidate",
        "role": "Exact placeholder/escape normalization.",
        "allowed_under_strict_principles": "Automatic candidate only for deterministic placeholders such as the reviewed `NEWLINENEWLINE` case.",
    },
    "E003": {
        "tier": "strict_action_candidate",
        "role": "Literal markup/script/CMS residue removal.",
        "allowed_under_strict_principles": "Automatic candidate only for exact markup/script/shortcode spans, not for content about code.",
    },
    "E013": {
        "tier": "strict_action_candidate",
        "role": "Exact duplicate-span removal.",
        "allowed_under_strict_principles": "Automatic candidate only for exact/near-exact duplicate spans where one complete copy remains.",
    },
    "E018": {
        "tier": "strict_action_candidate",
        "role": "Exact duplicated body/tail removal.",
        "allowed_under_strict_principles": "Automatic candidate only when duplicate blocks are explicit and one good copy is preserved.",
    },
}


REVIEW_GUIDANCE = {
    "E001": {
        "current_evidence": "Confirmed rows contain literal U+FFFD replacement characters, plus the class definition also covers private-use/control characters if later found.",
        "look_for": [
            "Visible replacement glyphs, control-code residue, or private-use characters embedded in otherwise readable text.",
            "A localized bad character whose removal or deterministic replacement preserves the surrounding sentence.",
        ],
        "not_this": [
            "A noisy or low-quality document merely because it has one rare bad character.",
            "OCR/encoding corruption that affects many normal-looking Greek letters; keep only isolated exact bad codepoints here.",
        ],
        "review_decision": "Use `normalize_or_trim_span` only for localized bad character positions; otherwise leave unmodified for review.",
    },
    "E002": {
        "current_evidence": "The current reviewed evidence is specifically literal `NEWLINENEWLINE` placeholders in a legal-text row. The broader label is reserved for similarly deterministic escaped/placeholder residue, but that broader class is not yet broadly evidenced.",
        "look_for": [
            "Machine placeholders such as literal `NEWLINE`, `NEWLINENEWLINE`, escaped unicode, HTML entities, or percent-encoding that appear as text rather than formatting.",
            "A one-to-one deterministic repair: for example placeholder text that should become whitespace or a normal character.",
        ],
        "not_this": [
            "Valid legal/source notation, URL text, or programming/code examples.",
            "Ambiguous escapes where the intended text cannot be recovered mechanically.",
        ],
        "review_decision": "Approve only exact deterministic normalization; otherwise leave unmodified for review rather than guessing.",
    },
    "E003": {
        "current_evidence": "Reviewed examples include ad/script snippets and CMS shortcode residue inside otherwise useful articles.",
        "look_for": [
            "Literal JavaScript/CSS/HTML/XML fragments, ad tags, shortcode names, layout tags, or CMS control text in the extracted body.",
            "Markup spans whose start and end can be selected without deleting the article.",
        ],
        "not_this": [
            "Articles about programming/web technology where code is the real content.",
            "Plain punctuation or braces in legitimate text.",
        ],
        "review_decision": "Prefer `trim_span` or deterministic normalization; pure extraction failures with no usable body are outside the active C1 transformation set.",
    },
    "E013": {
        "current_evidence": "Reviewed examples include duplicated notes or duplicated opening sentences caused by extraction repetition.",
        "look_for": [
            "An identical or near-identical sentence/paragraph/note repeated by extraction failure.",
            "A second copy whose removal leaves one complete version of the content.",
        ],
        "not_this": [
            "Deliberate refrain, poetry, song lyrics, legal formula repetition, rhetorical repetition, or repeated table labels.",
            "Two different but similar paragraphs.",
        ],
        "review_decision": "Usually `trim_span`; preserve the first or best copy and only remove the redundant duplicate.",
    },
    "E018": {
        "current_evidence": "Reviewed examples include repeated credit/tail/body blocks separated by boilerplate.",
        "look_for": [
            "A main body, credit block, note, or tail repeated after boilerplate or separator text.",
            "Two copies where one can be removed while preserving complete content.",
        ],
        "not_this": [
            "Intentional refrain, repeated source citations, repeated headings in a structured document, or article summaries that are not duplicates.",
        ],
        "review_decision": "Use span/suffix trim only when one complete copy is unambiguously preserved; otherwise leave unmodified for review.",
    },
}


PRIORITY_SOURCE_DOCS = {
    "E001": [
        "hplt::8_1.jsonl.zst::2875495ae31f87a51541a6545b1f0044",
        "hplt::8_1.jsonl.zst::37390486240cefe6edaae18793baa139",
    ],
    "E002": ["hplt::8_1.jsonl.zst::c03d84ae0889c6d7dfbef47c370d0988"],
    "E003": [
        "hplt::8_2.jsonl.zst::9be7c5d51863bde76c3743b975c9f3bd",
        "hplt::9_1.jsonl.zst::01b125fe9c6c5ca695127c0bff963a5d",
    ],
    "E013": ["hplt::8_1.jsonl.zst::530781bdead0f36e586733123e9eb86d"],
    "E018": ["hplt::8_1.jsonl.zst::20b5b15b8f661eabdcb26bf0d6b84dbe"],
}


REVIEWED_PATTERNS = [
    "reports/*annotations*.jsonl",
    "reports/*decision_manifest.jsonl",
    "reports/*evidence*.jsonl",
]

TEXT_SOURCE_PATTERNS = REVIEWED_PATTERNS + [
    "reports/*review_pack*.jsonl",
    "reports/*validation_slice*.jsonl",
    "reports/*shadow_records.jsonl",
]


@dataclass
class EvidenceRecord:
    source_doc_id: str
    error_ids: list[str]
    action: str | None
    host: str | None
    url: str | None
    quality_bin: Any
    notes: str
    label: str | None
    source_file: str
    source_line: int
    raw: dict[str, Any] = field(repr=False)


@dataclass
class TextSource:
    text: str | None = None
    doc_text_path: str | None = None
    source_file: str | None = None
    source_line: int | None = None


def iter_jsonl(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield line_no, json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"warn: skip invalid json {path}:{line_no}: {exc}", file=sys.stderr)
    except FileNotFoundError:
        return


def source_doc_id(obj: dict[str, Any]) -> str | None:
    for key in ("parent_source_doc_id", "source_doc_id", "doc_key"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def action_of(obj: dict[str, Any]) -> str | None:
    for key in ("review_correct_action", "reviewed_action", "boundary_review_action", "action", "candidate_action"):
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def notes_of(obj: dict[str, Any]) -> str:
    for key in ("review_notes", "boundary_notes", "evidence_excerpt", "review_span_or_split_notes"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    contexts = obj.get("e001_contexts")
    if isinstance(contexts, list):
        parts = []
        for context in contexts[:3]:
            if isinstance(context, dict) and context.get("context"):
                parts.append(str(context["context"]))
        if parts:
            return " | ".join(parts)
    return ""


def reviewed_error_ids(obj: dict[str, Any], path: Path) -> list[str]:
    label = obj.get("review_label")
    if label in {"false_positive", "clean"}:
        return []

    ids: list[str] = []
    for key in ("review_true_error_type_ids", "true_error_type_ids", "candidate_error_type_ids_reviewed_as_true"):
        value = obj.get(key)
        if isinstance(value, list):
            ids.extend(str(v) for v in value)

    boundary_status = obj.get("boundary_review_status")
    action_status = obj.get("action_status")
    if boundary_status in {"reviewed_accept", "reviewed_quarantine"} or action_status in {"applied", "quarantine"}:
        value = obj.get("error_type_ids")
        if isinstance(value, list):
            ids.extend(str(v) for v in value)

    if path.name.endswith("_evidence.jsonl"):
        value = obj.get("error_type_ids")
        if isinstance(value, list):
            ids.extend(str(v) for v in value)

    return sorted({v for v in ids if re.fullmatch(r"E\d{3}", v) and v in ERRORS})


def embedded_document_text(obj: dict[str, Any]) -> str | None:
    for key in ("document_text", "text"):
        value = obj.get(key)
        if isinstance(value, str):
            return value

    # Review packs use full_text for the original source row. Shadow records
    # also use full_text, but that is already-modified text and should not be
    # preferred for source examples.
    value = obj.get("full_text")
    if isinstance(value, str) and not obj.get("is_shadow_record"):
        return value

    review_windows = obj.get("review_windows")
    if isinstance(review_windows, list):
        for window in review_windows:
            if not isinstance(window, dict):
                continue
            text = window.get("text")
            if isinstance(text, str) and window.get("label") == "full_text":
                return text
        for window in review_windows:
            if not isinstance(window, dict):
                continue
            text = window.get("text")
            if isinstance(text, str):
                return text
    return None


def collect_text_sources() -> dict[str, TextSource]:
    by_source: dict[str, TextSource] = {}
    for pattern in TEXT_SOURCE_PATTERNS:
        for path in ROOT.glob(pattern):
            for line_no, obj in iter_jsonl(path):
                src = source_doc_id(obj)
                if not src:
                    continue
                current = by_source.setdefault(src, TextSource())
                embedded_text = embedded_document_text(obj)
                if isinstance(embedded_text, str) and not current.text:
                    current.text = embedded_text
                    current.source_file = str(path.relative_to(ROOT))
                    current.source_line = line_no
                doc_text_path = obj.get("doc_text_path")
                if isinstance(doc_text_path, str) and doc_text_path and not current.doc_text_path:
                    current.doc_text_path = doc_text_path
                    current.source_file = current.source_file or str(path.relative_to(ROOT))
                    current.source_line = current.source_line or line_no
    return by_source


def collect_evidence_records() -> list[EvidenceRecord]:
    records: list[EvidenceRecord] = []
    for pattern in REVIEWED_PATTERNS:
        for path in ROOT.glob(pattern):
            for line_no, obj in iter_jsonl(path):
                src = source_doc_id(obj)
                if not src:
                    continue
                ids = reviewed_error_ids(obj, path)
                if not ids:
                    continue
                records.append(
                    EvidenceRecord(
                        source_doc_id=src,
                        error_ids=ids,
                        action=action_of(obj),
                        host=obj.get("host") if isinstance(obj.get("host"), str) else None,
                        url=obj.get("url") if isinstance(obj.get("url"), str) else None,
                        quality_bin=obj.get("quality_bin"),
                        notes=notes_of(obj),
                        label=obj.get("review_label") or obj.get("boundary_review_status") or obj.get("action_status"),
                        source_file=str(path.relative_to(ROOT)),
                        source_line=line_no,
                        raw=obj,
                    )
                )
    return records


def remote_cat(path: str) -> str | None:
    if not path.startswith("/"):
        return None
    try:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "clariden", "cat", path],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def read_text_for(record: EvidenceRecord, text_sources: dict[str, TextSource]) -> tuple[str | None, str]:
    embedded = embedded_document_text(record.raw)
    if isinstance(embedded, str):
        return embedded, f"{record.source_file}:{record.source_line}:embedded_text"

    local_path = record.raw.get("doc_text_path")
    candidates = [local_path] if isinstance(local_path, str) and local_path else []
    source = text_sources.get(record.source_doc_id)
    if source:
        if source.text is not None:
            return source.text, f"{source.source_file}:{source.source_line}:embedded_text"
        if source.doc_text_path:
            candidates.append(source.doc_text_path)

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        p = Path(candidate)
        if p.exists():
            return p.read_text(encoding="utf-8", errors="replace"), str(p)
        remote_text = remote_cat(candidate)
        if remote_text is not None:
            return remote_text, f"clariden:{candidate}"
    return None, "missing"


def score_record(record: EvidenceRecord, error_id: str, text_sources: dict[str, TextSource]) -> tuple[int, str]:
    score = 0
    priority = PRIORITY_SOURCE_DOCS.get(error_id, [])
    if record.source_doc_id in priority:
        score += 10_000 - priority.index(record.source_doc_id)
    if record.label in {"reviewed_accept", "applied", "true_positive", "partial_true_positive", "false_negative_found"}:
        score += 500
    if record.action in {"trim_suffix", "trim_prefix", "trim_span", "normalize_or_trim_span"}:
        score += 100
    if record.host:
        score += 25
    if record.notes:
        score += min(80, len(record.notes) // 20)
    if record.raw.get("good_text_loss_estimate") == 0.0:
        score += 20
    if error_id == "E001" and record.raw.get("e001_contexts"):
        score += 1_000
    source = text_sources.get(record.source_doc_id)
    if record.raw.get("document_text") or (source and (source.text or source.doc_text_path)) or record.raw.get("doc_text_path"):
        score += 150
    if "boundary_manual" in record.source_file:
        score += 50
    if "materialization" in record.source_file:
        score += 20
    return score, record.source_file


def slugify(value: str | None, fallback: str = "unknown") -> str:
    if not value:
        return fallback
    value = value.lower()
    value = re.sub(r"^www\.", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return value or fallback


def shard_and_hash(source_doc_id: str) -> tuple[str, str]:
    match = re.search(r"hplt::([^:]+)::([0-9a-fA-F]+)$", source_doc_id)
    if match:
        shard = match.group(1).replace(".jsonl.zst", "").replace("_", "-")
        return shard, match.group(2)[:12]
    digest = hashlib.sha256(source_doc_id.encode("utf-8")).hexdigest()
    return "source", digest[:12]


def marker_strings(record: EvidenceRecord) -> list[str]:
    notes = record.notes or ""
    markers: list[str] = []
    for pattern in [
        r"`([^`]{2,120})`",
        r"marker '([^']{2,120})'",
        r'marker "([^"]{2,120})"',
        r"near \"([^\"]{2,120})\"",
        r"around `([^`]{2,120})`",
    ]:
        for match in re.finditer(pattern, notes):
            markers.append(match.group(1))
    if "U+FFFD" in notes:
        markers.append("\ufffd")
    # Longer literal markers are more specific.
    deduped = []
    for marker in sorted(markers, key=len, reverse=True):
        if marker not in deduped:
            deduped.append(marker)
    return deduped


def range_tags(record: EvidenceRecord, text_len: int) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    for key in ("span_ranges", "dropped_span_ranges"):
        value = record.raw.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, dict):
                continue
            start, end = item.get("start"), item.get("end")
            if isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= text_len:
                ranges.append((start, end, key))
    return ranges


def insert_tags(text: str, record: EvidenceRecord, error_id: str) -> tuple[str, str]:
    ranges = range_tags(record, len(text))
    tag_notes: list[str] = []

    if ranges:
        inserts: list[tuple[int, str]] = []
        for start, end, kind in ranges:
            inserts.append((start, f"[[ERROR_{error_id}_START kind={kind}]]"))
            inserts.append((end, f"[[ERROR_{error_id}_END kind={kind}]]"))
        for pos, marker in sorted(inserts, key=lambda item: item[0], reverse=True):
            text = text[:pos] + marker + text[pos:]
        tag_notes.append(f"tagged {len(ranges)} reviewed character range(s)")
        return text, "; ".join(tag_notes)

    if error_id == "E001":
        contexts = record.raw.get("e001_contexts")
        if isinstance(contexts, list):
            inserts: list[tuple[int, str]] = []
            for context in contexts:
                if not isinstance(context, dict):
                    continue
                index = context.get("char_index")
                if not isinstance(index, int) or index < 0:
                    continue
                codepoint = context.get("codepoint") or "U+FFFD"
                if index < len(text) and text[index] == "\ufffd":
                    inserts.append((index, f"[[ERROR_{error_id}_START codepoint={codepoint}]]"))
                    inserts.append((index + 1, f"[[ERROR_{error_id}_END codepoint={codepoint}]]"))
                elif index <= len(text):
                    # The reviewed E001 shadow overlay stores text after the bad
                    # character was removed. Reinsert the reviewed codepoint as
                    # a tag-local marker so the example still shows the source
                    # error without changing any HPLT source row.
                    bad_char = "\ufffd" if codepoint == "U+FFFD" else ""
                    inserts.append((index, f"[[ERROR_{error_id}_START codepoint={codepoint}]]{bad_char}[[ERROR_{error_id}_END codepoint={codepoint}]]"))
            if inserts:
                for pos, marker in sorted(inserts, key=lambda item: item[0], reverse=True):
                    text = text[:pos] + marker + text[pos:]
                return text, "tagged reviewed E001 character position(s)"

    for marker in marker_strings(record):
        idx = text.find(marker)
        if idx == -1:
            continue
        end = idx + len(marker)
        if record.action == "trim_suffix":
            text = text[:idx] + f"[[ERROR_{error_id}_START marker={json.dumps(marker, ensure_ascii=False)}]]" + text[idx:] + f"[[ERROR_{error_id}_END marker={json.dumps(marker, ensure_ascii=False)}]]"
            tag_notes.append("tagged suffix from reviewed literal marker to document end")
        elif record.action == "trim_prefix":
            text = f"[[ERROR_{error_id}_START marker={json.dumps(marker, ensure_ascii=False)}]]" + text[:end] + f"[[ERROR_{error_id}_END marker={json.dumps(marker, ensure_ascii=False)}]]" + text[end:]
            tag_notes.append("tagged prefix/marker residue")
        else:
            text = text[:idx] + f"[[ERROR_{error_id}_START marker={json.dumps(marker, ensure_ascii=False)}]]" + text[idx:end] + f"[[ERROR_{error_id}_END marker={json.dumps(marker, ensure_ascii=False)}]]" + text[end:]
            tag_notes.append("tagged reviewed literal marker")
        return text, "; ".join(tag_notes)

    return text, "no exact span marker available; inspect metadata notes"


def short_notes(notes: str, limit: int = 500) -> str:
    notes = re.sub(r"\s+", " ", notes).strip()
    if len(notes) <= limit:
        return notes
    return notes[: limit - 3].rstrip() + "..."


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def choose_examples(
    records: list[EvidenceRecord],
    text_sources: dict[str, TextSource],
    error_id: str,
    max_examples: int,
) -> list[tuple[EvidenceRecord, str, str]]:
    allowed_actions = CONFIDENT_ALLOWED_ACTIONS.get(error_id)
    candidates = [
        record
        for record in records
        if error_id in record.error_ids
        and (allowed_actions is None or record.action in allowed_actions)
    ]
    candidates.sort(key=lambda r: score_record(r, error_id, text_sources), reverse=True)
    selected: list[tuple[EvidenceRecord, str, str]] = []
    seen_sources: set[str] = set()
    for record in candidates:
        if record.source_doc_id in seen_sources:
            continue
        text, text_source = read_text_for(record, text_sources)
        if text is None:
            continue
        selected.append((record, text, text_source))
        seen_sources.add(record.source_doc_id)
        if len(selected) >= max_examples:
            break
    return selected


def build_library(output_dir: Path, max_examples: int) -> dict[str, int]:
    text_sources = collect_text_sources()
    records = collect_evidence_records()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest_rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {}

    for error_id in CONFIDENT_ERROR_IDS:
        meta = ERRORS[error_id]
        folder = output_dir / f"{error_id}_{meta['slug']}"
        folder.mkdir(parents=True)
        examples = choose_examples(records, text_sources, error_id, max_examples)
        counts[error_id] = len(examples)
        guidance = REVIEW_GUIDANCE[error_id]
        policy = STRICT_POLICY[error_id]
        ranking = PRESENTATION_RANKING[error_id]
        category = next(item for item in PRESENTATION_CATEGORIES if item["id"] == ranking["category"])

        lines = [
            f"# {error_id} - {meta['name']}",
            "",
            f"- Presentation category: `{category['id']}` - {category['name']}",
            f"- Rank within category: {ranking['rank']}",
            f"- Proposed transformation: {ranking['transformation']}",
            f"- False-positive confidence: {ranking['fp_confidence']}",
            f"- False-negative confidence: {ranking['fn_confidence']}",
            f"- Confidence note: {ranking['confidence_note']}",
            f"- Definition: {meta['definition']}",
            f"- Scope: {meta['scope']}",
            f"- False-positive boundary: {meta['fp_boundary']}",
            f"- Strict policy tier: `{policy['tier']}`",
            f"- Strict role: {policy['role']}",
            f"- Allowed under strict principles: {policy['allowed_under_strict_principles']}",
            f"- Current evidence: {guidance['current_evidence']}",
            f"- Review decision: {guidance['review_decision']}",
            f"- Generated examples: {len(examples)}",
            "",
            "## What To Look For",
            "",
        ]
        for item in guidance["look_for"]:
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "## What Not To Count",
                "",
            ]
        )
        for item in guidance["not_this"]:
            lines.append(f"- {item}")
        lines.extend(
            [
                "",
                "## Example Files",
                "",
            ]
        )

        for index, (record, raw_text, text_source) in enumerate(examples, 1):
            tagged_text, tagging_note = insert_tags(raw_text, record, error_id)
            shard, short_hash = shard_and_hash(record.source_doc_id)
            action_slug = slugify(record.action, "action")
            host_slug = slugify(record.host, "unknown-host")
            file_name = f"{index:02d}_{host_slug}_{shard}_{short_hash}_{action_slug}.txt"
            out_path = folder / file_name
            header = [
                "# HPLT cleaning error example",
                f"error_id: {error_id}",
                f"error_name: {meta['name']}",
                f"presentation_category: {category['id']} - {category['name']}",
                f"rank_within_category: {ranking['rank']}",
                f"proposed_transformation: {ranking['transformation']}",
                f"false_positive_confidence: {ranking['fp_confidence']}",
                f"false_negative_confidence: {ranking['fn_confidence']}",
                f"strict_policy_tier: {policy['tier']}",
                f"strict_policy_role: {policy['role']}",
                f"allowed_under_strict_principles: {policy['allowed_under_strict_principles']}",
                f"source_doc_id: {record.source_doc_id}",
                f"host: {record.host or ''}",
                f"url: {record.url or ''}",
                f"quality_bin: {record.quality_bin if record.quality_bin is not None else ''}",
                f"reviewed_action: {record.action or ''}",
                f"review_label: {record.label or ''}",
                f"review_source: {record.source_file}:{record.source_line}",
                f"text_source: {text_source}",
                f"tagging: {tagging_note}",
                f"notes: {short_notes(record.notes)}",
                "",
                "[[DOCUMENT_TEXT_UTF8]]",
                "",
            ]
            write_text(out_path, "\n".join(header) + tagged_text)
            lines.append(
                f"- `{file_name}` - `{record.host or 'unknown-host'}`; action `{record.action or ''}`; "
                f"source `{record.source_doc_id}`; tagging: {tagging_note}."
            )
            manifest_rows.append(
                {
                    "error_id": error_id,
                    "error_name": meta["name"],
                    "presentation_category_id": category["id"],
                    "presentation_category_name": category["name"],
                    "rank_within_category": ranking["rank"],
                    "proposed_transformation": ranking["transformation"],
                    "false_positive_confidence": ranking["fp_confidence"],
                    "false_negative_confidence": ranking["fn_confidence"],
                    "confidence_note": ranking["confidence_note"],
                    "strict_policy_tier": policy["tier"],
                    "strict_policy_role": policy["role"],
                    "allowed_under_strict_principles": policy["allowed_under_strict_principles"],
                    "file": str(out_path.relative_to(output_dir)),
                    "source_doc_id": record.source_doc_id,
                    "host": record.host,
                    "url": record.url,
                    "action": record.action,
                    "review_source": f"{record.source_file}:{record.source_line}",
                    "text_source": text_source,
                    "tagging": tagging_note,
                }
            )

        if not examples:
            lines.extend(
                [
                    "- No full-text example was generated. Reviewed evidence exists, but the generator could not resolve a document text source.",
                ]
            )
        write_text(folder / "ERROR.md", "\n".join(lines) + "\n")

    root_lines = [
        "# HPLT Cleaning Error Example Library",
        "",
        "This generated library is limited to the cleaning categories we are",
        "currently confident enough to keep in the active approach. It gives each",
        "retained evidence class a stable folder, a short description file, and",
        "concrete UTF-8 text files containing the underlying HPLT document text.",
        "The original HPLT rows are not modified.",
        "",
        "The governing policy is `../STRICT_CLEANING_PRINCIPLES.md`. The dropped",
        "observed surfaces are intentionally absent from this presentation library.",
        "",
        "## Naming Scheme",
        "",
        "- Folder: `E001_replacement_control_chars/`",
        "- Description: `ERROR.md`",
        "- Example file: `01_host_shard_hash_action.txt`",
        "- Tags: `[[ERROR_E003_START]]...[[ERROR_E003_END]]` or",
        "  `[[ERROR_E013_START]]...[[ERROR_E013_END]]` when exact reviewed",
        "  evidence is available.",
        "",
        "## Active Presentation Category And Ranking",
        "",
        "The active category is precision-first: higher rank means",
        "higher confidence that a fired rule is not a false positive. The",
        "false-negative confidence column states how much coverage we expect from",
        "the narrow rule; very precise rules can still have low coverage.",
        "",
    ]

    def append_error_ids(error_ids: tuple[str, ...]) -> None:
        root_lines.extend(
            [
                "| Rank | Evidence Class | Proposed Transformation | FP Confidence | FN Confidence | Note |",
                "| ---: | --- | --- | --- | --- | --- |",
            ]
        )
        for error_id in error_ids:
            meta = ERRORS[error_id]
            count = counts.get(error_id, 0)
            noun = "example file" if count == 1 else "example files"
            ranking = PRESENTATION_RANKING[error_id]
            root_lines.append(
                f"| {ranking['rank']} | `{error_id}_{meta['slug']}/` - {meta['name']} ({count} {noun}) | "
                f"{ranking['transformation']} | {ranking['fp_confidence']} | {ranking['fn_confidence']} | "
                f"{ranking['confidence_note']} |"
            )

    for category in PRESENTATION_CATEGORIES:
        root_lines.extend(
            [
                f"### `{category['id']}` - {category['name']}",
                "",
                category["description"],
                "",
            ]
        )
        append_error_ids(category["error_ids"])
        root_lines.append("")
    root_lines.extend(
        [
            "## Caveat",
            "",
            "These are reviewed examples for presentation and inspection. A folder",
            "being present here still does not approve a full automatic overlay:",
            "fresh-heldout precision and good-text-loss gates are required before",
            "full-corpus application.",
            "",
        ]
    )
    write_text(output_dir / "README.md", "\n".join(root_lines))
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-examples-per-error", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    counts = build_library(args.output_dir, args.max_examples_per_error)
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
