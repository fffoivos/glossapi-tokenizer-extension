#!/usr/bin/env python3
"""Pure, explainable deterministic ToC and bibliography structure detection.

This module is research code.  It emits evidence and proposed structure spans;
it deliberately has no corpus-mutation or deletion API.

``evaluate_r0_section`` is a dependency-free reproduction of GlossAPI's
historical ``_compute_likely_index_for_section`` rule.  The R1/R2 analysers are
separate: they recognise local evidence and conservative finite-state decoders
then decide whether that evidence forms a coherent ToC or bibliography block.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

RULES_ID = "deterministic_structural_rules_v1"
DECODER_ID = "confirmed_blocks_typed_gaps_v1"


class StructureKind(str, Enum):
    TOC = "TOC"
    BIB = "BIB"


class TocRole(str, Enum):
    HEADING = "TOC_HEADING"
    STRONG_ENTRY = "STRONG_TOC_ENTRY"
    WEAK_ENTRY = "WEAK_TOC_ENTRY"
    POSSIBLE_CONTINUATION = "POSSIBLE_TOC_CONTINUATION"
    HARD_OTHER = "HARD_OTHER"
    OTHER = "OTHER"


class BibRole(str, Enum):
    HEADING = "BIB_HEADING"
    SUBHEADING = "BIB_SUBHEADING"
    STRONG_ENTRY_START = "STRONG_ENTRY_START"
    WEAK_ENTRY_START = "WEAK_ENTRY_START"
    POSSIBLE_CONTINUATION = "POSSIBLE_CONTINUATION"
    HARD_OTHER = "HARD_OTHER"
    OTHER = "OTHER"


@dataclass(frozen=True)
class TocLineEvidence:
    """Local ToC evidence; never an instruction to remove the line."""

    line_index: int
    text: str
    role: TocRole
    score: float
    reason_codes: tuple[str, ...]
    hard_negative: bool
    token_count: int
    page_number: int | None = None
    page_kind: str | None = None
    section_prefix: str | None = None
    format_signature: str | None = None


@dataclass(frozen=True)
class BibLineEvidence:
    """Local bibliography evidence; never an instruction to remove the line."""

    line_index: int
    text: str
    role: BibRole
    score: float
    reason_codes: tuple[str, ...]
    hard_negative: bool
    token_count: int
    evidence_families: tuple[str, ...] = ()
    citation_styles: tuple[str, ...] = ()
    entry_number: int | None = None
    year: str | None = None


@dataclass(frozen=True)
class StructureSpan:
    """An inclusive, evidence-backed structural block proposal.

    ``line_indices`` makes non-contiguous physical coordinates explicit.  The
    start/end fields are inclusive physical line indexes, not character offsets.
    Consumers must make any later cleaning decision outside this module.
    """

    kind: StructureKind
    start_line_index: int
    end_line_index: int
    line_indices: tuple[int, ...]
    supporting_line_indices: tuple[int, ...]
    bridged_line_indices: tuple[int, ...]
    seed_line_indices: tuple[int, ...]
    seed_kind: str
    reason_codes: tuple[str, ...]
    terminator_line_index: int | None = None


@dataclass(frozen=True)
class StructureConflict:
    """Overlapping ToC/BIB proposals withheld by the fail-closed combiner."""

    toc_span: StructureSpan
    bib_span: StructureSpan
    overlapping_line_indices: tuple[int, ...]
    reason_codes: tuple[str, ...] = ("TOC_BIB_OVERLAP_FAIL_CLOSED",)


@dataclass(frozen=True)
class StructureDecision:
    """Evidence, non-conflicting proposals, and explicit withheld conflicts."""

    toc_evidence: tuple[TocLineEvidence, ...]
    bib_evidence: tuple[BibLineEvidence, ...]
    spans: tuple[StructureSpan, ...]
    conflicts: tuple[StructureConflict, ...]


# ---------------------------------------------------------------------------
# Frozen R0 compatibility rule
# ---------------------------------------------------------------------------


_R0_NUMBER = re.compile(r"\b(\d{1,3}(?:\.\d+)?)\b")
_R0_RANGE = re.compile(r"\b(\d{1,3})\s*-\s*(\d{1,3})\b")
_R0_ALPHA = re.compile(r"[A-Za-zΑ-Ωα-ω]")


def _is_missing_scalar(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def evaluate_r0_section(
    text: object,
    min_seq_length: int = 7,
    allowed_consecutive_equals: int = 4,
) -> tuple[int, list[int]]:
    """Reproduce the historical GlossAPI ToC section heuristic exactly.

    Compatibility intentionally includes the original table-only scope, page
    ceiling, nearly-all-table shortcut, and sequence-counter behaviour.
    """

    if _is_missing_scalar(text) or not isinstance(text, str):
        return 0, []

    numbers: list[int] = []
    lines = text.splitlines()
    index = 0
    table_lines_count = 0
    total_lines = len(lines)

    while index < len(lines):
        line = lines[index].strip()
        combined_line = line
        if line.startswith("|") and not line.endswith("|") and index + 1 < len(lines):
            next_line = lines[index + 1].strip()
            if not next_line.startswith("|") and next_line.endswith("|"):
                combined_line = line + " " + next_line
                index += 1

        if combined_line.startswith("|") and combined_line.endswith("|"):
            table_lines_count += 1
            cleaned_line = " ".join(combined_line.split())
            cleaned_line = re.sub(r"(?:(?:[.\-_]\s?){3,})", " ", cleaned_line)
            midpoint = len(cleaned_line) // 2
            first_half = cleaned_line[:midpoint]
            if _R0_ALPHA.search(first_half) is None:
                index += 1
                continue

            second_half = cleaned_line[midpoint:]
            range_match = _R0_RANGE.search(second_half)
            if range_match:
                candidate = range_match.group(1)
                after_candidate = second_half[range_match.end(1) :]
                if "," not in candidate and _R0_ALPHA.search(after_candidate) is None:
                    try:
                        number = int(candidate)
                        if number < 400:
                            numbers.append(number)
                            index += 1
                            continue
                    except ValueError:
                        pass

            matches = list(_R0_NUMBER.finditer(second_half))
            if matches:
                for match in reversed(matches):
                    candidate = match.group(1)
                    if "," in candidate:
                        continue
                    if _R0_ALPHA.search(second_half[match.end() :]):
                        continue
                    try:
                        number = int(candidate)
                    except ValueError:
                        continue
                    if number < 400:
                        numbers.append(number)
                        break
        index += 1

    if len(numbers) < min_seq_length:
        if table_lines_count >= total_lines - 1 and len(numbers) >= 4:
            if all(numbers[i] >= numbers[i - 1] for i in range(1, len(numbers))):
                return 1, numbers

    if len(numbers) < min_seq_length:
        return 0, numbers

    current_sequence = 1
    equal_count = 1
    for index in range(1, len(numbers)):
        if numbers[index] > numbers[index - 1]:
            current_sequence += 1
            equal_count = 1
        elif numbers[index] == numbers[index - 1]:
            equal_count += 1
            if equal_count <= allowed_consecutive_equals:
                current_sequence += 1
            else:
                current_sequence = 1
                equal_count = 1
        else:
            current_sequence = 1
            equal_count = 1
        if current_sequence >= min_seq_length:
            return 1, numbers
    return 0, numbers


# ---------------------------------------------------------------------------
# R1 local ToC evidence
# ---------------------------------------------------------------------------


_TOC_HEADINGS = {
    "αναλυτικα περιεχομενα",
    "contents",
    "table of contents",
    "toc",
    "περιεχομενα",
    "πινακας περιεχομενων",
}
_TOC_LEADER = re.compile(r"(?:(?:[._…·]\s*){3,})")
_TOC_PREFIX = re.compile(
    r"^\s*(?P<prefix>(?:\d+|[IVXLCDM]+)(?:\s*[.)]|(?:\.\d+){1,5}\.?))\s+",
    re.IGNORECASE,
)
_TOC_ARABIC_TAIL = re.compile(
    r"(?<!\w)(?P<first>\d{1,5})(?:\s*[-–—]\s*(?P<last>\d{1,5}))?\s*[\]).,:;|]*\s*$"
)
_TOC_ROMAN_TAIL = re.compile(
    r"(?<!\w)(?P<roman>[ivxlcdm]{1,10})\s*[\]).,:;|]*\s*$", re.IGNORECASE
)
_CANONICAL_ROMAN = re.compile(
    r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
)
_DATE = re.compile(r"\b(?:[0-3]?\d[./-][01]?\d[./-](?:19|20)?\d{2}|(?:19|20)\d{2})\b")
_STATISTICAL = re.compile(
    r"(?:%|€|\$|£|\b(?:mean|median|sd|p\s*[<=>])\b)", re.IGNORECASE
)
_EQUATION = re.compile(r"(?:\b\w+\s*=\s*[-+]?\d|[∑∫≈≤≥])")
_CAPTION = re.compile(
    r"^(?:πινακας|σχημα|εικονα|figure|table|fig\.)\s+\d+\b", re.IGNORECASE
)
_LEGAL_START = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*|[α-ω])[.)]\s*)?"
    r"(?:αρθρο|άρθρο|παραγραφος|παράγραφος|ο\s+αιτων|ο\s+αιτών|η\s+αιτουσα|η\s+αιτούσα)\b",
    re.IGNORECASE,
)
_ATX_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")


def _fold(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    folded = "".join(char for char in decomposed if not unicodedata.combining(char))
    # casefold maps Greek final sigma to σ; restore it at word boundaries so
    # readable Greek rule lexicons can use normal orthography.
    return re.sub(r"σ(?=\W|$)", "ς", folded)


def _heading_key(value: str) -> str:
    value = re.sub(r"^[\s#*_|:;\-–—]+|[\s#*_|:;.!?\-–—]+$", "", value)
    return re.sub(r"\s+", " ", _fold(value)).strip()


def _token_count(value: str) -> int:
    return len(re.findall(r"\S+", value))


def _has_letters(value: str) -> bool:
    return any(char.isalpha() for char in value)


def _roman_to_int(value: str) -> int | None:
    value = value.upper()
    if not value or not _CANONICAL_ROMAN.fullmatch(value):
        return None
    units = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = previous = 0
    for char in reversed(value):
        current = units[char]
        if current < previous:
            total -= current
        else:
            total += current
            previous = current
    return total if 0 < total <= 3999 else None


def analyze_toc_line(text: str, line_index: int = 0) -> TocLineEvidence:
    """Return explainable local ToC evidence without using document context."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if line_index < 0:
        raise ValueError("line_index must be non-negative")
    normalized = unicodedata.normalize("NFKC", text)
    stripped = normalized.strip()
    tokens = _token_count(stripped)
    reasons: list[str] = []

    if not stripped:
        return TocLineEvidence(
            line_index,
            text,
            TocRole.POSSIBLE_CONTINUATION,
            0.0,
            ("TOC_BLANK_FORMATTING_GAP",),
            False,
            0,
        )

    if _heading_key(stripped) in _TOC_HEADINGS:
        return TocLineEvidence(
            line_index,
            text,
            TocRole.HEADING,
            4.0,
            ("TOC_EXACT_HEADING",),
            False,
            tokens,
            format_signature="heading",
        )
    if _ATX_HEADING.match(stripped):
        return TocLineEvidence(
            line_index,
            text,
            TocRole.HARD_OTHER,
            -4.0,
            ("TOC_NEGATIVE_NONSTRUCTURAL_MARKDOWN_HEADING",),
            True,
            tokens,
        )

    table_row = stripped.count("|") >= 2
    table_separator = bool(
        table_row
        and re.fullmatch(r"\|?[\s:|-]+\|?", stripped)
        and stripped.count("-") >= 3
    )
    if table_separator:
        return TocLineEvidence(
            line_index,
            text,
            TocRole.POSSIBLE_CONTINUATION,
            0.5,
            ("TOC_MARKDOWN_SEPARATOR",),
            False,
            tokens,
            format_signature="table",
        )

    prefix_match = _TOC_PREFIX.match(stripped)
    section_prefix = prefix_match.group("prefix") if prefix_match else None
    leader = bool(_TOC_LEADER.search(stripped))

    page_number: int | None = None
    page_kind: str | None = None
    tail_start: int | None = None
    arabic = _TOC_ARABIC_TAIL.search(stripped)
    if arabic:
        page_number = int(arabic.group("first"))
        page_kind = "arabic"
        tail_start = arabic.start()
    else:
        roman = _TOC_ROMAN_TAIL.search(stripped)
        if roman:
            possible = _roman_to_int(roman.group("roman"))
            preceding = stripped[: roman.start()].rstrip(" |.\t")
            if possible is not None and _has_letters(preceding):
                page_number = possible
                page_kind = "roman"
                tail_start = roman.start()

    title_part = stripped[:tail_start] if tail_start is not None else stripped
    title_part = _TOC_LEADER.sub(" ", title_part)
    title_part = re.sub(r"[|._…·]+", " ", title_part)
    if prefix_match:
        title_part = (
            title_part[prefix_match.end() :]
            if prefix_match.end() <= len(title_part)
            else title_part
        )
    has_title = _has_letters(title_part)

    if leader:
        reasons.append("TOC_LEADER")
    if table_row:
        reasons.append("TOC_MARKDOWN_ROW")
    if section_prefix:
        reasons.append("TOC_HIERARCHICAL_PREFIX")
    if page_number is not None:
        reasons.append(
            "TOC_TERMINAL_ROMAN_PAGE"
            if page_kind == "roman"
            else "TOC_TERMINAL_ARABIC_PAGE"
        )
    if has_title:
        reasons.append("TOC_TITLE_TEXT")

    hard_reasons: list[str] = []
    folded = _fold(stripped)
    year_tail = (
        page_kind == "arabic"
        and page_number is not None
        and 1800 <= page_number <= 2099
    )
    if _STATISTICAL.search(stripped):
        hard_reasons.append("TOC_NEGATIVE_STATISTICAL_OR_CURRENCY")
    if _EQUATION.search(stripped):
        hard_reasons.append("TOC_NEGATIVE_EQUATION")
    if _CAPTION.match(folded):
        hard_reasons.append("TOC_NEGATIVE_CAPTION")
    if year_tail and not leader and not section_prefix:
        hard_reasons.append("TOC_NEGATIVE_TERMINAL_YEAR")
    elif _DATE.search(stripped) and page_number is None:
        hard_reasons.append("TOC_NEGATIVE_DATE")
    if _LEGAL_START.match(folded) and not leader:
        hard_reasons.append("TOC_NEGATIVE_LEGAL_PROCEDURE")
    if tokens >= 16 and not leader and not table_row and page_number is None:
        hard_reasons.append("TOC_NEGATIVE_RUNNING_PROSE")

    if hard_reasons:
        return TocLineEvidence(
            line_index,
            text,
            TocRole.HARD_OTHER,
            -3.0,
            tuple(reasons + hard_reasons),
            True,
            tokens,
            page_number,
            page_kind,
            section_prefix,
            "table" if table_row else "plain",
        )

    score = (
        (2.0 if page_number is not None else 0.0)
        + (1.0 if has_title else 0.0)
        + (2.0 if leader else 0.0)
        + (1.0 if table_row else 0.0)
        + (1.0 if section_prefix else 0.0)
    )
    signature = (
        "table"
        if table_row
        else "leader"
        if leader
        else "numbered"
        if section_prefix
        else "plain"
    )
    if (
        page_number is not None
        and has_title
        and (leader or table_row or section_prefix)
    ):
        role = TocRole.STRONG_ENTRY
    elif page_number is not None and has_title:
        role = TocRole.WEAK_ENTRY
    elif has_title and leader:
        role = TocRole.POSSIBLE_CONTINUATION
        reasons.append("TOC_OCR_OR_WRAPPED_ENTRY")
    elif has_title and tokens <= 10 and (stripped.isupper() or text[:1].isspace()):
        role = TocRole.POSSIBLE_CONTINUATION
        reasons.append("TOC_SHORT_SUBHEADING_OR_WRAP")
    else:
        role = TocRole.OTHER
        if not reasons:
            reasons.append("TOC_NO_LOCAL_EVIDENCE")

    return TocLineEvidence(
        line_index,
        text,
        role,
        score,
        tuple(reasons),
        False,
        tokens,
        page_number,
        page_kind,
        section_prefix,
        signature,
    )


def analyze_toc_lines(
    lines: Sequence[str], *, start_line_index: int = 0
) -> tuple[TocLineEvidence, ...]:
    return tuple(
        analyze_toc_line(text, start_line_index + offset)
        for offset, text in enumerate(lines)
    )


# ---------------------------------------------------------------------------
# R2 local bibliography evidence
# ---------------------------------------------------------------------------


_BIB_HEADINGS = {
    "bibliography",
    "bibliographical references",
    "bibliographic references",
    "references",
    "works cited",
    "literature cited",
    "reference list",
    "βιβλιογραφια",
    "βιβλιογραφικες αναφορες",
    "αναφορες",
    "πηγες και βιβλιογραφια",
    "βιβλιογραφια και πηγες",
}
_BIB_SUBHEADINGS = {
    "ελληνικη",
    "ελληνικη βιβλιογραφια",
    "ελληνογλωσση",
    "ελληνογλωσση βιβλιογραφια",
    "ξενη βιβλιογραφια",
    "ξενογλωσση",
    "ξενογλωσση βιβλιογραφια",
    "δικτυογραφια",
    "ιστοσελιδες",
    "ηλεκτρονικες πηγες",
    "νομοθεσια",
    "νομολογια",
    "πηγες",
    "primary sources",
    "secondary sources",
    "further reading",
    "greek references",
    "foreign references",
    "web references",
    "sources",
}
_CV_HEADINGS = {
    "βιογραφικο σημειωμα",
    "βιογραφικο",
    "δημοσιευσεις",
    "επιλεγμενες δημοσιευσεις",
    "ανακοινωσεις",
    "curriculum vitae",
    "publications",
    "selected publications",
}
_NOTES_HEADINGS = {
    "σημειωσεις",
    "υποσημειωσεις",
    "τελικες σημειωσεις",
    "end notes",
    "notes",
    "footnotes",
    "endnotes",
}
_BODY_HEADING = re.compile(
    r"^(?:κεφαλαιο|chapter|παραρτημα|appendix|συμπερασματα|conclusions|μεθοδολογια|methodology)\b",
    re.IGNORECASE,
)
_BIB_YEAR = re.compile(
    r"(?:\((?P<paren>(?:1[5-9]|20)\d{2}[a-zα-ω]?)\)|\b(?P<bare>(?:1[5-9]|20)\d{2}[a-zα-ω]?)\b|\b(?P<nd>n\.\s*d\.|χ\.\s*χ\.))",
    re.IGNORECASE,
)
_BIB_NUMBER = re.compile(
    r"^\s*(?:\[(?P<bracket>\d{1,4})\]|\((?P<paren>\d{1,4})\)|(?P<plain>\d{1,4})[.)])\s+"
)
_BIB_AUTHOR_COMMA = re.compile(
    r"^\s*[^\W\d_][\w'’\-]{1,40},\s*(?:[A-ZΑ-ΩΆΈΉΊΌΎΏ]\.?\s*)+",
    re.UNICODE,
)
_BIB_AUTHOR_YEAR = re.compile(
    r"^\s*[A-ZΑ-ΩΆΈΉΊΌΎΏ][^.!?]{1,90}?\s*\((?:1[5-9]|20)\d{2}[a-zα-ω]?\)",
    re.UNICODE,
)
_BIB_JOURNAL = re.compile(
    r"(?:\b(?:vol(?:ume)?|no\.|issue|journal|τεύχ(?:ος|\.)|τχ\.|τόμ(?:ος|\.)|vol\.)\b|\b\d+\s*\(\d+\)\s*[:,]\s*\d|\b(?:pp?\.|σσ?\.)\s*\d)",
    re.IGNORECASE,
)
_BIB_PUBLISHER = re.compile(
    r"(?:\b(?:εκδ(?:όσεις|\.)?|publisher|press|μτφρ\.|επιμ\.)\b|(?:Αθήνα|Θεσσαλονίκη|Athens|London|New York)\s*:)",
    re.IGNORECASE,
)
_BIB_IDENTIFIER = re.compile(
    r"(?:https?://|www\.|doi\s*:|\b10\.\d{4,9}/\S+|\bISBN(?:-1[03])?\b|\bISSN\b)",
    re.IGNORECASE,
)
_BIB_LEGAL = re.compile(
    r"(?:\b(?:ν(?:όμος|\.)?|law)\s*\d+[/-]\d{2,4}\b|\bΦΕΚ\s+[A-ZΑ-Ω]?\s*\d+[/-]\d{2,4}\b)",
    re.IGNORECASE,
)
_INLINE_CITATION = re.compile(
    r"\b(?:οπως|όπως|σύμφωνα|συμφωνα|κατα|κατά|as|according)\b.{0,100}\((?:19|20)\d{2}",
    re.IGNORECASE,
)
_NARRATIVE_AUTHOR_YEAR = re.compile(
    r"^\s*[^\W\d_][\w'’\-]{1,40}\s+\((?:1[5-9]|20)\d{2}[a-zα-ω]?\)\s+[^\W\d_]",
    re.IGNORECASE | re.UNICODE,
)
_FOOTNOTE = re.compile(r"^\s*(?:\^?\d{1,3}|[*†‡])[.):]?\s+")


def analyze_bib_line(text: str, line_index: int = 0) -> BibLineEvidence:
    """Return typed local citation evidence without deciding document structure."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if line_index < 0:
        raise ValueError("line_index must be non-negative")
    normalized = unicodedata.normalize("NFKC", text)
    stripped = normalized.strip()
    tokens = _token_count(stripped)
    key = _heading_key(stripped)

    if not stripped:
        return BibLineEvidence(
            line_index,
            text,
            BibRole.POSSIBLE_CONTINUATION,
            0.0,
            ("BIB_BLANK_FORMATTING_GAP",),
            False,
            0,
        )
    if key in _BIB_HEADINGS:
        return BibLineEvidence(
            line_index,
            text,
            BibRole.HEADING,
            4.0,
            ("BIB_EXACT_HEADING",),
            False,
            tokens,
        )
    if key in _BIB_SUBHEADINGS:
        styles: tuple[str, ...] = (
            ("web",)
            if key
            in {
                "δικτυογραφια",
                "ιστοσελιδες",
                "ηλεκτρονικες πηγες",
                "web references",
            }
            else ("legal",)
            if key in {"νομοθεσια", "νομολογια"}
            else ("formal_source",)
            if key
            not in {
                "πηγες",
                "sources",
                "primary sources",
                "secondary sources",
                "further reading",
            }
            else ("sources",)
        )
        return BibLineEvidence(
            line_index,
            text,
            BibRole.SUBHEADING,
            2.5,
            ("BIB_TYPED_SUBHEADING",),
            False,
            tokens,
            citation_styles=styles,
        )
    if key in _CV_HEADINGS:
        return BibLineEvidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            -4.0,
            ("BIB_NEGATIVE_CV_PUBLICATIONS_HEADING",),
            True,
            tokens,
        )
    if key in _NOTES_HEADINGS:
        return BibLineEvidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            -4.0,
            ("BIB_NEGATIVE_NOTES_HEADING",),
            True,
            tokens,
        )
    if _ATX_HEADING.match(stripped):
        return BibLineEvidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            -4.0,
            ("BIB_NEGATIVE_NONSTRUCTURAL_MARKDOWN_HEADING",),
            True,
            tokens,
        )

    folded = _fold(stripped)
    hard_reasons: list[str] = []
    if _BODY_HEADING.match(folded):
        hard_reasons.append("BIB_NEGATIVE_BODY_HEADING")
    if _STATISTICAL.search(stripped) or (
        stripped.count("|") >= 2 and re.search(r"\d", stripped)
    ):
        hard_reasons.append("BIB_NEGATIVE_STATISTICAL_TABLE")
    if _EQUATION.search(stripped):
        hard_reasons.append("BIB_NEGATIVE_EQUATION")
    if _FOOTNOTE.match(stripped) and not _BIB_NUMBER.match(stripped):
        hard_reasons.append("BIB_NEGATIVE_FOOTNOTE")
    if _INLINE_CITATION.search(folded):
        hard_reasons.append("BIB_NEGATIVE_INLINE_CITATION_PROSE")
    if _NARRATIVE_AUTHOR_YEAR.match(stripped):
        hard_reasons.append("BIB_NEGATIVE_NARRATIVE_AUTHOR_YEAR_PROSE")
    if _LEGAL_START.match(folded) and not _BIB_LEGAL.search(stripped):
        hard_reasons.append("BIB_NEGATIVE_LEGAL_PROCEDURE")

    number_match = _BIB_NUMBER.match(stripped)
    entry_number = None
    if number_match:
        entry_number = int(
            next(value for value in number_match.groupdict().values() if value)
        )
    year_match = _BIB_YEAR.search(stripped)
    year = (
        next((value for value in year_match.groupdict().values() if value), None)
        if year_match
        else None
    )
    author_text = stripped[number_match.end() :] if number_match else stripped
    author_inverted = bool(_BIB_AUTHOR_COMMA.match(author_text))
    author_year_match = _BIB_AUTHOR_YEAR.match(author_text)
    author_year_skeleton = bool(
        author_year_match
        and re.match(r"\s*[.,;:]", author_text[author_year_match.end() :])
    )
    author = bool(author_inverted or author_year_match)
    journal = bool(_BIB_JOURNAL.search(stripped))
    publisher = bool(_BIB_PUBLISHER.search(stripped))
    identifier = bool(_BIB_IDENTIFIER.search(stripped))
    legal = bool(_BIB_LEGAL.search(stripped))

    family_flags = (
        ("author", author),
        ("year", year is not None),
        ("numbered", entry_number is not None),
        ("journal_pages", journal),
        ("publisher_place", publisher),
        ("identifier", identifier),
        ("legal_source", legal),
    )
    families = tuple(name for name, present in family_flags if present)
    reasons = [f"BIB_{family.upper()}" for family in families]

    if tokens >= 16 and not author and entry_number is None and not legal:
        sentence_like = (
            stripped.endswith((".", "!", "?", ";", "·")) or stripped.count(",") >= 2
        )
        if sentence_like:
            hard_reasons.append("BIB_NEGATIVE_RUNNING_PROSE")
    if hard_reasons:
        return BibLineEvidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            -3.0,
            tuple(reasons + hard_reasons),
            True,
            tokens,
            families,
            (),
            entry_number,
            year,
        )

    styles: list[str] = []
    if author_inverted or author_year_skeleton:
        styles.append("citation_author_skeleton")
    if entry_number is not None:
        styles.append("numbered")
    if author and year is not None:
        styles.append("author_year")
    if legal:
        styles.append("legal")
    if identifier:
        styles.append(
            "web"
            if re.search(r"https?://|www\.", stripped, re.IGNORECASE)
            else "identifier"
        )

    strong_anchor = author or entry_number is not None or legal
    strong_auxiliary = identifier and (year is not None or publisher or journal)
    strong = len(families) >= 2 and (strong_anchor or strong_auxiliary)
    punctuation_skeleton = (
        stripped.count(".") + stripped.count(",") + stripped.count(":")
    )
    continuation_family = bool(identifier or journal or publisher) and not strong_anchor

    if strong:
        styles.append("formal_source")
        role = BibRole.STRONG_ENTRY_START
        score = 2.0 + len(families)
        reasons.append("BIB_MULTI_FAMILY_ENTRY")
    elif continuation_family:
        role = BibRole.POSSIBLE_CONTINUATION
        score = 1.0 + len(families)
        reasons.append("BIB_PUBLICATION_TAIL_OR_URL")
    elif len(families) == 1 and punctuation_skeleton >= 2:
        role = BibRole.WEAK_ENTRY_START
        score = 1.5
        reasons.append("BIB_SINGLE_FAMILY_CITATION_SHAPE")
    elif text[:1].isspace() and tokens <= 30 and stripped.endswith((".", ",", ";")):
        role = BibRole.POSSIBLE_CONTINUATION
        score = 0.5
        reasons.append("BIB_INDENTED_WRAPPED_LINE")
    else:
        role = BibRole.OTHER
        score = 0.0
        if not reasons:
            reasons.append("BIB_NO_LOCAL_EVIDENCE")

    return BibLineEvidence(
        line_index,
        text,
        role,
        score,
        tuple(reasons),
        False,
        tokens,
        families,
        tuple(styles),
        entry_number,
        year,
    )


def analyze_bib_lines(
    lines: Sequence[str], *, start_line_index: int = 0
) -> tuple[BibLineEvidence, ...]:
    return tuple(
        analyze_bib_line(text, start_line_index + offset)
        for offset, text in enumerate(lines)
    )


# ---------------------------------------------------------------------------
# Conservative block decoders
# ---------------------------------------------------------------------------


_MAX_SOFT_GAP_LINES = 2
_MAX_SOFT_GAP_TOKENS = 80


def _validate_evidence(evidence: Sequence[object]) -> None:
    indexes = [int(getattr(item, "line_index")) for item in evidence]
    if any(index < 0 for index in indexes):
        raise ValueError("line indexes must be non-negative")
    if any(right <= left for left, right in zip(indexes, indexes[1:])):
        raise ValueError("evidence must have strictly increasing line indexes")


def _toc_pages_progress(evidence: Sequence[TocLineEvidence]) -> bool:
    pages = [item for item in evidence if item.page_number is not None]
    previous: TocLineEvidence | None = None
    equal_run = 1
    for item in pages:
        if previous is None:
            previous = item
            continue
        if previous.page_kind == "arabic" and item.page_kind == "roman":
            return False
        if previous.page_kind == item.page_kind:
            assert previous.page_number is not None and item.page_number is not None
            if item.page_number < previous.page_number:
                return False
            if item.page_number == previous.page_number:
                equal_run += 1
                if equal_run > 4:
                    return False
            else:
                equal_run = 1
        else:
            # Roman front matter followed by Arabic body pages is coherent.
            equal_run = 1
        previous = item
    return True


def _soft_gap_ok(items: Sequence[object]) -> bool:
    return (
        len(items) <= _MAX_SOFT_GAP_LINES
        and sum(int(getattr(item, "token_count")) for item in items)
        <= _MAX_SOFT_GAP_TOKENS
    )


def _toc_seed(
    evidence: Sequence[TocLineEvidence], position: int, n_physical_lines: int
) -> tuple[int, str, tuple[int, ...]] | None:
    item = evidence[position]
    if item.role == TocRole.HEADING:
        candidates: list[TocLineEvidence] = []
        gaps: list[TocLineEvidence] = []
        for cursor in range(position + 1, min(len(evidence), position + 9)):
            current = evidence[cursor]
            if current.hard_negative or current.role == TocRole.OTHER:
                break
            if current.role in {TocRole.STRONG_ENTRY, TocRole.WEAK_ENTRY}:
                if gaps and not _soft_gap_ok(gaps):
                    break
                gaps.clear()
                candidates.append(current)
                if sum(entry.role == TocRole.STRONG_ENTRY for entry in candidates) >= 2:
                    if _toc_pages_progress(candidates):
                        seeds = (item.line_index,) + tuple(
                            entry.line_index for entry in candidates
                        )
                        return cursor, "toc_heading_confirmed", seeds
                    break
            elif current.role == TocRole.POSSIBLE_CONTINUATION:
                gaps.append(current)
                if not _soft_gap_ok(gaps):
                    break
        return None

    cutoff = min(300, int(0.30 * n_physical_lines))
    if item.line_index >= cutoff or item.role != TocRole.STRONG_ENTRY:
        return None
    candidates: list[TocLineEvidence] = []
    gaps: list[TocLineEvidence] = []
    for cursor in range(position, min(len(evidence), position + 10)):
        current = evidence[cursor]
        if current.hard_negative or current.role in {TocRole.OTHER, TocRole.HEADING}:
            break
        if current.role == TocRole.STRONG_ENTRY:
            if gaps and not _soft_gap_ok(gaps):
                break
            gaps.clear()
            candidates.append(current)
            if len(candidates) >= 4:
                signatures = Counter(
                    entry.format_signature
                    for entry in candidates
                    if entry.format_signature
                )
                compatible = bool(signatures and signatures.most_common(1)[0][1] >= 3)
                if compatible and _toc_pages_progress(candidates):
                    return (
                        cursor,
                        "toc_headerless_dense_run",
                        tuple(entry.line_index for entry in candidates),
                    )
                break
        elif current.role in {TocRole.WEAK_ENTRY, TocRole.POSSIBLE_CONTINUATION}:
            gaps.append(current)
            if not _soft_gap_ok(gaps):
                break
    return None


def _expand_toc_span(
    evidence: Sequence[TocLineEvidence],
    start: int,
    seed: tuple[int, str, tuple[int, ...]],
) -> tuple[StructureSpan, int]:
    _confirmed_at, seed_kind, seed_indexes = seed
    support: list[TocLineEvidence] = []
    bridged: list[TocLineEvidence] = []
    pending: list[TocLineEvidence] = []
    last_support_position: int | None = None
    pages: list[TocLineEvidence] = []
    terminator: int | None = None
    cursor = start

    while cursor < len(evidence):
        current = evidence[cursor]
        if current.hard_negative or current.role == TocRole.OTHER:
            terminator = current.line_index
            break
        if current.role in {TocRole.HEADING, TocRole.STRONG_ENTRY, TocRole.WEAK_ENTRY}:
            trial_pages = pages + ([current] if current.page_number is not None else [])
            if not _toc_pages_progress(trial_pages):
                terminator = current.line_index
                break
            if pending:
                if not _soft_gap_ok(pending):
                    terminator = pending[0].line_index
                    break
                bridged.extend(pending)
                pending.clear()
            support.append(current)
            pages = trial_pages
            last_support_position = cursor
        elif current.role == TocRole.POSSIBLE_CONTINUATION:
            pending.append(current)
            if not _soft_gap_ok(pending):
                terminator = pending[0].line_index
                break
        cursor += 1

    assert last_support_position is not None
    included = list(evidence[start : last_support_position + 1])
    included_indexes = tuple(item.line_index for item in included)
    reason_codes = ["TOC_COHERENT_BLOCK", seed_kind.upper()]
    if bridged:
        reason_codes.append("TOC_TYPED_SOFT_GAP_BRIDGED")
    span = StructureSpan(
        StructureKind.TOC,
        included[0].line_index,
        included[-1].line_index,
        included_indexes,
        tuple(item.line_index for item in support),
        tuple(item.line_index for item in bridged),
        seed_indexes,
        seed_kind,
        tuple(reason_codes),
        terminator,
    )
    return span, last_support_position + 1


def decode_toc_blocks(
    evidence: Sequence[TocLineEvidence], *, n_physical_lines: int | None = None
) -> tuple[StructureSpan, ...]:
    """Decode coherent ToC blocks; isolated line evidence never forms a span."""

    if not evidence:
        return ()
    _validate_evidence(evidence)
    inferred_lines = evidence[-1].line_index + 1
    physical_lines = inferred_lines if n_physical_lines is None else n_physical_lines
    if physical_lines < inferred_lines:
        raise ValueError("n_physical_lines does not cover the final evidence line")

    spans: list[StructureSpan] = []
    position = 0
    while position < len(evidence):
        seed = _toc_seed(evidence, position, physical_lines)
        if seed is None:
            position += 1
            continue
        span, next_position = _expand_toc_span(evidence, position, seed)
        spans.append(span)
        position = max(next_position, position + 1)
    return tuple(spans)


def _bib_primary_styles(item: BibLineEvidence) -> tuple[str, ...]:
    return tuple(
        style
        for style in item.citation_styles
        if style in {"author_year", "numbered", "legal", "web"}
    )


def _bib_headerless_citation_specific(item: BibLineEvidence) -> bool:
    return bool(
        set(item.evidence_families).intersection(
            {
                "journal_pages",
                "publisher_place",
                "identifier",
                "legal_source",
            }
        )
        or "citation_author_skeleton" in item.citation_styles
    )


def _bib_seed(
    evidence: Sequence[BibLineEvidence], position: int, scope_denied: bool
) -> tuple[int, str, tuple[int, ...]] | None:
    item = evidence[position]
    if scope_denied and item.role != BibRole.HEADING:
        return None
    if item.role in {BibRole.HEADING, BibRole.SUBHEADING}:
        candidates: list[BibLineEvidence] = []
        gaps: list[BibLineEvidence] = []
        soft_evidence: list[BibLineEvidence] = []
        anchor_styles = set(item.citation_styles)
        for cursor in range(position + 1, min(len(evidence), position + 11)):
            current = evidence[cursor]
            if current.hard_negative or current.role in {
                BibRole.OTHER,
                BibRole.HEADING,
            }:
                break
            if current.role in {BibRole.STRONG_ENTRY_START, BibRole.WEAK_ENTRY_START}:
                if gaps and not _soft_gap_ok(gaps):
                    break
                gaps.clear()
                candidates.append(current)
            elif current.role in {BibRole.POSSIBLE_CONTINUATION, BibRole.SUBHEADING}:
                gaps.append(current)
                soft_evidence.append(current)
                if not _soft_gap_ok(gaps):
                    break
            observed = candidates + soft_evidence
            strong = [
                entry
                for entry in candidates
                if entry.role == BibRole.STRONG_ENTRY_START
            ]
            weak_or_continuation = [
                entry
                for entry in observed
                if entry.role
                in {BibRole.WEAK_ENTRY_START, BibRole.POSSIBLE_CONTINUATION}
            ]
            if anchor_styles:
                matching = (
                    [
                        entry
                        for entry in observed
                        if _bib_headerless_citation_specific(entry)
                    ]
                    if "sources" in anchor_styles
                    else [
                        entry
                        for entry in observed
                        if anchor_styles.intersection(entry.citation_styles)
                    ]
                )
                if len(matching) < 2:
                    continue
            if len(strong) >= 2 or (
                len(strong) >= 1 and len(weak_or_continuation) >= 2
            ):
                seeds = (item.line_index,) + tuple(
                    sorted(entry.line_index for entry in observed)
                )
                return cursor, "bib_heading_confirmed", seeds
        return None

    if item.role != BibRole.STRONG_ENTRY_START:
        return None
    candidates: list[BibLineEvidence] = []
    gaps: list[BibLineEvidence] = []
    for cursor in range(position, min(len(evidence), position + 10)):
        current = evidence[cursor]
        if current.hard_negative or current.role in {
            BibRole.OTHER,
            BibRole.HEADING,
            BibRole.SUBHEADING,
        }:
            break
        if current.role == BibRole.STRONG_ENTRY_START:
            if gaps and not _soft_gap_ok(gaps):
                break
            gaps.clear()
            candidates.append(current)
            specific = [
                entry
                for entry in candidates
                if _bib_headerless_citation_specific(entry)
            ]
            by_style = Counter(
                style for entry in specific for style in _bib_primary_styles(entry)
            )
            if by_style and by_style.most_common(1)[0][1] >= 4:
                matching_style = by_style.most_common(1)[0][0]
                seeds = tuple(
                    entry.line_index
                    for entry in candidates
                    if matching_style in entry.citation_styles
                )
                return cursor, "bib_headerless_dense_run", seeds
        elif current.role in {BibRole.WEAK_ENTRY_START, BibRole.POSSIBLE_CONTINUATION}:
            gaps.append(current)
            if not _soft_gap_ok(gaps):
                break
    return None


def _expand_bib_span(
    evidence: Sequence[BibLineEvidence],
    start: int,
    seed: tuple[int, str, tuple[int, ...]],
) -> tuple[StructureSpan, int]:
    _confirmed_at, seed_kind, seed_indexes = seed
    support: list[BibLineEvidence] = []
    bridged: list[BibLineEvidence] = []
    pending: list[BibLineEvidence] = []
    last_support_position: int | None = None
    terminator: int | None = None
    cursor = start

    while cursor < len(evidence):
        current = evidence[cursor]
        if current.hard_negative or current.role == BibRole.OTHER:
            terminator = current.line_index
            break
        if current.role in {
            BibRole.HEADING,
            BibRole.SUBHEADING,
            BibRole.STRONG_ENTRY_START,
            BibRole.WEAK_ENTRY_START,
        }:
            if pending:
                if not _soft_gap_ok(pending):
                    terminator = pending[0].line_index
                    break
                bridged.extend(pending)
                pending.clear()
            support.append(current)
            last_support_position = cursor
        elif current.role == BibRole.POSSIBLE_CONTINUATION:
            pending.append(current)
            if not _soft_gap_ok(pending):
                terminator = pending[0].line_index
                break
        cursor += 1

    assert last_support_position is not None
    # Explicit publication-tail evidence may close an entry at EOF/a barrier;
    # arbitrary blanks and formatting gaps are never appended blindly.
    trailing = [
        item
        for item in pending
        if "BIB_PUBLICATION_TAIL_OR_URL" in item.reason_codes
        or "BIB_INDENTED_WRAPPED_LINE" in item.reason_codes
    ]
    if pending and len(trailing) == len(pending) and _soft_gap_ok(pending):
        bridged.extend(pending)
        last_support_position += len(pending)

    included = list(evidence[start : last_support_position + 1])
    reason_codes = ["BIB_COHERENT_BLOCK", seed_kind.upper()]
    if bridged:
        reason_codes.append("BIB_TYPED_CONTINUATION_BRIDGED")
    span = StructureSpan(
        StructureKind.BIB,
        included[0].line_index,
        included[-1].line_index,
        tuple(item.line_index for item in included),
        tuple(item.line_index for item in support),
        tuple(item.line_index for item in bridged),
        seed_indexes,
        seed_kind,
        tuple(reason_codes),
        terminator,
    )
    return span, last_support_position + 1


def decode_bib_blocks(evidence: Sequence[BibLineEvidence]) -> tuple[StructureSpan, ...]:
    """Decode formal bibliography blocks without heading-to-EOF expansion."""

    if not evidence:
        return ()
    _validate_evidence(evidence)
    spans: list[StructureSpan] = []
    position = 0
    denied_scope: str | None = None
    while position < len(evidence):
        item = evidence[position]
        if "BIB_NEGATIVE_CV_PUBLICATIONS_HEADING" in item.reason_codes:
            denied_scope = "cv_publications"
        elif "BIB_NEGATIVE_NOTES_HEADING" in item.reason_codes:
            denied_scope = "notes_or_footnotes"
        elif "BIB_NEGATIVE_BODY_HEADING" in item.reason_codes:
            denied_scope = None
        seed = _bib_seed(evidence, position, denied_scope is not None)
        if seed is None:
            position += 1
            continue
        if item.role == BibRole.HEADING:
            # Only a confirmed formal bibliography heading reopens a denied
            # CV/notes scope; an orphan heading changes no scope state.
            denied_scope = None
        span, next_position = _expand_bib_span(evidence, position, seed)
        spans.append(span)
        position = max(next_position, position + 1)
    return tuple(spans)


def detect_structure(
    lines: Sequence[str],
    *,
    start_line_index: int = 0,
    n_physical_lines: int | None = None,
) -> StructureDecision:
    """Return evidence and only non-conflicting structure proposals.

    Any ToC/BIB overlap withholds both candidate spans and is surfaced as an
    explicit conflict.  Consumers therefore cannot accidentally treat two
    contradictory proposals as cleaning actions.
    """

    toc = analyze_toc_lines(lines, start_line_index=start_line_index)
    bib = analyze_bib_lines(lines, start_line_index=start_line_index)
    inferred = start_line_index + len(lines)
    if n_physical_lines is not None and n_physical_lines < inferred:
        raise ValueError("n_physical_lines does not cover the final input line")
    physical = inferred if n_physical_lines is None else n_physical_lines
    toc_spans = decode_toc_blocks(toc, n_physical_lines=physical)
    bib_spans = decode_bib_blocks(bib)
    conflicts: list[StructureConflict] = []
    conflicting_ids: set[int] = set()
    for toc_span in toc_spans:
        toc_lines = set(toc_span.line_indices)
        for bib_span in bib_spans:
            overlap = tuple(sorted(toc_lines.intersection(bib_span.line_indices)))
            if overlap:
                conflicts.append(StructureConflict(toc_span, bib_span, overlap))
                conflicting_ids.update((id(toc_span), id(bib_span)))
    spans = tuple(
        span for span in toc_spans + bib_spans if id(span) not in conflicting_ids
    )
    return StructureDecision(toc, bib, spans, tuple(conflicts))
