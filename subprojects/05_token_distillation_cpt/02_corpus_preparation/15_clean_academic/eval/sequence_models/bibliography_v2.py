#!/usr/bin/env python3
"""Inspectable deterministic bibliography features and coherent block proposals.

This is a research-only successor to the frozen R2 bibliography rules in
``deterministic_structure``.  It deliberately lives beside R2 so comparisons
retain a real control and the Python/Rust parity contract is not changed before
the new rules have earned that change.

The detector has two stages:

* extract explicit, countable citation features from each physical line;
* propose a bibliography block only when citation-like lines form a coherent
  local run (or confirm a formal bibliography heading).

Nothing in this module mutates a corpus or authorises removal.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Sequence

from .deterministic_structure import (
    BibRole,
    StructureKind,
    StructureSpan,
    analyze_bib_line,
)


RULES_ID = "deterministic_bibliography_explicit_features_v2"
DECODER_ID = "bibliography_anchor_clusters_typed_gaps_v2"


_LATIN_UPPER = "A-ZÀ-ÖØ-Þ"
_LATIN_LOWER = "a-zà-öø-ÿ"
_GREEK_UPPER = "Α-ΩΆΈΉΊΌΎΏΪΫ"
_GREEK_LOWER = "α-ωάέήίόύώϊϋΐΰς"
_UPPER = _LATIN_UPPER + _GREEK_UPPER
_LOWER = _LATIN_LOWER + _GREEK_LOWER
_LETTER = _UPPER + _LOWER

_YEAR = re.compile(r"(?<!\d)(?:1[5-9]\d{2}|20\d{2})(?:[a-zα-ω])?(?!\d)", re.I)
_NO_DATE = re.compile(r"\b(?:n\s*\.\s*d\s*\.|s\s*\.\s*d\s*\.|χ\s*\.\s*χ\s*\.)", re.I)
_NUMERIC_DATE = re.compile(
    r"(?<!\d)(?:"
    r"(?:0?[1-9]|[12]\d|3[01])\s*(?P<date_sep_dmy>[-./])\s*"
    r"(?:0?[1-9]|1[0-2])\s*(?P=date_sep_dmy)\s*(?:\d{2}|\d{4})"
    r"|(?:1[5-9]\d{2}|20\d{2})\s*(?P<date_sep_ymd>[-./])\s*"
    r"(?:0?[1-9]|1[0-2])\s*(?P=date_sep_ymd)\s*(?:0?[1-9]|[12]\d|3[01])"
    r")(?!\d)"
)
_MONTH_DATE = re.compile(
    r"\b(?:0?[1-9]|[12]\d|3[01])(?:\s*[-–—]\s*(?:0?[1-9]|[12]\d|3[01]))?\s+"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|"
    r"ιαν(?:ουαρίου)?|φεβ(?:ρουαρίου)?|μαρ(?:τίου)?|απρ(?:ιλίου)?|μαΐου|ιουν(?:ίου)?|"
    r"ιουλ(?:ίου)?|αυγ(?:ούστου)?|σεπ(?:τεμβρίου)?|οκτ(?:ωβρίου)?|νοε(?:μβρίου)?|δεκ(?:εμβρίου)?)"
    r"\s+(?:1[5-9]\d{2}|20\d{2})\b",
    re.I,
)
_ACCESS_DATE = re.compile(
    r"\b(?:accessed|retrieved|last\s+access(?:ed)?|πρόσβαση|ανακτήθηκε|"
    r"τελευταία\s+πρόσβαση|ημερομηνία\s+πρόσβασης)\b",
    re.I,
)

_URL = re.compile(r"(?:https?://|ftp://|www\.)[^\s<>\]\[{}]+", re.I)
_DOI = re.compile(
    r"(?:\bdoi\s*:\s*|https?://doi\.org/)?\b10\.\d{4,9}/[^\s<>\]\[{}]+", re.I
)
_ISBN = re.compile(
    r"\bISBN(?:-1[03])?\s*:?[ \t]*(?:97[89][ -]?)?[0-9X](?:[0-9X -]{7,20})[0-9X]\b",
    re.I,
)
_ISSN = re.compile(r"\bISSN\s*:?[ \t]*\d{4}[ -]?\d{3}[\dX]\b", re.I)

_INITIAL_ATOM = rf"(?:[{_UPPER}]\s*\.|[{_UPPER}][{_LOWER}]\s*\.)"
_INITIAL = re.compile(rf"(?<![{_LETTER}]){_INITIAL_ATOM}")
_PROPER_WORD = re.compile(
    rf"(?<![{_LETTER}])[{_UPPER}][{_LOWER}]{{2,}}(?:[-’'][{_UPPER}]?[{_LOWER}]{{2,}})?"
    rf"(?![{_LETTER}])(?!(?:[ \t]*\.))"
)
_INVERTED_AUTHOR = re.compile(
    rf"(?<![{_LETTER}])[{_UPPER}][{_LETTER}’'\-]{{1,45}}"
    rf"(?:\s+[{_UPPER}][{_LETTER}’'\-]{{1,45}})?"
    rf",\s*(?:{_INITIAL_ATOM}\s*){{1,4}}"
)
_AUTHOR_PREFIX = re.compile(
    r"^\s*(?:[-–—•\uf0a0]\s*)?(?:\[?\d{1,4}\]?[.)]?\s+)?$"
)
_NAME_INITIAL_PAIR = re.compile(
    rf"(?<![{_LETTER}])[{_UPPER}][{_LETTER}’'\-]{{1,40}}\s+"
    rf"(?:[{_UPPER}]{{1,3}}|(?:{_INITIAL_ATOM}\s*){{1,3}})"
    rf"(?=\s*(?:,|;|&|&amp;|and\b|και\b|\(|$))"
)
_DIRECT_AUTHOR = re.compile(
    rf"(?<![{_LETTER}])(?:{_INITIAL_ATOM}\s*){{1,4}}"
    rf"[{_UPPER}][{_LETTER}’'\-]{{1,45}}"
    rf"(?:\s+[{_UPPER}][{_LETTER}’'\-]{{1,45}})?"
    rf"(?=\s*(?:,|;|&|&amp;|[Aa]nd\b|[Κκ]αι\b|\(|(?:1[5-9]\d{{2}}|20\d{{2}})|$))"
)
_AMPERSAND = re.compile(r"(?:&|&amp;)", re.I)
_QUOTED = re.compile(r"(?:«[^»]{3,}»|“[^”]{3,}”|„[^“]{3,}“|\"[^\"]{3,}\"|'[^'\n]{3,}')")

_EDITOR_TERMS = re.compile(
    rf"(?<![{_LETTER}])(?:eds?\.|editors?|edited\s+by|trans\.|translator|translated\s+by|"
    r"επιμ(?:έλεια|ελητής|ελητές)?\.?|εκδ\.?\s*επιμ\.?|μτφρ\.?|μετάφραση|"
    rf"μεταφραστ(?:ής|ές|ρια))(?![{_LETTER}])",
    re.I,
)
_THESIS_TERMS = re.compile(
    r"\b(?:ph\.?\s*d\.?|doctoral\s+(?:thesis|dissertation)|master'?s?\s+thesis|"
    r"dissertation|thesis|διδακτορικ(?:ή|ης)\s+διατριβ(?:ή|ής)|"
    r"μεταπτυχιακ(?:ή|ης)\s+(?:εργασία|διατριβή)|διατριβ(?:ή|ής))\b",
    re.I,
)
_IN_CONTAINER = re.compile(
    r"(?:^|[,.;:]\s+)(?P<container_term>in|στο|στη|στις|στων)"
    r"(?=\s+(?:[{upper}]|«|\"|'))".format(
        upper=_UPPER
    ),
    re.I,
)
_EDITION_TERMS = re.compile(
    rf"(?<![{_LETTER}])(?:\d+(?:st|nd|rd|th)\s+ed(?:ition)?\.?|revised\s+edition|"
    r"edition|edn\.?|έκδ(?:οση|\.)|αναθ(?:εωρημένη|\.)\s+έκδ(?:οση|\.))\b",
    re.I,
)

# Abbreviated words are a feature, not an answer: sentence-final ``Press.``
# may contribute one count, whereas journal strings such as ``Nat. Chem.
# Biol.`` produce the more discriminative sequence feature.
_DOTTED_WORD = re.compile(
    rf"(?<![{_LETTER}])[{_UPPER}][{_LETTER}]{{2,5}}\.(?![{_LETTER}])"
)
_VOLUME_MARKER = re.compile(
    rf"(?<![{_LETTER}])(?:vol(?:ume)?|issue|no|number|τόμ(?:ος|ου)?|"
    rf"τεύχ(?:ος|ους)?|τχ)\s*\.?(?=\s*\d+)",
    re.I,
)
_VOLUME_SHAPE = re.compile(
    r"(?<!\d)(?P<volume>\d{1,4})\s*\(\s*(?P<issue>\d{1,4})\s*\)(?!\d)"
)
_JOURNAL_YEAR_VOLUME = re.compile(
    r"(?P<journal_year>1[5-9]\d{2}|20\d{2})\s*[,;]\s*"
    r"(?P<journal_volume>\d{1,4})(?P<journal_issue>\s*\(\s*\d{1,4}\s*\))?",
    re.I,
)
_PAGE_MARKER = re.compile(r"(?<!\w)(?:pp?|σσ?|σελ)\s*\.\s*(?=\d)", re.I)
_PAGE_RANGE = re.compile(
    r"(?<![\d,])(?<!\d[,.])(?P<page_start>\d{1,5})\s*[-–—]\s*"
    r"(?P<page_end>\d{1,5})(?![\d,])(?!\.\d)"
)
_PUBLISHER_TERMS = re.compile(
    rf"(?<![{_LETTER}])(?:press|publisher|publishing|publications?|εκδ(?:όσεις|οτικός|\.)|"
    r"πανεπιστημιακές\s+εκδόσεις|springer|elsevier|routledge|wiley|sage|"
    rf"πατάκ(?:η|ης)|μεταίχμιο|κάκτος|παπαζήση|gutenberg)(?![{_LETTER}])",
    re.I,
)
_PLACE_NAMES = re.compile(
    r"\b(?:Athens|London|New\s+York|Boston|Cambridge|Oxford|Chicago|Paris|Berlin|"
    r"Amsterdam|Brussels|Rome|Milan|Munich|Thessaloniki|Αθήνα|Αθήναι|"
    r"Θεσσαλονίκη|Πάτρα|Ιωάννινα|Ηράκλειο|Λευκωσία|Ρώμη|Παρίσι|Λονδίνο|"
    r"Βερολίνο|Νέα\s+Υόρκη)\b",
    re.I,
)
_PLACE_PUBLISHER_SHAPE = re.compile(
    r"(?:\b(?:Athens|London|New\s+York|Boston|Cambridge|Oxford|Chicago|Paris|Berlin|"
    r"Amsterdam|Brussels|Rome|Milan|Munich|Thessaloniki|Αθήνα|Αθήναι|Θεσσαλονίκη|"
    r"Πάτρα|Ιωάννινα|Ηράκλειο|Λευκωσία|Ρώμη|Παρίσι|Λονδίνο|Βερολίνο|Νέα\s+Υόρκη)\s*"
    r":\s*(?:[A-ZΑ-ΩΆΈΉΊΌΎΏ][\w.'’\-]+\s+){0,3}"
    r"(?:Press|University\s+Press|Publishing|Publisher|Εκδόσεις|Εκδ\.)"
    r"|:\s*(?:[A-ZΑ-ΩΆΈΉΊΌΎΏ][\w.'’\-]+\s+){0,3}"
    r"(?:Press|University\s+Press|Publishing|Publisher|Εκδόσεις|Εκδ\.))",
    re.I,
)

_PROSE_LEAD = re.compile(
    r"^\s*(?:όπως|σύμφωνα|κατά\s+τη|στην\s+παρούσα|η\s+παρούσα|"
    r"σε\s+αυτό|as\s+shown|according\s+to|this\s+(?:paper|study|chapter))\b",
    re.I,
)
_BIB_HEADING_WORD = re.compile(
    r"^(?:bibliography|references|works\s+cited|literature\s+cited|"
    r"βιβλιογραφία|βιβλιογραφικές\s+αναφορές|πηγές(?:\s+και\s+βιβλιογραφία)?)$",
    re.I,
)
_BIB_EXTENDED_HEADING = re.compile(
    r"^(?:βιβλιογραφ[ιί]α\s+κεφαλα[ιί]ου\s+[A-ZΑ-ΩΆΈΉΊΌΎΏ0-9.΄ʹ]+|"
    r"chapter\s+[A-Z0-9.]+\s+(?:bibliography|references))$",
    re.I,
)
_BIB_EXTENDED_SUBHEADING = re.compile(
    r"^(?:(?:ελληνόγλωσσες|ξενόγλωσσες)\s+βιβλιογραφικές\s+πηγές|"
    r"(?:greek|foreign[- ]language)\s+bibliographic\s+sources)$",
    re.I,
)
_FIGURE_CAPTION_START = re.compile(
    r"^\s*(?:[-–—•]\s*)?(?:εικ(?:όνα|\.)?|σχ(?:ήμα|\.)?|fig(?:ure|\.)?)\s*\d",
    re.I,
)
_ENUMERATED_PROSE_START = re.compile(
    r"^\s*(?:[-–—•\uf0a0]\s*)?(?:\d{1,3}|[A-Za-zΑ-Ωα-ω])\s*[.)]\s+"
)


@dataclass(frozen=True)
class BibliographyFeatures:
    """Countable local evidence.  Counts are intentionally not clipped."""

    token_count: int
    year_count: int
    no_date_count: int
    numeric_date_count: int
    month_date_count: int
    access_date_count: int
    url_count: int
    doi_count: int
    isbn_count: int
    issn_count: int
    initial_count: int
    proper_name_word_count: int
    inverted_author_count: int
    author_year_count: int
    name_initial_pair_count: int
    direct_author_count: int
    numbered_entry_count: int
    ampersand_count: int
    quoted_span_count: int
    editor_term_count: int
    thesis_term_count: int
    in_container_count: int
    edition_term_count: int
    dotted_word_count: int
    dotted_sequence_count: int
    volume_marker_count: int
    volume_shape_count: int
    journal_year_volume_count: int
    page_marker_count: int
    page_range_count: int
    publisher_term_count: int
    place_name_count: int
    place_publisher_shape_count: int
    punctuation_count: int
    prose_lead_count: int
    table_row_count: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class BibliographyFeatureMatch:
    """One review-only feature match in NFKC-normalized character offsets."""

    feature: str
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class BibliographyFeatureReview:
    """Diagnostic spans for review UIs; never used by scoring or decoding."""

    normalized_text: str
    features: BibliographyFeatures
    matches: tuple[BibliographyFeatureMatch, ...]


@dataclass(frozen=True)
class BibliographyV2Evidence:
    line_index: int
    text: str
    role: BibRole
    score: float
    reason_codes: tuple[str, ...]
    hard_negative: bool
    token_count: int
    evidence_families: tuple[str, ...]
    citation_styles: tuple[str, ...]
    features: BibliographyFeatures


_Span = tuple[int, int]


def _pattern_spans(
    pattern: re.Pattern[str],
    text: str,
    *,
    offset: int = 0,
    first_only: bool = False,
    group: int | str = 0,
) -> list[_Span]:
    spans: list[_Span] = []
    for match in pattern.finditer(text):
        spans.append((offset + match.start(group), offset + match.end(group)))
        if first_only:
            break
    return spans


def _overlaps(left: _Span, right: _Span) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _without_overlaps(spans: Sequence[_Span], blockers: Sequence[_Span]) -> list[_Span]:
    """Keep only spans not already owned by a more specific detector."""

    return [span for span in spans if not any(_overlaps(span, hit) for hit in blockers)]


def _dotted_sequences(value: str, dotted_words: Sequence[_Span]) -> list[_Span]:
    """Group adjacent residual dotted words into non-overlapping sequences."""

    sequences: list[_Span] = []
    run: list[_Span] = []
    for span in dotted_words:
        if run and value[run[-1][1] : span[0]].strip():
            if len(run) >= 2:
                sequences.append((run[0][0], run[-1][1]))
            run = []
        run.append(span)
    if len(run) >= 2:
        sequences.append((run[0][0], run[-1][1]))
    return sequences


def _joined_spans(spans: dict[str, list[_Span]], names: Sequence[str]) -> list[_Span]:
    return [span for name in names for span in spans[name]]


def _analysis_bounds(value: str) -> tuple[int, int]:
    start = len(value) - len(value.lstrip())
    end = len(value.rstrip())
    if start < end and value[start] == "|" and value[end - 1] == "|":
        start += 1
        end -= 1
        while start < end and value[start].isspace():
            start += 1
        while end > start and value[end - 1].isspace():
            end -= 1
    return start, end


def _numbered_entry_span(text: str) -> tuple[int, int] | None:
    """Return a decorated leading-number span with a strict linear scan."""

    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    start = index
    while index < len(text) and not text[index].isalnum():
        index += 1
    if index >= len(text) or not text[index].isdigit():
        return None
    digits_start = index
    while index < len(text) and text[index].isdigit():
        index += 1
    if (
        index + 1 < len(text)
        and text[index] in ".,"
        and text[index + 1].isdigit()
    ):
        return None
    # A leading four-digit year is publication/date evidence, not a list index.
    if index - digits_start == 4 and 1500 <= int(text[digits_start:index]) <= 2099:
        return None
    while index < len(text) and not text[index].isalnum():
        index += 1
    return start, index


def _feature_spans(value: str) -> dict[str, list[_Span]]:
    """Extract spans and assign broad evidence to one most-specific owner.

    Relational shapes such as ``author_year`` may still contain their atomic
    evidence. Catch-all lexical and numeric detectors are residual: a span
    already explained by a more specific detector is not counted again.
    """

    start, end = _analysis_bounds(value)
    analysis_value = value[start:end]
    patterns = {
        "year_count": _YEAR,
        "no_date_count": _NO_DATE,
        "numeric_date_count": _NUMERIC_DATE,
        "month_date_count": _MONTH_DATE,
        "access_date_count": _ACCESS_DATE,
        "url_count": _URL,
        "doi_count": _DOI,
        "isbn_count": _ISBN,
        "issn_count": _ISSN,
        "initial_count": _INITIAL,
        "proper_name_word_count": _PROPER_WORD,
        "ampersand_count": _AMPERSAND,
        "quoted_span_count": _QUOTED,
        "editor_term_count": _EDITOR_TERMS,
        "thesis_term_count": _THESIS_TERMS,
        "in_container_count": _IN_CONTAINER,
        "edition_term_count": _EDITION_TERMS,
        "dotted_word_count": _DOTTED_WORD,
        "volume_marker_count": _VOLUME_MARKER,
        "volume_shape_count": _VOLUME_SHAPE,
        "journal_year_volume_count": _JOURNAL_YEAR_VOLUME,
        "page_marker_count": _PAGE_MARKER,
        "publisher_term_count": _PUBLISHER_TERMS,
        "place_name_count": _PLACE_NAMES,
        "place_publisher_shape_count": _PLACE_PUBLISHER_SHAPE,
        "prose_lead_count": _PROSE_LEAD,
    }
    spans = {name: _pattern_spans(pattern, value) for name, pattern in patterns.items()}
    spans["dotted_sequence_count"] = []
    spans["prose_lead_count"] = spans["prose_lead_count"][:1]
    spans["in_container_count"] = _pattern_spans(
        _IN_CONTAINER, value, group="container_term"
    )
    spans["inverted_author_count"] = _pattern_spans(
        _INVERTED_AUTHOR, analysis_value, offset=start
    )
    spans["author_year_count"] = []
    spans["name_initial_pair_count"] = _pattern_spans(
        _NAME_INITIAL_PAIR, analysis_value, offset=start
    )
    spans["direct_author_count"] = _pattern_spans(
        _DIRECT_AUTHOR, analysis_value, offset=start
    )
    author_noise = _joined_spans(
        spans,
        (
            "url_count",
            "doi_count",
            "isbn_count",
            "issn_count",
            "no_date_count",
            "numeric_date_count",
            "month_date_count",
            "quoted_span_count",
            "editor_term_count",
            "thesis_term_count",
            "edition_term_count",
            "volume_marker_count",
            "page_marker_count",
        ),
    )
    spans["inverted_author_count"] = _without_overlaps(
        spans["inverted_author_count"], author_noise
    )
    spans["direct_author_count"] = _without_overlaps(
        spans["direct_author_count"], author_noise
    )
    for orientation in ("inverted_author_count", "direct_author_count"):
        hits = spans[orientation]
        if hits and _AUTHOR_PREFIX.fullmatch(value[start : hits[0][0]]) is None:
            spans[orientation] = []

    # Comma boundaries make a direct list such as ``L. Caires, G. F.
    # Italiano`` look locally like the inverted author ``Caires, G. F.``.
    # Evaluate both orientations over the whole line, then retain only the
    # hypothesis explaining more authors. Coverage and earliest occurrence are
    # deterministic tie-breakers for equally sized hypotheses.
    inverted = spans["inverted_author_count"]
    direct = spans["direct_author_count"]
    if inverted and direct:
        inverted_key = (
            len(inverted),
            sum(hit_end - hit_start for hit_start, hit_end in inverted),
            -inverted[0][0],
        )
        direct_key = (
            len(direct),
            sum(hit_end - hit_start for hit_start, hit_end in direct),
            -direct[0][0],
        )
        if direct_key > inverted_key:
            spans["inverted_author_count"] = []
        else:
            spans["direct_author_count"] = []

    numbered = _numbered_entry_span(analysis_value)
    spans["numbered_entry_count"] = (
        [(start + numbered[0], start + numbered[1])] if numbered is not None else []
    )

    # Page ranges exclude identifiers and full dates. When both endpoints look
    # like years, retain the range as pages only if another publication year is
    # present outside it; this avoids both treating 1980–1990 as pages and
    # treating citation pages 1535–1548 as two publication years.
    page_blockers = _joined_spans(
        spans,
        (
            "url_count",
            "doi_count",
            "isbn_count",
            "issn_count",
            "numeric_date_count",
            "month_date_count",
        ),
    )
    page_ranges: list[_Span] = []
    raw_years = spans["year_count"]
    for match in _PAGE_RANGE.finditer(value):
        span = (match.start(), match.end())
        if any(_overlaps(span, blocker) for blocker in page_blockers):
            continue
        page_start = int(match.group("page_start"))
        page_end = int(match.group("page_end"))
        both_yearlike = 1500 <= page_start <= 2099 and 1500 <= page_end <= 2099
        outside_year = any(not _overlaps(span, year) for year in raw_years)
        if both_yearlike and not outside_year:
            continue
        page_ranges.append(span)
    spans["page_range_count"] = page_ranges

    # A year followed by the first endpoint of a page range is not a genuine
    # year-volume pair (e.g. ``1988:255-263``).
    journal_spans: list[_Span] = []
    for match in _JOURNAL_YEAR_VOLUME.finditer(value):
        span = (match.start(), match.end())
        volume_span = match.span("journal_volume")
        overlaps_page_range = any(
            _overlaps(span, page_range) for page_range in page_ranges
        )
        volume_is_page_start = any(
            _overlaps(volume_span, page_range) for page_range in page_ranges
        )
        if overlaps_page_range and (
            volume_is_page_start or match.group("journal_issue") is None
        ):
            continue
        journal_spans.append(span)
    spans["journal_year_volume_count"] = journal_spans

    # Issue numbers cannot themselves be four-digit publication years.
    spans["volume_shape_count"] = [
        match.span()
        for match in _VOLUME_SHAPE.finditer(value)
        if not 1500 <= int(match.group("issue")) <= 2099
    ]

    # Build author–year only from an author hypothesis that begins after a
    # legal list/bullet prefix. This replaces the former catch-all expression
    # that could consume running prose up to an inline citation.
    author_hits = spans["inverted_author_count"] + spans["direct_author_count"]
    if not author_hits and len(spans["name_initial_pair_count"]) >= 2:
        first_pair = spans["name_initial_pair_count"][0]
        if _AUTHOR_PREFIX.fullmatch(value[start : first_pair[0]]) is not None:
            author_hits = spans["name_initial_pair_count"]
    if author_hits:
        first_author = min(author_hits)
        date_blockers = _joined_spans(
            spans,
            ("numeric_date_count", "month_date_count", "page_range_count"),
        )
        date_candidates = sorted(
            span
            for span in spans["year_count"] + spans["no_date_count"]
            if span[0] >= first_author[1]
            and span[0] - first_author[0] <= 350
            and not any(_overlaps(span, blocker) for blocker in date_blockers)
        )
        if date_candidates:
            spans["author_year_count"] = [
                (first_author[0], date_candidates[0][1])
            ]

    # More-specific ownership rules. These turn the broad detectors into
    # fallback evidence rather than repeated points for one textual event.
    spans["url_count"] = _without_overlaps(spans["url_count"], spans["doi_count"])
    spans["year_count"] = _without_overlaps(
        spans["year_count"],
        _joined_spans(
            spans,
            (
                "numeric_date_count",
                "month_date_count",
                "url_count",
                "doi_count",
                "isbn_count",
                "issn_count",
                "journal_year_volume_count",
                "page_range_count",
                "author_year_count",
            ),
        ),
    )
    spans["editor_term_count"] = _without_overlaps(
        spans["editor_term_count"], spans["edition_term_count"]
    )
    spans["volume_shape_count"] = _without_overlaps(
        spans["volume_shape_count"], spans["journal_year_volume_count"]
    )
    spans["place_name_count"] = _without_overlaps(
        spans["place_name_count"], spans["place_publisher_shape_count"]
    )
    spans["publisher_term_count"] = _without_overlaps(
        spans["publisher_term_count"],
        spans["place_publisher_shape_count"]
        + spans["url_count"]
        + spans["doi_count"],
    )
    spans["ampersand_count"] = _without_overlaps(
        spans["ampersand_count"], spans["url_count"] + spans["doi_count"]
    )
    spans["numbered_entry_count"] = _without_overlaps(
        spans["numbered_entry_count"],
        _joined_spans(
            spans,
            ("numeric_date_count", "month_date_count", "page_range_count"),
        ),
    )
    spans["initial_count"] = _without_overlaps(
        spans["initial_count"],
        _joined_spans(
            spans,
            (
                "no_date_count",
                "quoted_span_count",
                "editor_term_count",
                "thesis_term_count",
                "edition_term_count",
                "volume_marker_count",
                "page_marker_count",
            ),
        ),
    )
    dotted_blockers = _joined_spans(
        spans,
        (
            "url_count",
            "doi_count",
            "no_date_count",
            "initial_count",
            "quoted_span_count",
            "editor_term_count",
            "thesis_term_count",
            "edition_term_count",
            "volume_marker_count",
            "page_marker_count",
            "publisher_term_count",
            "place_name_count",
            "place_publisher_shape_count",
        ),
    )
    residual_dotted = _without_overlaps(spans["dotted_word_count"], dotted_blockers)
    spans["dotted_sequence_count"] = _dotted_sequences(value, residual_dotted)
    spans["dotted_word_count"] = _without_overlaps(
        residual_dotted, spans["dotted_sequence_count"]
    )

    proper_blockers = _joined_spans(
        spans,
        (
            "url_count",
            "doi_count",
            "isbn_count",
            "issn_count",
            "numeric_date_count",
            "month_date_count",
            "access_date_count",
            "quoted_span_count",
            "editor_term_count",
            "thesis_term_count",
            "in_container_count",
            "edition_term_count",
            "dotted_word_count",
            "dotted_sequence_count",
            "volume_marker_count",
            "page_marker_count",
            "publisher_term_count",
            "place_name_count",
            "place_publisher_shape_count",
            "prose_lead_count",
        ),
    )
    spans["proper_name_word_count"] = _without_overlaps(
        spans["proper_name_word_count"], proper_blockers
    )

    stripped_start = len(value) - len(value.lstrip())
    stripped_end = len(value.rstrip())
    spans["table_row_count"] = (
        [(stripped_start, stripped_end)]
        if value.strip().startswith("|") and value.strip().endswith("|")
        else []
    )
    semantic_spans = [
        span
        for name, feature_spans in spans.items()
        if name != "punctuation_count"
        for span in feature_spans
    ]
    spans["punctuation_count"] = _without_overlaps(
        [
            (index, index + 1)
            for index, character in enumerate(value)
            if character in set('.,;:()[]«»“”"')
        ],
        semantic_spans,
    )
    return spans


def _features_and_spans(value: str) -> tuple[BibliographyFeatures, dict[str, list[_Span]]]:
    spans = _feature_spans(value)
    tokens = re.findall(r"[^\W_]+(?:[’'\-][^\W_]+)*", value, re.UNICODE)
    counts = {
        "year_count": len(spans["year_count"]),
        "no_date_count": len(spans["no_date_count"]),
        "numeric_date_count": len(spans["numeric_date_count"]),
        "month_date_count": len(spans["month_date_count"]),
        "access_date_count": len(spans["access_date_count"]),
        "url_count": len(spans["url_count"]),
        "doi_count": len(spans["doi_count"]),
        "isbn_count": len(spans["isbn_count"]),
        "issn_count": len(spans["issn_count"]),
        "initial_count": len(spans["initial_count"]),
        "proper_name_word_count": len(spans["proper_name_word_count"]),
        "inverted_author_count": len(spans["inverted_author_count"]),
        "author_year_count": len(spans["author_year_count"]),
        "name_initial_pair_count": len(spans["name_initial_pair_count"]),
        "direct_author_count": len(spans["direct_author_count"]),
        "numbered_entry_count": len(spans["numbered_entry_count"]),
        "ampersand_count": len(spans["ampersand_count"]),
        "quoted_span_count": len(spans["quoted_span_count"]),
        "editor_term_count": len(spans["editor_term_count"]),
        "thesis_term_count": len(spans["thesis_term_count"]),
        "in_container_count": len(spans["in_container_count"]),
        "edition_term_count": len(spans["edition_term_count"]),
        "dotted_word_count": len(spans["dotted_word_count"]),
        "dotted_sequence_count": len(spans["dotted_sequence_count"]),
        "volume_marker_count": len(spans["volume_marker_count"]),
        "volume_shape_count": len(spans["volume_shape_count"]),
        "journal_year_volume_count": len(spans["journal_year_volume_count"]),
        "page_marker_count": len(spans["page_marker_count"]),
        "page_range_count": len(spans["page_range_count"]),
        "publisher_term_count": len(spans["publisher_term_count"]),
        "place_name_count": len(spans["place_name_count"]),
        "place_publisher_shape_count": len(spans["place_publisher_shape_count"]),
        "punctuation_count": len(spans["punctuation_count"]),
        "prose_lead_count": len(spans["prose_lead_count"]),
        "table_row_count": len(spans["table_row_count"]),
    }
    return BibliographyFeatures(token_count=len(tokens), **counts), spans


def extract_bibliography_features(text: str) -> BibliographyFeatures:
    """Extract the requested English/Greek bibliography signals from one line."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFKC", text)
    features, _ = _features_and_spans(value)
    return features


def extract_bibliography_feature_review(text: str) -> BibliographyFeatureReview:
    """Expose exact, ownership-resolved feature spans for review UIs."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    value = unicodedata.normalize("NFKC", text)
    features, spans = _features_and_spans(value)
    matches = [
        BibliographyFeatureMatch(feature, start, end, value[start:end])
        for feature, feature_spans in spans.items()
        for start, end in feature_spans
    ]
    matches.sort(key=lambda match: (match.start, match.end, match.feature))
    return BibliographyFeatureReview(value, features, tuple(matches))


def _has_date_evidence(features: BibliographyFeatures) -> bool:
    """Include dates owned by composite detectors, not only residual years."""

    return bool(
        features.year_count
        or features.no_date_count
        or features.numeric_date_count
        or features.month_date_count
        or features.author_year_count
        or features.journal_year_volume_count
    )


def _feature_families(features: BibliographyFeatures) -> tuple[str, ...]:
    flags = (
        (
            "author",
            features.inverted_author_count > 0
            or features.author_year_count > 0
            or features.name_initial_pair_count > 0
            or features.direct_author_count > 0,
        ),
        ("names", features.initial_count > 0 or features.proper_name_word_count >= 2),
        (
            "year_date",
            bool(
                features.year_count
                or features.no_date_count
                or features.author_year_count
                or features.journal_year_volume_count
            ),
        ),
        ("full_date", features.numeric_date_count > 0 or features.month_date_count > 0),
        (
            "identifier",
            features.doi_count > 0
            or features.isbn_count > 0
            or features.issn_count > 0,
        ),
        ("web", features.url_count > 0),
        (
            "container",
            features.editor_term_count > 0 or features.in_container_count > 0,
        ),
        (
            "publication",
            features.publisher_term_count > 0
            or features.place_publisher_shape_count > 0,
        ),
        (
            "journal_volume",
            features.volume_marker_count > 0
            or features.volume_shape_count > 0
            or features.journal_year_volume_count > 0,
        ),
        ("pages", features.page_marker_count > 0 or features.page_range_count > 0),
        ("abbreviations", features.dotted_sequence_count > 0),
        ("quoted_title", features.quoted_span_count > 0),
        ("thesis", features.thesis_term_count > 0),
        ("numbered", features.numbered_entry_count > 0),
    )
    return tuple(name for name, enabled in flags if enabled)


def score_bibliography_features(
    features: BibliographyFeatures,
) -> tuple[float, tuple[str, ...]]:
    """Return an additive, inspectable score and the contributions that fired."""

    score = 0.0
    reasons: list[str] = []

    def add(condition: bool, weight: float, code: str) -> None:
        nonlocal score
        if condition:
            score += weight
            reasons.append(code)

    add(features.inverted_author_count > 0, 2.8, "BIB2_AUTHOR_INVERTED")
    add(features.author_year_count > 0, 2.4, "BIB2_AUTHOR_YEAR_SKELETON")
    add(
        features.name_initial_pair_count >= 2,
        2.8,
        "BIB2_REPEATED_NAME_INITIAL_PAIRS",
    )
    add(features.name_initial_pair_count == 1, 0.8, "BIB2_NAME_INITIAL_PAIR")
    add(features.direct_author_count > 0, 1.8, "BIB2_DIRECT_AUTHOR_SHAPE")
    add(features.initial_count >= 2, 1.4, "BIB2_MULTIPLE_INITIALS")
    add(features.proper_name_word_count >= 2, 0.6, "BIB2_PROPER_NAME_SHAPE")
    add(
        features.year_count > 0 or features.no_date_count > 0,
        1.0,
        "BIB2_YEAR_OR_NO_DATE",
    )
    add(
        features.numeric_date_count > 0 or features.month_date_count > 0,
        0.7,
        "BIB2_FULL_DATE",
    )
    add(features.access_date_count > 0, 0.8, "BIB2_ACCESS_DATE")
    add(features.doi_count > 0, 3.2, "BIB2_DOI")
    add(features.isbn_count > 0, 3.0, "BIB2_ISBN")
    add(features.issn_count > 0, 2.7, "BIB2_ISSN")
    add(features.url_count > 0, 1.4, "BIB2_URL")
    add(features.numbered_entry_count > 0, 1.0, "BIB2_NUMBERED_ENTRY")
    add(features.ampersand_count > 0, 0.3, "BIB2_AMPERSAND")
    add(features.quoted_span_count > 0, 0.6, "BIB2_QUOTED_TITLE_SHAPE")
    add(features.editor_term_count > 0, 1.1, "BIB2_EDITOR_TRANSLATOR_TERM")
    add(features.thesis_term_count > 0, 1.2, "BIB2_THESIS_TERM")
    add(features.in_container_count > 0, 0.5, "BIB2_IN_CONTAINER")
    add(features.edition_term_count > 0, 0.7, "BIB2_EDITION_TERM")
    add(features.dotted_sequence_count > 0, 1.2, "BIB2_DOTTED_ABBREVIATION_SEQUENCE")
    add(features.dotted_word_count >= 3, 0.5, "BIB2_DOTTED_WORD_DENSITY")
    add(
        features.volume_marker_count > 0 or features.volume_shape_count > 0,
        1.0,
        "BIB2_VOLUME_ISSUE",
    )
    add(features.journal_year_volume_count > 0, 1.5, "BIB2_YEAR_VOLUME_SEQUENCE")
    add(features.page_marker_count > 0, 0.9, "BIB2_PAGE_MARKER")
    add(features.page_range_count > 0, 0.9, "BIB2_PAGE_RANGE")
    add(features.publisher_term_count > 0, 0.9, "BIB2_PUBLISHER_TERM")
    add(features.place_name_count > 0, 0.4, "BIB2_PLACE_LEXICON")
    add(features.place_publisher_shape_count > 0, 1.0, "BIB2_PLACE_PUBLISHER_SHAPE")

    authorish = (
        features.inverted_author_count > 0
        or features.author_year_count > 0
        or features.name_initial_pair_count >= 2
        or features.direct_author_count > 0
        or features.initial_count >= 2
    )
    dated = _has_date_evidence(features)
    publication_tail = bool(
        features.publisher_term_count
        or features.place_publisher_shape_count
        or features.volume_marker_count
        or features.volume_shape_count
        or features.page_marker_count
        or features.page_range_count
        or features.doi_count
        or features.isbn_count
    )
    add(authorish and dated, 1.5, "BIB2_INTERACTION_AUTHOR_DATE")
    add(dated and publication_tail, 1.0, "BIB2_INTERACTION_DATE_PUBLICATION")
    add(
        features.dotted_sequence_count > 0
        and dated
        and (
            features.volume_marker_count > 0 or features.journal_year_volume_count > 0
        ),
        1.4,
        "BIB2_INTERACTION_JOURNAL_DATE_VOLUME",
    )
    add(
        features.url_count > 0 and features.access_date_count > 0,
        1.0,
        "BIB2_INTERACTION_URL_ACCESS_DATE",
    )
    add(
        features.quoted_span_count > 0 and (authorish or dated),
        0.6,
        "BIB2_INTERACTION_QUOTED_CITATION",
    )

    if features.prose_lead_count:
        score -= 3.0
        reasons.append("BIB2_NEGATIVE_PROSE_LEAD")
    if features.token_count >= 40 and not authorish and not publication_tail:
        score -= 1.5
        reasons.append("BIB2_NEGATIVE_LONG_UNANCHORED_PROSE")
    if features.table_row_count:
        score -= 2.0
        reasons.append("BIB2_NEGATIVE_TABLE_ROW")
    return score, tuple(reasons)


def analyze_bibliography_line_v2(
    text: str, line_index: int = 0
) -> BibliographyV2Evidence:
    """Classify local evidence while retaining R2's conservative hard vetoes."""

    if line_index < 0:
        raise ValueError("line_index must be non-negative")
    features = extract_bibliography_features(text)
    base = analyze_bib_line(text, line_index)
    stripped = unicodedata.normalize("NFKC", text).strip()
    heading_text = stripped.lstrip("#").strip()

    if not stripped:
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.POSSIBLE_CONTINUATION,
            0.0,
            ("BIB2_BLANK_FORMATTING_GAP",),
            False,
            0,
            (),
            (),
            features,
        )
    if base.role in {BibRole.HEADING, BibRole.SUBHEADING}:
        return BibliographyV2Evidence(
            line_index,
            text,
            base.role,
            base.score,
            tuple(code.replace("BIB_", "BIB2_") for code in base.reason_codes),
            False,
            features.token_count,
            base.evidence_families,
            base.citation_styles,
            features,
        )
    if _BIB_EXTENDED_HEADING.fullmatch(heading_text):
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.HEADING,
            4.0,
            ("BIB2_EXTENDED_HEADING",),
            False,
            features.token_count,
            (),
            (),
            features,
        )
    if _BIB_EXTENDED_SUBHEADING.fullmatch(heading_text):
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.SUBHEADING,
            2.5,
            ("BIB2_EXTENDED_SUBHEADING",),
            False,
            features.token_count,
            (),
            ("formal_source",),
            features,
        )
    if _BIB_HEADING_WORD.fullmatch(
        unicodedata.normalize("NFKD", stripped).casefold().replace("́", "")
    ):
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.HEADING,
            4.0,
            ("BIB2_EXACT_HEADING",),
            False,
            features.token_count,
            (),
            (),
            features,
        )
    strong_table_citation = bool(
        features.table_row_count
        and _has_date_evidence(features)
        and (
            features.inverted_author_count
            or features.author_year_count
            or features.name_initial_pair_count >= 2
        )
    )
    if (
        features.table_row_count
        and not strong_table_citation
        and not (features.doi_count or features.isbn_count or features.issn_count)
    ):
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            -4.0,
            ("BIB2_NEGATIVE_NONCITATION_TABLE_ROW",),
            True,
            features.token_count,
            _feature_families(features),
            (),
            features,
        )

    author_specific = bool(
        features.inverted_author_count
        or features.author_year_count
        or features.name_initial_pair_count >= 2
    )
    identifier_specific = bool(
        features.doi_count or features.isbn_count or features.issn_count
    )
    if _FIGURE_CAPTION_START.match(stripped):
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            -4.0,
            ("BIB2_NEGATIVE_FIGURE_CAPTION",),
            True,
            features.token_count,
            _feature_families(features),
            (),
            features,
        )
    if (
        features.token_count >= 22
        and _ENUMERATED_PROSE_START.match(stripped)
        and not author_specific
        and not identifier_specific
    ):
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            -4.0,
            ("BIB2_NEGATIVE_LONG_ENUMERATED_PROSE",),
            True,
            features.token_count,
            _feature_families(features),
            (),
            features,
        )

    base_negative_codes = {code for code in base.reason_codes if "NEGATIVE" in code}
    override_running_prose = base_negative_codes == {
        "BIB_NEGATIVE_RUNNING_PROSE"
    } and bool(
        features.name_initial_pair_count >= 2
        or features.doi_count
        or features.isbn_count
        or (
            features.journal_year_volume_count
            and (features.page_marker_count or features.page_range_count)
        )
        or (
            (features.publisher_term_count or features.place_publisher_shape_count)
            and (
                _has_date_evidence(features)
                or features.page_marker_count
                or features.page_range_count
            )
        )
    )
    override_statistical_table = strong_table_citation and base_negative_codes == {
        "BIB_NEGATIVE_STATISTICAL_TABLE"
    }
    if base.hard_negative and not (
        override_running_prose or override_statistical_table
    ):
        return BibliographyV2Evidence(
            line_index,
            text,
            BibRole.HARD_OTHER,
            base.score,
            tuple(code.replace("BIB_", "BIB2_") for code in base.reason_codes),
            True,
            features.token_count,
            _feature_families(features),
            (),
            features,
        )

    score, reasons = score_bibliography_features(features)
    families = _feature_families(features)
    styles: list[str] = []
    if features.numbered_entry_count:
        styles.append("numbered")
    if (
        features.inverted_author_count
        or features.author_year_count
        or features.name_initial_pair_count
        or features.direct_author_count
    ):
        styles.append("author")
    if features.author_year_count or (
        features.inverted_author_count
        and _has_date_evidence(features)
    ):
        styles.append("author_year")
    if features.url_count:
        styles.append("web")
    if features.doi_count or features.isbn_count or features.issn_count:
        styles.append("identifier")
    if (
        features.volume_marker_count
        or features.volume_shape_count
        or features.journal_year_volume_count
    ):
        styles.append("journal")

    specific_identifier = identifier_specific
    publication_tail = bool(
        features.url_count
        or specific_identifier
        or features.publisher_term_count
        or features.place_publisher_shape_count
        or features.volume_marker_count
        or features.volume_shape_count
        or features.journal_year_volume_count
        or features.page_marker_count
        or features.page_range_count
    )
    dated = _has_date_evidence(features)
    anchor = bool(
        features.inverted_author_count
        or features.author_year_count
        or (features.name_initial_pair_count >= 2 and (dated or publication_tail))
        or (features.direct_author_count and (dated or publication_tail))
        or (features.numbered_entry_count and (dated or publication_tail))
        or (features.initial_count >= 2 and dated)
        or specific_identifier
    )
    if score >= 4.5 and anchor and len(families) >= 2:
        role = BibRole.STRONG_ENTRY_START
        reasons = reasons + ("BIB2_STRONG_MULTI_FAMILY_ENTRY",)
    elif score >= 2.4 and anchor:
        role = BibRole.WEAK_ENTRY_START
        reasons = reasons + ("BIB2_WEAK_MULTI_FAMILY_ENTRY",)
    elif publication_tail and score >= 1.2:
        role = BibRole.POSSIBLE_CONTINUATION
        reasons = reasons + ("BIB2_PUBLICATION_TAIL",)
    elif features.token_count <= 8 and (
        text[:1].isspace() or stripped.endswith((",", ";", "."))
    ):
        role = BibRole.POSSIBLE_CONTINUATION
        reasons = reasons + ("BIB2_SHORT_WRAPPED_FRAGMENT",)
    else:
        role = BibRole.OTHER
        if not reasons:
            reasons = ("BIB2_NO_LOCAL_EVIDENCE",)

    return BibliographyV2Evidence(
        line_index,
        text,
        role,
        score,
        reasons,
        False,
        features.token_count,
        families,
        tuple(styles),
        features,
    )


def analyze_bibliography_lines_v2(
    lines: Sequence[str], *, start_line_index: int = 0
) -> tuple[BibliographyV2Evidence, ...]:
    return tuple(
        analyze_bibliography_line_v2(text, start_line_index + offset)
        for offset, text in enumerate(lines)
    )


def hard_gap_evidence(line_index: int) -> BibliographyV2Evidence:
    """Create a barrier for unrepresented physical content."""

    features = extract_bibliography_features("")
    return BibliographyV2Evidence(
        line_index,
        "",
        BibRole.HARD_OTHER,
        -4.0,
        ("BIB2_NEGATIVE_UNREPRESENTED_PHYSICAL_GAP",),
        True,
        0,
        (),
        (),
        features,
    )


def _is_anchor(item: BibliographyV2Evidence) -> bool:
    return item.role == BibRole.STRONG_ENTRY_START or (
        item.role == BibRole.WEAK_ENTRY_START and item.score >= 3.0
    )


def _gap_is_bridgeable(items: Sequence[BibliographyV2Evidence]) -> bool:
    if len(items) > 3 or any(item.hard_negative for item in items):
        return False
    if sum(item.token_count for item in items) > 45:
        return False
    for item in items:
        if item.role in {BibRole.HEADING, BibRole.SUBHEADING}:
            continue
        if item.role in {BibRole.WEAK_ENTRY_START, BibRole.POSSIBLE_CONTINUATION}:
            continue
        if item.token_count <= 12:
            continue
        return False
    return True


def _denied_positions(evidence: Sequence[BibliographyV2Evidence]) -> set[int]:
    denied: set[int] = set()
    active = False
    for position, item in enumerate(evidence):
        codes = set(item.reason_codes)
        if (
            "BIB2_NEGATIVE_CV_PUBLICATIONS_HEADING" in codes
            or "BIB2_NEGATIVE_NOTES_HEADING" in codes
        ):
            active = True
        elif item.role == BibRole.HEADING or "BIB2_NEGATIVE_BODY_HEADING" in codes:
            active = False
        if active:
            denied.add(position)
    return denied


def _anchor_clusters(
    evidence: Sequence[BibliographyV2Evidence], denied: set[int]
) -> list[list[int]]:
    anchors = [
        position
        for position, item in enumerate(evidence)
        if position not in denied and _is_anchor(item)
    ]
    if not anchors:
        return []
    clusters: list[list[int]] = [[anchors[0]]]
    for position in anchors[1:]:
        previous = clusters[-1][-1]
        between = evidence[previous + 1 : position]
        if _gap_is_bridgeable(between):
            clusters[-1].append(position)
        else:
            clusters.append([position])
    return clusters


def decode_bibliography_blocks_v2(
    evidence: Sequence[BibliographyV2Evidence],
) -> tuple[StructureSpan, ...]:
    """Propose only heading-confirmed or dense headerless bibliography runs.

    Neutral short fragments are included *only* when bounded by citation
    anchors.  This is the deterministic coherence check: a weak line becomes a
    bibliography line because of its position inside a confirmed run, not
    because it happens to contain a year or a capitalised word.
    """

    if not evidence:
        return ()
    indexes = [item.line_index for item in evidence]
    if any(right <= left for left, right in zip(indexes, indexes[1:])):
        raise ValueError("evidence must have strictly increasing line indexes")

    denied = _denied_positions(evidence)
    spans: list[StructureSpan] = []
    for cluster in _anchor_clusters(evidence, denied):
        first_anchor = cluster[0]
        last_anchor = cluster[-1]
        heading_position: int | None = None
        for position in range(first_anchor - 1, max(-1, first_anchor - 13), -1):
            item = evidence[position]
            if item.hard_negative:
                break
            if item.role == BibRole.HEADING:
                heading_position = position
                break
            if not _gap_is_bridgeable(evidence[position:first_anchor]):
                break

        heading_confirmed = heading_position is not None and len(cluster) >= 2
        headerless_confirmed = len(cluster) >= 3
        if not (heading_confirmed or headerless_confirmed):
            continue

        start = heading_position if heading_confirmed else first_anchor
        end = last_anchor
        # Include only explicit citation tails after the final anchor; arbitrary
        # neutral text is never appended at an open edge.
        for position in range(last_anchor + 1, min(len(evidence), last_anchor + 4)):
            item = evidence[position]
            if item.hard_negative:
                break
            if item.role in {
                BibRole.SUBHEADING,
                BibRole.WEAK_ENTRY_START,
                BibRole.POSSIBLE_CONTINUATION,
            } and (item.score >= 1.2 or "BIB2_PUBLICATION_TAIL" in item.reason_codes):
                end = position
                continue
            break

        included = evidence[start : end + 1]
        support_positions = set(cluster)
        if heading_position is not None:
            support_positions.add(heading_position)
        supporting = tuple(
            evidence[position].line_index for position in sorted(support_positions)
        )
        bridged = tuple(
            item.line_index
            for position, item in enumerate(evidence[start : end + 1], start)
            if position not in support_positions
        )
        seed_kind = (
            "bib2_heading_confirmed"
            if heading_confirmed
            else "bib2_headerless_dense_run"
        )
        reason_codes = ["BIB2_COHERENT_BLOCK", seed_kind.upper()]
        if bridged:
            reason_codes.append("BIB2_BOUNDED_CONTEXT_BRIDGED")
        terminator = evidence[end + 1].line_index if end + 1 < len(evidence) else None
        spans.append(
            StructureSpan(
                StructureKind.BIB,
                included[0].line_index,
                included[-1].line_index,
                tuple(item.line_index for item in included),
                supporting,
                bridged,
                supporting,
                seed_kind,
                tuple(reason_codes),
                terminator,
            )
        )

    # Anchor clusters cannot overlap, but a heading may be close to two runs.
    # Merge any such adjacent proposals into a single unambiguous block.
    merged: list[StructureSpan] = []
    for span in spans:
        if not merged or span.start_line_index > merged[-1].end_line_index:
            merged.append(span)
            continue
        previous = merged.pop()
        line_indices = tuple(dict.fromkeys(previous.line_indices + span.line_indices))
        supporting = tuple(
            dict.fromkeys(
                previous.supporting_line_indices + span.supporting_line_indices
            )
        )
        bridged = tuple(index for index in line_indices if index not in set(supporting))
        merged.append(
            StructureSpan(
                StructureKind.BIB,
                min(previous.start_line_index, span.start_line_index),
                max(previous.end_line_index, span.end_line_index),
                line_indices,
                supporting,
                bridged,
                supporting,
                "bib2_merged_coherent_runs",
                ("BIB2_COHERENT_BLOCK", "BIB2_MERGED_ADJACENT_RUNS"),
                span.terminator_line_index,
            )
        )
    return tuple(merged)
