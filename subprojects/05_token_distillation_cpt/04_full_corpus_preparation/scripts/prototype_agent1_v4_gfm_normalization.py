#!/usr/bin/env python3
"""Prototype loss-aware HTML-to-GFM normalization on the Agent 1 review sample.

This is deliberately outside GlossAPI.  It reads the immutable raw-review site,
applies the current GlossAPI complex-repetition cleaner, converts recognized HTML
structure to GitHub-Flavored Markdown, and writes lazy presentation artifacts.
Raw review documents are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import html.entities
import importlib.util
import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


SCHEMA = "agent1_v4_gfm_normalization_audit_v4"
DOCUMENT_SCHEMA = "agent1_v4_gfm_normalized_document_v4"
SITE_SCHEMA = "agent1_v4_raw_review_site_manifest_v1"
AUDIT_RELATIVE_PATH = Path("data/gfm_normalization_audit.json")
OUTPUT_DOCUMENT_DIR = Path("data/gfm/documents")
DEFAULT_GLOSSAPI_ROOT = Path.home() / "Projects/glossapi-development"
REPETITION_COMMENT = "<!-- repeating-text-removed -->"
DESCRIPTION_OF_REMOVED_IMAGE_COMMENT = "<!-- description-of-removed-image -->"
LITERAL_AMPERSAND_SENTINEL = "\ue000GFM_LITERAL_AMPERSAND\ue001"
PORTABLE_BULK_PREFIXES = ("data/documents/", "data/gfm/documents/")
PRESERVED_MARKDOWN_TOKEN_TYPES = (
    "heading_open",
    "fence",
    "code_block",
    "code_inline",
    "link_open",
    "image",
    "bullet_list_open",
    "ordered_list_open",
    "blockquote_open",
    "strong_open",
    "em_open",
    "s_open",
    "table_open",
)

AUTOLINK_RE = re.compile(
    r"^<(?:[A-Za-z][A-Za-z0-9+.-]{1,31}:[^<>\s]*|"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?)>$"
)
FIXED_PIPELINE_COMMENT_PATTERN = r"(?:repeating-text-removed|text-missing|table-removed)"
DESCRIPTION_OF_REMOVED_IMAGE_PATTERN = r"description-of-removed-image(?:[ \t]*:[ \t]*[^\r\n]*?)?"
ALLOWED_COMMENT_RE = re.compile(
    rf"^<!--[ \t]*(?:{FIXED_PIPELINE_COMMENT_PATTERN}|{DESCRIPTION_OF_REMOVED_IMAGE_PATTERN})[ \t]*-->$",
    re.IGNORECASE,
)
ALLOWED_COMMENT_TOKEN_RE = re.compile(
    rf"<!--[ \t]*(?:{FIXED_PIPELINE_COMMENT_PATTERN}|{DESCRIPTION_OF_REMOVED_IMAGE_PATTERN})[ \t]*-->",
    re.IGNORECASE,
)
DESCRIPTION_OF_REMOVED_IMAGE_TOKEN_RE = re.compile(
    rf"<!--[ \t]*{DESCRIPTION_OF_REMOVED_IMAGE_PATTERN}[ \t]*-->",
    re.IGNORECASE,
)
FIXED_PIPELINE_COMMENT_TOKEN_RE = re.compile(
    rf"<!--[ \t]*{FIXED_PIPELINE_COMMENT_PATTERN}[ \t]*-->",
    re.IGNORECASE,
)
RESIDUAL_ANGLE_RE = re.compile(r"<[^<>\n]+>")
ALIGN_RE = re.compile(r"text-align\s*:\s*(left|center|right)", re.IGNORECASE)
GENERATED_IMAGE_BASENAME_RE = re.compile(
    r"(?i)^[0-9a-f]{32,64}(?:_[0-9]+)+_img\.(?:avif|bmp|gif|jpe?g|png|tiff?|webp)$"
)
GENERATED_IMAGE_TOKEN_PATTERN = (
    r"(?<![0-9A-Za-z])(?:[^\s()<>{}\[\]]*/)?"
    r"[0-9a-f]{32,64}(?:_[0-9]+)+_img\.(?:avif|bmp|gif|jpe?g|png|tiff?|webp)"
    r"(?![0-9A-Za-z])"
)
GENERATED_IMAGE_TOKEN_RE = re.compile(GENERATED_IMAGE_TOKEN_PATTERN, re.IGNORECASE)
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)

TABLE_TAGS = {"table", "thead", "tbody", "tfoot", "tr", "th", "td", "caption"}
VOID_TAGS = {"br", "hr", "img", "input", "meta", "link", "source", "track", "wbr"}
DROP_WITH_CONTENT_TAGS = {
    "script",
    "style",
    "canvas",
    "svg",
    "head",
    "template",
}
BLOCK_WRAPPER_TAGS = {
    "address",
    "article",
    "aside",
    "center",
    "details",
    "dialog",
    "div",
    "figcaption",
    "figure",
    "footer",
    "header",
    "main",
    "nav",
    "p",
    "section",
    "summary",
    "iframe",
    "object",
    "embed",
    "video",
    "audio",
    "picture",
    "form",
    "button",
    "select",
    "option",
    "textarea",
    "title",
    "noscript",
}
FLATTEN_TAGS = {
    "abbr",
    "acronym",
    "big",
    "cite",
    "dfn",
    "font",
    "ins",
    "kbd",
    "mark",
    "q",
    "ruby",
    "rp",
    "rt",
    "samp",
    "small",
    "span",
    "time",
    "u",
    "var",
}
INLINE_TAGS = {"a", "b", "strong", "i", "em", "del", "s", "strike", "code", "sup", "sub", "math"}
LIST_TAGS = {"ul", "ol", "li", "dl", "dt", "dd"}
HEADING_TAGS = {f"h{level}" for level in range(1, 7)}
KNOWN_HTML_TAGS = (
    TABLE_TAGS
    | VOID_TAGS
    | DROP_WITH_CONTENT_TAGS
    | BLOCK_WRAPPER_TAGS
    | FLATTEN_TAGS
    | INLINE_TAGS
    | LIST_TAGS
    | HEADING_TAGS
    | {"blockquote", "pre"}
)
KNOWN_HTML_TAG_RE = re.compile(
    r"</?(?:" + "|".join(sorted(map(re.escape, KNOWN_HTML_TAGS), key=len, reverse=True)) + r")(?:\s[^<>]*?)?/?>",
    re.IGNORECASE,
)


TRANSFORMATION_POLICY: list[dict[str, object]] = [
    {
        "tags": ["table", "thead", "tbody", "tfoot", "tr", "th", "td"],
        "target": "GFM pipe table or readable line fallback",
        "content_policy": "Keep every cell once. Convert rectangular geometry, expand spans with empty cells, use the first header row, and synthesize an empty header only when needed. Nested or damaged geometry becomes one cell per line with blank lines between rows.",
    },
    {
        "tags": ["caption"],
        "target": "italic paragraph before table",
        "content_policy": "Keep caption text because GFM tables have no caption field.",
    },
    {
        "tags": ["b", "strong"],
        "target": "**strong emphasis**",
        "content_policy": "Keep content and surrounding whitespace.",
    },
    {
        "tags": ["i", "em"],
        "target": "*emphasis*",
        "content_policy": "Keep content and surrounding whitespace.",
    },
    {
        "tags": ["del", "s", "strike"],
        "target": "~~strikethrough~~",
        "content_policy": "Keep content.",
    },
    {
        "tags": ["br"],
        "target": "GFM hard break; whitespace inside table cells",
        "content_policy": "Keep the boundary without retaining raw <br> HTML or inventing visible punctuation.",
    },
    {
        "tags": ["p", "div", "section", "article", "main", "header", "footer", "aside", "center"],
        "target": "Markdown paragraph/block boundary",
        "content_policy": "Keep textual and Markdown content; discard layout attributes.",
    },
    {
        "tags": ["h1", "h2", "h3", "h4", "h5", "h6"],
        "target": "# through ###### ATX headings",
        "content_policy": "Keep heading level and content; discard HTML attributes.",
    },
    {
        "tags": ["ul", "ol", "li"],
        "target": "Markdown list",
        "content_policy": "Keep list order and items; retain item text separated by whitespace inside table cells.",
    },
    {
        "tags": ["a"],
        "target": "[label](destination)",
        "content_policy": "Keep safe destinations; flatten links with missing, javascript:, or data: destinations to their labels.",
    },
    {
        "tags": ["code", "pre"],
        "target": "inline code or fenced code block",
        "content_policy": "Choose a delimiter longer than any backtick run already in the payload.",
    },
    {
        "tags": ["blockquote", "hr"],
        "target": "> block quote or --- thematic break",
        "content_policy": "Keep quoted content and structural boundaries.",
    },
    {
        "tags": ["sup", "sub", "u", "span"],
        "target": "plain inline content",
        "content_policy": "GFM has no pure non-HTML equivalent for these styles; remove the element and attributes but retain its text to avoid corpus loss.",
    },
    {
        "tags": ["math"],
        "target": "$inline math$ or $$display math$$",
        "content_policy": "Keep the existing TeX-like payload.",
    },
    {
        "tags": ["img"],
        "target": "description-of-removed-image provenance comment or ![alt](source)",
        "content_policy": "Generated extraction images and source-less image alt text become hidden provenance comments. Non-artifact sources remain Markdown images; adjacent prose is never classified semantically.",
    },
    {
        "tags": ["input"],
        "target": "removed",
        "content_policy": "The observed checkboxes are inline OCR artifacts, not GFM task-list items.",
    },
    {
        "tags": ["script", "style", "canvas", "svg", "head", "template"],
        "target": "removed with content",
        "content_policy": "Executable, styling, vector-path, metadata, and template payloads have no document-text representation.",
    },
    {
        "tags": ["iframe", "object", "embed", "video", "audio", "picture", "form", "button", "select", "option", "textarea", "title", "noscript"],
        "target": "plain block content",
        "content_policy": "Remove the unsupported container and attributes but retain readable fallback text nodes.",
    },
    {
        "tags": ["unknown tag-like angle text"],
        "target": "escaped literal text",
        "content_policy": "Preserve OCR text and angle-bracketed citations literally; preserve valid GFM URI/email autolinks unchanged.",
    },
    {
        "tags": ["HTML comments"],
        "target": "removed",
        "content_policy": "Preserve only explicit GlossAPI removal markers and description-of-removed-image provenance comments.",
    },
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: object) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _generated_image_destination(value: str) -> bool:
    candidate = value.strip()
    if candidate.startswith("<") and ">" in candidate:
        candidate = candidate[1 : candidate.index(">")]
    else:
        candidate = candidate.split(None, 1)[0] if candidate else ""
    candidate = candidate.split("?", 1)[0].split("#", 1)[0].replace("\\", "/")
    return bool(GENERATED_IMAGE_BASENAME_RE.fullmatch(candidate.rsplit("/", 1)[-1]))


def _balanced_end(text: str, start: int, opening: str, closing: str) -> int | None:
    depth = 1
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return index
    return None


class _SingleImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.attrs: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.casefold() == "img" and not self.attrs:
            self.attrs = {key.casefold(): value or "" for key, value in attrs}

    handle_startendtag = handle_starttag


def _normalize_image_description(value: str) -> str:
    """Make image alt text readable and safe inside one HTML comment line."""

    normalized = re.sub(r"\s+", " ", html.unescape(value)).strip()
    normalized = normalized.replace("<", "&lt;").replace(">", "&gt;")
    return normalized.replace("--", "&#45;&#45;")


def _image_description_comment(value: str) -> tuple[str, dict[str, int | bool]]:
    """Wrap a description without nesting existing pipeline comments."""

    pieces: list[str] = []
    description_characters = 0
    comment_count = 0
    pipeline_marker_count = 0
    cursor = 0
    for marker in FIXED_PIPELINE_COMMENT_TOKEN_RE.finditer(value):
        description = _normalize_image_description(value[cursor : marker.start()])
        if description:
            pieces.append(f"<!-- description-of-removed-image: {description} -->")
            description_characters += len(description)
            comment_count += 1
        pieces.append(marker.group(0))
        pipeline_marker_count += 1
        cursor = marker.end()
    description = _normalize_image_description(value[cursor:])
    if description:
        pieces.append(f"<!-- description-of-removed-image: {description} -->")
        description_characters += len(description)
        comment_count += 1
    empty_description = not pieces
    if empty_description:
        pieces.append(DESCRIPTION_OF_REMOVED_IMAGE_COMMENT)
        comment_count = 1
    return " ".join(pieces), {
        "description_character_count": description_characters,
        "description_comment_count": comment_count,
        "pipeline_marker_count": pipeline_marker_count,
        "description_split_around_pipeline_marker": pipeline_marker_count > 0,
        "empty_description": empty_description,
    }


def _image_event(
    *,
    rule: str,
    original: str,
    replacement: str,
    artifact_characters_removed: int,
    comment_metrics: Mapping[str, int | bool] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "rule": rule,
        "original": original,
        "replacement": replacement,
        "artifact_characters_removed": max(0, artifact_characters_removed),
    }
    if comment_metrics is not None:
        event.update(comment_metrics)
        event["image_description_commented"] = True
    return event


def _clean_html_artifact_images(text: str, events: list[dict[str, object]]) -> str:
    def replace(match: re.Match[str]) -> str:
        parser = _SingleImageParser()
        parser.feed(match.group(0))
        source = parser.attrs.get("src", "")
        alt = parser.attrs.get("alt", "").strip()
        generated_source = _generated_image_destination(source)
        source_less_description = not source and bool(alt)
        if not generated_source and not source_less_description:
            return match.group(0)
        replacement, comment_metrics = _image_description_comment(alt)
        events.append(
            _image_event(
                rule=(
                    "html_generated_image_to_description_comment"
                    if generated_source
                    else "source_less_html_image_to_description_comment"
                ),
                original=match.group(0),
                replacement=replacement,
                artifact_characters_removed=len(match.group(0)) - len(html.unescape(alt)),
                comment_metrics=comment_metrics,
            )
        )
        return replacement

    return HTML_IMAGE_RE.sub(replace, text)


def _clean_markdown_artifact_destinations(text: str, events: list[dict[str, object]]) -> str:
    pieces: list[str] = []
    cursor = 0
    index = 0
    while index < len(text):
        image = text[index] == "!" and index + 1 < len(text) and text[index + 1] == "["
        link = text[index] == "["
        if not image and not link:
            index += 1
            continue
        label_start = index + 2 if image else index + 1
        label_end = _balanced_end(text, label_start, "[", "]")
        if label_end is None:
            index += 1
            continue
        destination_start = label_end + 1
        if destination_start >= len(text) or text[destination_start] != "(":
            index = label_end + 1
            continue
        destination_end = _balanced_end(text, destination_start + 1, "(", ")")
        if destination_end is None:
            index = label_end + 1
            continue
        destination = text[destination_start + 1 : destination_end]
        if not _generated_image_destination(destination):
            index = destination_end + 1
            continue
        label = text[label_start:label_end]
        original = text[index : destination_end + 1]
        if image:
            replacement, comment_metrics = _image_description_comment(label)
            rule = "markdown_generated_image_to_description_comment"
        else:
            replacement = label
            comment_metrics = None
            rule = "markdown_generated_image_link_to_label"
        pieces.extend((text[cursor:index], replacement))
        events.append(
            _image_event(
                rule=rule,
                original=original,
                replacement=replacement,
                artifact_characters_removed=len(original) - len(label),
                comment_metrics=comment_metrics,
            )
        )
        cursor = destination_end + 1
        index = cursor
    pieces.append(text[cursor:])
    return "".join(pieces)


def clean_generated_image_artifacts(
    text: str, *, metrics: MutableMapping[str, object] | None = None
) -> str:
    """Remove generated-image targets and mark image descriptions as provenance."""

    events: list[dict[str, object]] = []
    cleaned = _clean_html_artifact_images(str(text or ""), events)
    cleaned = _clean_markdown_artifact_destinations(cleaned, events)

    parenthesized = re.compile(
        r"\(\s*(?P<target>" + GENERATED_IMAGE_TOKEN_PATTERN + r")\s*\)", re.IGNORECASE
    )

    def remove_parenthesized(match: re.Match[str]) -> str:
        events.append(
            _image_event(
                rule="parenthesized_generated_image_target_removed",
                original=match.group(0),
                replacement="",
                artifact_characters_removed=len(match.group(0)),
            )
        )
        return ""

    cleaned = parenthesized.sub(remove_parenthesized, cleaned)

    def remove_bare(match: re.Match[str]) -> str:
        events.append(
            _image_event(
                rule="bare_generated_image_target_removed",
                original=match.group(0),
                replacement="",
                artifact_characters_removed=len(match.group(0)),
            )
        )
        return ""

    cleaned = GENERATED_IMAGE_TOKEN_RE.sub(remove_bare, cleaned)
    if metrics is not None:
        counts = Counter(str(event["rule"]) for event in events)
        metrics.clear()
        metrics.update(
            {
                "generated_image_artifact_count": len(events),
                "generated_image_rule_counts": dict(sorted(counts.items())),
                "generated_image_events": events,
                "generated_image_characters_removed": sum(
                    int(event["artifact_characters_removed"]) for event in events
                ),
                "image_description_elements_commented": sum(
                    bool(event.get("image_description_commented")) for event in events
                ),
                "image_description_comments_emitted": sum(
                    int(event.get("description_comment_count", 0)) for event in events
                ),
                "image_description_characters_preserved": sum(
                    int(event.get("description_character_count", 0)) for event in events
                ),
                "empty_image_description_comments": sum(
                    bool(event.get("empty_description")) for event in events
                ),
                "image_descriptions_split_around_pipeline_markers": sum(
                    bool(event.get("description_split_around_pipeline_marker")) for event in events
                ),
                "pipeline_markers_preserved_inside_image_descriptions": sum(
                    int(event.get("pipeline_marker_count", 0)) for event in events
                ),
            }
        )
    return cleaned


@dataclass
class Node:
    tag: str | None
    attrs: dict[str, str] = field(default_factory=dict)
    children: list[Node | str] = field(default_factory=list)
    malformed: bool = False
    source_line: int | None = None
    source_column: int | None = None


@dataclass
class RenderContext:
    in_table_cell: bool = False
    list_depth: int = 0


@dataclass
class NormalizationMetrics:
    tag_counts: Counter[str] = field(default_factory=Counter)
    attribute_counts: Counter[str] = field(default_factory=Counter)
    transformations: Counter[str] = field(default_factory=Counter)
    pseudo_tags_escaped: int = 0
    comments_removed: int = 0
    comments_preserved: int = 0
    content_characters_removed: int = 0
    parser_unmatched_end_tags: int = 0
    parser_implicitly_closed_tags: int = 0
    table_fallback_events: list[dict[str, object]] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "tag_counts": dict(sorted(self.tag_counts.items())),
            "attribute_counts": dict(sorted(self.attribute_counts.items())),
            "transformations": dict(sorted(self.transformations.items())),
            "pseudo_tags_escaped": self.pseudo_tags_escaped,
            "comments_removed": self.comments_removed,
            "comments_preserved": self.comments_preserved,
            "content_characters_removed": self.content_characters_removed,
            "parser_unmatched_end_tags": self.parser_unmatched_end_tags,
            "parser_implicitly_closed_tags": self.parser_implicitly_closed_tags,
            "table_fallback_events": self.table_fallback_events,
        }


def _escape_angle(value: str) -> str:
    return value.replace("<", "&lt;").replace(">", "&gt;")


class MixedMarkupParser(HTMLParser):
    def __init__(self, metrics: NormalizationMetrics):
        super().__init__(convert_charrefs=False)
        self.metrics = metrics
        self.root = Node(None)
        self.stack = [self.root]

    def _append(self, value: Node | str) -> None:
        self.stack[-1].children.append(value)

    def _start(self, tag: str, attrs: list[tuple[str, str | None]], *, self_closing: bool) -> None:
        raw = self.get_starttag_text() or f"<{tag}>"
        if AUTOLINK_RE.fullmatch(raw):
            self._append(raw)
            self.metrics.transformations["gfm_autolinks_preserved"] += 1
            return
        if tag not in KNOWN_HTML_TAGS:
            self._append(_escape_angle(raw))
            self.metrics.pseudo_tags_escaped += 1
            return
        normalized_attrs = {key.casefold(): value or "" for key, value in attrs}
        self.metrics.tag_counts[tag] += 1
        self.metrics.attribute_counts.update(f"{tag}.{key}" for key in normalized_attrs)
        source_line, source_column = self.getpos()
        node = Node(
            tag,
            normalized_attrs,
            source_line=source_line,
            source_column=source_column,
        )
        self._append(node)
        if not self_closing and tag not in VOID_TAGS:
            self.stack.append(node)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag.casefold(), attrs, self_closing=False)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._start(tag.casefold(), attrs, self_closing=True)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.casefold()
        if tag not in KNOWN_HTML_TAGS:
            self._append(f"&lt;/{tag}&gt;")
            self.metrics.pseudo_tags_escaped += 1
            return
        matching = next((index for index in range(len(self.stack) - 1, 0, -1) if self.stack[index].tag == tag), None)
        if matching is None:
            self.metrics.parser_unmatched_end_tags += 1
            return
        implicitly_closed = self.stack[matching + 1 :]
        self.metrics.parser_implicitly_closed_tags += len(implicitly_closed)
        for node in implicitly_closed:
            node.malformed = True
        del self.stack[matching:]

    def handle_data(self, data: str) -> None:
        self._append(data)

    def handle_entityref(self, name: str) -> None:
        self._append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        raw = f"<!--{data}-->"
        if ALLOWED_COMMENT_RE.fullmatch(raw):
            self._append(raw)
            self.metrics.comments_preserved += 1
        else:
            self.metrics.comments_removed += 1
            self.metrics.content_characters_removed += len(raw)

    def handle_decl(self, decl: str) -> None:
        self.metrics.transformations["declarations_removed"] += 1
        self.metrics.content_characters_removed += len(decl) + 3

    def handle_pi(self, data: str) -> None:
        self.metrics.transformations["processing_instructions_removed"] += 1
        self.metrics.content_characters_removed += len(data) + 4

    def finish(self) -> Node:
        unclosed = self.stack[1:]
        self.metrics.parser_implicitly_closed_tags += len(unclosed)
        for node in unclosed:
            node.malformed = True
        self.stack = [self.root]
        return self.root


def _plain_text(value: Node | str) -> str:
    if isinstance(value, str):
        return value
    return "".join(_plain_text(child) for child in value.children)


def _block(value: str) -> str:
    content = value.strip()
    return f"\n\n{content}\n\n" if content else ""


def _wrap_inline(value: str, marker: str) -> str:
    if not value or not value.strip():
        return value
    leading = value[: len(value) - len(value.lstrip())]
    trailing = value[len(value.rstrip()) :]
    core = value.strip()
    return f"{leading}{marker}{core}{marker}{trailing}"


def _escape_destination(value: str) -> str:
    return value.strip().replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace(" ", "%20")


def _escape_table_pipes(value: str) -> str:
    output: list[str] = []
    for character in value:
        if character == "|":
            slash_count = 0
            for prior in reversed(output):
                if prior != "\\":
                    break
                slash_count += 1
            if slash_count % 2 == 0:
                output.append("\\")
        output.append(character)
    return "".join(output)


def _normalize_cell(value: str) -> str:
    compact = re.sub(r"\s*\n\s*", " ", value.replace("\r", " "))
    compact = re.sub(r"[ \t]{2,}", " ", compact).strip()
    return _escape_table_pipes(compact)


def _alignment(attrs: Mapping[str, str]) -> str | None:
    direct = attrs.get("align", "").casefold()
    if direct in {"left", "center", "right"}:
        return direct
    match = ALIGN_RE.search(attrs.get("style", ""))
    return match.group(1).casefold() if match else None


@dataclass
class TableCell:
    content: str
    header: bool
    alignment: str | None


def _row_nodes(table: Node) -> list[tuple[Node, bool]]:
    rows: list[tuple[Node, bool]] = []

    def visit(node: Node, in_head: bool = False) -> None:
        for child in node.children:
            if not isinstance(child, Node):
                continue
            if child.tag == "table":
                continue
            child_in_head = in_head or child.tag == "thead"
            if child.tag == "tr":
                rows.append((child, child_in_head))
            else:
                visit(child, child_in_head)

    visit(table)
    return rows


def _cell_nodes(row: Node) -> list[Node]:
    cells: list[Node] = []
    for child in row.children:
        if isinstance(child, Node) and child.tag in {"th", "td"}:
            cells.append(child)
    return cells


def _table_comments_outside_cells(table: Node) -> list[str]:
    comments: list[str] = []

    def visit(value: Node | str, in_cell: bool = False) -> None:
        if isinstance(value, str):
            if not in_cell:
                comments.extend(match.group(0) for match in ALLOWED_COMMENT_TOKEN_RE.finditer(value))
            return
        child_in_cell = in_cell or value.tag in {"th", "td"}
        for child in value.children:
            visit(child, child_in_cell)

    visit(table)
    return comments


def _positive_span(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return parsed if 1 <= parsed <= 100 else 1


def _render_children(node: Node, metrics: NormalizationMetrics, context: RenderContext) -> str:
    rendered = [[child, _render_node(child, metrics, context)] for child in node.children]
    for index, (child, value) in enumerate(rendered):
        if not isinstance(child, Node) or child.tag not in {"sup", "sub"} or not value:
            continue
        prior = next((str(piece[1]) for piece in reversed(rendered[:index]) if piece[1]), "")
        prefix = " " if prior.rstrip().endswith(("*", "_", "~", "`")) and value[0].isalnum() else ""
        if prefix:
            metrics.transformations["flattened_style_markdown_boundaries_spaced"] += 1
            rendered[index][1] = prefix + value
    return "".join(str(piece[1]) for piece in rendered)


def _render_list(node: Node, metrics: NormalizationMetrics, context: RenderContext) -> str:
    ordered = node.tag == "ol"
    items = [child for child in node.children if isinstance(child, Node) and child.tag == "li"]
    if context.in_table_cell:
        rendered = []
        for item in items:
            content = _normalize_cell(_render_children(item, metrics, RenderContext(True, context.list_depth + 1)))
            if content:
                rendered.append(content)
        metrics.transformations["lists_flattened_inside_table_cells"] += int(bool(rendered))
        return " ".join(rendered)
    lines: list[str] = []
    for index, item in enumerate(items, 1):
        content = _render_children(item, metrics, RenderContext(False, context.list_depth + 1)).strip()
        if not content:
            continue
        prefix = f"{index}. " if ordered else "- "
        parts = content.splitlines()
        lines.append(prefix + parts[0])
        lines.extend("  " + part for part in parts[1:] if part.strip())
    return _block("\n".join(lines))


def _descendant_nodes(node: Node) -> Iterable[Node]:
    yield node
    for child in node.children:
        if isinstance(child, Node):
            yield from _descendant_nodes(child)


def _fallback_plain_text(value: Node | str) -> str:
    if isinstance(value, str):
        return ALLOWED_COMMENT_TOKEN_RE.sub("", value)
    if value.tag == "br":
        return "\n"
    if value.tag == "table":
        rows: list[str] = []
        for row, _ in _row_nodes(value):
            cells = [
                re.sub(r"\s+", " ", _fallback_plain_text(cell)).strip()
                for cell in _cell_nodes(row)
            ]
            rows.extend(cell for cell in cells if cell)
            if cells:
                rows.append("")
        return "\n".join(rows).rstrip()
    return "".join(_fallback_plain_text(child) for child in value.children)


def _fallback_table(
    table: Node,
    metrics: NormalizationMetrics,
    *,
    reason: str,
) -> str:
    rows = _row_nodes(table)
    lines: list[str] = []
    captions = [child for child in table.children if isinstance(child, Node) and child.tag == "caption"]
    for caption in captions:
        value = re.sub(r"\s+", " ", _fallback_plain_text(caption)).strip()
        if value:
            lines.extend((value, ""))
    for row, _ in rows:
        cells = _cell_nodes(row)
        if cells:
            for cell in cells:
                value = _fallback_plain_text(cell).strip()
                parts = [part.strip() for part in value.splitlines()]
                while parts and not parts[0]:
                    parts.pop(0)
                while parts and not parts[-1]:
                    parts.pop()
                lines.extend(parts)
            lines.append("")
        else:
            value = _fallback_plain_text(row).strip()
            if value:
                lines.extend((value, ""))
    if not rows:
        value = _fallback_plain_text(table).strip()
        if value:
            lines.append(value)
    comments = [match.group(0) for match in ALLOWED_COMMENT_TOKEN_RE.finditer(_plain_text(table))]
    if comments:
        if lines and lines[-1] != "":
            lines.append("")
        lines.extend(comments)
    table_nodes = list(_descendant_nodes(table))
    table_count = sum(node.tag == "table" for node in table_nodes)
    cell_count = sum(node.tag in {"th", "td"} for node in table_nodes)
    anchors: list[str] = []
    for line in lines:
        candidate = re.sub(r"\s+", " ", ALLOWED_COMMENT_TOKEN_RE.sub("", line)).strip()
        if candidate and candidate not in anchors:
            anchors.append(candidate[:500])
    anchors.sort(key=lambda value: (-len(value), value))
    metrics.table_fallback_events.append(
        {
            "reason": reason,
            "table_count": table_count,
            "cell_count": cell_count,
            "anchor_candidates": anchors[:12],
            "plain_text_preview": "\n".join(lines).strip()[:2000],
            "source_line": table.source_line,
            "source_column": table.source_column,
        }
    )
    metrics.transformations["html_tables_fallback_to_text"] += table_count
    metrics.transformations["html_table_fallback_cells_preserved"] += cell_count
    metrics.transformations[f"table_fallback_reason_{reason}"] += 1
    if comments:
        metrics.transformations["table_comments_relocated_after_table"] += len(comments)
    return _block("\n".join(lines).strip())


def _render_table(table: Node, metrics: NormalizationMetrics, context: RenderContext) -> str:
    out_of_cell_comments = _table_comments_outside_cells(table)
    if context.in_table_cell:
        return _fallback_table(table, metrics, reason="nested_table")

    descendants = list(_descendant_nodes(table))
    if any(node is not table and node.tag == "table" for node in descendants):
        return _fallback_table(table, metrics, reason="nested_table")
    if any(node.malformed for node in descendants):
        return _fallback_table(table, metrics, reason="malformed_html")

    captions = [child for child in table.children if isinstance(child, Node) and child.tag == "caption"]
    caption = " / ".join(
        value for value in (_normalize_cell(_render_children(node, metrics, RenderContext())) for node in captions) if value
    )
    source_rows = _row_nodes(table)
    if not source_rows:
        if ALLOWED_COMMENT_TOKEN_RE.sub("", _plain_text(table)).strip():
            return _fallback_table(table, metrics, reason="missing_rows")
        metrics.transformations["empty_tables_removed"] += 1
        metrics.content_characters_removed += len(ALLOWED_COMMENT_TOKEN_RE.sub("", _plain_text(table)))
        if out_of_cell_comments:
            metrics.transformations["table_comments_relocated_after_table"] += len(out_of_cell_comments)
            return _block("\n".join(out_of_cell_comments))
        return ""
    if any(not _cell_nodes(row) for row, _ in source_rows):
        return _fallback_table(table, metrics, reason="row_without_cells")

    occupied: dict[tuple[int, int], TableCell] = {}
    grid: list[list[TableCell | None]] = []
    header_flags: list[bool] = []
    alignments: defaultdict[int, list[str]] = defaultdict(list)
    for row_index, (row, in_thead) in enumerate(source_rows):
        cells = _cell_nodes(row)
        row_values: list[TableCell | None] = []

        def ensure(column: int) -> None:
            while len(row_values) <= column:
                row_values.append(None)

        for (occupied_row, column), value in sorted(occupied.items()):
            if occupied_row == row_index:
                ensure(column)
                row_values[column] = value
        column = 0
        for cell_node in cells:
            while (row_index, column) in occupied:
                column += 1
            colspan = _positive_span(cell_node.attrs.get("colspan", "1"))
            rowspan = _positive_span(cell_node.attrs.get("rowspan", "1"))
            content = _normalize_cell(_render_children(cell_node, metrics, RenderContext(True, context.list_depth)))
            value = TableCell(content, cell_node.tag == "th", _alignment(cell_node.attrs))
            ensure(column + colspan - 1)
            row_values[column] = value
            for offset in range(1, colspan):
                row_values[column + offset] = TableCell("", value.header, value.alignment)
            for row_offset in range(1, rowspan):
                for column_offset in range(colspan):
                    occupied[(row_index + row_offset, column + column_offset)] = TableCell("", value.header, value.alignment)
            if colspan > 1:
                metrics.transformations["colspan_cells_expanded"] += 1
            if rowspan > 1:
                metrics.transformations["rowspan_cells_expanded"] += 1
            for offset in range(colspan):
                if value.alignment:
                    alignments[column + offset].append(value.alignment)
            column += colspan
        grid.append(row_values)
        header_flags.append(in_thead or bool(cells) and all(cell.tag == "th" for cell in cells))

    if any(row_index >= len(source_rows) for row_index, _ in occupied):
        return _fallback_table(table, metrics, reason="rowspan_outside_rows")

    column_count = max((len(row) for row in grid), default=0)
    if column_count == 0:
        metrics.transformations["empty_tables_removed"] += 1
        if out_of_cell_comments:
            metrics.transformations["table_comments_relocated_after_table"] += len(out_of_cell_comments)
            return _block("\n".join(out_of_cell_comments))
        return ""
    blank = TableCell("", False, None)
    normalized_grid: list[list[TableCell]] = [
        [cell if cell is not None else blank for cell in row + [None] * (column_count - len(row))]
        for row in grid
    ]
    leading_header_count = 0
    for flag in header_flags:
        if not flag:
            break
        leading_header_count += 1
    if leading_header_count:
        headers = [cell.content for cell in normalized_grid[0]]
        body = normalized_grid[1:]
        if leading_header_count > 1:
            metrics.transformations["additional_header_rows_preserved"] += leading_header_count - 1
    else:
        headers = ["" for _ in range(column_count)]
        body = normalized_grid
        metrics.transformations["synthetic_empty_table_headers"] += 1

    body_rows: list[list[str]] = []
    for row in body:
        body_rows.append([
            _wrap_inline(cell.content, "**") if cell.header and cell.content else cell.content
            for cell in row
        ])
    delimiters = []
    for column in range(column_count):
        values = alignments.get(column, [])
        alignment = values[0] if values and all(value == values[0] for value in values) else None
        if values and alignment is None:
            metrics.transformations["conflicting_table_alignments_dropped"] += 1
        delimiters.append({"left": ":---", "center": ":---:", "right": "---:"}.get(alignment, "---"))

    def line(cells: Sequence[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    lines = [line(headers), line(delimiters), *(line(row) for row in body_rows)]
    if any(len(row) != column_count for row in [headers, delimiters, *body_rows]):
        raise RuntimeError("generated GFM table has inconsistent columns")
    metrics.transformations["html_tables_to_gfm"] += 1
    metrics.transformations["html_table_rows_emitted"] += len(lines) - 1
    metrics.transformations["html_table_cells_preserved"] += sum(len(_cell_nodes(row)) for row, _ in source_rows)
    table_markdown = "\n".join(lines)
    if caption:
        table_markdown = f"*{caption}*\n\n{table_markdown}"
        metrics.transformations["table_captions_preserved"] += 1
    if out_of_cell_comments:
        metrics.transformations["table_comments_relocated_after_table"] += len(out_of_cell_comments)
        table_markdown += "\n\n" + "\n".join(out_of_cell_comments)
    return _block(table_markdown)


def _render_node(value: Node | str, metrics: NormalizationMetrics, context: RenderContext) -> str:
    if isinstance(value, str):
        return value
    tag = value.tag
    if tag is None:
        return _render_children(value, metrics, context)
    if tag in DROP_WITH_CONTENT_TAGS or tag in {"meta", "link", "source", "track"}:
        removed = _plain_text(value)
        metrics.content_characters_removed += len(removed)
        metrics.transformations["unsupported_elements_removed_with_content"] += 1
        return ""
    if tag == "table":
        return _render_table(value, metrics, context)
    if tag in TABLE_TAGS:
        if tag in {"th", "td"}:
            # Malformed OCR HTML can leave cells outside any table.  There is
            # no recoverable rectangular geometry, so preserve their content
            # as a readable block and account for the structural downgrade.
            metrics.transformations["orphan_table_cells_flattened"] += 1
            return _block(_render_children(value, metrics, context))
        return _render_children(value, metrics, context)
    content = _render_children(value, metrics, context)
    if tag in {"b", "strong"}:
        return _wrap_inline(content, "**")
    if tag in {"i", "em"}:
        return _wrap_inline(content, "*")
    if tag in {"del", "s", "strike"}:
        return _wrap_inline(content, "~~")
    if tag in {"sup", "sub", "u", "span", "ins"}:
        metrics.transformations[f"{tag}_formatting_flattened"] += 1
        return content
    if tag == "math":
        payload = content.strip()
        if not payload:
            return ""
        metrics.transformations["math_to_github_math"] += 1
        if value.attrs.get("display", "").casefold() == "block" and not context.in_table_cell:
            return _block(f"$$\n{payload}\n$$")
        return f"${payload}$"
    if tag == "br":
        metrics.transformations["html_breaks_converted"] += 1
        return " " if context.in_table_cell else "  \n"
    if tag == "hr":
        return _block("---")
    if tag in BLOCK_WRAPPER_TAGS:
        if context.in_table_cell:
            return content.strip() + ("; " if content.strip() else "")
        return _block(content)
    if tag in HEADING_TAGS:
        level = int(tag[1])
        return _block(f"{'#' * level} {content.strip()}")
    if tag in {"ul", "ol"}:
        return _render_list(value, metrics, context)
    if tag == "li":
        return content
    if tag in {"dl", "dt", "dd"}:
        return _block(content) if not context.in_table_cell else content
    if tag == "blockquote":
        lines = content.strip().splitlines()
        return _block("\n".join(f"> {line}" if line else ">" for line in lines))
    if tag == "code":
        fence = "`" * (max((len(match.group(0)) for match in re.finditer(r"`+", content)), default=0) + 1)
        fence = fence if len(fence) >= 1 else "`"
        return f"{fence}{content.strip()}{fence}"
    if tag == "pre":
        payload = _plain_text(value).strip("\n")
        fence = "`" * max(3, max((len(match.group(0)) for match in re.finditer(r"`+", payload)), default=0) + 1)
        return _block(f"{fence}\n{payload}\n{fence}")
    if tag == "a":
        label = content.strip()
        href = value.attrs.get("href", "").strip()
        if label and href and not href.casefold().startswith(("javascript:", "data:")):
            return f"[{label}]({_escape_destination(href)})"
        metrics.transformations["links_flattened_without_safe_destination"] += 1
        return label
    if tag == "img":
        source = value.attrs.get("src", "").strip()
        if not source:
            alt = value.attrs.get("alt", "").strip()
            if alt:
                replacement, _ = _image_description_comment(alt)
                metrics.transformations["source_less_images_to_description_comment"] += 1
                return replacement
            metrics.transformations["source_less_images_removed"] += 1
            return ""
        alt = value.attrs.get("alt", "").replace("[", "\\[").replace("]", "\\]")
        metrics.transformations["html_images_to_markdown"] += 1
        return f"![{alt}]({_escape_destination(source)})"
    if tag == "input":
        metrics.transformations["inline_inputs_removed"] += 1
        return ""
    if tag in FLATTEN_TAGS:
        return content
    return content


def _escape_residual_angles(text: str, metrics: NormalizationMetrics) -> str:
    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        if AUTOLINK_RE.fullmatch(raw) or ALLOWED_COMMENT_RE.fullmatch(raw):
            return raw
        metrics.pseudo_tags_escaped += 1
        return _escape_angle(raw)

    # OCR commonly emits doubled or nested guillemet-like ASCII brackets such
    # as ``<<company name>>``.  A single regex pass sees only the inner pair;
    # iterate until every non-autolink layer is escaped in this same pass.
    while True:
        escaped = RESIDUAL_ANGLE_RE.sub(replace, text)
        if escaped == text:
            break
        text = escaped

    # A malformed link emitted from damaged source markup can contain an
    # unterminated known-tag prefix such as ``<a href=`` and only encounter a
    # ``>`` many lines later in embedded PHP.  The same-line regex above must
    # not consume across those lines, so shield every remaining ``<`` unless
    # the complete same-line candidate is an approved comment or GFM autolink.
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        opener = text.find("<", cursor)
        if opener < 0:
            output.append(text[cursor:])
            break
        output.append(text[cursor:opener])
        suffix = text[opener:]
        line_end = suffix.find("\n")
        candidate_end = suffix.find(">", 1) if line_end < 0 else suffix.find(">", 1, line_end)
        candidate = suffix[: candidate_end + 1] if candidate_end >= 0 else ""
        if candidate and (AUTOLINK_RE.fullmatch(candidate) or ALLOWED_COMMENT_RE.fullmatch(candidate)):
            output.append("<")
        else:
            output.append("&lt;")
            metrics.pseudo_tags_escaped += 1
        cursor = opener + 1
    return "".join(output)


def _protect_literal_ampersands(text: str) -> str:
    """Shield bare ampersands from HTMLParser's permissive entity parsing."""

    if LITERAL_AMPERSAND_SENTINEL in text:
        raise ValueError("input contains the reserved literal-ampersand sentinel")
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        ampersand = text.find("&", cursor)
        if ampersand < 0:
            output.append(text[cursor:])
            break
        output.append(text[cursor:ampersand])
        suffix = text[ampersand + 1 :]
        numeric = re.match(r"#(?:[0-9]+|[xX][0-9A-Fa-f]+);", suffix)
        named = re.match(r"([A-Za-z][A-Za-z0-9]+);", suffix)
        if numeric is not None or (
            named is not None and named.group(1) + ";" in html.entities.html5
        ):
            output.append("&")
        else:
            output.append(LITERAL_AMPERSAND_SENTINEL)
        cursor = ampersand + 1
    return "".join(output)


def _protect_non_html_angle_syntax(text: str, metrics: NormalizationMetrics) -> str:
    """Escape pseudo-tags before HTMLParser can consume later document content.

    ``HTMLParser`` permits a start tag to span newlines.  A literal unmatched
    expression such as ``<proles-is = source)`` can therefore absorb many
    paragraphs and a later ``>`` as one unknown tag, including approved
    provenance comments between those points.  First escape complete same-line
    angle expressions that are neither HTML nor GFM autolinks, then shield any
    remaining ``<`` that cannot begin recognized HTML syntax.
    """

    def escape_complete(match: re.Match[str]) -> str:
        raw = match.group(0)
        if (
            AUTOLINK_RE.fullmatch(raw)
            or KNOWN_HTML_TAG_RE.fullmatch(raw)
            or raw.startswith("<!--")
            or raw.startswith("<!")
            or raw.startswith("<?")
        ):
            return raw
        metrics.pseudo_tags_escaped += 1
        return _escape_angle(raw)

    # Iterate so nested OCR brackets such as ``<<name>>`` are fully escaped.
    while True:
        escaped = RESIDUAL_ANGLE_RE.sub(escape_complete, text)
        if escaped == text:
            break
        text = escaped

    known_prefix = re.compile(
        r"</?(?:"
        + "|".join(sorted(map(re.escape, KNOWN_HTML_TAGS), key=len, reverse=True))
        + r")(?=[\s/>])",
        re.IGNORECASE,
    )
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        opener = text.find("<", cursor)
        if opener < 0:
            output.append(text[cursor:])
            break
        output.append(text[cursor:opener])
        suffix = text[opener:]
        line_end = suffix.find("\n")
        candidate_end = suffix.find(">", 1) if line_end < 0 else suffix.find(">", 1, line_end)
        candidate = suffix[: candidate_end + 1] if candidate_end >= 0 else ""
        if (
            suffix.startswith("<!--")
            or suffix.startswith("<!")
            or suffix.startswith("<?")
            or known_prefix.match(suffix)
            or (candidate and AUTOLINK_RE.fullmatch(candidate))
        ):
            output.append("<")
        else:
            output.append("&lt;")
            metrics.pseudo_tags_escaped += 1
        cursor = opener + 1
    return "".join(output)


def _split_gfm_pipe_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return None
    inner = stripped[1:-1]
    cells: list[str] = []
    start = 0
    for index, character in enumerate(inner):
        if character != "|":
            continue
        slash_count = 0
        scan = index - 1
        while scan >= 0 and inner[scan] == "\\":
            slash_count += 1
            scan -= 1
        if slash_count % 2:
            continue
        cells.append(inner[start:index].strip())
        start = index + 1
    cells.append(inner[start:].strip())
    return cells


def _gfm_delimiter_cells(line: str) -> list[str] | None:
    cells = _split_gfm_pipe_row(line)
    if cells is None or not cells:
        return None
    return cells if all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells) else None


def normalize_existing_gfm_tables(text: str, metrics: NormalizationMetrics) -> str:
    """Pad structurally obvious ragged pipe tables without interpreting cell meaning."""

    lines = text.split("\n")
    index = 1
    while index < len(lines):
        delimiters = _gfm_delimiter_cells(lines[index])
        headers = _split_gfm_pipe_row(lines[index - 1])
        if delimiters is not None and headers is None:
            marker_header = re.fullmatch(
                r"\s*\|\s*(<!--\s*repeating-text-removed\s*-->)\s*\|?\s*",
                lines[index - 1],
                re.IGNORECASE,
            )
            if marker_header is not None:
                headers = [marker_header.group(1)]
                metrics.transformations["repetition_marker_table_headers_repaired"] += 1
        if delimiters is None or headers is None:
            index += 1
            continue
        end = index + 1
        rows: list[list[str]] = []
        while end < len(lines):
            cells = _split_gfm_pipe_row(lines[end])
            if cells is None:
                break
            rows.append(cells)
            end += 1
        all_rows = [headers, delimiters, *rows]
        column_count = max(map(len, all_rows))
        if any(len(row) != column_count for row in all_rows):
            padded = sum(column_count - len(row) for row in all_rows)
            headers += [""] * (column_count - len(headers))
            delimiters += ["---"] * (column_count - len(delimiters))
            for row in rows:
                row += [""] * (column_count - len(row))

            def render(cells: Sequence[str]) -> str:
                return "| " + " | ".join(cells) + " |"

            lines[index - 1 : end] = [render(headers), render(delimiters), *(render(row) for row in rows)]
            metrics.transformations["existing_gfm_tables_repaired"] += 1
            metrics.transformations["existing_gfm_table_cells_padded"] += padded
        index = end
    return "\n".join(lines)


def normalize_mixed_markup_to_gfm(text: str, *, metrics: MutableMapping[str, object] | None = None) -> str:
    """Convert recognized HTML to a conservative, HTML-free GFM representation."""
    state = NormalizationMetrics()
    parser = MixedMarkupParser(state)
    protected = _protect_non_html_angle_syntax(str(text or ""), state)
    protected = _protect_literal_ampersands(protected)
    parser.feed(protected)
    parser.close()
    root = parser.finish()
    normalized = _render_node(root, state, RenderContext())
    normalized = normalized.replace(LITERAL_AMPERSAND_SENTINEL, "&")
    normalized = _escape_residual_angles(normalized, state)
    normalized = normalize_existing_gfm_tables(normalized, state)
    if KNOWN_HTML_TAG_RE.search(normalized):
        raise RuntimeError("recognized HTML tag remains after GFM normalization")
    if metrics is not None:
        metrics.clear()
        metrics.update(state.as_dict())
    return normalized


def _apply_comment_safe_repetition_pass(
    text: str,
    pass_record: Mapping[str, object],
    *,
    pass_index: int,
) -> tuple[str, dict[str, object]]:
    """Expand repetition cuts to whole provenance comments before replacing."""

    comment_intervals = [
        (match.start(), match.end()) for match in DESCRIPTION_OF_REMOVED_IMAGE_TOKEN_RE.finditer(text)
    ]
    adjusted: list[dict[str, object]] = []
    for span_value in pass_record.get("spans", []):
        span = dict(span_value)
        original_start = int(span["start_index"])
        original_end = int(span["end_index"])
        start = original_start
        end = original_end
        for comment_start, comment_end in comment_intervals:
            if comment_start < start < comment_end:
                start = comment_start
            if comment_start < end < comment_end:
                end = comment_end
        span["start_index"] = start
        span["end_index"] = end
        span["removed_char_count"] = end - start
        if start != original_start or end != original_end:
            span["comment_boundary_expansion"] = {
                "original_start_index": original_start,
                "original_end_index": original_end,
                "expanded_start_index": start,
                "expanded_end_index": end,
            }
        if adjusted and start <= int(adjusted[-1]["end_index"]):
            prior = adjusted[-1]
            prior["end_index"] = max(int(prior["end_index"]), end)
            prior["removed_char_count"] = int(prior["end_index"]) - int(prior["start_index"])
            prior["rules"] = sorted(set(map(str, prior.get("rules", []))) | set(map(str, span.get("rules", []))))
            prior["findings"] = [*prior.get("findings", []), *span.get("findings", [])]
            prior.setdefault("merged_comment_boundary_spans", []).append(
                {
                    "start_index": start,
                    "end_index": end,
                }
            )
        else:
            adjusted.append(span)

    pieces: list[str] = []
    cursor = 0
    for span in adjusted:
        start = int(span["start_index"])
        end = int(span["end_index"])
        if start < cursor or end <= start or end > len(text):
            raise RuntimeError(f"invalid comment-safe repetition span: {start}:{end}")
        pieces.extend((text[cursor:start], REPETITION_COMMENT))
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), {"pass_index": pass_index, "spans": adjusted}


def _replace_repetitions_comment_safe(text: str, repetition_cleaner: object) -> tuple[str, dict[str, object]]:
    """Run the detector while preventing cuts through provenance comments."""

    current = text
    details: list[dict[str, object]] = []
    rule_counts: Counter[str] = Counter()
    preserved_findings: list[dict[str, object]] = []
    for pass_index in range(1, 33):
        detected_metrics: dict[str, object] = {}
        repetition_cleaner(current, metrics=detected_metrics)
        pass_records = list(detected_metrics["complex_repetition_replacement_details"])
        if not pass_records:
            preserved_findings = [
                dict(value) for value in detected_metrics["complex_repetition_preserved_findings"]
            ]
            break
        current, adjusted_record = _apply_comment_safe_repetition_pass(
            current,
            dict(pass_records[0]),
            pass_index=pass_index,
        )
        if not adjusted_record["spans"]:
            raise RuntimeError("repetition detector reported a pass without replaceable spans")
        details.append(adjusted_record)
        for span in adjusted_record["spans"]:
            rule_counts.update(map(str, span.get("rules", [])))
    else:
        raise RuntimeError("comment-safe repetition cleaning exceeded 32 passes")
    return current, {
        "complex_repetition_passes": len(details),
        "complex_repetition_replacements": sum(len(record["spans"]) for record in details),
        "complex_repetition_characters_removed": sum(
            int(span["removed_char_count"])
            for record in details
            for span in record["spans"]
        ),
        "complex_repetition_rule_counts": dict(sorted(rule_counts.items())),
        "complex_repetition_replacement_details": details,
        "complex_repetition_preserved_findings": preserved_findings,
    }


def clean_then_normalize_to_gfm(text: str, *, repetition_cleaner: object) -> dict[str, object]:
    """Apply the frozen extraction-artifact cleaning order and emit audited GFM."""

    if not callable(repetition_cleaner):
        raise TypeError("repetition_cleaner must be callable")
    first_repetition_metrics: dict[str, object] = {}
    repeated_cleaned = repetition_cleaner(str(text or ""), metrics=first_repetition_metrics)
    image_metrics: dict[str, object] = {}
    image_cleaned = clean_generated_image_artifacts(repeated_cleaned, metrics=image_metrics)
    cleaned_text, second_repetition_metrics = _replace_repetitions_comment_safe(
        image_cleaned,
        repetition_cleaner,
    )
    retained_image_comments = cleaned_text.count("<!-- description-of-removed-image")
    emitted_image_comments = int(image_metrics["image_description_comments_emitted"])
    if retained_image_comments > emitted_image_comments:
        raise RuntimeError("follow-up repetition cleaning invented image-description comments")
    image_metrics["image_description_comments_retained_after_repetition"] = retained_image_comments
    image_metrics["image_description_comments_removed_as_repetition"] = (
        emitted_image_comments - retained_image_comments
    )

    repetition_details: list[dict[str, object]] = []
    repetition_rule_counts: Counter[str] = Counter()
    for stage, stage_metrics in (
        ("before_generated_image_cleanup", first_repetition_metrics),
        ("after_generated_image_cleanup", second_repetition_metrics),
    ):
        repetition_rule_counts.update(
            {str(key): int(value) for key, value in dict(stage_metrics["complex_repetition_rule_counts"]).items()}
        )
        for pass_record in stage_metrics["complex_repetition_replacement_details"]:
            repetition_details.append({**dict(pass_record), "cleaning_stage": stage})
    repetition_metrics: dict[str, object] = {
        "complex_repetition_passes": int(first_repetition_metrics["complex_repetition_passes"])
        + int(second_repetition_metrics["complex_repetition_passes"]),
        "complex_repetition_replacements": int(first_repetition_metrics["complex_repetition_replacements"])
        + int(second_repetition_metrics["complex_repetition_replacements"]),
        "complex_repetition_characters_removed": int(
            first_repetition_metrics["complex_repetition_characters_removed"]
        )
        + int(second_repetition_metrics["complex_repetition_characters_removed"]),
        "complex_repetition_rule_counts": dict(sorted(repetition_rule_counts.items())),
        "complex_repetition_replacement_details": repetition_details,
        "complex_repetition_preserved_findings": second_repetition_metrics[
            "complex_repetition_preserved_findings"
        ],
    }
    markup_metrics: dict[str, object] = {}
    normalized = normalize_mixed_markup_to_gfm(cleaned_text, metrics=markup_metrics)
    if GENERATED_IMAGE_TOKEN_RE.search(normalized):
        raise RuntimeError("generated image artifact remains after normalization")
    return {
        "cleaned_text": cleaned_text,
        "normalized_markdown": normalized,
        "repetition_metrics": repetition_metrics,
        "generated_image_metrics": image_metrics,
        "markup_metrics": markup_metrics,
    }


def markdown_structure_counts(text: str) -> dict[str, int]:
    lines = text.splitlines()
    return {
        "atx_headings": sum(bool(re.match(r"^ {0,3}#{1,6}\s+\S", line)) for line in lines),
        "fence_lines": sum(bool(re.match(r"^ {0,3}(?:`{3,}|~{3,})", line)) for line in lines),
        "gfm_table_delimiters": sum(bool(re.match(r"^\s*\|?\s*:?-{3,}", line)) and "|" in line for line in lines),
        "markdown_images": len(re.findall(r"!\[[^\]\n]*\]\([^\n)]*\)", text)),
        "markdown_links": len(re.findall(r"(?<!!)\[[^\]\n]+\]\([^\n)]*\)", text)),
        "strong_delimiter_pairs": text.count("**") // 2,
    }


def markdown_token_counts(renderer: object, text: str) -> dict[str, int]:
    counts: Counter[str] = Counter()

    def visit(tokens: Iterable[object]) -> None:
        for token in tokens:
            token_type = str(getattr(token, "type", ""))
            if token_type in PRESERVED_MARKDOWN_TOKEN_TYPES:
                counts[token_type] += 1
            children = getattr(token, "children", None)
            if children:
                visit(children)

    visit(renderer.parse(text))
    return {token_type: counts[token_type] for token_type in PRESERVED_MARKDOWN_TOKEN_TYPES}


def _load_repetition_module(glossapi_root: Path) -> tuple[object, Path]:
    module_path = (glossapi_root / "src/glossapi/ocr/utils/repetition.py").resolve()
    specification = importlib.util.spec_from_file_location("agent1_v4_gfm_glossapi_repetition", module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"could not import GlossAPI repetition module: {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    if not callable(getattr(module, "replace_complex_repetitions", None)):
        raise RuntimeError("GlossAPI repetition module lacks replace_complex_repetitions")
    if getattr(module, "TEXT_REMOVED_COMMENT", None) != REPETITION_COMMENT:
        raise RuntimeError("GlossAPI repetition marker does not match the normalization contract")
    return module, module_path


def _cards_by_opaque_id(site_dir: Path) -> dict[str, Mapping[str, object]]:
    cards: dict[str, Mapping[str, object]] = {}
    for source_path in sorted((site_dir / "data/sources").glob("*.json")):
        for card in _read_json(source_path).get("cards", []):
            if not isinstance(card, Mapping):
                raise ValueError(f"{source_path}: invalid source card")
            opaque_id = card.get("opaque_id")
            if not isinstance(opaque_id, str) or opaque_id in cards:
                raise ValueError(f"{source_path}: invalid or duplicate opaque_id")
            cards[opaque_id] = card
    return cards


def _markdown_renderer() -> tuple[object, str]:
    from markdown_it import MarkdownIt, __version__ as markdown_it_version

    renderer = MarkdownIt("commonmark", {"html": True, "linkify": True}).enable("table").enable("strikethrough")
    return renderer, f"markdown-it-py {markdown_it_version}; commonmark + table + strikethrough"


def _inventory_without_manifest(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative == "site_manifest.json":
            continue
        if path.is_symlink():
            raise ValueError(f"review site contains a forbidden symlink: {relative}")
        if path.is_file():
            payload = path.read_bytes()
            records.append({"path": relative, "bytes": len(payload), "sha256": sha256_bytes(payload)})
    return records


def _is_bulk_asset(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in PORTABLE_BULK_PREFIXES)


def _remove_stale_normalized_documents(
    site_dir: Path, expected_relative_paths: Iterable[str]
) -> int:
    """Remove generated payloads no longer referenced by the current audit."""

    expected = set(expected_relative_paths)
    output_dir = site_dir / OUTPUT_DOCUMENT_DIR
    if not output_dir.exists():
        return 0
    removed = 0
    for path in sorted(output_dir.glob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"invalid generated normalized-document artifact: {path}")
        relative = path.relative_to(site_dir).as_posix()
        if relative not in expected:
            path.unlink()
            removed += 1
    return removed


def normalize_site(*, site_dir: Path, glossapi_root: Path) -> dict[str, object]:
    site_dir = site_dir.resolve()
    manifest = _read_json(site_dir / "site_manifest.json")
    if manifest.get("schema_version") != SITE_SCHEMA or manifest.get("status") != "passed":
        raise ValueError("review site manifest is not passed")
    cards = _cards_by_opaque_id(site_dir)
    document_paths = sorted((site_dir / "data/documents").glob("*.json"))
    if len(document_paths) != len(cards):
        raise ValueError("review site card/document count mismatch")
    repetition, repetition_path = _load_repetition_module(glossapi_root)
    renderer, renderer_description = _markdown_renderer()

    rows: list[dict[str, object]] = []
    aggregate_tags: Counter[str] = Counter()
    aggregate_attributes: Counter[str] = Counter()
    aggregate_transformations: Counter[str] = Counter()
    aggregate_markdown_before: Counter[str] = Counter()
    aggregate_markdown_after: Counter[str] = Counter()
    aggregate_markdown_tokens_before: Counter[str] = Counter()
    aggregate_markdown_tokens_after: Counter[str] = Counter()
    changed_by_source: Counter[str] = Counter()
    documents_with_html = 0
    documents_changed = 0
    documents_with_repetition_replacement = 0
    replacement_count = 0
    repetition_characters_removed = 0
    content_characters_removed = 0
    pseudo_tags_escaped = 0
    comments_removed = 0
    comments_preserved = 0
    generated_image_artifacts_removed = 0
    generated_image_characters_removed = 0
    image_description_elements_commented = 0
    image_description_comments_emitted = 0
    image_description_comments_retained_after_repetition = 0
    image_description_comments_removed_as_repetition = 0
    image_description_characters_preserved = 0
    empty_image_description_comments = 0
    image_descriptions_split_around_pipeline_markers = 0
    pipeline_markers_preserved_inside_image_descriptions = 0
    aggregate_generated_image_rules: Counter[str] = Counter()

    for document_index, document_path in enumerate(document_paths, 1):
        if document_index == 1 or document_index % 25 == 0 or document_index == len(document_paths):
            print(f"normalizing sample document {document_index}/{len(document_paths)}", flush=True)
        payload = _read_json(document_path)
        opaque_id = payload.get("opaque_id")
        raw_text = payload.get("text")
        if not isinstance(opaque_id, str) or document_path.stem != opaque_id or opaque_id not in cards:
            raise ValueError(f"{document_path}: opaque identity mismatch")
        if not isinstance(raw_text, str):
            raise ValueError(f"{document_path}: text must be a string")

        result = clean_then_normalize_to_gfm(
            raw_text,
            repetition_cleaner=repetition.replace_complex_repetitions,
        )
        cleaned_text = str(result["cleaned_text"])
        normalized = str(result["normalized_markdown"])
        repetition_metrics = dict(result["repetition_metrics"])
        generated_image_metrics = dict(result["generated_image_metrics"])
        markup_metrics = dict(result["markup_metrics"])
        if "<!-- text-removed -->" in normalized:
            raise RuntimeError("obsolete repetition marker remains in normalized output")
        if KNOWN_HTML_TAG_RE.search(normalized):
            raise RuntimeError(f"{document_path}: recognized HTML remains")
        if normalized.count(REPETITION_COMMENT) != cleaned_text.count(REPETITION_COMMENT):
            raise RuntimeError(f"{document_path}: repetition removal comment was not preserved")
        cleaned_image_comments = cleaned_text.count("<!-- description-of-removed-image")
        if cleaned_image_comments != int(
            generated_image_metrics["image_description_comments_retained_after_repetition"]
        ):
            raise RuntimeError(f"{document_path}: retained image-description comment accounting did not close")
        if normalized.count("<!-- description-of-removed-image") != cleaned_image_comments:
            raise RuntimeError(f"{document_path}: image-description provenance comment was not preserved")
        if re.search(
            r"<!--\s*description-of-removed-image:(?:(?!-->)[^\r\n])*<!--",
            normalized,
            re.IGNORECASE,
        ):
            raise RuntimeError(f"{document_path}: image-description provenance contains a nested comment")
        if len(ALLOWED_COMMENT_TOKEN_RE.findall(normalized)) != int(markup_metrics["comments_preserved"]):
            raise RuntimeError(f"{document_path}: approved removal comments were not preserved exactly once")
        before_structures = markdown_structure_counts(cleaned_text)
        after_structures = markdown_structure_counts(normalized)
        before_tokens = markdown_token_counts(renderer, cleaned_text)
        after_tokens = markdown_token_counts(renderer, normalized)
        repetition_idempotence_metrics: dict[str, object] = {}
        if repetition.replace_complex_repetitions(
            normalized, metrics=repetition_idempotence_metrics
        ) != normalized:
            raise RuntimeError(f"{document_path}: normalization exposed a new complex repetition")
        if clean_generated_image_artifacts(normalized) != normalized:
            raise RuntimeError(f"{document_path}: generated image cleaning is not idempotent")
        if normalize_mixed_markup_to_gfm(normalized) != normalized:
            raise RuntimeError(f"{document_path}: HTML-to-GFM normalization is not idempotent")
        aggregate_markdown_before.update(before_structures)
        aggregate_markdown_after.update(after_structures)
        aggregate_markdown_tokens_before.update(before_tokens)
        aggregate_markdown_tokens_after.update(after_tokens)

        tag_counts = Counter({str(key): int(value) for key, value in dict(markup_metrics["tag_counts"]).items()})
        attribute_counts = Counter({str(key): int(value) for key, value in dict(markup_metrics["attribute_counts"]).items()})
        transformations = Counter({str(key): int(value) for key, value in dict(markup_metrics["transformations"]).items()})
        for structure in ("atx_headings", "fence_lines", "markdown_links"):
            if after_structures[structure] != before_structures[structure]:
                raise RuntimeError(f"{document_path}: existing {structure} were not preserved")
        if after_structures["markdown_images"] != before_structures["markdown_images"] + transformations.get("html_images_to_markdown", 0):
            raise RuntimeError(f"{document_path}: existing Markdown images were not preserved")
        if after_structures["gfm_table_delimiters"] != before_structures["gfm_table_delimiters"] + transformations.get("html_tables_to_gfm", 0):
            raise RuntimeError(f"{document_path}: existing or converted GFM tables did not close")
        if after_structures["strong_delimiter_pairs"] < before_structures["strong_delimiter_pairs"]:
            raise RuntimeError(f"{document_path}: existing strong-emphasis structures were lost")
        for token_type in PRESERVED_MARKDOWN_TOKEN_TYPES:
            if after_tokens[token_type] < before_tokens[token_type]:
                raise RuntimeError(f"{document_path}: existing Markdown token {token_type} was lost")
        aggregate_tags.update(tag_counts)
        aggregate_attributes.update(attribute_counts)
        aggregate_transformations.update(transformations)
        has_html = bool(tag_counts)
        changed = normalized != raw_text
        repetition_replacements = int(repetition_metrics["complex_repetition_replacements"])
        if has_html:
            documents_with_html += 1
        if changed:
            documents_changed += 1
            source_id = str(cards[opaque_id]["source_id"])
            changed_by_source[source_id] += 1
        if repetition_replacements:
            documents_with_repetition_replacement += 1
        replacement_count += repetition_replacements
        repetition_characters_removed += int(repetition_metrics["complex_repetition_characters_removed"])
        content_characters_removed += int(markup_metrics["content_characters_removed"])
        pseudo_tags_escaped += int(markup_metrics["pseudo_tags_escaped"])
        comments_removed += int(markup_metrics["comments_removed"])
        comments_preserved += int(markup_metrics["comments_preserved"])
        generated_image_artifacts_removed += int(generated_image_metrics["generated_image_artifact_count"])
        generated_image_characters_removed += int(generated_image_metrics["generated_image_characters_removed"])
        image_description_elements_commented += int(
            generated_image_metrics["image_description_elements_commented"]
        )
        image_description_comments_emitted += int(
            generated_image_metrics["image_description_comments_emitted"]
        )
        image_description_comments_retained_after_repetition += int(
            generated_image_metrics["image_description_comments_retained_after_repetition"]
        )
        image_description_comments_removed_as_repetition += int(
            generated_image_metrics["image_description_comments_removed_as_repetition"]
        )
        image_description_characters_preserved += int(
            generated_image_metrics["image_description_characters_preserved"]
        )
        empty_image_description_comments += int(
            generated_image_metrics["empty_image_description_comments"]
        )
        image_descriptions_split_around_pipeline_markers += int(
            generated_image_metrics["image_descriptions_split_around_pipeline_markers"]
        )
        pipeline_markers_preserved_inside_image_descriptions += int(
            generated_image_metrics["pipeline_markers_preserved_inside_image_descriptions"]
        )
        aggregate_generated_image_rules.update(
            {
                str(key): int(value)
                for key, value in dict(generated_image_metrics["generated_image_rule_counts"]).items()
            }
        )

        output_relative: str | None = None
        normalized_sha = sha256_bytes(normalized.encode("utf-8"))
        if changed:
            rendered_html = renderer.render(normalized)
            if re.search(r"<(?:script|style|iframe|object|embed)\b", rendered_html, re.IGNORECASE):
                raise RuntimeError(f"{document_path}: unsafe element reached the Markdown preview")
            output_path = site_dir / OUTPUT_DOCUMENT_DIR / f"{opaque_id}.json"
            _write_json(
                output_path,
                {
                    "schema_version": DOCUMENT_SCHEMA,
                    "opaque_id": opaque_id,
                    "original_text_sha256": sha256_bytes(raw_text.encode("utf-8")),
                    "cleaned_text_sha256": sha256_bytes(cleaned_text.encode("utf-8")),
                    "normalized_markdown_sha256": normalized_sha,
                    "cleaned_text": cleaned_text,
                    "normalized_markdown": normalized,
                    "rendered_html": rendered_html,
                    "renderer": renderer_description,
                },
            )
            output_relative = output_path.relative_to(site_dir).as_posix()

        card = cards[opaque_id]
        rows.append(
            {
                "opaque_id": opaque_id,
                "source_id": str(card["source_id"]),
                "source_doc_id": str(card["source_doc_id"]),
                "changed": changed,
                "has_recognized_html": has_html,
                "normalized_document_path": output_relative,
                "original_char_count": len(raw_text),
                "normalized_char_count": len(normalized),
                "normalized_markdown_sha256": normalized_sha,
                "tag_counts": dict(sorted(tag_counts.items())),
                "transformations": dict(sorted(transformations.items())),
                "content_characters_removed": int(markup_metrics["content_characters_removed"]),
                "pseudo_tags_escaped": int(markup_metrics["pseudo_tags_escaped"]),
                "comments_removed": int(markup_metrics["comments_removed"]),
                "generated_image_artifacts_removed": int(
                    generated_image_metrics["generated_image_artifact_count"]
                ),
                "generated_image_characters_removed": int(
                    generated_image_metrics["generated_image_characters_removed"]
                ),
                "generated_image_rule_counts": generated_image_metrics["generated_image_rule_counts"],
                "generated_image_events": generated_image_metrics["generated_image_events"],
                "image_description_elements_commented": int(
                    generated_image_metrics["image_description_elements_commented"]
                ),
                "image_description_comments_emitted": int(
                    generated_image_metrics["image_description_comments_emitted"]
                ),
                "image_description_comments_retained_after_repetition": int(
                    generated_image_metrics["image_description_comments_retained_after_repetition"]
                ),
                "image_description_comments_removed_as_repetition": int(
                    generated_image_metrics["image_description_comments_removed_as_repetition"]
                ),
                "image_description_characters_preserved": int(
                    generated_image_metrics["image_description_characters_preserved"]
                ),
                "empty_image_description_comments": int(
                    generated_image_metrics["empty_image_description_comments"]
                ),
                "image_descriptions_split_around_pipeline_markers": int(
                    generated_image_metrics["image_descriptions_split_around_pipeline_markers"]
                ),
                "pipeline_markers_preserved_inside_image_descriptions": int(
                    generated_image_metrics["pipeline_markers_preserved_inside_image_descriptions"]
                ),
                "table_fallback_events": markup_metrics["table_fallback_events"],
                "repetition_replacements": repetition_replacements,
                "repetition_characters_removed": int(repetition_metrics["complex_repetition_characters_removed"]),
                "repetition_details": repetition_metrics["complex_repetition_replacement_details"],
                "markdown_structures_before": before_structures,
                "markdown_structures_after": after_structures,
                "markdown_tokens_before": before_tokens,
                "markdown_tokens_after": after_tokens,
            }
        )

    if len(rows) != int(manifest.get("document_count", -1)):
        raise ValueError("normalization document count does not close with site manifest")
    _remove_stale_normalized_documents(
        site_dir,
        (
            str(row["normalized_document_path"])
            for row in rows
            if row["normalized_document_path"] is not None
        ),
    )
    policy_tags = {
        tag
        for policy in TRANSFORMATION_POLICY
        for tag in map(str, policy["tags"])
        if tag not in {"unknown tag-like angle text", "HTML comments"}
    }
    uncovered_tags = sorted(set(aggregate_tags) - policy_tags)
    if uncovered_tags:
        raise RuntimeError(f"observed HTML tags lack an explicit transformation decision: {uncovered_tags}")
    handled_tables = sum(
        aggregate_transformations.get(key, 0)
        for key in ("html_tables_to_gfm", "html_tables_fallback_to_text", "empty_tables_removed")
    )
    if handled_tables != aggregate_tags.get("table", 0):
        raise RuntimeError("HTML table handling count does not close")
    handled_cells = (
        aggregate_transformations.get("html_table_cells_preserved", 0)
        + aggregate_transformations.get("html_table_fallback_cells_preserved", 0)
        + aggregate_transformations.get("orphan_table_cells_flattened", 0)
    )
    if handled_cells != aggregate_tags.get("td", 0) + aggregate_tags.get("th", 0):
        raise RuntimeError("HTML table-cell handling count does not close")
    decision_rows = []
    for policy in TRANSFORMATION_POLICY:
        tags = [str(tag) for tag in policy["tags"]]
        if tags == ["unknown tag-like angle text"]:
            observed = pseudo_tags_escaped
        elif tags == ["HTML comments"]:
            observed = comments_removed + comments_preserved
        else:
            observed = sum(aggregate_tags.get(tag, 0) for tag in tags)
        decision_rows.append({**policy, "observed_start_tag_count": observed})
    return {
        "schema_version": SCHEMA,
        "status": "passed",
        "prototype_scope": "Dry-run review sample only; no source or raw-review document was overwritten.",
        "renderer": renderer_description,
        "glossapi_reuse_review": {
            "reused_now": [
                "glossapi.ocr.utils.repetition.replace_complex_repetitions",
            ],
            "reuse_when_ported": [
                "glossapi.ocr.utils.cleaning.canonicalize_markdown for whitespace, dehyphenation, placeholder-cell and empty-table cleanup after structural conversion",
                "glossapi_rs_cleaner.table_analysis_module for validation of emitted pipe tables",
                "glossapi.gloss_section for downstream consumption of existing GFM headings, lists, and tables",
            ],
            "must_precede": "Run HTML-to-GFM conversion before Rust strip_tags_custom; that Rust function removes tags but does not preserve their structure.",
            "do_not_reuse_for_conversion": "The Rust malformed-table stage removes tables; it does not convert HTML tables.",
            "complex_repetition_module_sha256": sha256_bytes(repetition_path.read_bytes()),
        },
        "transformation_decisions": decision_rows,
        "summary": {
            "document_count": len(rows),
            "documents_changed": documents_changed,
            "documents_with_recognized_html": documents_with_html,
            "documents_with_repetition_replacement": documents_with_repetition_replacement,
            "repetition_replacement_count": replacement_count,
            "repetition_characters_removed": repetition_characters_removed,
            "html_start_tag_count": sum(aggregate_tags.values()),
            "html_tag_counts": dict(sorted(aggregate_tags.items())),
            "html_attribute_counts": dict(sorted(aggregate_attributes.items())),
            "transformation_counts": dict(sorted(aggregate_transformations.items())),
            "content_characters_removed": content_characters_removed,
            "pseudo_tags_escaped": pseudo_tags_escaped,
            "html_comments_removed": comments_removed,
            "html_comments_preserved": comments_preserved,
            "generated_image_artifacts_removed": generated_image_artifacts_removed,
            "generated_image_characters_removed": generated_image_characters_removed,
            "generated_image_rule_counts": dict(sorted(aggregate_generated_image_rules.items())),
            "image_description_elements_commented": image_description_elements_commented,
            "image_description_comments_emitted": image_description_comments_emitted,
            "image_description_comments_retained_after_repetition": image_description_comments_retained_after_repetition,
            "image_description_comments_removed_as_repetition": image_description_comments_removed_as_repetition,
            "image_description_characters_preserved": image_description_characters_preserved,
            "empty_image_description_comments": empty_image_description_comments,
            "image_descriptions_split_around_pipeline_markers": image_descriptions_split_around_pipeline_markers,
            "pipeline_markers_preserved_inside_image_descriptions": pipeline_markers_preserved_inside_image_descriptions,
            "residual_recognized_html_tags": 0,
            "uncovered_html_tags": uncovered_tags,
            "normalization_idempotence_failures": 0,
            "markdown_structures_before": dict(sorted(aggregate_markdown_before.items())),
            "markdown_structures_after": dict(sorted(aggregate_markdown_after.items())),
            "markdown_tokens_before": dict(sorted(aggregate_markdown_tokens_before.items())),
            "markdown_tokens_after": dict(sorted(aggregate_markdown_tokens_after.items())),
            "documents_changed_by_source": dict(sorted(changed_by_source.items())),
        },
        "documents": rows,
    }


def write_normalization(*, site_dir: Path, glossapi_root: Path) -> dict[str, object]:
    site_dir = site_dir.resolve()
    import build_agent1_v4_review_site as site_assets

    site_assets._write_file(site_dir / "normalization.html", site_assets._normalization_html().encode("utf-8"))
    site_assets._write_file(
        site_dir / "assets/normalization.css", site_assets._normalization_css().encode("utf-8")
    )
    site_assets._write_file(
        site_dir / "assets/normalization.js", site_assets._normalization_js().encode("utf-8")
    )
    audit = normalize_site(site_dir=site_dir, glossapi_root=glossapi_root)
    audit_path = site_dir / AUDIT_RELATIVE_PATH
    _write_json(audit_path, audit)
    manifest_path = site_dir / "site_manifest.json"
    manifest = _read_json(manifest_path)
    inventory = _inventory_without_manifest(site_dir)
    portable_assets = [item for item in inventory if not _is_bulk_asset(str(item["path"]))]
    portable_asset_bytes = sum(int(item["bytes"]) for item in portable_assets)
    if portable_asset_bytes > int(manifest["max_portable_assets_bytes"]):
        raise ValueError("portable site assets exceed the frozen size limit")
    audit_payload = audit_path.read_bytes()
    manifest["gfm_normalization_audit"] = {
        "path": AUDIT_RELATIVE_PATH.as_posix(),
        "sha256": sha256_bytes(audit_payload),
        "bytes": len(audit_payload),
    }
    manifest["portable_assets"] = portable_assets
    manifest["portable_asset_bytes"] = portable_asset_bytes
    manifest["files"] = inventory
    _write_json(manifest_path, manifest)
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-dir", type=Path, required=True)
    parser.add_argument("--glossapi-root", type=Path, default=DEFAULT_GLOSSAPI_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit = write_normalization(site_dir=args.site_dir, glossapi_root=args.glossapi_root)
    print(json.dumps(audit["summary"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
