#!/usr/bin/env python3
"""Inspect Wave-4 tokenizer vocabularies for residual cleaner artifacts.

The continuous tokenizer inherits a base vocabulary, so this script can report
both the full vocabulary and an added-token slice selected by token id.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


GREEK_RANGES = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
CTRL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}

POSTSCRIPT_NAMES = {
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "a",
    "alpha",
    "Alpha",
    "b",
    "beta",
    "Beta",
    "c",
    "colon",
    "comma",
    "d",
    "delta",
    "Delta",
    "e",
    "ellipsis",
    "elipsis",
    "f",
    "g",
    "gamma",
    "Gamma",
    "h",
    "hyphen",
    "hyphenminus",
    "i",
    "j",
    "k",
    "l",
    "m",
    "minus",
    "n",
    "o",
    "omega",
    "Omega",
    "p",
    "parenleft",
    "parenright",
    "period",
    "pi",
    "q",
    "quotedbl",
    "quoteleft",
    "quoteright",
    "r",
    "s",
    "semicolon",
    "sigma",
    "Sigma",
    "space",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
}


def bytes_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


UNICODE_TO_BYTE = {v: k for k, v in bytes_to_unicode().items()}


def escape_controls(text: str) -> str:
    return "".join(CTRL_ESCAPES.get(ch, ch) for ch in text)


def decode_bytelevel(raw: str) -> tuple[str, bool, bool, str, str]:
    try:
        body_bytes = bytes(UNICODE_TO_BYTE[ch] for ch in raw)
    except KeyError:
        return escape_controls(raw), False, False, "", ""
    word_initial = body_bytes.startswith(b" ")
    body = body_bytes[1:] if word_initial else body_bytes
    try:
        return escape_controls(body.decode("utf-8")), word_initial, True, "", ""
    except UnicodeDecodeError:
        head_end = 0
        while head_end < len(body) and 0x80 <= body[head_end] <= 0xBF:
            head_end += 1
        lead = body[:head_end].hex()
        rest = body[head_end:]
        n = len(rest)
        text = ""
        while n > 0:
            try:
                text = rest[:n].decode("utf-8")
                break
            except UnicodeDecodeError:
                n -= 1
        return escape_controls(text), word_initial, False, lead, rest[n:].hex()


def has_greek(text: str) -> bool:
    stripped = "".join(
        ch for ch in unicodedata.normalize("NFD", text) if unicodedata.category(ch) != "Mn"
    )
    return any(any(lo <= ord(ch) <= hi for lo, hi in GREEK_RANGES) for ch in stripped)


def mojibake_recovers_greek(text: str) -> bool:
    if not text:
        return False
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    return has_greek(fixed)


@dataclass(frozen=True)
class TokenRow:
    tokenizer: str
    scope: str
    token_id: int
    raw: str
    decoded: str
    word_initial: bool
    decodable: bool
    leading_bytes: str
    trailing_bytes: str


PS_SLASH_RE = re.compile(r"/([A-Za-z][A-Za-z0-9]{0,32})\b")
PS_FONT_SUBSET_RE = re.compile(r"/[A-Z]{2,6}(?:\+[A-Za-z][A-Za-z0-9_.-]*)?\b")
GLYPH_UPPER_RE = re.compile(r"GLYPH")
GLYPH_REPEAT_RE = re.compile(r"(?:GLYPH){2,}")
GLYPH_STRUCTURED_RE = re.compile(r"glyph\[[A-Za-z0-9_.=-]{1,100}\]", re.IGNORECASE)
POSTSCRIPT_STRUCTURED_RE = re.compile(r"(?:font=/|FontName=|/uni[0-9A-Fa-f]{4,}|/gid[0-9]+|/g[0-9]+)")


def has_postscript_name(text: str) -> bool:
    for match in PS_SLASH_RE.finditer(text):
        if match.group(1) in POSTSCRIPT_NAMES:
            return True
    return bool(PS_FONT_SUBSET_RE.search(text))


CategoryPredicate = Callable[[str, TokenRow], bool]


CATEGORIES: dict[str, CategoryPredicate] = {
    "glyph_upper_stem": lambda text, row: bool(GLYPH_UPPER_RE.search(text)),
    "glyph_upper_repeat": lambda text, row: bool(GLYPH_REPEAT_RE.search(text)),
    "glyph_structured_marker": lambda text, row: bool(GLYPH_STRUCTURED_RE.search(text)),
    "postscript_hyphenminus": lambda text, row: "/hyphenminus" in text,
    "postscript_named_glyph": lambda text, row: has_postscript_name(text),
    "postscript_structured_pdf": lambda text, row: bool(POSTSCRIPT_STRUCTURED_RE.search(text)),
    "mojibake_marker_char": lambda text, row: any(ch in text for ch in "ÎÏïâÂÃ�"),
    "latin1_utf8_mojibake_recovers_greek": lambda text, row: mojibake_recovers_greek(text),
    "latin_ext_a_or_b": lambda text, row: any(0x0100 <= ord(ch) <= 0x024F for ch in text),
    "pua": lambda text, row: any(
        0xE000 <= ord(ch) <= 0xF8FF
        or 0xF0000 <= ord(ch) <= 0xFFFFD
        or 0x100000 <= ord(ch) <= 0x10FFFD
        for ch in text
    ),
    "replacement_char": lambda text, row: "\uFFFD" in text,
    "math_alphanumeric": lambda text, row: any(0x1D400 <= ord(ch) <= 0x1D7FF for ch in text),
    "cyrillic": lambda text, row: any(0x0400 <= ord(ch) <= 0x04FF for ch in text),
    "nondecodable_boundary_fragment": lambda text, row: not row.decodable,
}


def load_vocab(name: str, tokenizer_json: Path, added_threshold: int | None) -> list[TokenRow]:
    data = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    vocab = data.get("model", {}).get("vocab", {})
    if not isinstance(vocab, dict):
        vocab = {token: token_id for token, token_id in vocab}
    rows: list[TokenRow] = []
    for raw, token_id_value in vocab.items():
        token_id = int(token_id_value)
        decoded, word_initial, decodable, lead, trail = decode_bytelevel(raw)
        rows.append(
            TokenRow(
                tokenizer=name,
                scope="all",
                token_id=token_id,
                raw=raw,
                decoded=decoded,
                word_initial=word_initial,
                decodable=decodable,
                leading_bytes=lead,
                trailing_bytes=trail,
            )
        )
        if added_threshold is not None and token_id >= added_threshold:
            rows.append(
                TokenRow(
                    tokenizer=name,
                    scope=f"added_id_ge_{added_threshold}",
                    token_id=token_id,
                    raw=raw,
                    decoded=decoded,
                    word_initial=word_initial,
                    decodable=decodable,
                    leading_bytes=lead,
                    trailing_bytes=trail,
                )
            )
    return sorted(rows, key=lambda row: (row.tokenizer, row.scope, row.token_id))


def parse_name_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.name, path
    name, path = value.split("=", 1)
    if not name:
        raise argparse.ArgumentTypeError("tokenizer name cannot be empty")
    return name, Path(path)


def parse_threshold(value: str) -> tuple[str, int]:
    name, raw = value.split("=", 1)
    return name, int(raw)


def analyze(rows: Iterable[TokenRow]) -> tuple[dict, list[dict]]:
    summary: dict[str, dict[str, dict]] = defaultdict(dict)
    suspicious: list[dict] = []
    grouped_rows: dict[tuple[str, str], list[TokenRow]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row.tokenizer, row.scope)].append(row)
        categories = [name for name, pred in CATEGORIES.items() if pred(row.decoded, row)]
        for category in categories:
            suspicious.append(
                {
                    "tokenizer": row.tokenizer,
                    "scope": row.scope,
                    "category": category,
                    "id": row.token_id,
                    "decoded": row.decoded,
                    "raw": row.raw,
                    "word_initial": row.word_initial,
                    "decodable": row.decodable,
                    "leading_bytes": row.leading_bytes,
                    "trailing_bytes": row.trailing_bytes,
                }
            )
    for (tokenizer, scope), group in grouped_rows.items():
        scope_summary = {"vocab_rows": len(group), "categories": {}}
        for category, pred in CATEGORIES.items():
            matches = [row for row in group if pred(row.decoded, row)]
            scope_summary["categories"][category] = {
                "count": len(matches),
                "sample": [
                    {
                        "id": row.token_id,
                        "decoded": row.decoded[:100],
                        "raw": row.raw[:120],
                        "decodable": row.decodable,
                    }
                    for row in matches[:20]
                ],
            }
        summary[tokenizer][scope] = scope_summary
    suspicious.sort(key=lambda item: (item["tokenizer"], item["scope"], item["category"], item["id"]))
    return summary, suspicious


def write_markdown(summary: dict, output_md: Path) -> None:
    category_names = list(CATEGORIES)
    lines = [
        "# Wave-4 Tokenizer Artifact Analysis",
        "",
        "| tokenizer | scope | vocab rows | " + " | ".join(f"`{name}`" for name in category_names) + " |",
        "| --- | --- | ---: | " + " | ".join("---:" for _ in category_names) + " |",
    ]
    for tokenizer in sorted(summary):
        for scope in sorted(summary[tokenizer]):
            rec = summary[tokenizer][scope]
            counts = [str(rec["categories"][name]["count"]) for name in category_names]
            lines.append(
                f"| `{tokenizer}` | `{scope}` | {rec['vocab_rows']} | "
                + " | ".join(counts)
                + " |"
            )
    lines.append("")
    for tokenizer in sorted(summary):
        for scope in sorted(summary[tokenizer]):
            lines.append(f"## {tokenizer} / {scope}")
            lines.append("")
            for category in category_names:
                rec = summary[tokenizer][scope]["categories"][category]
                if rec["count"] == 0:
                    continue
                rendered_items = []
                for item in rec["sample"][:12]:
                    decoded = item["decoded"].replace("`", "\\`")
                    rendered_items.append(f"`{item['id']}:{decoded}`")
                rendered = ", ".join(rendered_items)
                lines.append(f"- `{category}`: {rec['count']} hits; {rendered}")
            lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", action="append", required=True, type=parse_name_path)
    parser.add_argument("--added-threshold", action="append", default=[], type=parse_threshold)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    thresholds = dict(args.added_threshold)
    rows: list[TokenRow] = []
    for name, path in args.tokenizer:
        rows.extend(load_vocab(name, path, thresholds.get(name)))

    summary, suspicious = analyze(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "wave4_tokenizer_artifact_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.out_dir / "wave4_suspicious_tokens.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in suspicious),
        encoding="utf-8",
    )
    write_markdown(summary, args.out_dir / "wave4_tokenizer_artifact_report.md")
    print(f"wrote {args.out_dir / 'wave4_tokenizer_artifact_report.md'}")
    print(f"wrote {args.out_dir / 'wave4_tokenizer_artifact_summary.json'}")
    print(f"wrote {args.out_dir / 'wave4_suspicious_tokens.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
