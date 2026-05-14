#!/usr/bin/env python3
"""Wave-2 tokenizer/corpus noise analysis.

This script intentionally does not depend on the existing F1 extractor
scripts. It decodes GPT-2 ByteLevel BPE vocab entries, classifies likely
noise families, compares arms, and can scan training parquet text for
the same families.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SPACE_MARKER = "Ġ"
GREEK_RANGES = ((0x0370, 0x03FF), (0x1F00, 0x1FFF))
CYRILLIC_HOMOGLYPHS = set(
    "\u0430\u0435\u043e\u0440\u0441\u0443\u0445\u0456"
    "\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0425"
)
DINGBAT_RANGES = ((0x2600, 0x27BF),)
PS_GLYPH_NAMES = {
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
    "n",
    "o",
    "omega",
    "Omega",
    "p",
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


def _bytes_to_unicode() -> dict[int, str]:
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
    return {b: chr(c) for b, c in zip(bs, cs)}


UNICODE_TO_BYTE = {v: k for k, v in _bytes_to_unicode().items()}
CTRL_ESCAPES = {"\n": "\\n", "\r": "\\r", "\t": "\\t"}


def escape_controls(text: str) -> str:
    return "".join(CTRL_ESCAPES.get(c, c) for c in text)


def decode_bytelevel_token(raw: str) -> tuple[str, bool, bool, str, str]:
    try:
        body_bytes = bytes(UNICODE_TO_BYTE[c] for c in raw)
    except KeyError:
        return raw, False, False, "", ""
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
        trail = rest[n:].hex()
        return escape_controls(text), word_initial, False, lead, trail


def has_greek(text: str) -> bool:
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    return any(any(lo <= ord(c) <= hi for lo, hi in GREEK_RANGES) for c in stripped)


def is_reserved(text: str, decodable: bool) -> bool:
    if not decodable or len(text) < 3:
        return False
    return (
        text.startswith("<")
        and text.endswith(">")
        and all(ord(c) < 128 for c in text)
    ) or (
        text.startswith("[")
        and text.endswith("]")
        and all(ord(c) < 128 for c in text)
    )


def has_letter_or_digit(text: str) -> bool:
    return any(unicodedata.category(c)[0] in {"L", "N"} for c in text)


def contains_dingbat(text: str) -> bool:
    return any(any(lo <= ord(c) <= hi for lo, hi in DINGBAT_RANGES) for c in text)


def md_code_text(text: str, limit: int = 60) -> str:
    return text.replace("`", "\\`")[:limit]


def mojibake_recovers_greek(text: str) -> str | None:
    if not text:
        return None
    try:
        fixed = text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return fixed if has_greek(fixed) else None


RUN_RE = re.compile(r"^(.)(?:\1){3,}$", re.DOTALL)
DASH_RE = re.compile(r"^[-\u2010\u2011\u2012\u2013\u2014\u2015]{4,}$")
DOT_RE = re.compile(r"^[.\u00b7\u2022\u2027\u22c5\u22ef\u2026]{4,}$")
ESCAPED_RUN_RE = re.compile(r"^(?:\\[_*#.=\\/\-~]){2,}$")
TABLE_FRAGMENT_RE = re.compile(r"^\|?[-:| ]{8,}\|?$")
BARE_GLYPH_RE = re.compile(r"^(?:GLYPH)+$", re.IGNORECASE)
PS_SLASH_RE = re.compile(r"^/([A-Za-z][A-Za-z0-9]{0,20})$")
PS_FONT_SUBSET_RE = re.compile(r"^/[GCBDF][A-Z]{1,3}$")
PAGE_MARKER_RE = re.compile(r"^(?:Page|page|\u03a3\u03b5\u03bb\u03af\u03b4\u03b1)$")


@dataclass
class TokenRow:
    arm: str
    token_id: int
    token: str
    word_initial: bool
    decodable: bool
    leading_bytes: str
    trailing_bytes: str


def load_vocab(tokenizer_dir: Path) -> list[TokenRow]:
    data = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = data["model"]["vocab"]
    rows = []
    arm = tokenizer_dir.name
    for raw, token_id in vocab.items():
        text, word_initial, decodable, lead, trail = decode_bytelevel_token(raw)
        rows.append(TokenRow(arm, int(token_id), text, word_initial, decodable, lead, trail))
    return sorted(rows, key=lambda r: r.token_id)


def classify_token(row: TokenRow) -> set[str]:
    text = row.token
    cats: set[str] = set()
    if is_reserved(text, row.decodable):
        return cats
    # ByteLevel BPE often has non-decodable boundary fragments that are
    # normal tokenizer mechanics, not corpus-cleaning evidence.
    if any(ord(c) < 32 or 0x7F <= ord(c) <= 0x9F or c == "\u00ad" for c in text):
        cats.add("control_or_soft_hyphen")
    if "\u00b5" in text:
        cats.add("micro_sign")
    if any(c in CYRILLIC_HOMOGLYPHS for c in text):
        cats.add("cyrillic_homoglyph")
    fixed = mojibake_recovers_greek(text)
    if fixed is not None:
        cats.add("latin1_utf8_mojibake_recovers_greek")
    if any(c in "\u00ce\u00cf\u00ef\u00e2" for c in text):
        cats.add("mojibake_marker_char")
    if BARE_GLYPH_RE.match(text):
        cats.add("bare_glyph_repeat")
    ps_match = PS_SLASH_RE.match(text)
    if ps_match and (ps_match.group(1) in PS_GLYPH_NAMES or PS_FONT_SUBSET_RE.match(text)):
        cats.add("postscript_glyph_name")
    if ESCAPED_RUN_RE.match(text):
        cats.add("escaped_markdown_run")
    if TABLE_FRAGMENT_RE.match(text) and ("|" in text or "-" in text):
        cats.add("table_separator_fragment")
    if DASH_RE.match(text) or DOT_RE.match(text) or RUN_RE.match(text):
        if len(text) >= 4:
            cats.add("long_punct_or_symbol_run")
    if text.startswith("$$") or text in {"$$", "$$\\", "$$("}:
        cats.add("math_fence_residue")
    if PAGE_MARKER_RE.match(text):
        cats.add("page_marker_residue")
    if contains_dingbat(text):
        cats.add("dingbat_or_symbol_kept")
    return cats


def write_vocab_report(tokenizer_dirs: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suspicious_rows: list[dict] = []
    by_arm_cat: dict[str, dict[str, list[TokenRow]]] = defaultdict(lambda: defaultdict(list))
    token_to_arms: dict[tuple[str, str], set[str]] = defaultdict(set)

    for tokenizer_dir in tokenizer_dirs:
        for row in load_vocab(tokenizer_dir):
            cats = classify_token(row)
            for cat in cats:
                by_arm_cat[row.arm][cat].append(row)
                token_to_arms[(cat, row.token)].add(row.arm)
                suspicious_rows.append(
                    {
                        "arm": row.arm,
                        "id": row.token_id,
                        "token": row.token,
                        "word_initial": row.word_initial,
                        "decodable": row.decodable,
                        "leading_bytes": row.leading_bytes,
                        "trailing_bytes": row.trailing_bytes,
                        "category": cat,
                    }
                )

    suspicious_rows.sort(key=lambda r: (r["category"], r["arm"], r["id"]))
    (out_dir / "suspicious_tokens.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in suspicious_rows),
        encoding="utf-8",
    )

    all_arms = sorted({d.name for d in tokenizer_dirs})
    all_cats = sorted({r["category"] for r in suspicious_rows})
    lines = [
        "# Wave-2 Bad-Token Vocab Analysis",
        "",
        "Independent ByteLevel decode and pattern classification over completed wave-2 tokenizer arms.",
        "",
        "| category | " + " | ".join(all_arms) + " | shared tokens |",
        "| --- | " + " | ".join("---:" for _ in all_arms) + " | ---: |",
    ]
    for cat in all_cats:
        counts = [len(by_arm_cat[arm].get(cat, [])) for arm in all_arms]
        shared = sum(1 for (c, _tok), arms in token_to_arms.items() if c == cat and len(arms) > 1)
        lines.append(
            f"| `{cat}` | " + " | ".join(str(c) for c in counts) + f" | {shared} |"
        )
    lines.append("")
    for cat in all_cats:
        lines.append(f"## {cat}")
        lines.append("")
        for arm in all_arms:
            sample = by_arm_cat[arm].get(cat, [])[:12]
            if not sample:
                continue
            rendered = ", ".join(f"`{r.token_id}:{md_code_text(r.token)}`" for r in sample)
            lines.append(f"- `{arm}`: {rendered}")
        shared_tokens = sorted(
            tok for (c, tok), arms in token_to_arms.items() if c == cat and len(arms) > 1
        )[:20]
        if shared_tokens:
            rendered = ", ".join(f"`{md_code_text(t)}`" for t in shared_tokens)
            lines.append(f"- shared examples: {rendered}")
        lines.append("")

    (out_dir / "vocab_bad_token_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'vocab_bad_token_report.md'}")
    print(f"wrote {out_dir / 'suspicious_tokens.jsonl'}")


CORPUS_PATTERNS = {
    "long_dash_run": re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015]{6,}"),
    "long_underscore_run": re.compile(r"_{4,}"),
    "long_asterisk_run": re.compile(r"\*{4,}"),
    "long_equal_run": re.compile(r"={4,}"),
    "hash_7plus_run": re.compile(r"#{7,}"),
    "long_slash_run": re.compile(r"/{4,}"),
    "long_backslash_run": re.compile(r"\\{4,}"),
    "long_pipe_run": re.compile(r"[|]{4,}"),
    "escaped_markdown_run": re.compile(r"(?:\\[_*#.=\\/\-~]){2,}"),
    "dot_leader_run": re.compile(r"[.\u00b7\u2022\u2027\u22c5\u22ef\u2026]{4,}"),
    "table_separator_fragment": re.compile(r"(?:\|[ :.-]{4,}){1,}\|?"),
    "bare_glyph_repeat": re.compile(
        r"\b(?:(?:GLYPH)+|glyph\[[A-Za-z0-9_.-]{1,80}\])\b",
        re.IGNORECASE,
    ),
    "postscript_glyph_name": re.compile(
        r"(?<![A-Za-z0-9.])/(?:hyphenminus|hyphen|ellipsis|elipsis|period|comma|space|colon|semicolon|pi|alpha|Alpha|sigma|Sigma|omega|Omega)\b|(?<![A-Za-z0-9.])/[GCBDF][A-Z]{1,3}\b"
    ),
    "control_or_soft_hyphen": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad]"),
    "micro_sign": re.compile(r"\u00b5"),
    "mojibake_marker": re.compile(r"[\u00ce\u00cf\u00ef\u00e2][\u0080-\u00ff]?"),
    "cyrillic_any": re.compile(r"[\u0400-\u04ff]"),
    "empty_math_fence": re.compile(r"\$\$\s*(?:\\\s*)?\$\$", re.DOTALL),
    "page_marker_line": re.compile(
        r"(?im)^\s*(?:Page|page|\u03a3\u03b5\u03bb\u03af\u03b4\u03b1)\s+\d+(?:\s+(?:of|\u03b1\u03c0\u03cc)\s+\d+)?\s*$"
    ),
}


DUCKDB_PATTERNS = {
    "table_separator_fragment": r"\|[ :.\-]{4,}",
    "dot_leader_run": r"[.·•‧⋅⋯…]{4,}",
    "long_dash_run": r"[-‐‑‒–—―]{6,}",
    "long_underscore_run": r"_{4,}",
    "escaped_markdown_run": r"(\\[_*#.=\\/\-~]){2,}",
    "bare_glyph_or_glyph_bracket": r"(?i)((GLYPH){1,}|glyph\[[A-Za-z0-9_.-]{1,80}\])",
    "micro_sign": "\u00b5",
    "page_marker_line": r"(?m)^\s*(Page|page|Σελίδα)\s+[0-9]+",
    "empty_math_fence": r"\$\$[[:space:]]*\$\$",
    "hash_7plus_run": r"#{7,}",
    "long_pipe_run": r"[|]{4,}",
}

MP_PATTERNS = {
    "table_separator_fragment": re.compile(r"\|[ :.\-]{4,}"),
    "dot_leader_run": re.compile(r"[.\u00b7\u2022\u2027\u22c5\u22ef\u2026]{4,}"),
    "long_dash_run": re.compile(r"[-\u2010\u2011\u2012\u2013\u2014\u2015]{6,}"),
    "long_underscore_run": re.compile(r"_{4,}"),
    "escaped_markdown_run": re.compile(r"(?:\\[_*#.=\\/\-~]){2,}"),
    "bare_glyph_or_glyph_bracket": re.compile(
        r"\b(?:(?:GLYPH)+|glyph\[[A-Za-z0-9_.-]{1,80}\])\b",
        re.IGNORECASE,
    ),
    "postscript_named_glyph_exact": re.compile(
        r"(?<![A-Za-z0-9.])/(?:hyphenminus|hyphen|ellipsis|elipsis|period|comma|space|colon|semicolon|pi|alpha|Alpha|sigma|Sigma|omega|Omega)\b"
    ),
    "html_comment_placeholder": re.compile(
        r"<!--\s*(?:image|text-missing|formula-not-decoded)\s*-->",
        re.IGNORECASE,
    ),
    "micro_sign": re.compile("\u00b5"),
    "page_marker_line": re.compile(
        r"(?m)^\s*(?:Page|page|\u03a3\u03b5\u03bb\u03af\u03b4\u03b1)\s+[0-9]+"
    ),
    "hash_7plus_run": re.compile(r"#{7,}"),
    "long_pipe_run": re.compile(r"[|]{4,}"),
    "long_slash_run": re.compile(r"/{4,}"),
    "long_equal_run": re.compile(r"={4,}"),
    "long_asterisk_run": re.compile(r"\*{4,}"),
    "soft_hyphen_or_non_newline_control": re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u00ad]"),
    "cyrillic_any": re.compile(r"[\u0400-\u04ff]"),
    "mojibake_marker": re.compile(r"[\u00ce\u00cf\u00ef\u00e2]"),
}


def sql_string_literal(text: str) -> str:
    return "'" + text.replace("'", "''") + "'"


def snippet(text: str, start: int, end: int, radius: int = 120) -> str:
    lo = max(0, start - radius)
    hi = min(len(text), end + radius)
    return escape_controls(" ".join(text[lo:hi].split()))


def scan_corpus(parquet_paths: list[Path], output: Path, max_docs: int | None, batch_size: int) -> None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyarrow is required for corpus scanning") from exc

    stats = {
        name: {"matches": 0, "docs": 0, "examples": []}
        for name in CORPUS_PATTERNS
    }
    total_docs = 0
    for parquet_path in parquet_paths:
        pf = pq.ParquetFile(parquet_path)
        for batch in pf.iter_batches(columns=["text"], batch_size=batch_size):
            texts = batch.column(0).to_pylist()
            for text in texts:
                if text is None:
                    continue
                total_docs += 1
                for name, pattern in CORPUS_PATTERNS.items():
                    matches = list(pattern.finditer(text))
                    if not matches:
                        continue
                    stats[name]["matches"] += len(matches)
                    stats[name]["docs"] += 1
                    if len(stats[name]["examples"]) < 8:
                        m = matches[0]
                        stats[name]["examples"].append(
                            {
                                "path": str(parquet_path),
                                "doc_index_seen": total_docs,
                                "match": escape_controls(m.group(0)[:200]),
                                "snippet": snippet(text, m.start(), m.end()),
                            }
                        )
                if max_docs is not None and total_docs >= max_docs:
                    break
            if total_docs and total_docs % 5000 == 0:
                print(f"scanned {total_docs:,} docs", file=sys.stderr)
            if max_docs is not None and total_docs >= max_docs:
                break
        if max_docs is not None and total_docs >= max_docs:
            break

    payload = {"total_docs_scanned": total_docs, "patterns": stats}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Wave-2 Corpus Noise Scan",
        "",
        f"Docs scanned: {total_docs:,}",
        "",
        "| pattern | docs | matches |",
        "| --- | ---: | ---: |",
    ]
    for name, item in sorted(stats.items(), key=lambda kv: (-kv[1]["docs"], kv[0])):
        lines.append(f"| `{name}` | {item['docs']:,} | {item['matches']:,} |")
    lines.append("")
    for name, item in sorted(stats.items(), key=lambda kv: (-kv[1]["docs"], kv[0])):
        if not item["examples"]:
            continue
        lines.append(f"## {name}")
        lines.append("")
        for ex in item["examples"]:
            lines.append(f"- `{ex['match']}` in `{ex['path']}` doc_seen={ex['doc_index_seen']}: {ex['snippet']}")
        lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output}")
    print(f"wrote {output.with_suffix('.json')}")


def duckdb_doc_counts(parquet_paths: list[Path], output_json: Path, threads: int) -> None:
    try:
        import duckdb
    except ModuleNotFoundError as exc:
        raise SystemExit("duckdb is required for full doc-count scanning") from exc

    results = {}
    for parquet_path in parquet_paths:
        select_items = ["count(*) as rows", "sum(length(text)) as chars"]
        for name, pattern in DUCKDB_PATTERNS.items():
            select_items.append(
                "sum(case when regexp_matches(text, "
                + sql_string_literal(pattern)
                + f") then 1 else 0 end) as {name}"
            )
        sql = (
            "select "
            + ", ".join(select_items)
            + " from read_parquet("
            + sql_string_literal(str(parquet_path))
            + ")"
        )
        con = duckdb.connect()
        con.execute(f"PRAGMA threads={int(threads)}")
        row = con.execute(sql).fetchone()
        cols = [desc[0] for desc in con.description]
        results[str(parquet_path)] = dict(zip(cols, row))
        con.close()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {output_json}")


def scan_text_chunk(task: tuple[int, str, list[str]]) -> dict:
    first_doc_index, source_path, texts = task
    result = {
        "docs": len(texts),
        "chars": 0,
        "patterns": {
            name: {"docs": 0, "examples": []}
            for name in MP_PATTERNS
        },
    }
    for offset, text in enumerate(texts):
        if text is None:
            continue
        result["chars"] += len(text)
        doc_index = first_doc_index + offset
        for name, pattern in MP_PATTERNS.items():
            match = pattern.search(text)
            if match is None:
                continue
            bucket = result["patterns"][name]
            bucket["docs"] += 1
            if len(bucket["examples"]) < 3:
                bucket["examples"].append(
                    {
                        "path": source_path,
                        "doc_index": doc_index,
                        "match": escape_controls(match.group(0)[:200]),
                        "snippet": snippet(text, match.start(), match.end()),
                    }
                )
    return result


def merge_mp_result(total: dict, item: dict) -> None:
    total["docs"] += item["docs"]
    total["chars"] += item["chars"]
    for name, bucket in item["patterns"].items():
        out = total["patterns"][name]
        out["docs"] += bucket["docs"]
        slots = max(0, 8 - len(out["examples"]))
        if slots:
            out["examples"].extend(bucket["examples"][:slots])


def write_mp_report(total: dict, output_md: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.with_suffix(".json").write_text(
        json.dumps(total, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Wave-2 Multiprocess Corpus Noise Scan",
        "",
        f"Docs scanned: {total['docs']:,}",
        f"Chars scanned: {total['chars']:,}",
        "",
        "| pattern | docs | doc % |",
        "| --- | ---: | ---: |",
    ]
    denom = max(1, total["docs"])
    for name, item in sorted(total["patterns"].items(), key=lambda kv: (-kv[1]["docs"], kv[0])):
        pct = item["docs"] * 100.0 / denom
        lines.append(f"| `{name}` | {item['docs']:,} | {pct:.2f}% |")
    lines.append("")
    for name, item in sorted(total["patterns"].items(), key=lambda kv: (-kv[1]["docs"], kv[0])):
        if not item["examples"]:
            continue
        lines.append(f"## {name}")
        lines.append("")
        for ex in item["examples"]:
            lines.append(
                f"- `{ex['match']}` doc_index={ex['doc_index']}: {ex['snippet']}"
            )
        lines.append("")
    output_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output_md}")
    print(f"wrote {output_md.with_suffix('.json')}")


def mp_corpus_scan(
    parquet_paths: list[Path],
    output_md: Path,
    workers: int,
    batch_size: int,
    task_docs: int,
    max_inflight: int,
    max_docs: int | None,
) -> None:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise SystemExit("pyarrow is required for corpus scanning") from exc

    total = {
        "docs": 0,
        "chars": 0,
        "patterns": {
            name: {"docs": 0, "examples": []}
            for name in MP_PATTERNS
        },
        "config": {
            "workers": workers,
            "batch_size": batch_size,
            "task_docs": task_docs,
            "max_inflight": max_inflight,
            "max_docs": max_docs,
            "parquet_paths": [str(p) for p in parquet_paths],
        },
    }

    doc_seen = 0
    submitted = 0
    completed = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures: set[concurrent.futures.Future] = set()

        def drain_one() -> None:
            nonlocal completed
            done, _pending = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for fut in done:
                futures.remove(fut)
                merge_mp_result(total, fut.result())
                completed += 1
            if completed and completed % 100 == 0:
                print(
                    f"completed_tasks={completed:,} submitted={submitted:,} "
                    f"docs={total['docs']:,} chars={total['chars']:,}",
                    file=sys.stderr,
                    flush=True,
                )

        for parquet_path in parquet_paths:
            pf = pq.ParquetFile(parquet_path)
            for batch in pf.iter_batches(columns=["text"], batch_size=batch_size, use_threads=True):
                texts = batch.column(0).to_pylist()
                for start in range(0, len(texts), task_docs):
                    chunk = texts[start:start + task_docs]
                    if max_docs is not None:
                        remaining = max_docs - doc_seen
                        if remaining <= 0:
                            break
                        chunk = chunk[:remaining]
                    if not chunk:
                        continue
                    while len(futures) >= max_inflight:
                        drain_one()
                    futures.add(pool.submit(scan_text_chunk, (doc_seen + 1, str(parquet_path), chunk)))
                    submitted += 1
                    doc_seen += len(chunk)
                if max_docs is not None and doc_seen >= max_docs:
                    break
            if max_docs is not None and doc_seen >= max_docs:
                break

        while futures:
            drain_one()

    write_mp_report(total, output_md)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    vocab = sub.add_parser("vocab")
    vocab.add_argument("--tokenizer-dir", action="append", type=Path, required=True)
    vocab.add_argument("--out-dir", type=Path, required=True)

    corpus = sub.add_parser("corpus")
    corpus.add_argument("--parquet", action="append", type=Path, required=True)
    corpus.add_argument("--output-md", type=Path, required=True)
    corpus.add_argument("--max-docs", type=int)
    corpus.add_argument("--batch-size", type=int, default=256)

    counts = sub.add_parser("duckdb-counts")
    counts.add_argument("--parquet", action="append", type=Path, required=True)
    counts.add_argument("--output-json", type=Path, required=True)
    counts.add_argument("--threads", type=int, default=8)

    mp = sub.add_parser("mp-corpus")
    mp.add_argument("--parquet", action="append", type=Path, required=True)
    mp.add_argument("--output-md", type=Path, required=True)
    mp.add_argument("--workers", type=int, default=48)
    mp.add_argument("--batch-size", type=int, default=512)
    mp.add_argument("--task-docs", type=int, default=16)
    mp.add_argument("--max-inflight", type=int, default=192)
    mp.add_argument("--max-docs", type=int)

    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cmd == "vocab":
        write_vocab_report(args.tokenizer_dir, args.out_dir)
    elif args.cmd == "corpus":
        scan_corpus(args.parquet, args.output_md, args.max_docs, args.batch_size)
    elif args.cmd == "duckdb-counts":
        duckdb_doc_counts(args.parquet, args.output_json, args.threads)
    elif args.cmd == "mp-corpus":
        mp_corpus_scan(
            args.parquet,
            args.output_md,
            args.workers,
            args.batch_size,
            args.task_docs,
            args.max_inflight,
            args.max_docs,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
