#!/usr/bin/env python3
"""Compare the local polytonic Greek corpus with FineWeb-2 grc_Grek."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow.parquet as pq
import regex


WORD_RE = regex.compile(r"\p{L}[\p{L}\p{M}\p{N}'’·-]*")
GREEK_WORD_RE = regex.compile(r"(?:\p{Script=Greek}|\p{M})+")
GREEK_CHAR_RE = regex.compile(r"\p{Script=Greek}")
DISTINCTIVE_POLYTONIC_MARKS = {"\u0300", "\u0313", "\u0314", "\u0342", "\u0345"}


def _polytonic_char_pattern() -> regex.Pattern[str]:
    chars: list[str] = []
    for start, end in [(0x0370, 0x03FF), (0x1F00, 0x1FFF)]:
        for codepoint in range(start, end + 1):
            ch = chr(codepoint)
            if any(mark in DISTINCTIVE_POLYTONIC_MARKS for mark in unicodedata.normalize("NFD", ch)):
                chars.append(regex.escape(ch))
    chars.extend(regex.escape(ch) for ch in sorted(DISTINCTIVE_POLYTONIC_MARKS))
    return regex.compile("[" + "".join(chars) + "]")


POLYTONIC_CHAR_RE = _polytonic_char_pattern()


def normalized_text_hash(text: str) -> str:
    norm = " ".join(text.split()).casefold()
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def has_distinctive_polytonic_signal(s: str) -> bool:
    return any(ch in DISTINCTIVE_POLYTONIC_MARKS for ch in unicodedata.normalize("NFD", s))


def greek_char_counts(text: str) -> tuple[int, int]:
    greek_chars = len(GREEK_CHAR_RE.findall(text))
    distinctive = len(POLYTONIC_CHAR_RE.findall(text))
    if greek_chars:
        distinctive = min(distinctive, greek_chars)
    return greek_chars, distinctive


def text_metrics(text: str) -> dict[str, int | float]:
    words = WORD_RE.findall(text)
    greek_words = GREEK_WORD_RE.findall(text)
    poly_words = sum(1 for w in greek_words if has_distinctive_polytonic_signal(w))
    greek_chars, poly_chars = greek_char_counts(text)
    return {
        "chars": len(text),
        "utf8_bytes": len(text.encode("utf-8")),
        "unicode_words": len(words),
        "greek_words": len(greek_words),
        "polytonic_greek_words": poly_words,
        "greek_chars": greek_chars,
        "polytonic_greek_chars": poly_chars,
        "polytonic_word_ratio": poly_words / len(greek_words) if greek_words else 0.0,
        "polytonic_char_ratio": poly_chars / greek_chars if greek_chars else 0.0,
    }


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    values = sorted(values)

    def q(p: float) -> float:
        idx = min(len(values) - 1, max(0, round((len(values) - 1) * p)))
        return values[idx]

    return {
        "min": values[0],
        "p10": q(0.10),
        "p25": q(0.25),
        "p50": q(0.50),
        "p75": q(0.75),
        "p90": q(0.90),
        "p95": q(0.95),
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def host_from_url(url: str | None) -> str | None:
    if not url:
        return None
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def safe_json_loads(value: Any) -> Any:
    if not value or not isinstance(value, str):
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def collect_urls(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and ("url" in key.lower() or value.startswith(("http://", "https://"))):
                if value.startswith(("http://", "https://")):
                    out.append(value)
            else:
                collect_urls(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_urls(item, out)


def summarize_parquet(
    path: Path,
    kind: str,
    fineweb_hashes: dict[str, dict[str, str]] | None = None,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    pf = pq.ParquetFile(path)
    columns = set(pf.schema_arrow.names)
    wanted = ["text"]
    for col in ["source_dataset", "source_doc_id", "source_metadata_json", "id", "url", "file_path"]:
        if col in columns:
            wanted.append(col)

    stats: dict[str, Any] = {
        "path": str(path),
        "parquet_bytes": path.stat().st_size,
        "rows": 0,
        "chars": 0,
        "utf8_bytes": 0,
        "unicode_words": 0,
        "greek_words": 0,
        "polytonic_greek_words": 0,
        "greek_chars": 0,
        "polytonic_greek_chars": 0,
        "docs_passing_w050_c010": 0,
        "docs_with_no_distinctive_polytonic_words": 0,
        "source_counts": Counter(),
        "domain_counts": Counter(),
        "text_char_quantiles": [],
        "unicode_word_quantiles": [],
        "greek_word_quantiles": [],
        "polytonic_word_ratio_quantiles": [],
        "polytonic_char_ratio_quantiles": [],
        "exact_normalized_matches_with_fineweb": 0,
        "exact_normalized_match_samples": [],
    }
    hashes: dict[str, dict[str, str]] = {}

    for batch in pf.iter_batches(columns=wanted, batch_size=512):
        data = batch.to_pydict()
        rows = len(data["text"])
        for i in range(rows):
            text = data["text"][i] or ""
            metrics = text_metrics(text)
            stats["rows"] += 1
            for key in [
                "chars",
                "utf8_bytes",
                "unicode_words",
                "greek_words",
                "polytonic_greek_words",
                "greek_chars",
                "polytonic_greek_chars",
            ]:
                stats[key] += metrics[key]
            if metrics["polytonic_word_ratio"] >= 0.50 and metrics["polytonic_char_ratio"] >= 0.10:
                stats["docs_passing_w050_c010"] += 1
            if metrics["polytonic_greek_words"] == 0:
                stats["docs_with_no_distinctive_polytonic_words"] += 1

            stats["text_char_quantiles"].append(float(metrics["chars"]))
            stats["unicode_word_quantiles"].append(float(metrics["unicode_words"]))
            stats["greek_word_quantiles"].append(float(metrics["greek_words"]))
            stats["polytonic_word_ratio_quantiles"].append(float(metrics["polytonic_word_ratio"]))
            stats["polytonic_char_ratio_quantiles"].append(float(metrics["polytonic_char_ratio"]))

            if kind == "ours":
                source = data.get("source_dataset", [None] * rows)[i] or "unknown"
                stats["source_counts"][source] += 1
                meta = safe_json_loads(data.get("source_metadata_json", [None] * rows)[i])
                urls: list[str] = []
                collect_urls(meta, urls)
                for url in urls:
                    host = host_from_url(url)
                    if host:
                        stats["domain_counts"][host] += 1
            else:
                url = data.get("url", [None] * rows)[i]
                host = host_from_url(url)
                if host:
                    stats["domain_counts"][host] += 1

            h = normalized_text_hash(text)
            if kind == "fineweb":
                hashes[h] = {
                    "id": str(data.get("id", [""] * rows)[i] or ""),
                    "url": str(data.get("url", [""] * rows)[i] or ""),
                }
            elif fineweb_hashes and h in fineweb_hashes:
                stats["exact_normalized_matches_with_fineweb"] += 1
                if len(stats["exact_normalized_match_samples"]) < 10:
                    stats["exact_normalized_match_samples"].append(
                        {
                            "source_dataset": str(data.get("source_dataset", [""] * rows)[i] or ""),
                            "source_doc_id": str(data.get("source_doc_id", [""] * rows)[i] or ""),
                            "fineweb": fineweb_hashes[h],
                        }
                    )

    for key in [
        "text_char_quantiles",
        "unicode_word_quantiles",
        "greek_word_quantiles",
        "polytonic_word_ratio_quantiles",
        "polytonic_char_ratio_quantiles",
    ]:
        stats[key] = quantiles(stats[key])

    for ratio_key, num, den in [
        ("polytonic_word_ratio", "polytonic_greek_words", "greek_words"),
        ("polytonic_char_ratio", "polytonic_greek_chars", "greek_chars"),
    ]:
        stats[ratio_key] = stats[num] / stats[den] if stats[den] else math.nan

    stats["source_counts"] = dict(stats["source_counts"].most_common())
    stats["top_domains"] = dict(stats["domain_counts"].most_common(25))
    del stats["domain_counts"]
    return stats, hashes


def write_markdown(out: Path, result: dict[str, Any]) -> None:
    ours = result["ours"]
    fine = result["fineweb_main_grc_Grek"]
    published = result["fineweb_published_main_grc_Grek"]
    lines = [
        "# FineWeb-2 Ancient Greek Comparison",
        "",
        "## Version Note",
        "",
        "The `28,539` documents / `33,850,484` words / `340.80MB` figure matches the",
        "current FineWeb-2 language-distribution CSV for `grc_Grek` train, not the",
        "older `v2.0.1` README table, which lists `10,500` documents and",
        "`9,397,616` words for `grc_Grek`.",
        "",
        "## Scale",
        "",
        "| Corpus | Docs | Unicode words | Greek words | UTF-8 bytes | Parquet bytes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        f"| FineWeb-2 `grc_Grek` published | {published['documents']:,} | {published['words']:,} | n/a | {published['utf8_bytes']:,} | {published['parquet_bytes']:,} |",
        f"| FineWeb-2 `grc_Grek` local count | {fine['rows']:,} | {fine['unicode_words']:,} | {fine['greek_words']:,} | {fine['utf8_bytes']:,} | {fine['parquet_bytes']:,} |",
        f"| Our kept corpus | {ours['rows']:,} | {ours['unicode_words']:,} | {ours['greek_words']:,} | {ours['utf8_bytes']:,} | {ours['parquet_bytes']:,} |",
        "",
        "## Strict Polytonic Signal",
        "",
        "| Corpus | Docs passing w>=0.50/c>=0.10 | Docs with no distinctive polytonic words | Global word ratio | Global char ratio |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| FineWeb-2 `grc_Grek` | {fine['docs_passing_w050_c010']:,} | {fine['docs_with_no_distinctive_polytonic_words']:,} | {fine['polytonic_word_ratio']:.3f} | {fine['polytonic_char_ratio']:.3f} |",
        f"| Our kept corpus | {ours['docs_passing_w050_c010']:,} | {ours['docs_with_no_distinctive_polytonic_words']:,} | {ours['polytonic_word_ratio']:.3f} | {ours['polytonic_char_ratio']:.3f} |",
        "",
        "## Exact Normalized Overlap",
        "",
        f"Our corpus has `{ours['exact_normalized_matches_with_fineweb']}` full-document",
        "matches against FineWeb-2 after case-folding and whitespace normalization.",
        "",
        "## Our Rows By Source",
        "",
    ]
    for source, count in ours["source_counts"].items():
        lines.append(f"- `{source}`: {count:,}")
    lines.extend(
        [
            "",
            "## FineWeb Top Domains",
            "",
        ]
    )
    for domain, count in fine["top_domains"].items():
        lines.append(f"- `{domain}`: {count:,}")
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--fineweb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fine_stats, fine_hashes = summarize_parquet(args.fineweb, "fineweb")
    our_stats, _ = summarize_parquet(args.ours, "ours", fine_hashes)
    result = {
        "fineweb_published_main_grc_Grek": {
            "documents": 28539,
            "words": 33850484,
            "utf8_bytes": 357352609,
            "parquet_bytes": 110536477,
            "source": "https://raw.githubusercontent.com/huggingface/fineweb-2/main/fineweb2-language-distribution.csv",
        },
        "fineweb_v2_0_1_readme_grc_Grek": {
            "documents": 10500,
            "words": 9397616,
            "disk_size_human": "30.04MB",
            "source": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/raw/v2.0.1/README.md",
        },
        "fineweb_main_grc_Grek": fine_stats,
        "ours": our_stats,
    }
    (args.out_dir / "fineweb2_comparison_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(args.out_dir / "fineweb2_comparison_report.md", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
