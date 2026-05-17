#!/usr/bin/env python3
"""Cross-corpus overlap checks against FineWeb-2 Ancient Greek.

This script intentionally separates symmetric document-level dedup from
directional fragment containment. FineWeb web documents are often much
shorter than the curated texts in this subproject, so full-doc Jaccard
or SimHash can miss the important case: "FineWeb document is contained
inside one of our longer documents".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

import pyarrow.parquet as pq
import regex


WORD_RE = regex.compile(r"\p{L}[\p{L}\p{M}\p{N}'’·-]*")
WS_RE = re.compile(r"\s+")
GREEK_RE = regex.compile(r"\p{Script=Greek}")


@dataclass(frozen=True)
class Doc:
    corpus: str
    doc_id: str
    source: str
    title: str
    author: str
    url: str
    domain: str
    text: str


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = WS_RE.sub(" ", text).strip().casefold()
    return text


def text_hash(text: str) -> str:
    return hashlib.blake2b(normalize_text(text).encode("utf-8"), digest_size=16).hexdigest()


def token_hash(token: str) -> int:
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), "little")


def shingle_hash(tokens: list[str], start: int, n: int) -> int:
    h = hashlib.blake2b(digest_size=8)
    for tok in tokens[start : start + n]:
        h.update(tok.encode("utf-8"))
        h.update(b"\0")
    return int.from_bytes(h.digest(), "little")


def tokenize(text: str) -> list[str]:
    return [m.group(0).casefold() for m in WORD_RE.finditer(unicodedata.normalize("NFC", text or ""))]


def simhash(tokens: Iterable[str]) -> int:
    weights = [0] * 64
    for token in tokens:
        h = token_hash(token)
        for bit in range(64):
            if (h >> bit) & 1:
                weights[bit] += 1
            else:
                weights[bit] -= 1
    out = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            out |= 1 << bit
    return out


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def host(url: str) -> str:
    if not url:
        return ""
    h = urlparse(url).netloc.lower()
    return h[4:] if h.startswith("www.") else h


def canonical_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = unquote(parsed.path or "/").rstrip("/")
    return f"{netloc}{path}".casefold()


def safe_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def collect_urls(obj: Any, out: list[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str):
                if value.startswith(("http://", "https://")):
                    out.append(value)
                elif "url" in key.lower() and value.startswith(("http://", "https://")):
                    out.append(value)
            else:
                collect_urls(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_urls(item, out)


def load_docs(path: Path, corpus: str) -> list[Doc]:
    pf = pq.ParquetFile(path)
    columns = set(pf.schema_arrow.names)
    wanted = ["text"]
    for col in [
        "doc_key",
        "source_dataset",
        "source_doc_id",
        "title",
        "author",
        "source_metadata_json",
        "id",
        "url",
    ]:
        if col in columns:
            wanted.append(col)

    docs: list[Doc] = []
    for batch in pf.iter_batches(columns=wanted, batch_size=512):
        data = batch.to_pydict()
        for i, text in enumerate(data["text"]):
            if corpus == "fineweb":
                url = data.get("url", [""] * len(data["text"]))[i] or ""
                doc_id = data.get("id", [""] * len(data["text"]))[i] or f"fineweb_{len(docs):06d}"
                docs.append(
                    Doc(
                        corpus=corpus,
                        doc_id=str(doc_id),
                        source="fineweb2_grc_Grek",
                        title="",
                        author="",
                        url=url,
                        domain=host(url),
                        text=text or "",
                    )
                )
            else:
                meta = safe_json((data.get("source_metadata_json") or [None] * len(data["text"]))[i])
                urls: list[str] = []
                collect_urls(meta, urls)
                url = urls[0] if urls else ""
                source = data.get("source_dataset", [""] * len(data["text"]))[i] or ""
                source_doc_id = data.get("source_doc_id", [""] * len(data["text"]))[i] or ""
                docs.append(
                    Doc(
                        corpus=corpus,
                        doc_id=str(source_doc_id),
                        source=str(source),
                        title=str((data.get("title") or [""] * len(data["text"]))[i] or ""),
                        author=str((data.get("author") or [""] * len(data["text"]))[i] or ""),
                        url=url,
                        domain=host(url),
                        text=text or "",
                    )
                )
    return docs


def source_bucket_for_domain(domain: str) -> str:
    d = domain.lower()
    if "wikisource.org" in d:
        return "wikisource"
    if "perseus" in d or "perseids.org" in d:
        return "perseus_classics"
    if "goarch.org" in d:
        return "goarch_liturgical"
    if any(s in d for s in ["bible", "bibbia", "sacred-texts", "catholic", "orthodox", "agia-grafi", "textusreceptus"]):
        return "biblical_patristic_church"
    if any(s in d for s in ["digitalathenaeus", "hellas.bab2min", "skuolasprint"]):
        return "classical_other"
    return "other"


def exact_and_simhash(ours: list[Doc], fine: list[Doc]) -> dict[str, Any]:
    fine_hashes: dict[str, list[Doc]] = defaultdict(list)
    for doc in fine:
        fine_hashes[text_hash(doc.text)].append(doc)

    exact = []
    for doc in ours:
        for hit in fine_hashes.get(text_hash(doc.text), []):
            exact.append(
                {
                    "our_source": doc.source,
                    "our_doc_id": doc.doc_id,
                    "fineweb_id": hit.doc_id,
                    "fineweb_url": hit.url,
                }
            )

    # Fast full-document near-dedup smoke test. This is symmetric and
    # intentionally not used as a fragment detector.
    fine_bands: dict[tuple[int, int], list[tuple[Doc, int, int]]] = defaultdict(list)
    for doc in fine:
        tokens = tokenize(doc.text)
        if len(tokens) < 50:
            continue
        sig = simhash(tokens)
        for band in range(4):
            key = (band, (sig >> (band * 16)) & 0xFFFF)
            fine_bands[key].append((doc, sig, len(tokens)))

    near = []
    seen: set[tuple[str, str]] = set()
    for doc in ours:
        tokens = tokenize(doc.text)
        if len(tokens) < 50:
            continue
        sig = simhash(tokens)
        candidates: list[tuple[Doc, int, int]] = []
        for band in range(4):
            candidates.extend(fine_bands.get((band, (sig >> (band * 16)) & 0xFFFF), []))
        for fw, fw_sig, fw_len in candidates:
            key = (doc.doc_id, fw.doc_id)
            if key in seen:
                continue
            seen.add(key)
            dist = hamming(sig, fw_sig)
            length_ratio = min(len(tokens), fw_len) / max(len(tokens), fw_len)
            if dist <= 8 and length_ratio >= 0.70:
                near.append(
                    {
                        "our_source": doc.source,
                        "our_doc_id": doc.doc_id,
                        "our_words": len(tokens),
                        "fineweb_id": fw.doc_id,
                        "fineweb_url": fw.url,
                        "fineweb_words": fw_len,
                        "hamming": dist,
                        "length_ratio": length_ratio,
                    }
                )
    near.sort(key=lambda x: (x["hamming"], -x["length_ratio"]))
    return {"exact_full_doc_matches": exact, "simhash_near_full_doc_matches": near[:200], "simhash_near_full_doc_count": len(near)}


def metadata_analysis(ours: list[Doc], fine: list[Doc]) -> dict[str, Any]:
    fine_urls = defaultdict(list)
    for doc in fine:
        c = canonical_url(doc.url)
        if c:
            fine_urls[c].append(doc)

    exact_url_matches = []
    our_url_counts = Counter()
    for doc in ours:
        c = canonical_url(doc.url)
        if c:
            our_url_counts[c] += 1
        for hit in fine_urls.get(c, []):
            exact_url_matches.append(
                {
                    "our_source": doc.source,
                    "our_doc_id": doc.doc_id,
                    "our_url": doc.url,
                    "fineweb_id": hit.doc_id,
                    "fineweb_url": hit.url,
                }
            )

    fine_domains = Counter(doc.domain for doc in fine if doc.domain)
    fine_buckets = Counter(source_bucket_for_domain(doc.domain) for doc in fine if doc.domain)
    our_sources = Counter(doc.source for doc in ours)
    our_domains = Counter(doc.domain for doc in ours if doc.domain)
    return {
        "exact_url_match_count": len(exact_url_matches),
        "exact_url_match_samples": exact_url_matches[:50],
        "our_docs_with_metadata_url": sum(1 for doc in ours if doc.url),
        "our_source_counts": dict(our_sources.most_common()),
        "our_metadata_domains": dict(our_domains.most_common(50)),
        "fineweb_top_domains": dict(fine_domains.most_common(80)),
        "fineweb_domain_buckets": dict(fine_buckets.most_common()),
    }


def directional_fragment_map(
    ours: list[Doc],
    fine: list[Doc],
    *,
    shingle_size: int,
    stride: int,
    min_fine_tokens: int,
    min_anchor_hits: int,
) -> dict[str, Any]:
    # Index sampled shingles from our larger documents.
    index: dict[int, list[int]] = defaultdict(list)
    our_token_counts: list[int] = []
    for idx, doc in enumerate(ours):
        tokens = tokenize(doc.text)
        our_token_counts.append(len(tokens))
        if len(tokens) < shingle_size:
            continue
        seen_for_doc: set[int] = set()
        for start in range(0, len(tokens) - shingle_size + 1, stride):
            h = shingle_hash(tokens, start, shingle_size)
            if h not in seen_for_doc:
                seen_for_doc.add(h)
                index[h].append(idx)

    # Remove very common sampled shingles; they are formulaic and noisy.
    common_cutoff = 25
    noisy = {h for h, docs in index.items() if len(docs) > common_cutoff}
    for h in noisy:
        del index[h]

    matches = []
    matched_fine = 0
    for fw in fine:
        fw_tokens = tokenize(fw.text)
        if len(fw_tokens) < min_fine_tokens:
            continue
        anchors: list[int] = []
        for start in range(0, len(fw_tokens) - shingle_size + 1, stride):
            h = shingle_hash(fw_tokens, start, shingle_size)
            if h in index:
                anchors.append(h)
        if not anchors:
            continue
        counts: Counter[int] = Counter()
        for h in set(anchors):
            for our_idx in index.get(h, []):
                counts[our_idx] += 1
        if not counts:
            continue
        best_idx, best_hits = counts.most_common(1)[0]
        if best_hits < min_anchor_hits:
            continue
        sampled_anchor_count = max(1, (len(fw_tokens) - shingle_size) // stride + 1)
        containment_estimate = best_hits / sampled_anchor_count
        if containment_estimate >= 0.25:
            matched_fine += 1
            our = ours[best_idx]
            matches.append(
                {
                    "fineweb_id": fw.doc_id,
                    "fineweb_url": fw.url,
                    "fineweb_domain": fw.domain,
                    "fineweb_words": len(fw_tokens),
                    "our_source": our.source,
                    "our_doc_id": our.doc_id,
                    "our_title": our.title,
                    "our_author": our.author,
                    "our_words": our_token_counts[best_idx],
                    "sampled_anchor_hits": best_hits,
                    "sampled_anchor_count": sampled_anchor_count,
                    "containment_estimate": containment_estimate,
                }
            )

    matches.sort(key=lambda x: (-x["containment_estimate"], -x["sampled_anchor_hits"]))
    by_domain = Counter(m["fineweb_domain"] for m in matches)
    by_our_source = Counter(m["our_source"] for m in matches)
    return {
        "params": {
            "shingle_size": shingle_size,
            "stride": stride,
            "min_fine_tokens": min_fine_tokens,
            "min_anchor_hits": min_anchor_hits,
            "noisy_anchor_doc_frequency_cutoff": common_cutoff,
            "indexed_anchor_count": len(index),
            "removed_noisy_anchor_count": len(noisy),
        },
        "matched_fineweb_docs": matched_fine,
        "match_count": len(matches),
        "by_fineweb_domain": dict(by_domain.most_common(50)),
        "by_our_source": dict(by_our_source.most_common()),
        "top_matches": matches[:200],
    }


def write_report(path: Path, result: dict[str, Any]) -> None:
    meta = result["metadata"]
    dedup = result["dedup"]
    frag = result["directional_fragment_map"]
    lines = [
        "# FineWeb-2 Overlap Analysis",
        "",
        "## Step 1: Cross-Dedup",
        "",
        f"- Exact normalized full-document matches: `{len(dedup['exact_full_doc_matches'])}`",
        f"- Fast SimHash near full-document matches: `{dedup['simhash_near_full_doc_count']}`",
        "",
        "This symmetric pass is only meant to catch same-sized or nearly same-sized",
        "documents. It does not answer whether a FineWeb fragment is contained in",
        "one of our longer curated documents.",
        "",
        "## Step 2: Metadata Matches",
        "",
        f"- Our docs with URL-like metadata: `{meta['our_docs_with_metadata_url']}`",
        f"- Exact canonical URL matches: `{meta['exact_url_match_count']}`",
        "",
        "## Step 3: Source Coverage",
        "",
        "Our source counts:",
        "",
    ]
    for source, count in meta["our_source_counts"].items():
        lines.append(f"- `{source}`: {count:,}")
    lines.extend(["", "FineWeb domain buckets:", ""])
    for bucket, count in meta["fineweb_domain_buckets"].items():
        lines.append(f"- `{bucket}`: {count:,}")
    lines.extend(["", "FineWeb top domains:", ""])
    for domain, count in list(meta["fineweb_top_domains"].items())[:25]:
        lines.append(f"- `{domain}`: {count:,}")
    lines.extend(
        [
            "",
            "## Step 4: Directional Fragment Mapping",
            "",
            f"- FineWeb docs matched as likely fragments of our docs: `{frag['matched_fineweb_docs']}`",
            f"- Match rows emitted: `{frag['match_count']}`",
            f"- Indexed sampled anchors: `{frag['params']['indexed_anchor_count']:,}`",
            f"- Removed noisy anchors: `{frag['params']['removed_noisy_anchor_count']:,}`",
            "",
            "Matches by our source:",
            "",
        ]
    )
    for source, count in frag["by_our_source"].items():
        lines.append(f"- `{source}`: {count:,}")
    lines.extend(["", "Matches by FineWeb domain:", ""])
    for domain, count in list(frag["by_fineweb_domain"].items())[:25]:
        lines.append(f"- `{domain}`: {count:,}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", type=Path, required=True)
    parser.add_argument("--fineweb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--shingle-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--min-fine-tokens", type=int, default=80)
    parser.add_argument("--min-anchor-hits", type=int, default=5)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ours = load_docs(args.ours, "ours")
    fine = load_docs(args.fineweb, "fineweb")
    result = {
        "inputs": {"ours": str(args.ours), "fineweb": str(args.fineweb)},
        "corpus_rows": {"ours": len(ours), "fineweb": len(fine)},
        "dedup": exact_and_simhash(ours, fine),
        "metadata": metadata_analysis(ours, fine),
        "directional_fragment_map": directional_fragment_map(
            ours,
            fine,
            shingle_size=args.shingle_size,
            stride=args.stride,
            min_fine_tokens=args.min_fine_tokens,
            min_anchor_hits=args.min_anchor_hits,
        ),
    }
    (args.out_dir / "fineweb2_overlap_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(args.out_dir / "fineweb2_overlap_report.md", result)
    print(json.dumps(result["corpus_rows"], ensure_ascii=False))
    print(json.dumps(result["dedup"], ensure_ascii=False)[:1000])
    print(json.dumps(result["metadata"], ensure_ascii=False)[:1000])
    print(json.dumps(result["directional_fragment_map"], ensure_ascii=False)[:1000])


if __name__ == "__main__":
    main()
