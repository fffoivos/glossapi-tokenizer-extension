#!/usr/bin/env python3
"""Triage false-negative control rows for manual HPLT cleaning review.

The input pack contains rows the current policy would keep. This script does
not clean or rewrite anything; it ranks rows by simple residual-artifact signals
so reviewers can focus on likely missed trim/split/drop cases while still
preserving the random-control frame.
"""

from __future__ import print_function

import argparse
import collections
import csv
import hashlib
import json
import os
import re


GREEK_RE = re.compile(r"[Α-Ωα-ωΆΈΉΊΌΎΏάέήίόύώΪΫϊϋΐΰ]{2,}")
WORD_RE = re.compile(r"\w+", re.U)
URL_RE = re.compile(r"https?://|www\.", re.I)
EMAIL_RE = re.compile(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{2,}\\b")
COMMENT_RE = re.compile(
    r"leave a reply|leave a comment|δημοσίευση σχολίου|δεν υπάρχουν σχόλια|"
    r"σχολιασμός|πείτε τη γνώμη σας|σχολιάστε|disqus|σχόλια",
    re.I,
)
RELATED_RE = re.compile(
    r"διαβάστε επίσης|δείτε επίσης|τα πιο δημοφιλή|διαβάζονται πάντα|"
    r"σχετικά άρθρα|related|read more|περισσότερα",
    re.I,
)
ARCHIVE_URL_RE = re.compile(r"/tag/|/category/|/search/label/|archive|itemlist/date|limit=|start=|page/[0-9]+", re.I)
PRICE_RE = re.compile(r"(€|\\bτιμή\\b|αρχική αξία|έκπτωση|καλάθι|προσφορά|αγορές)", re.I)
PRODUCT_URL_RE = re.compile(r"/product/|/proion/|/shop/|itemid=|option=com_", re.I)
FEED_META_RE = re.compile(r"tag:blogger\\.com|noreply@blogger\\.com|blogger[0-9]{4}|atom|rss", re.I)
HTML_RE = re.compile(r"</?\\w+|&[a-zA-Z]{2,12};|&#\\d+;|#x[0-9a-f]+", re.I)
MULTIDOC_RE = re.compile(r"(δευτέρα|τρίτη|τετάρτη|πέμπτη|παρασκευή|σάββατο|κυριακή),?\\s+\\d{1,2}\\s+[α-ω]+\\s+\\d{4}|\\b\\d{1,2}/\\d{1,2}/\\d{2,4}\\b", re.I)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as exc:
                raise RuntimeError("Invalid JSON at %s:%s: %s" % (path, line_number, exc))


def read_doc(path):
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def read_pack_markdown(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        text = handle.read()
    records = {}
    for match in re.finditer(r"^### (pe_[^\n]+)\n", text, flags=re.M):
        start = match.start()
        next_match = re.search(r"^### pe_", text[match.end():], flags=re.M)
        section = text[start:match.end() + next_match.start()] if next_match else text[start:]
        doc_match = re.search(r"Document text:\n\n```text\n(.*?)\n```", section, flags=re.S)
        if doc_match:
            records[match.group(1)] = doc_match.group(1)
    return records


def stable_key(value):
    return hashlib.sha1(value.encode("utf-8", "replace")).hexdigest()[:10]


def ratio(num, den):
    return float(num) / float(den or 1)


def text_features(text, url):
    words = WORD_RE.findall(text)
    greek_words = GREEK_RE.findall(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_lines = sum(1 for line in lines if len(line) <= 45)
    list_lines = sum(1 for line in lines if line.startswith("-") or re.match(r"^\\d+[.)]", line))
    url_hits = len(URL_RE.findall(text))
    email_hits = len(EMAIL_RE.findall(text))
    comment_hits = len(COMMENT_RE.findall(text))
    related_hits = len(RELATED_RE.findall(text))
    archive_url = 1 if ARCHIVE_URL_RE.search(url or "") else 0
    product_url = 1 if PRODUCT_URL_RE.search(url or "") else 0
    price_hits = len(PRICE_RE.findall(text))
    feed_hits = len(FEED_META_RE.findall(text))
    html_hits = len(HTML_RE.findall(text))
    date_hits = len(MULTIDOC_RE.findall(text))
    greek_share = ratio(len(greek_words), len(words))
    short_density = ratio(short_lines, len(lines))
    list_density = ratio(list_lines, len(lines))
    reasons = []
    score = 0.0
    if archive_url:
        score += 0.25
        reasons.append("archive_url")
    if product_url:
        score += 0.18
        reasons.append("product_url")
    if comment_hits:
        score += min(0.24, 0.08 * comment_hits)
        reasons.append("comment_chrome")
    if related_hits:
        score += min(0.20, 0.07 * related_hits)
        reasons.append("related_chrome")
    if url_hits >= 2 or email_hits:
        score += min(0.18, 0.04 * (url_hits + email_hits))
        reasons.append("url_email_residue")
    if price_hits >= 2:
        score += min(0.20, 0.05 * price_hits)
        reasons.append("product_or_price_list")
    if feed_hits:
        score += 0.35
        reasons.append("feed_metadata")
    if html_hits:
        score += min(0.22, 0.06 * html_hits)
        reasons.append("html_entity_residue")
    if date_hits >= 3:
        score += min(0.22, 0.04 * date_hits)
        reasons.append("many_date_boundaries")
    if len(words) < 120 and (comment_hits or related_hits or archive_url):
        score += 0.12
        reasons.append("short_with_chrome")
    if short_density >= 0.55 and len(lines) >= 8:
        score += 0.12
        reasons.append("short_line_dense")
    if list_density >= 0.30 and len(lines) >= 8:
        score += 0.12
        reasons.append("list_dense")
    if greek_share < 0.65 and len(words) >= 120:
        score += 0.18
        reasons.append("low_greek_share")
    return {
        "triage_score": round(min(score, 1.0), 4),
        "triage_reasons": reasons,
        "word_count": len(words),
        "greek_word_count": len(greek_words),
        "greek_word_share": round(greek_share, 4),
        "line_count": len(lines),
        "short_line_density": round(short_density, 4),
        "list_line_density": round(list_density, 4),
        "url_hits": url_hits,
        "email_hits": email_hits,
        "comment_hits": comment_hits,
        "related_hits": related_hits,
        "price_hits": price_hits,
        "feed_hits": feed_hits,
        "html_hits": html_hits,
        "date_boundary_hits": date_hits,
    }


def make_excerpt(text, max_chars):
    text = text.replace("\r\n", "\n")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n...[SNIP]...\n" + text[-half:]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path, rows):
    fields = [
        "false_negative_triage_id",
        "policy_evaluation_sample_id",
        "host",
        "quality_bin",
        "chars_before",
        "triage_score",
        "triage_reasons",
        "url",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ",".join(row[key]) if isinstance(row.get(key), list) else row.get(key) for key in fields})


def write_md(path, rows, args):
    with open(path, "w", encoding="utf-8") as out:
        out.write("# False-Negative Control Triage\n\n")
        out.write("This is a non-destructive triage of candidate `keep` rows. It does not clean source rows; it ranks likely missed trim/split/drop cases for manual review.\n\n")
        out.write("- input: `%s`\n" % args.false_negative_pack)
        out.write("- rows: `%d`\n" % len(rows))
        out.write("- high-risk threshold: `%s`\n\n" % args.high_risk_threshold)
        out.write("## Triage Counts\n\n")
        buckets = collections.Counter("high" if r["triage_score"] >= args.high_risk_threshold else "low" for r in rows)
        for key, count in sorted(buckets.items()):
            out.write("- `%s`: `%d`\n" % (key, count))
        out.write("\n## Reason Counts\n\n| Reason | Rows |\n| --- | ---: |\n")
        reasons = collections.Counter(reason for row in rows for reason in row["triage_reasons"])
        for reason, count in reasons.most_common():
            out.write("| `%s` | %d |\n" % (reason, count))
        out.write("\n## Top Rows\n\n")
        for row in sorted(rows, key=lambda r: (-r["triage_score"], str(r.get("host")), str(r.get("policy_evaluation_sample_id"))))[: args.markdown_rows]:
            out.write("### %s\n\n" % row["false_negative_triage_id"])
            out.write("- policy_evaluation_sample_id: `%s`\n" % row.get("policy_evaluation_sample_id"))
            out.write("- source_doc_id: `%s`\n" % row.get("source_doc_id"))
            out.write("- host: `%s`\n" % row.get("host"))
            out.write("- url: `%s`\n" % row.get("url"))
            out.write("- qbin: `%s`\n" % row.get("quality_bin"))
            out.write("- chars_before: `%s`\n" % row.get("chars_before"))
            out.write("- triage_score: `%s`\n" % row.get("triage_score"))
            out.write("- triage_reasons: `%s`\n" % "`, `".join(row.get("triage_reasons") or []))
            out.write("- doc_text_path: `%s`\n\n" % row.get("doc_text_path"))
            out.write("Excerpt:\n\n```text\n%s\n```\n\n" % row.get("review_excerpt", ""))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--false-negative-pack", required=True)
    parser.add_argument("--review-pack-md", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--high-risk-threshold", type=float, default=0.35)
    parser.add_argument("--excerpt-chars", type=int, default=3000)
    parser.add_argument("--markdown-rows", type=int, default=80)
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    pack_docs = read_pack_markdown(args.review_pack_md)
    rows = []
    for index, row in enumerate(read_jsonl(args.false_negative_pack), 1):
        text = pack_docs.get(row.get("policy_evaluation_sample_id")) or read_doc(row.get("doc_text_path"))
        features = text_features(text, row.get("url") or "")
        out = dict(row)
        out.update(features)
        out["false_negative_triage_id"] = "fn_triage_%s_%03d_%s" % (
            args.timestamp,
            index,
            stable_key(row.get("policy_evaluation_sample_id") or row.get("source_doc_id") or str(index)),
        )
        out["review_excerpt"] = make_excerpt(text, args.excerpt_chars)
        rows.append(out)
    rows = sorted(rows, key=lambda r: (-r["triage_score"], str(r.get("host")), str(r.get("policy_evaluation_sample_id"))))

    base = "false_negative_control_triage_%s" % args.timestamp
    jsonl_path = os.path.join(args.output_dir, base + ".jsonl")
    csv_path = os.path.join(args.output_dir, base + ".csv")
    md_path = os.path.join(args.output_dir, base + ".md")
    summary_path = os.path.join(args.output_dir, base + "_summary.json")
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)
    write_md(md_path, rows, args)
    summary = {
        "input": args.false_negative_pack,
        "review_pack_md": args.review_pack_md,
        "docs_from_markdown": sum(1 for row in rows if row.get("review_excerpt")),
        "rows": len(rows),
        "jsonl": jsonl_path,
        "csv": csv_path,
        "markdown": md_path,
        "high_risk_threshold": args.high_risk_threshold,
        "high_risk_rows": sum(1 for row in rows if row["triage_score"] >= args.high_risk_threshold),
        "reason_counts": dict(collections.Counter(reason for row in rows for reason in row["triage_reasons"]).most_common()),
        "host_top": dict(collections.Counter(row.get("host") for row in rows).most_common(20)),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
