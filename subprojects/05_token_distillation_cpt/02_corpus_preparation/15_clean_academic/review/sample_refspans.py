#!/usr/bin/env python3
"""Review sampler for the academic reference detector.

Produces inspection artifacts that follow the project's review conventions:
  - FULL document body, no truncation.
  - The detected reference/citation spans are spliced INLINE into the continuous
    passage as <match kind="..." gated="...">…</match> (never before/match/after blocks).
  - STRATIFIED over a chosen per-doc counter: 10 zones across the metric range,
    sampled proportional to population, floor 5/zone, 100–200 docs total.
  - Filenames are prefixed with the zero-padded metric value so `ls` order == metric order.

It samples for AUDIT, not deletion: every span is shown in context (kept and flagged
alike) so the reviewer can set the keep/drop policy from real evidence. A dedicated
`_audit/` subset over-samples the risky gate decisions (kept-non-bib β, footnote
prose-or-hybrid kept, boundaries rejected as TOC/colophon).

Usage (whole-doc source):
  sample_refspans.py --source greek_phd --mode wholedoc \
     --text-input '/…/greek_phd/.../*.jsonl.zst' \
     --spans out/greek_phd/refspans/greek_phd.spans.jsonl \
     --counters out/greek_phd/greek_phd.counters.jsonl \
     --metric footnote_citation_only --out-dir out/greek_phd/review_samples
"""
import argparse
import glob
import io
import json
import os
import subprocess
import sys
from collections import defaultdict

ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}


def stratified_pick(rows, metric, total_target=150, zones=10, floor=5):
    """rows: list of (doc_id, metric_value). Returns chosen doc_ids."""
    vals = [(d, float(v)) for d, v in rows if v is not None]
    if not vals:
        return []
    lo = min(v for _, v in vals)
    hi = max(v for _, v in vals)
    if hi <= lo:
        return [d for d, _ in vals[:total_target]]
    span = (hi - lo) / zones
    buckets = defaultdict(list)
    for d, v in vals:
        z = min(zones - 1, int((v - lo) / span))
        buckets[z].append((d, v))
    n = len(vals)
    chosen = []
    for z in range(zones):
        pop = buckets.get(z, [])
        if not pop:
            continue
        want = max(floor, round(total_target * len(pop) / n))
        want = min(want, len(pop))
        step = max(1, len(pop) // want)
        chosen += [pop[i][0] for i in range(0, len(pop), step)][:want]
    return chosen


def load_counters(path):
    out = []
    for l in open(path, encoding="utf-8"):
        if l.strip():
            out.append(json.loads(l))
    return out


def load_spans(path, wanted_ids):
    by_doc = defaultdict(list)
    for l in open(path, encoding="utf-8"):
        if not l.strip():
            continue
        d = json.loads(l)
        if d["doc_id"] in wanted_ids:
            by_doc[d["doc_id"]].append(d)
    return by_doc


def iter_jsonl_zst(path):
    if path.endswith(".zst"):
        p = subprocess.Popen(["zstd", "-dc", path], stdout=subprocess.PIPE)
        for raw in io.TextIOWrapper(p.stdout, encoding="utf-8"):
            yield raw
        p.wait()
    else:
        for raw in open(path, encoding="utf-8"):
            yield raw


def pick(d, keys):
    for k in keys:
        if k in d and isinstance(d[k], str):
            return d[k]
    return None


def load_texts(text_inputs, wanted_ids, id_keys, text_keys):
    texts = {}
    files = []
    for ti in text_inputs:
        files += sorted(glob.glob(ti)) if any(c in ti for c in "*?[") else [ti]
    for fp in files:
        for raw in iter_jsonl_zst(fp):
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except Exception:
                continue
            did = pick(d, id_keys)
            if did in wanted_ids and did not in texts:
                t = pick(d, text_keys)
                if t is not None:
                    texts[did] = t
        if len(texts) >= len(wanted_ids):
            break
    return texts


def splice_inline(text, spans):
    """Insert <match kind gated>…</match> at char offsets. Non-overlapping (outermost kept)."""
    chars = list(text)
    spans = sorted(spans, key=lambda s: (s["char_start"], -(s["char_end"])))
    inserts = []  # (pos, string, is_open)
    last_end = -1
    for s in spans:
        cs, ce = s["char_start"], s["char_end"]
        if cs < last_end:  # overlaps a kept span → skip
            continue
        if ce > len(chars):
            ce = len(chars)
        if cs >= ce:
            continue
        kind = s["kind"].replace('"', "")
        gated = s.get("gated_by", "").replace('"', "")
        inserts.append((cs, f'<match kind="{kind}" gated="{gated}">', True))
        inserts.append((ce, "</match>", False))
        last_end = ce
    # walk the chars once, emitting tag(s) at each boundary offset and escaping body chars
    out = []
    inserts_by_pos = defaultdict(list)
    for p, strv, is_open in inserts:
        inserts_by_pos[p].append((is_open, strv))
    for i, c in enumerate(chars):
        for is_open, strv in inserts_by_pos.get(i, []):
            out.append(strv)
        out.append(ESC.get(c, c))
    for is_open, strv in inserts_by_pos.get(len(chars), []):
        out.append(strv)
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--mode", choices=["wholedoc", "sections"], default="wholedoc")
    ap.add_argument("--text-input", action="append", required=True, help="jsonl(.zst) glob with the doc text")
    ap.add_argument("--spans", required=True)
    ap.add_argument("--counters", required=True)
    ap.add_argument("--metric", default="footnote_citation_only")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--total", type=int, default=150)
    ap.add_argument("--id-keys", default="id,source_doc_id,doc_id,filename,md_filename")
    ap.add_argument("--text-keys", default="text,document,content,markdown,md")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    counters = load_counters(a.counters)
    rows = [(c["doc_id"], c.get(a.metric)) for c in counters]
    chosen = set(stratified_pick(rows, a.metric, a.total))
    if not chosen:
        sys.exit("no docs selected (empty metric?)")
    print(f"[sampler] {len(chosen)} docs stratified on '{a.metric}'", file=sys.stderr)

    spans = load_spans(a.spans, chosen)
    texts = load_texts(a.text_input, chosen, a.id_keys.split(","), a.text_keys.split(","))
    cmap = {c["doc_id"]: c for c in counters}

    written = 0
    for did in chosen:
        if did not in texts:
            continue
        mv = cmap[did].get(a.metric) or 0
        prefix = f"{int(round(float(mv))):07d}"
        marked = splice_inline(texts[did], spans.get(did, []))
        header = (
            f"<!-- source={a.source} doc_id={did} metric={a.metric}={mv} "
            f"dominant_regime={cmap[did].get('dominant_regime')} "
            f"endmatter_bib_chars={cmap[did].get('endmatter_bib_chars')} "
            f"footnote_citation_only={cmap[did].get('footnote_citation_only')} "
            f"footnote_prose_kept_chars={cmap[did].get('footnote_prose_or_hybrid_kept_chars')} -->\n\n"
        )
        safe = str(did).replace("/", "_")[:40]
        path = os.path.join(a.out_dir, f"{prefix}_{safe}.md")
        open(path, "w", encoding="utf-8").write(header + marked)
        written += 1
    print(f"[sampler] wrote {written} inline-<match> review files → {a.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
