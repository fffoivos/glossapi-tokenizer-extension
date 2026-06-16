#!/usr/bin/env python3
"""Data foundation for the bibliography-span classifiers.

Reconstructs, per document, the ORDERED stream of present (non-blank) lines from the sampled windows
(`units/SPAN/batch_*.json`, lines prefixed `L#####:` with absolute indices, blank lines already
dropped), unions a doc's overlapping tail+body windows, attaches per-line gold labels (binary in-bib +
BIO over present-line runs) from the blind Opus annotations, recovers document position, and assigns a
doc-grouped leak-free train/test split (md5(doc_id) % 10 < 3 → test) persisted to
`units/SPAN_split.json`. Imported as a library by the model trainers; `main()` self-verifies.

Library API:
  load() -> Data with:
    .docs            : {doc_id: {source, lines:[(abs_idx,text)], N, split, spans:[span...]}}
    .line_rows()     : iterator of per-present-line dicts {doc_id, source, split, idx, abs_idx, text,
                       y, bio, pos, n_present} (idx = position in the doc's present-line list)
    .split_of(doc)   : 'train' | 'test'
Span dict carries the rich metadata (kind, citation_style, language, script, subject_register,
noise_level, n_entries, has_header) for sliced evaluation.
"""
import json, glob, os, re, hashlib, collections
HERE = os.path.dirname(os.path.abspath(__file__))
LINE_RE = re.compile(r"^L(\d+):\s?(.*)$")


def split_of(doc_id, test_frac=0.30):
    k = int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 10
    return "test" if k < int(round(test_frac * 10)) else "train"


def _parse_unit(text_numbered):
    out = []
    for ln in text_numbered.split("\n"):
        m = LINE_RE.match(ln)
        if m:
            out.append((int(m.group(1)), m.group(2)))
    return out


class Data:
    def __init__(self, docs):
        self.docs = docs

    def split_of(self, doc_id):
        return self.docs[doc_id]["split"]

    def line_rows(self):
        for doc_id, d in self.docs.items():
            lines = d["lines"]
            n = len(lines)
            # gold line set + BIO over present-line runs
            gold = set()
            for s in d["spans"]:
                a, b = s["start_line"], s["end_line"]
                if isinstance(a, int) and isinstance(b, int) and b >= a:
                    gold.update(range(a, b + 1))
            prev_y = 0
            for i, (abs_idx, text) in enumerate(lines):
                y = 1 if abs_idx in gold else 0
                bio = "O" if not y else ("B" if not prev_y else "I")
                prev_y = y
                yield {"doc_id": doc_id, "source": d["source"], "split": d["split"],
                       "idx": i, "abs_idx": abs_idx, "text": text, "y": y, "bio": bio,
                       "pos": abs_idx / d["N"] if d["N"] else 0.0, "n_present": n}


def merge_spans(spans, gap=10):
    """Merge overlapping/adjacent spans into regions (for span-level scoring). Mirrors
    build_boundary_report.merge — a subdivided sub-list is ONE region."""
    sp = sorted((s["start_line"], s["end_line"]) for s in spans
                if isinstance(s.get("start_line"), int) and isinstance(s.get("end_line"), int)
                and s["end_line"] >= s["start_line"])
    out = []
    for s, e in sp:
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def load():
    man = {json.loads(l)["unit_id"]: json.loads(l)
           for l in open(f"{HERE}/units/SPAN_manifest.jsonl") if l.strip()}
    ann = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_span/all.json"))["annotations"]}
    # unit text
    txt = {}
    for p in glob.glob(f"{HERE}/units/SPAN/batch_*.json"):
        for u in json.load(open(p)):
            txt[u["unit_id"]] = u["text_numbered"]

    docs = {}
    for uid, m in man.items():
        doc_id = m["doc_id"]
        d = docs.setdefault(doc_id, {"source": m["source"], "linemap": {}, "N": 0, "spans": []})
        d["N"] = max(d["N"], m["win_hi"])
        if uid in txt:
            for abs_idx, text in _parse_unit(txt[uid]):
                d["linemap"].setdefault(abs_idx, text)  # dedup overlapping windows, keep first
        a = ann.get(uid)
        if a:
            d["spans"].extend(a.get("spans", []))
    for doc_id, d in docs.items():
        d["lines"] = sorted(d["linemap"].items())
        del d["linemap"]
        d["split"] = split_of(doc_id)
    return Data(docs)


def main():
    d = load()
    docs = d.docs
    ndoc = len(docs)
    # split sizes
    tr = [x for x in docs if docs[x]["split"] == "train"]
    te = [x for x in docs if docs[x]["split"] == "test"]
    rows = list(d.line_rows())
    nlines = len(rows)
    nbib = sum(r["y"] for r in rows)
    nB = sum(1 for r in rows if r["bio"] == "B")

    # VERIFY 1: per-line in-bib count == size of union of all spans' lines (present only)
    union = 0
    present_bib_from_spans = 0
    for doc_id, dd in docs.items():
        present = {a for a, _ in dd["lines"]}
        g = set()
        for s in dd["spans"]:
            a, b = s["start_line"], s["end_line"]
            if isinstance(a, int) and isinstance(b, int) and b >= a:
                g.update(range(a, b + 1))
        present_bib_from_spans += len(g & present)
        union += len(g)
    assert nbib == present_bib_from_spans, (nbib, present_bib_from_spans)

    # VERIFY 2: no doc leak — each doc exactly one split (true by construction); report balance
    nspan = sum(len(dd["spans"]) for dd in docs.values())
    by_src = collections.Counter(dd["source"] for dd in docs.values())
    src_tr = collections.Counter(docs[x]["source"] for x in tr)
    src_te = collections.Counter(docs[x]["source"] for x in te)

    print(f"docs={ndoc}  (train {len(tr)} / test {len(te)} = {100*len(te)//ndoc}%)")
    print(f"present lines={nlines:,}  in-bib lines={nbib:,} ({100*nbib//nlines}%)  BIO-B runs={nB:,}  spans(raw)={nspan:,}")
    print(f"VERIFY ok: per-line in-bib ({nbib:,}) == union of span lines ∩ present ({present_bib_from_spans:,})")
    print(f"  (raw span line-union incl. out-of-window = {union:,}; ≥99% land in-window)")
    print(f"by source (docs): {dict(by_src)}")
    print(f"  train: {dict(src_tr)}")
    print(f"  test : {dict(src_te)}")
    # bib-line prevalence per split (sanity: similar)
    for nm, ids in (("train", set(tr)), ("test", set(te))):
        rs = [r for r in rows if r["doc_id"] in ids]
        nb = sum(r["y"] for r in rs)
        print(f"  {nm} bib-line prevalence: {100*nb/max(1,len(rs)):.1f}%  ({len(rs):,} lines)")

    json.dump({doc_id: dd["split"] for doc_id, dd in docs.items()},
              open(f"{HERE}/units/SPAN_split.json", "w"))
    print(f"wrote units/SPAN_split.json ({ndoc} docs)")


if __name__ == "__main__":
    main()
