#!/usr/bin/env python3
"""Unified per-line data foundation for the TWO-HEAD structural tagger (bibliography + table-of-contents),
on the 2000-doc gold `units/STRUCT_2K_gold.jsonl`. Mirrors the `span_seq_data.Data` interface so the
existing bib machinery (line_lr / decode_spans / operating_point / score_span_models) is reused unchanged,
one class at a time.

The gold stores, per doc, the ORDERED present (non-blank) lines `[abs_idx, text, label]` with
label ∈ {0 other, 1 bibliography, 2 table_of_contents}, plus `n_lines` (true full-doc line count) and the
doc-grouped leak-free split. We featurize ONCE (reusing `line_lr.doc_features` + `line_lr.FEATS`) and carry
BOTH per-line label vectors so each head trains on the same matrix.

API:
  load() -> Data with .docs[doc_id] = {source, lines:[(abs_idx,text)], labels:[0/1/2], N, split,
                                        spans_bib:[{start_line,end_line}], spans_toc:[...]}
  set_class(data, 'bib'|'toc')         -> sets each doc's d['spans'] to that class's runs (for decode/score reuse)
  build_matrix(data, rebuild=False)    -> {X, y_bib, y_toc, tr, rows, col}   (cached STRUCT_linemat.npz)
  toc_gate(data)                       -> {doc_id: bool-mask over present lines}  (abs_idx < min(300, 0.30·N))
  apply_toc_gate(pline_toc, gate)      -> gated pline (p*=mask) so ToC never opens a run outside the front window
"""
import json, os, collections
import numpy as np
import line_lr as L
import span_signals as S
HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = f"{HERE}/units/STRUCT_2K_gold.jsonl"
MCACHE = f"{HERE}/units/STRUCT_linemat.npz"
CLASS = {"other": 0, "bibliography": 1, "table_of_contents": 2}
BIB_FEATS = L.FEATS                       # bib head keeps the existing 22-feature set (apples-to-apples regression gate)
TOC_FEATS = L.FEATS + S.TOC_KEYS          # ToC head = 22 bib feats (incl. pos/year as anti-features) + 5 ToC signals
ALLFEATS = L.FEATS + S.TOC_KEYS           # one matrix carries both; each head selects its columns


def _runs(lines, labels, cls):
    """Maximal consecutive (present-line order) runs with label==cls → [{start_line,end_line}] in abs coords."""
    spans = []; i = 0; n = len(lines)
    while i < n:
        if labels[i] == cls:
            j = i
            while j + 1 < n and labels[j + 1] == cls:
                j += 1
            spans.append({"start_line": lines[i][0], "end_line": lines[j][0]})
            i = j + 1
        else:
            i += 1
    return spans


class Data:
    def __init__(self, docs):
        self.docs = docs

    def split_of(self, doc_id):
        return self.docs[doc_id]["split"]


def load():
    docs = {}
    for ln in open(GOLD, encoding="utf-8"):
        if not ln.strip():
            continue
        r = json.loads(ln)
        lines = [(int(n), t) for n, t, _lab in r["lines"]]
        labels = [int(lab) for _n, _t, lab in r["lines"]]
        N = r.get("n_lines") or (max((n for n, _ in lines), default=0) + 1)
        docs[r["doc_id"]] = {
            "source": r.get("source"), "split": r.get("split", "train"),
            "lines": lines, "labels": labels, "N": N,
            "spans_bib": _runs(lines, labels, 1), "spans_toc": _runs(lines, labels, 2),
        }
    return Data(docs)


def set_class(data, cls):
    """Point d['spans'] at the per-class gold runs so decode_spans / score_span_models read the right gold."""
    key = "spans_bib" if cls == "bib" else "spans_toc"
    for d in data.docs.values():
        d["spans"] = d[key]


def build_matrix(data, rebuild=False):
    """Featurize once: line_lr.doc_features (22 bib feats) + span_signals.toc_signals (5 ToC feats) per line;
    carry BOTH per-line label vectors. col indexes ALLFEATS; each head selects BIB_FEATS / TOC_FEATS. Cached."""
    col = {k: j for j, k in enumerate(ALLFEATS)}
    if os.path.exists(MCACHE) and not rebuild:
        z = np.load(MCACHE, allow_pickle=True)
        docids = list(z["docids"])
        rows = [(docids[di], li) for di, li in z["rowdoc"]]
        return dict(X=z["X"], y_bib=z["y_bib"], y_toc=z["y_toc"], tr=z["tr"], rows=rows, col=col)
    X, ybib, ytoc, tr, rowdoc = [], [], [], [], []
    docids = list(data.docs.keys()); didx = {d: i for i, d in enumerate(docids)}
    for doc_id, d in data.docs.items():
        feats = L.doc_features(d)            # reuses span_signals.line_signals + CTX/position; needs d['lines'],d['N']
        istr = d["split"] == "train"
        for i, (f, lab) in enumerate(zip(feats, d["labels"])):
            tocs = S.toc_signals(d["lines"][i][1])
            X.append([f[k] for k in L.FEATS] + [tocs[k] for k in S.TOC_KEYS])
            ybib.append(1 if lab == 1 else 0); ytoc.append(1 if lab == 2 else 0)
            tr.append(istr); rowdoc.append((didx[doc_id], i))
    X = np.asarray(X, np.float32); ybib = np.asarray(ybib, np.int8); ytoc = np.asarray(ytoc, np.int8)
    tr = np.asarray(tr, bool); rowdoc = np.asarray(rowdoc, np.int32)
    np.savez(MCACHE, X=X, y_bib=ybib, y_toc=ytoc, tr=tr, rowdoc=rowdoc,
             docids=np.array(docids, dtype=object))
    rows = [(docids[di], li) for di, li in rowdoc]
    return dict(X=X, y_bib=ybib, y_toc=ytoc, tr=tr, rows=rows, col=col)


def toc_gate(data):
    """Front-window mask per doc: a present line is ToC-eligible iff abs_idx < min(300, 0.30·N)."""
    g = {}
    for doc_id, d in data.docs.items():
        cut = min(300, int(0.30 * d["N"]))
        g[doc_id] = np.array([abs_idx < cut for abs_idx, _ in d["lines"]], dtype=bool)
    return g


def apply_toc_gate(pline_toc, gate):
    """Zero ToC probability outside the front window (so the decoder never opens a run there)."""
    return {doc: np.asarray(p) * gate[doc] for doc, p in pline_toc.items()}


def main():
    data = load()
    docs = data.docs
    ndoc = len(docs)
    tr = [x for x in docs if docs[x]["split"] == "train"]
    te = [x for x in docs if docs[x]["split"] == "test"]
    nlines = sum(len(d["lines"]) for d in docs.values())
    cnt = collections.Counter()
    for d in docs.values():
        cnt.update(d["labels"])
    sb = sum(len(d["spans_bib"]) for d in docs.values()); st = sum(len(d["spans_toc"]) for d in docs.values())
    print(f"docs={ndoc}  (train {len(tr)} / test {len(te)} = {100*len(te)//ndoc}%)")
    print(f"present lines={nlines:,}  labels: other {cnt[0]:,} ({100*cnt[0]//nlines}%) · "
          f"bib {cnt[1]:,} ({100*cnt[1]//nlines}%) · toc {cnt[2]:,} ({100*cnt[2]//nlines}%)")
    print(f"reconstructed spans: bib {sb:,} · toc {st:,}")
    # split-balance + leak check (each doc one split by construction)
    import hashlib
    def expect(doc): return "test" if int(hashlib.md5(doc.encode()).hexdigest(), 16) % 10 < 3 else "train"
    leaks = sum(1 for doc in docs if docs[doc]["split"] != expect(doc))
    print(f"split matches md5%10<3: {'OK' if leaks == 0 else f'{leaks} MISMATCH'}")
    by_src = collections.Counter(d["source"] for d in docs.values())
    print(f"by source: {dict(by_src)}")
    # build matrix + sanity
    b = build_matrix(data, rebuild=True)
    print(f"matrix: X{b['X'].shape}  feats={len(L.FEATS)}  "
          f"bib-pos {int(b['y_bib'].sum()):,}  toc-pos {int(b['y_toc'].sum()):,}  "
          f"train rows {int(b['tr'].sum()):,} / test {int((~b['tr']).sum()):,}")
    # front-gate coverage: fraction of gold toc lines that survive the gate (should be ~all)
    g = toc_gate(data); kept = tot = 0
    for doc, d in docs.items():
        for i, lab in enumerate(d["labels"]):
            if lab == 2:
                tot += 1; kept += int(g[doc][i])
    print(f"ToC front-gate keeps {kept:,}/{tot:,} gold ToC lines ({100*kept//max(tot,1)}%)  [rest are ToCs past min(300,0.30N)]")


if __name__ == "__main__":
    main()
