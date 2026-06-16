#!/usr/bin/env python3
"""Tier-1 coarse WINDOW classifier (is the window a bibliography region?) — numpy LR over ≤8 justified
window features. Two roles in the synergy: (1) a high-recall GATE (skip the line tagger where no bib
region is plausible → kills prose false positives), and (2) p_win injected as a per-line CONTEXT feature
for the line model. Label = annotation has_bib. Doc-grouped split shared with the line model.

Exposes:
  predict(data) -> ({unit_id: p_win}, model)
  line_pwin(data, pw) -> {doc_id: np.array aligned with doc lines}  (max p_win over covering windows)
Run directly: window-level precision/recall + ablation.
"""
import json, os, glob, math, collections
import numpy as np
import span_signals as S
import span_seq_data as D
import line_lr as L
HERE = os.path.dirname(os.path.abspath(__file__))
WFEATS = ["log_entry", "entry_density", "author_d", "year_paren_d", "page_d",
          "header_bib", "header_cv", "win_pos"]


def _load_units():
    man = {json.loads(l)["unit_id"]: json.loads(l)
           for l in open(f"{HERE}/units/SPAN_manifest.jsonl") if l.strip()}
    ann = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_span/all.json"))["annotations"]}
    txt = {}
    for p in glob.glob(f"{HERE}/units/SPAN/batch_*.json"):
        for u in json.load(open(p)):
            txt[u["unit_id"]] = u["text_numbered"]
    Ndoc = collections.defaultdict(int)
    for m in man.values():
        Ndoc[m["doc_id"]] = max(Ndoc[m["doc_id"]], m["win_hi"])
    return man, ann, txt, Ndoc


def _win_feats(text_numbered, win_lo, win_hi, Nd):
    lines = []
    for x in text_numbered.split("\n"):
        m = D.LINE_RE.match(x)
        if m and m.group(2).strip():
            lines.append(m.group(2))
    nl = max(1, len(lines))
    ent = sum(1 for l in lines if S.is_entry(l))
    return {
        "log_entry": math.log1p(ent),
        "entry_density": ent / nl,
        "author_d": sum(1 for l in lines if S.AUTHOR.search(l)) / nl,
        "year_paren_d": sum(1 for l in lines if S.YEAR_PAREN.search(l)) / nl,
        "page_d": sum(1 for l in lines if S.PAGE.search(l)) / nl,
        "header_bib": 1.0 if any(S.header_family(l) == "bib" for l in lines) else 0.0,
        "header_cv": 1.0 if any(S.header_family(l) == "cv" for l in lines) else 0.0,
        "win_pos": ((win_lo + win_hi) / 2) / (Nd or 1),
    }


def _build(data):
    man, ann, txt, Ndoc = _load_units()
    rows = []
    for uid, m in man.items():
        if uid not in txt or uid not in ann:
            continue
        f = _win_feats(txt[uid], m["win_lo"], m["win_hi"], Ndoc[m["doc_id"]])
        rows.append(dict(uid=uid, doc=m["doc_id"], split=data.split_of(m["doc_id"]),
                         y=int(bool(ann[uid].get("has_bib"))), f=f,
                         lo=m["win_lo"], hi=m["win_hi"]))
    return rows


def predict(data, keys=WFEATS, cost_fp=1.0, export=True):
    rows = _build(data)
    X = np.array([[r["f"][k] for k in keys] for r in rows])
    y = np.array([r["y"] for r in rows])
    tr = np.array([r["split"] == "train" for r in rows])
    w = np.where(y == 1, 1.0, cost_fp)
    mu, sd, W, b = L.fit_lr(X[tr], y[tr], w[tr])
    P = L._prob(X, mu, sd, W, b)
    pw = {r["uid"]: float(p) for r, p in zip(rows, P)}
    model = {"features": keys, "mu": [round(float(x), 5) for x in mu], "sd": [round(float(x), 5) for x in sd],
             "weight": [round(float(x), 5) for x in W], "bias": round(float(b), 5),
             "note": "window is-bib LR; standardize then sigmoid(w·z+b)."}
    if export:
        json.dump(model, open(f"{HERE}/window_lr_model.json", "w"), ensure_ascii=False, indent=1)
    return pw, model, (rows, X, y, tr, keys)


def line_pwin(data, pw):
    """Per-line p_win = max p_win over windows of the doc covering that line."""
    man = {json.loads(l)["unit_id"]: json.loads(l)
           for l in open(f"{HERE}/units/SPAN_manifest.jsonl") if l.strip()}
    bydoc = collections.defaultdict(list)
    for uid, m in man.items():
        if uid in pw:
            bydoc[m["doc_id"]].append((m["win_lo"], m["win_hi"], pw[uid]))
    out = {}
    for doc_id, d in data.docs.items():
        wins = bydoc.get(doc_id, [])
        arr = np.zeros(len(d["lines"]))
        for i, (abs_idx, _) in enumerate(d["lines"]):
            cov = [p for lo, hi, p in wins if lo <= abs_idx < hi]
            arr[i] = max(cov) if cov else 0.0
        out[doc_id] = arr
    return out


def main():
    data = D.load()
    pw, model, (rows, X, y, tr, keys) = predict(data)
    te = ~tr
    P = np.array([pw[r["uid"]] for r in rows])
    yte, Pte = y[te], P[te]
    print(f"window classifier: {len(keys)} features, {int(tr.sum())} train / {int(te.sum())} test windows")
    print(f"  has_bib prevalence test: {yte.mean():.2f}")
    print("\nwindow precision/recall sweep on TEST:")
    print("  thr   precision  recall")
    for thr in (0.2, 0.3, 0.5, 0.7, 0.9):
        pred = Pte >= thr
        tp = int((pred & (yte == 1)).sum()); fp = int((pred & (yte == 0)).sum()); fn = int((~pred & (yte == 1)).sum())
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        print(f"  {thr:.2f}   {pr:.3f}      {rc:.3f}")
    # high-recall gate threshold (recall>=0.98) for the synergy
    order = np.argsort(Pte)
    print("\nweights (standardized): " + json.dumps(dict(zip(keys, [round(float(x), 2) for x in model["weight"]])), ensure_ascii=False))


if __name__ == "__main__":
    main()
