#!/usr/bin/env python3
"""Iteration engine for the β-gate. Implements a library of candidate feature sets / models,
fits on TRAIN, scores on the FROZEN held-out TEST (weighted to corpus). One command per round.

Design values it serves (the standing critics):
  - explainable: every feature is justified by what a bibliography IS — a LIST OF REFERENCE ENTRIES
    (name + year + title + publisher/pages), extracted by Docling and cleaned by GlossAPI (so noisy).
  - parsimonious: prefers the fewest features that work; reports model size alongside accuracy.
  - idea-based: includes non-incremental candidates (an 'entry-likeness' composite; header-family).
"""
import glob, json, os, re
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = {"end_bibliography", "chapter_bibliography", "subdivided_sublist",
       "archival_primary_sources", "web_sources", "further_reading"}

YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
YEAR_PAREN = re.compile(r"\((1[5-9]\d{2}|20\d{2})\)")
PAGE = re.compile(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d")
PLACE_PUB = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]")
NAME = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]{2,}")
AUTHOR_INIT = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,?\s*[Α-ΩA-ZΆ-Ώ]\.")
URL = re.compile(r"https?://|doi\.|www\.")
LOC = ["σσ.", "σ.", "τ.", "τόμ.", "εκδ.", "εκδόσεις", "επιμ.", "μτφρ.", "vol.", "pp.", "eds.", "ed.", "et al"]
# header families (the section's own label says what it is — maximally explainable)
H_BIB = ["βιβλιογραφ", "αναφορ", "πηγ", "references", "bibliograph", "δικτυογραφ", "ελληνογλωσσ", "ξενογλωσσ", "ιστοσελιδ"]
H_CV = ["δημοσιευσ", "ανακοινωσ", "βιογραφικο", "σπουδες", "εκπαιδευσ", "εμπειρια", "ξενες γλωσσ", "σεμιναρ"]
H_APP = ["δραστηριοτ", "λεξεις-κλειδ", "λεξεις κλειδ", "απαντησ", "λυσ", "προαπαιτ", "ασκησ", "περιληψ", "συνοψ", "κριτηρι"]
COLO = "βιβλιογραφικη αναφορα"


def is_entry_line(l):
    # a bibliographic entry: a capitalised name AND a year or page-range or publisher-colon or locator
    if not NAME.search(l):
        return False
    return bool(YEAR.search(l) or PAGE.search(l) or PLACE_PUB.search(l) or any(x in l.lower() for x in LOC))


def feats(header, text):
    low = (header + " " + text).lower()
    lines = [l for l in text.split("\n") if l.strip()]
    nl = max(1, len(lines)); toks = text.split(); nt = max(1, len(toks))
    f = {
        # idea-based composite: "fraction of lines that ARE reference entries" (the definition of a bib)
        "entry_density": sum(1 for l in lines if is_entry_line(l)) / nl,
        # supporting justified signals
        "year_line": sum(1 for l in lines if YEAR.search(l)) / nl,
        "author_init": sum(1 for l in lines if AUTHOR_INIT.search(l)) / nl,
        "year_paren": len(YEAR_PAREN.findall(text)) / nl,
        "comma_ano": (text.count(",") + text.count("·") + text.count(";")) / nt,
        "url": len(URL.findall(text)) / nl,
        # header-family one-hots (the section's own label)
        "header_bib": 1.0 if any(x in low for x in H_BIB) else 0.0,
        "header_cv": 1.0 if any(x in low for x in H_CV) else 0.0,
        "header_app": 1.0 if any(x in low for x in H_APP) else 0.0,
        "colophon": 1.0 if COLO in low else 0.0,
    }
    return f


def wm(y, pred, w):
    y, pred, w = np.array(y), np.array(pred), np.array(w, float)
    TP = w[(pred == 1) & (y == 1)].sum(); FP = w[(pred == 1) & (y == 0)].sum()
    FN = w[(pred == 0) & (y == 1)].sum(); TN = w[(pred == 0) & (y == 0)].sum()
    p = TP / (TP + FP) if TP + FP else 0; r = TP / (TP + FN) if TP + FN else 0
    return dict(precision=p, recall=r, f1=2 * p * r / (p + r) if p + r else 0,
                accuracy=(TP + TN) / (TP + FP + FN + TN))


def fit_lr(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.zeros(Ztr.shape[1]); b = 0.0
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(Ztr @ w + b)))
        w -= 0.3 * (Ztr.T @ (p - ytr) / len(ytr) + 1e-2 * w); b -= 0.3 * (p - ytr).mean()
    return (1 / (1 + np.exp(-(Zte @ w + b))) >= 0.5).astype(int), dict(zip_=list(w))


def main():
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/B_scale_manifest.jsonl")}
    ann = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_scale/all.json"))["annotations"]}
    txt = {}
    for p in glob.glob(f"{HERE}/units/B_scale/batch_*.json"):
        for u in json.load(open(p)):
            txt[u["unit_id"]] = (u.get("header_line", ""), u.get("section_text", ""))
    rows = []
    for uid, m in man.items():
        a = ann.get(uid)
        if not a or uid not in txt:
            continue
        tb = a.get("is_reference_list") if isinstance(a.get("is_reference_list"), bool) else (a.get("kind") in BIB)
        rows.append(dict(y=int(tb), gate=int(m["gate_decision"].startswith("bib")), w=m["weight"],
                         split=m["split"], f=feats(*txt[uid])))
    tr = [r for r in rows if r["split"] == "train"]; te = [r for r in rows if r["split"] == "test"]
    yte = [r["y"] for r in te]; wte = [r["w"] for r in te]; ytr = np.array([r["y"] for r in tr])

    def X(rs, keys): return np.array([[r["f"][k] for k in keys] for r in rs])

    cands = []
    cands.append(("M0 current gate", 0, [int(r["gate"]) for r in te]))
    # idea-based, parsimony-first: ONE feature — "is this a list of entries?" + colophon/CV/app deny
    def rule_entry(r, t):
        f = r["f"]
        if f["colophon"] or f["header_cv"] or f["header_app"]:
            return 0
        return int(f["entry_density"] >= t or f["header_bib"])
    best_t = max([0.2, 0.3, 0.4, 0.5], key=lambda t: wm(ytr, [rule_entry(r, t) for r in tr], [r["w"] for r in tr])["f1"])
    cands.append((f"M-idea: entry_density≥{best_t} + header deny (2 rules)", 1,
                  [rule_entry(r, best_t) for r in te]))
    # parsimonious LR on 3 justified features
    k3 = ["entry_density", "header_cv", "header_app"]
    p3, _ = fit_lr(X(tr, k3), ytr, X(te, k3)); cands.append(("M-LR3: entry+header_cv+header_app", 3, list(p3)))
    # 6-feature LR
    k6 = ["entry_density", "author_init", "year_paren", "url", "header_cv", "colophon"]
    p6, _ = fit_lr(X(tr, k6), ytr, X(te, k6)); cands.append(("M-LR6", 6, list(p6)))
    # full 10-feature LR (this round's superset)
    k10 = list(tr[0]["f"].keys())
    p10, _ = fit_lr(X(tr, k10), ytr, X(te, k10)); cands.append((f"M-LR{len(k10)} (all feats)", len(k10), list(p10)))

    out = ["# β-gate iteration — round 1 (held-out test, weighted to corpus)\n",
           f"train={len(tr)} test={len(te)}\n",
           "| candidate | #feats/rules | precision | recall | F1 | acc |",
           "|---|---:|---:|---:|---:|---:|"]
    res = []
    for name, sz, pred in cands:
        m = wm(yte, pred, wte)
        out.append(f"| {name} | {sz} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} |")
        res.append(dict(name=name, size=sz, **m))
    print("\n".join(out))
    json.dump(res, open(f"{HERE}/results_round1.json", "w"), indent=1)
    open(f"{HERE}/ITERATION_ROUND1.md", "w").write("\n".join(out))


if __name__ == "__main__":
    main()
