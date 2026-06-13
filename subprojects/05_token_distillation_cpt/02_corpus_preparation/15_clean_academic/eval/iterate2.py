#!/usr/bin/env python3
"""Round 2 — acts on the three standing critics:
  C3 (idea): add ABSOLUTE entry COUNT (log) — a bibliography is a LONG list of entries; a CV is
             entry-dense but entry-FEW; prose is short. Resolves round-1's entry_density failure.
  C1 (explainability): DROP `url` (DOI⇒reference is false; hurts the Greek-FN target; leakage risk).
  C1/C3: section header used as a DENY-only veto (51% of Greek bibs have no bib header → never require it).
  C2 (parsimony): keep ≤6 features AND add a PAIRED BOOTSTRAP of model-vs-model F1 deltas on the test,
             so we only 'win' when the delta CI excludes zero.
All models fit on TRAIN, scored on the FROZEN held-out TEST (weighted to corpus).
"""
import glob, json, math, os, re
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
EDITOR = re.compile(r"επιμ\.|\(επιμ|eds?\.|\(eds?\)|μτφρ\.")
LOC = ["σσ.", "σ.", "τ.", "τόμ.", "εκδ.", "εκδόσεις", "επιμ.", "μτφρ.", "vol.", "pp.", "eds.", "ed.", "et al"]
H_CV = ["δημοσιευσ", "ανακοινωσ", "βιογραφικο", "σπουδες", "εκπαιδευσ", "εμπειρια", "ξενες γλωσσ", "σεμιναρ", "συνεδρι", "καταλογος ανακοιν"]
H_APP = ["δραστηριοτ", "λεξεις-κλειδ", "λεξεις κλειδ", "απαντησ", "λυσ", "προαπαιτ", "ασκησ", "περιληψ", "συνοψ", "κριτηρι"]
H_BIB = ["βιβλιογραφ", "αναφορ", "πηγ", "references", "bibliograph", "δικτυογραφ", "ελληνογλωσσ", "ξενογλωσσ", "ιστοσελιδ"]
COLO = "βιβλιογραφικη αναφορα"


def is_entry(l):
    if not NAME.search(l):
        return False
    return bool(YEAR.search(l) or PAGE.search(l) or PLACE_PUB.search(l) or any(x in l.lower() for x in LOC))


def feats(header, text):
    low = (header + " " + text).lower()
    lines = [l for l in text.split("\n") if l.strip()]
    nl = max(1, len(lines)); toks = text.split(); nt = max(1, len(toks))
    entry_lines = sum(1 for l in lines if is_entry(l))
    return {
        "log_entry": math.log1p(entry_lines),                 # C3 idea: absolute entry COUNT
        "log_nlines": math.log1p(len(lines)),                 # list length
        "entry_density": entry_lines / nl,                    # (kept for comparison)
        "author_init": sum(1 for l in lines if AUTHOR_INIT.search(l)) / nl,
        "year_paren": len(YEAR_PAREN.findall(text)) / nl,
        "place_pub": len(PLACE_PUB.findall(text)) / nl,       # imprint metadata (C3 idea 2)
        "editor": len(EDITOR.findall(text)) / nl,
        "url": (len(re.findall(r"https?://|doi\.|www\.", text))) / nl,  # kept ONLY to show it should die
        "header_cv": 1.0 if any(x in low for x in H_CV) else 0.0,
        "header_app": 1.0 if any(x in low for x in H_APP) else 0.0,
        "header_bib": 1.0 if any(x in low for x in H_BIB) else 0.0,
        "colophon": 1.0 if COLO in low else 0.0,
    }


def wm(y, pred, w):
    y, pred, w = np.array(y), np.array(pred), np.array(w, float)
    TP = w[(pred == 1) & (y == 1)].sum(); FP = w[(pred == 1) & (y == 0)].sum()
    FN = w[(pred == 0) & (y == 1)].sum(); TN = w[(pred == 0) & (y == 0)].sum()
    p = TP / (TP + FP) if TP + FP else 0; r = TP / (TP + FN) if TP + FN else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0), (TP + TN) / (TP + FP + FN + TN)


def fit(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(5000):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= 0.3 * (Z.T @ (p - ytr) / len(ytr) + 1e-2 * w); b -= 0.3 * (p - ytr).mean()
    return (1 / (1 + np.exp(-(Zt @ w + b))) >= 0.5).astype(int), dict(zip(KEYS_LAST, np.round(w, 2).tolist()))


KEYS_LAST = []


def paired_boot(y, predA, predB, w, B=2000):
    y, predA, predB, w = map(lambda v: np.array(v, float), (y, predA, predB, w))
    n = len(y); deltas = []; seed = 999
    for b in range(B):
        s = seed + b * 2654435761; idx = []
        for _ in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF; idx.append(s % n)
        idx = np.array(idx)
        fA = wm(y[idx], predA[idx], w[idx])[2]; fB = wm(y[idx], predB[idx], w[idx])[2]
        deltas.append(fB - fA)
    deltas.sort()
    return deltas[int(0.025 * B)], deltas[int(0.975 * B)], float(np.mean(deltas))


def main():
    global KEYS_LAST
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
    ytr = np.array([r["y"] for r in tr]); yte = [r["y"] for r in te]; wte = [r["w"] for r in te]

    def X(rs, keys): return np.array([[r["f"][k] for k in keys] for r in rs])

    models = {}
    models["M0 gate"] = ([r["gate"] for r in te], 0, {})
    # Round-1 reference: LR-6 with url
    global KEYS_LAST
    k = ["entry_density", "author_init", "year_paren", "url", "header_cv", "colophon"]; KEYS_LAST = k
    pr, wts = fit(X(tr, k), ytr, X(te, k)); models["LR6 (round1, has url)"] = (list(pr), 6, wts)
    # Round-2 justified: entry COUNT, NO url, imprint, header DENY as features
    k = ["log_entry", "author_init", "year_paren", "place_pub", "header_cv", "header_app"]; KEYS_LAST = k
    pr, wts = fit(X(tr, k), ytr, X(te, k)); models["LR6* (entry-count, no url)"] = (list(pr), 6, wts)
    # parsimony: 4 features
    k = ["log_entry", "author_init", "place_pub", "header_cv"]; KEYS_LAST = k
    pr, wts = fit(X(tr, k), ytr, X(te, k)); models["LR4 (count+author+imprint+cv-deny)"] = (list(pr), 4, wts)
    # one-sentence RULE with length + header veto
    def rule(r):
        f = r["f"]
        if f["colophon"] or f["header_cv"] or f["header_app"]:
            return 0
        return int(f["log_entry"] >= math.log1p(4) and (f["author_init"] > 0.1 or f["year_paren"] > 0.1 or f["header_bib"]))
    models["RULE (≥4 entries & author/year, header-veto)"] = ([rule(r) for r in te], 1, {})

    out = ["# β-gate iteration — round 2 (held-out test, weighted to corpus)\n",
           f"train={len(tr)} test={len(te)}\n",
           "| model | size | precision | recall | F1 | acc |", "|---|---:|---:|---:|---:|---:|"]
    M = {}
    for name, (pred, sz, wts) in models.items():
        p, r, f1, ac = wm(yte, pred, wte); M[name] = pred
        out.append(f"| {name} | {sz} | {p:.3f} | {r:.3f} | {f1:.3f} | {ac:.3f} |")
    out.append("\n## Paired bootstrap of F1 deltas on the test (does the win exceed noise?)\n")
    base = "LR6 (round1, has url)"
    for name in ["LR6* (entry-count, no url)", "LR4 (count+author+imprint+cv-deny)", "RULE (≥4 entries & author/year, header-veto)"]:
        lo, hi, mn = paired_boot(yte, M[base], M[name], wte)
        sig = "REAL (excludes 0)" if (lo > 0 or hi < 0) else "noise (spans 0)"
        out.append(f"- {name} − {base}: ΔF1 = {mn:+.3f}  [95% {lo:+.3f}, {hi:+.3f}]  → {sig}")
    out.append("\n## Round-2 weights (justified model)\n")
    out.append("`LR6* (entry-count, no url)`: " + json.dumps(models["LR6* (entry-count, no url)"][2], ensure_ascii=False))
    print("\n".join(out))
    open(f"{HERE}/ITERATION_ROUND2.md", "w").write("\n".join(out))


if __name__ == "__main__":
    main()
