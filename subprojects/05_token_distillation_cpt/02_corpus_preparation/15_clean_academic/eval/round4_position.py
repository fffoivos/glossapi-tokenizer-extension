#!/usr/bin/env python3
"""Round 4 — the idea-based BREAK: re-attach document POSITION + neighbour context.

Hypothesis (pre-registered, mechanistic): CV publication lists sit in front-matter (pos≈0.16),
real bibliographies at the document tail (pos≈0.93) — nearly disjoint. The section-isolated model
is blind to this. `pos` is already logged in every span's trigger (100% join). Add position + a few
doc-level neighbour features and paired-bootstrap ΔF1 vs the section-internal model. The bootstrap is
the arbiter: CI excludes 0 → real break; CI spans 0 → β is at ceiling, pivot to Eval A.

Honors C1: AUTHOR/EDITOR broadened, PLACE original (reject the 0.66-precision place detector).
"""
import glob, json, math, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = {"end_bibliography", "chapter_bibliography", "subdivided_sublist",
       "archival_primary_sources", "web_sources", "further_reading"}
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b"); YP = re.compile(r"\((1[5-9]\d{2}|20\d{2})\)")
PAGE = re.compile(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d")
NAME = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]{2,}")
AUTHOR = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,\s*[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ.]+")  # broadened
PLACE = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]")  # original (C1)
LOC = ["σσ.", "σ.", "τ.", "εκδ.", "επιμ.", "vol.", "pp.", "eds.", "et al"]
H_CV = ["δημοσιευσ", "ανακοινωσ", "βιογραφικο", "σπουδες", "εκπαιδευσ", "εμπειρια", "σεμιναρ", "συνεδρι"]
H_APP = ["δραστηριοτ", "λεξεις-κλειδ", "λεξεις κλειδ", "απαντησ", "προαπαιτ", "ασκησ", "περιληψ", "κριτηρι"]
H_BIB = ["βιβλιογραφ", "αναφορ", "πηγ", "references", "δικτυογραφ", "ελληνογλωσσ", "ξενογλωσσ"]
TRIG = re.compile(r"pos=([\d.]+)")


def is_entry(l):
    if not NAME.search(l):
        return False
    return bool(YEAR.search(l) or PAGE.search(l) or AUTHOR.search(l) or PLACE.search(l) or any(x in l.lower() for x in LOC))


def internal_feats(header, text):
    low = (header + " " + text).lower(); lines = [l for l in text.split("\n") if l.strip()]; nl = max(1, len(lines))
    el = sum(1 for l in lines if is_entry(l))
    return {
        "log_entry": math.log1p(el),
        "author": sum(1 for l in lines if AUTHOR.search(l)) / nl,
        "year_paren": len(YP.findall(text)) / nl,
        "place": len(PLACE.findall(text)) / nl,
        "header_cv": 1.0 if any(x in low for x in H_CV) else 0.0,
        "header_app": 1.0 if any(x in low for x in H_APP) else 0.0,
    }


def main():
    # join: row_id -> (doc_id, pos, header_bib_stem) from the full corpus spans; doc_id -> [pos...]
    rid2 = {}; doc_pos = collections.defaultdict(list); doc_bibpos = collections.defaultdict(list)
    for src in ("kallipos", "pergamos"):
        for line in open(f"{HERE.rsplit('/eval',1)[0]}/out/{src}_full/refspans/{src}.spans.jsonl"):
            d = json.loads(line)
            if d.get("kind") != "beta_section":
                continue
            m = TRIG.search(d.get("trigger", ""))
            pos = float(m.group(1)) if m else 0.5
            hdr = d["trigger"].split("|", 1)[0].lower()
            rid2[d["row_id"]] = (d["doc_id"], pos)
            doc_pos[d["doc_id"]].append(pos)
            if any(x in hdr for x in H_BIB):
                doc_bibpos[d["doc_id"]].append(pos)

    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/B_scale_manifest.jsonl")}
    ann = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_scale/all.json"))["annotations"]}
    txt = {}
    for p in glob.glob(f"{HERE}/units/B_scale/batch_*.json"):
        for u in json.load(open(p)):
            txt[u["unit_id"]] = (u.get("header_line", ""), u.get("section_text", ""))

    rows = []
    for uid, m in man.items():
        a = ann.get(uid)
        if not a or uid not in txt or m["row_id"] not in rid2:
            continue
        doc_id, pos = rid2[m["row_id"]]
        sib = doc_pos[doc_id]; nb = len(sib); maxpos = max(sib) if sib else pos
        front = sum(1 for x in sib if x < 0.3) / max(1, nb)
        has_end_bib = 1.0 if any(x > 0.6 for x in doc_bibpos.get(doc_id, [])) else 0.0
        f = internal_feats(*txt[uid])
        pos_f = {"pos": pos, "dist_end": 1 - pos, "is_last": 1.0 if pos >= maxpos - 1e-6 else 0.0,
                 "doc_maxpos": maxpos, "front_cluster": front, "log_nsib": math.log1p(nb),
                 "doc_has_end_bib": has_end_bib}
        tb = a.get("is_reference_list") if isinstance(a.get("is_reference_list"), bool) else (a.get("kind") in BIB)
        rows.append(dict(y=int(tb), w=m["weight"], split=m["split"], f=f, pf=pos_f,
                         kind=a.get("kind"), lang=a.get("language"), pos=pos))
    tr = [r for r in rows if r["split"] == "train"]; te = [r for r in rows if r["split"] == "test"]
    ytr = np.array([r["y"] for r in tr]); yte = [r["y"] for r in te]; wte = [r["w"] for r in te]
    print(f"joined {len(rows)} units (row_id→doc/pos). CV-list pos median = "
          f"{np.median([r['pos'] for r in rows if r['kind']=='cv_publication_list']):.2f}, "
          f"end_bibliography pos median = {np.median([r['pos'] for r in rows if r['kind']=='end_bibliography']):.2f}")

    INT = list(tr[0]["f"].keys()); POS = list(tr[0]["pf"].keys())

    def fit(keys, usepos):
        def vec(r): return [r["f"][k] for k in INT] + ([r["pf"][k] for k in POS] if usepos else [])
        Xtr = np.array([vec(r) for r in tr]); Xte = np.array([vec(r) for r in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
        w = np.zeros(Z.shape[1]); b = 0.0
        for _ in range(5000):
            p = 1 / (1 + np.exp(-(Z @ w + b))); w -= 0.3 * (Z.T @ (p - ytr) / len(ytr) + 1e-2 * w); b -= 0.3 * (p - ytr).mean()
        names = INT + (POS if usepos else [])
        return (1 / (1 + np.exp(-(Zt @ w + b))) >= 0.5).astype(int), dict(zip(names, np.round(w, 2).tolist()))

    def wm(y, pred, w):
        y, pred, w = np.array(y), np.array(pred), np.array(w, float)
        TP = w[(pred == 1) & (y == 1)].sum(); FP = w[(pred == 1) & (y == 0)].sum()
        FN = w[(pred == 0) & (y == 1)].sum(); TN = w[(pred == 0) & (y == 0)].sum()
        p = TP / (TP + FP) if TP + FP else 0; r = TP / (TP + FN) if TP + FN else 0
        return p, r, 2 * p * r / (p + r) if p + r else 0

    p_int, _ = fit(INT, False); p_pos, wts = fit(INT + POS, True)
    pi, ri, fi = wm(yte, p_int, wte); pp, rp, fp = wm(yte, p_pos, wte)
    # paired bootstrap
    y, A, Bv, w = np.array(yte), np.array(p_int), np.array(p_pos), np.array(wte, float); n = len(y); ds = []; s = 31
    for b in range(3000):
        s = s + b * 2654435761; idx = []
        for _ in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF; idx.append(s % n)
        idx = np.array(idx); ds.append(wm(y[idx], Bv[idx], w[idx])[2] - wm(y[idx], A[idx], w[idx])[2])
    ds.sort(); lo, hi, mn = ds[75], ds[2925], float(np.mean(ds))

    out = ["# β-gate Round 4 — document position + neighbour context (held-out test, weighted)\n",
           "| model | precision | recall | F1 |", "|---|---:|---:|---:|",
           f"| section-internal (round-3 lock) | {pi:.3f} | {ri:.3f} | {fi:.3f} |",
           f"| **+ position/neighbour** | {pp:.3f} | {rp:.3f} | {fp:.3f} |", "",
           f"**Paired bootstrap ΔF1(+pos − internal) = {mn:+.3f}  [95% {lo:+.3f}, {hi:+.3f}]  → "
           f"{'REAL break (CI excludes 0)' if (lo>0 or hi<0) else 'noise (CI spans 0) → PIVOT to Eval A'}**\n",
           "## CV-FP before/after position\n",
           f"- CV-list FP (internal): {sum(1 for r,pr in zip(te,p_int) if pr and r['kind']=='cv_publication_list')}"
           f"  → (with position): {sum(1 for r,pr in zip(te,p_pos) if pr and r['kind']=='cv_publication_list')}",
           "\n## position-model weights\n" + json.dumps(wts, ensure_ascii=False)]
    print("\n".join(out))
    open(f"{HERE}/ITERATION_ROUND4.md", "w").write("\n".join(out))


if __name__ == "__main__":
    main()
