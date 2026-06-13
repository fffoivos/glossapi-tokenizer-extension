#!/usr/bin/env python3
"""Compare three β-gate predictors on the FROZEN held-out test (weighted to corpus):
  (1) current detector gate   (2) a tuned deterministic rule   (3) a logistic regression.
Rule + LR are fit on TRAIN only; all three scored on TEST. No sklearn — numpy LR by hand.
Writes eval/MODEL_COMPARISON.md + eval/results_compare.json.
"""
import glob, json, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = {"end_bibliography", "chapter_bibliography", "subdivided_sublist",
       "archival_primary_sources", "web_sources", "further_reading"}
GREEK_STOP = set("και να το τη την τον της των στο στη στην στον με από για που δεν ως είναι ήταν θα ένα μία μια αυτό αυτή αυτός αυτά οι τα ο η".split())
LOCATORS = ["σσ.", "σ.", "τ.", "τόμ.", "εκδ.", "εκδόσεις", "επιμ.", "μτφρ.", "ό.π.", "vol.", "pp.", "eds.", "ed.", "trans.", "et al"]
CV_MARK = ["δημοσιευσ", "ανακοινωσ", "καταλογος ανακοιν", "curriculum", "βιογραφικο σημει"]
COLO = ["βιβλιογραφικη αναφορα", "βιβλιογραφικη  αναφορα"]
AUTHOR_INIT = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,?\s*[Α-ΩA-ZΆ-Ώ]\.")
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
YEAR_PAREN = re.compile(r"\((1[5-9]\d{2}|20\d{2})\)")
PLACE_PUB = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]")
URL = re.compile(r"https?://|doi\.|www\.")


def feats(header, text):
    low = (header + " " + text).lower()
    lines = [l for l in text.split("\n") if l.strip()]
    nl = max(1, len(lines)); toks = text.split(); nt = max(1, len(toks)); ch = max(1, len(text))
    caps = sum(1 for t in toks if t[:1].isupper())
    mid = sum(1 for l in lines for t in l.split()[1:] if t[:1].isupper())
    f = {
        "year_line": sum(1 for l in lines if YEAR.search(l)) / nl,
        "author_init": sum(1 for l in lines if AUTHOR_INIT.search(l)) / nl,
        "year_paren": len(YEAR_PAREN.findall(text)) / nl,
        "locator": sum(low.count(x) for x in LOCATORS) / nl,
        "comma_ano": (text.count(",") + text.count("·") + text.count(";")) / nt,
        "period": text.count(".") / nt,
        "place_pub": len(PLACE_PUB.findall(text)) / nl,
        "midcap": mid / nt,
        "digit": sum(c.isdigit() for c in text) / ch,
        "stopword": sum(1 for t in toks if t.strip(".,·;:()").lower() in GREEK_STOP) / nt,
        "url": len(URL.findall(text)) / nl,
        "cv_marker": 1.0 if any(x in low for x in CV_MARK) else 0.0,
        "colophon": 1.0 if any(x in low for x in COLO) else 0.0,
    }
    return f


FKEYS = ["year_line", "author_init", "year_paren", "locator", "comma_ano", "period",
         "place_pub", "midcap", "digit", "stopword", "url", "cv_marker", "colophon"]


def wmetrics(y, pred, w):
    y, pred, w = np.array(y), np.array(pred), np.array(w)
    TP = w[(pred == 1) & (y == 1)].sum(); FP = w[(pred == 1) & (y == 0)].sum()
    FN = w[(pred == 0) & (y == 1)].sum(); TN = w[(pred == 0) & (y == 0)].sum()
    p = TP / (TP + FP) if TP + FP else 0; r = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * p * r / (p + r) if p + r else 0
    return dict(TP=float(TP), FP=float(FP), FN=float(FN), TN=float(TN), precision=p, recall=r, f1=f1,
                accuracy=float((TP + TN) / (TP + FP + FN + TN)))


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
        rows.append(dict(uid=uid, y=int(tb), gate=int(m["gate_decision"].startswith("bib")),
                         w=m["weight"], split=m["split"], f=feats(*txt[uid])))
    tr = [r for r in rows if r["split"] == "train"]
    te = [r for r in rows if r["split"] == "test"]
    Xtr = np.array([[r["f"][k] for k in FKEYS] for r in tr]); ytr = np.array([r["y"] for r in tr])
    Xte = np.array([[r["f"][k] for k in FKEYS] for r in te]); yte = np.array([r["y"] for r in te])
    wte = [r["w"] for r in te]

    # (1) current gate on test
    m_gate = wmetrics(yte, [r["gate"] for r in te], wte)

    # (2) tuned deterministic rule — deny CV/colophon, else include on year/author/locator signal
    def rule(f, Y, A):
        if f["cv_marker"] or f["colophon"]:
            return 0
        if f["author_init"] > 0 or f["year_line"] >= Y or (f["year_paren"] > 0 and f["comma_ano"] >= A):
            return 1
        return 0
    best = (-1, 0.45, 0.1)
    for Y in [0.3, 0.4, 0.5, 0.6]:
        for A in [0.05, 0.1, 0.15]:
            pr = [rule(r["f"], Y, A) for r in tr]
            f1 = wmetrics(ytr, pr, [r["w"] for r in tr])["f1"]
            if f1 > best[0]:
                best = (f1, Y, A)
    _, Y, A = best
    m_rule = wmetrics(yte, [rule(r["f"], Y, A) for r in te], wte)

    # (3) logistic regression (standardised, L2, numpy GD), fit on train
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = np.zeros(Ztr.shape[1]); b = 0.0; lam = 1e-2; lr = 0.3
    for _ in range(4000):
        p = 1 / (1 + np.exp(-(Ztr @ w + b)))
        g = Ztr.T @ (p - ytr) / len(ytr) + lam * w
        w -= lr * g; b -= lr * (p - ytr).mean()
    pte = 1 / (1 + np.exp(-(Zte @ w + b)))
    m_lr = wmetrics(yte, (pte >= 0.5).astype(int), wte)
    weights = sorted(zip(FKEYS, w.tolist()), key=lambda x: -abs(x[1]))

    res = dict(n_train=len(tr), n_test=len(te), gate=m_gate, rule=m_rule, lr=m_lr,
               rule_params=dict(year_line_thresh=Y, comma_thresh=A), lr_weights=weights)
    json.dump(res, open(f"{HERE}/results_compare.json", "w"), indent=1)

    L = ["# β-gate: current vs tuned-rule vs logistic-regression (held-out test, weighted to corpus)\n",
         f"train={len(tr)} · test={len(te)}\n",
         "| predictor | precision | recall | F1 | accuracy |",
         "|---|---:|---:|---:|---:|"]
    for name, m in [("current gate", m_gate), (f"tuned rule (Y≥{Y}, deny CV/colophon)", m_rule),
                    ("logistic regression (13 feats)", m_lr)]:
        L.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['accuracy']:.3f} |")
    L.append("")
    L.append("## Logistic-regression weights (standardised — interpretable, Rust-portable as a dot-product)\n")
    for k, v in weights:
        L.append(f"- `{k}`: {v:+.2f}")
    open(f"{HERE}/MODEL_COMPARISON.md", "w", encoding="utf-8").write("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
