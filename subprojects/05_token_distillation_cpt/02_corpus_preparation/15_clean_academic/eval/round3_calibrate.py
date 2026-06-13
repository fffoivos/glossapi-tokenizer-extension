#!/usr/bin/env python3
"""Round 3 (diagnostic) — calibrate our deterministic feature DETECTORS against the annotators'
ground-truth `syntax_features`, and error-analyse the round-2 winner's residual FP/FN.

Idea (not switch-toggling): the model plateaued because its justified features are noisy regex
proxies; the annotations carry ground-truth has_authors / has_publisher_place / has_editors /
has_year, so we can MEASURE each detector's recall/precision vs truth and fix the weak ones —
making the few features accurate instead of adding more.
"""
import glob, json, os, re, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIB = {"end_bibliography", "chapter_bibliography", "subdivided_sublist",
       "archival_primary_sources", "web_sources", "further_reading"}
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
# current detectors
AUTHOR_NOW = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,?\s*[Α-ΩA-ZΆ-Ώ]\.")
PLACE_NOW = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]")
EDITOR_NOW = re.compile(r"επιμ\.|\(επιμ|eds?\.|μτφρ\.")
# improved detectors (broadened to Greek conventions)
AUTHOR_NEW = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,\s*[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ.]+")  # Surname, Firstname/Init
PUB_WORDS = re.compile(r"(?i)εκδόσε|εκδ\.|publi|press|verlag|πανεπιστημ\w*\s+εκδ|university press|αποστολικη διακονια|gutenberg")
PLACE_NEW = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*[:,]\s*[Α-ΩA-ZΆ-Ώ]|εν\s+[Α-ΩΆ-Ώ]\w+αις")
EDITOR_NEW = re.compile(r"(?i)επιμ\.|\(επιμ|\bεπιμέλ|eds?\.|\(eds?\)|μτφρ\.|μετάφρ|trans\.|\(ed\)")


def line_has(rx, text):
    return any(rx.search(l) for l in text.split("\n") if l.strip())


def main():
    man = {json.loads(l)["unit_id"]: json.loads(l) for l in open(f"{HERE}/units/B_scale_manifest.jsonl")}
    ann = {a["unit_id"]: a for a in json.load(open(f"{HERE}/annotations_scale/all.json"))["annotations"]}
    txt = {}
    for p in glob.glob(f"{HERE}/units/B_scale/batch_*.json"):
        for u in json.load(open(p)):
            txt[u["unit_id"]] = u.get("header_line", "") + "\n" + u.get("section_text", "")

    # ---- calibration: detector vs ground-truth syntax_features ----
    def calib(name, detect, truth_key):
        tp = fp = fn = tn = 0
        for uid, a in ann.items():
            if uid not in txt:
                continue
            sf = a.get("syntax_features", {})
            if truth_key not in sf:
                continue
            d = detect(txt[uid]); t = bool(sf[truth_key])
            tp += d and t; fp += d and not t; fn += (not d) and t; tn += (not d) and not t
        rec = tp / (tp + fn) if tp + fn else 0; prec = tp / (tp + fp) if tp + fp else 0
        print(f"  {name:<28} vs {truth_key:<22} recall={rec:.2f} precision={prec:.2f} (tp={tp} fn={fn} fp={fp})")
        return rec, prec

    print("=== detector calibration vs ground-truth labels ===")
    calib("AUTHOR_NOW (Surname, Α.)", lambda t: line_has(AUTHOR_NOW, t), "has_authors")
    calib("AUTHOR_NEW (+Surname,First)", lambda t: line_has(AUTHOR_NEW, t), "has_authors")
    calib("PLACE_NOW (X: Y)", lambda t: line_has(PLACE_NOW, t), "has_publisher_place")
    calib("PLACE_NEW (+comma/pub-words)", lambda t: (line_has(PLACE_NEW, t) or bool(PUB_WORDS.search(t))), "has_publisher_place")
    calib("EDITOR_NOW", lambda t: bool(EDITOR_NOW.search(t)), "has_editors")
    calib("EDITOR_NEW", lambda t: bool(EDITOR_NEW.search(t)), "has_editors")

    # ---- refit LR-6* with NEW detectors, paired-bootstrap vs OLD detectors ----
    NAME = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]{2,}")
    PAGE = re.compile(r"\b\d+\s*[-–]\s*\d+\b|σσ?\.\s*\d|pp?\.\s*\d")
    YP = re.compile(r"\((1[5-9]\d{2}|20\d{2})\)")
    LOC = ["σσ.", "σ.", "τ.", "εκδ.", "επιμ.", "vol.", "pp.", "eds.", "et al"]
    H_CV = ["δημοσιευσ", "ανακοινωσ", "βιογραφικο", "σπουδες", "εκπαιδευσ", "εμπειρια", "σεμιναρ", "συνεδρι"]
    H_APP = ["δραστηριοτ", "λεξεις-κλειδ", "λεξεις κλειδ", "απαντησ", "προαπαιτ", "ασκησ", "περιληψ", "κριτηρι"]
    import math

    def entry(l, author_rx, place_fn):
        if not NAME.search(l):
            return False
        return bool(YEAR.search(l) or PAGE.search(l) or author_rx.search(l) or place_fn(l) or any(x in l.lower() for x in LOC))

    def feats(header, text, new):
        low = (header + " " + text).lower(); lines = [l for l in text.split("\n") if l.strip()]; nl = max(1, len(lines))
        arx = AUTHOR_NEW if new else AUTHOR_NOW
        place_fn = (lambda l: bool(PLACE_NEW.search(l))) if new else (lambda l: bool(PLACE_NOW.search(l)))
        el = sum(1 for l in lines if entry(l, arx, place_fn))
        return [math.log1p(el), sum(1 for l in lines if arx.search(l)) / nl, len(YP.findall(text)) / nl,
                (1.0 if (line_has(PLACE_NEW, text) or PUB_WORDS.search(text)) else 0.0) if new else len(PLACE_NOW.findall(text)) / nl,
                1.0 if any(x in low for x in H_CV) else 0.0, 1.0 if any(x in low for x in H_APP) else 0.0]

    rows = []
    for uid, m in man.items():
        a = ann.get(uid)
        if not a or uid not in txt:
            continue
        tb = a.get("is_reference_list") if isinstance(a.get("is_reference_list"), bool) else (a.get("kind") in BIB)
        rows.append(dict(uid=uid, y=int(tb), w=m["weight"], split=m["split"],
                         old=feats(*([txt[uid].split("\n", 1)[0], txt[uid]] + [False])),
                         new=feats(*([txt[uid].split("\n", 1)[0], txt[uid]] + [True])),
                         kind=a.get("kind"), style=a.get("citation_style"), lang=a.get("language")))
    tr = [r for r in rows if r["split"] == "train"]; te = [r for r in rows if r["split"] == "test"]
    ytr = np.array([r["y"] for r in tr]); yte = [r["y"] for r in te]; wte = [r["w"] for r in te]

    def fit(key):
        Xtr = np.array([r[key] for r in tr]); Xte = np.array([r[key] for r in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9; Z, Zt = (Xtr - mu) / sd, (Xte - mu) / sd
        w = np.zeros(Z.shape[1]); b = 0.0
        for _ in range(5000):
            p = 1 / (1 + np.exp(-(Z @ w + b))); w -= 0.3 * (Z.T @ (p - ytr) / len(ytr) + 1e-2 * w); b -= 0.3 * (p - ytr).mean()
        return (1 / (1 + np.exp(-(Zt @ w + b))) >= 0.5).astype(int)

    def wm(y, pred, w):
        y, pred, w = np.array(y), np.array(pred), np.array(w, float)
        TP = w[(pred == 1) & (y == 1)].sum(); FP = w[(pred == 1) & (y == 0)].sum()
        FN = w[(pred == 0) & (y == 1)].sum(); TN = w[(pred == 0) & (y == 0)].sum()
        p = TP / (TP + FP) if TP + FP else 0; r = TP / (TP + FN) if TP + FN else 0
        return p, r, 2 * p * r / (p + r) if p + r else 0

    pold, pnew = fit("old"), fit("new")
    print("\n=== LR-6* with OLD vs NEW detectors (held-out test, weighted) ===")
    for nm, pr in [("old detectors", pold), ("new detectors", pnew)]:
        p, r, f = wm(yte, pr, wte); print(f"  {nm:<16} P={p:.3f} R={r:.3f} F1={f:.3f}")
    # paired bootstrap
    y, A, Bv, w = np.array(yte), np.array(pold), np.array(pnew), np.array(wte, float); n = len(y); ds = []; s = 7
    for b in range(2000):
        s = (s + b * 2654435761); idx = []
        for _ in range(n):
            s = (1103515245 * s + 12345) & 0x7FFFFFFF; idx.append(s % n)
        idx = np.array(idx); ds.append(wm(y[idx], Bv[idx], w[idx])[2] - wm(y[idx], A[idx], w[idx])[2])
    ds.sort(); print(f"  ΔF1(new-old) = {np.mean(ds):+.3f} [95% {ds[50]:+.3f}, {ds[1950]:+.3f}]")

    # ---- error analysis of NEW model on test ----
    print("\n=== residual errors of NEW model (test) ===")
    fp = [(r["kind"], r["style"]) for r, pr in zip(te, pnew) if pr and not r["y"]]
    fn = [(r["kind"], r["lang"]) for r, pr in zip(te, pnew) if not pr and r["y"]]
    print("  FP kinds:", dict(collections.Counter(k for k, _ in fp)))
    print("  FN lang :", dict(collections.Counter(l for _, l in fn)))


if __name__ == "__main__":
    main()
