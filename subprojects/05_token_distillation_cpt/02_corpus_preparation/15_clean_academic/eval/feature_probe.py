#!/usr/bin/env python3
"""Measure which section-level features separate bibliography from non-bibliography,
on the 64 Opus-labelled β sections. Reports per-feature class means + AUC (separability),
so feature selection is data-driven, not asserted.

Feature families mirror established reference-detection pipelines (GROBID / ParsCit /
anystyle / CERMINE / refextract): token-shape, lexicon (locators/publishers), punctuation,
numerals, structural/line-regularity, and a function-word/verb anti-prose ratio — adapted
to Greek + OCR markdown + section granularity.
"""
import glob, json, os, re, math

HERE = os.path.dirname(os.path.abspath(__file__))

# --- Greek/Latin lexicons ----------------------------------------------------
GREEK_STOP = set("και να το τη την τον της των στο στη στην στον με από για που δεν ως είναι ήταν "
                 "θα ένα μία μια αυτό αυτή αυτός αυτά οι τα ο η τους μας σας πως όταν αλλά ή".split())
LOCATORS = ["σσ.", "σ.", "τ.", "τόμ.", "τομ.", "εκδ.", "εκδόσεις", "επιμ.", "μτφρ.", "μετάφρ.",
            "ό.π.", "στο ίδιο", "κ.ά.", "κ.συν.", "vol.", "pp.", "no.", "eds.", "ed.", "trans.", "et al"]
CV_MARK = ["δημοσιευσ", "ανακοινωσ", "καταλογος ανακοιν", "curriculum", "βιογραφικο"]
VERB_END = ("ει", "ουν", "εται", "ονται", "ηκε", "θηκε", "ουμε", "ετε", "ιζει", "ωσε", "ησαν", "ουσε")

AUTHOR_INIT = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώϊϋ]+\s*,?\s*[Α-ΩA-ZΆ-Ώ]\.")  # Surname, Α.
YEAR = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")
YEAR_PAREN = re.compile(r"\((1[5-9]\d{2}|20\d{2})\)")
PLACE_PUB = re.compile(r"[Α-ΩA-ZΆ-Ώ][α-ωa-zά-ώ]+\s*:\s*[Α-ΩA-ZΆ-Ώ]")  # Αθήνα: Εκδόσεις
PAGE_RANGE = re.compile(r"\b\d+\s*[-–]\s*\d+\b")
URL_DOI = re.compile(r"https?://|doi\.|www\.|\.gr/|\.com/")


def is_cap(tok):
    for c in tok:
        if c.isalpha():
            return c.isupper()
    return False


def features(text):
    lines = [l for l in text.split("\n")]
    nonempty = [l for l in lines if l.strip()]
    toks = text.split()
    nt = max(1, len(toks))
    nl = max(1, len(nonempty))
    chars = max(1, len(text))
    low = text.lower()

    commas = text.count(",") + text.count("·") + text.count(";")
    periods = text.count(".")
    digits = sum(c.isdigit() for c in text)
    caps = sum(1 for t in toks if is_cap(t))
    # mid-line caps = capitalised tokens that are not the first token of their line
    midcaps = 0
    for l in nonempty:
        lt = l.split()
        for t in lt[1:]:
            if is_cap(t):
                midcaps += 1
    stop = sum(1 for t in toks if t.strip(".,·;:()[]").lower() in GREEK_STOP)
    verbs = sum(1 for t in toks if t.strip(".,·;:()[]").lower().endswith(VERB_END) and len(t) > 4)
    loc = sum(low.count(x) for x in LOCATORS)
    author_lines = sum(1 for l in nonempty if AUTHOR_INIT.search(l))
    year_lines = sum(1 for l in nonempty if YEAR.search(l))
    dash_lines = sum(1 for l in nonempty if l.lstrip().startswith(("- ", "• ", "* ", "–", "—")))
    num_lines = sum(1 for l in nonempty if re.match(r"^\s*\d+[.\)]\s", l))
    end_period = sum(1 for l in nonempty if l.rstrip().endswith((".", ")")))
    llens = [len(l) for l in nonempty]
    mean_ll = sum(llens) / nl
    var_ll = sum((x - mean_ll) ** 2 for x in llens) / nl
    cv_ll = math.sqrt(var_ll) / mean_ll if mean_ll else 0

    return {
        # the user's explicit ideas
        "dashlist_density": dash_lines / nl,
        "punct_comma_density": commas / nt,
        "period_density": periods / nt,
        "verb_ratio(anti)": verbs / nt,
        "propernoun_cap_ratio": caps / nt,
        "midline_cap_ratio": midcaps / nt,
        # error-derived + pipeline-standard
        "author_initial_density": author_lines / nl,
        "year_line_density": year_lines / nl,
        "year_paren_density": len(YEAR_PAREN.findall(text)) / nl,
        "stopword_ratio(anti)": stop / nt,
        "locator_lexicon_per_line": loc / nl,
        "place_publisher_per_line": len(PLACE_PUB.findall(text)) / nl,
        "page_range_per_line": len(PAGE_RANGE.findall(text)) / nl,
        "digit_density": digits / chars,
        "numbered_line_density": num_lines / nl,
        "end_period_frac": end_period / nl,
        "line_len_cv(anti)": cv_ll,
        "url_doi_per_line": len(URL_DOI.findall(text)) / nl,
        "cv_marker": 1.0 if any(x in low for x in CV_MARK) else 0.0,
    }


def auc(pos, neg):
    # rank-based AUC: P(random pos > random neg)
    if not pos or not neg:
        return float("nan")
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    # average ranks
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = r
        i = j
    sr = sum(ranks[k] for k in range(len(allv)) if allv[k][1] == 1)
    np_, nn = len(pos), len(neg)
    return (sr - np_ * (np_ + 1) / 2) / (np_ * nn)


def main():
    # truth from annotations
    truth = {}
    for p in glob.glob(f"{HERE}/annotations/B_batch*.json"):
        for a in json.load(open(p, encoding="utf-8")):
            truth[a["unit_id"]] = bool(a.get("is_reference_list"))
    feats = {}
    for p in glob.glob(f"{HERE}/units/B_betagate/unit_*.json"):
        u = json.load(open(p, encoding="utf-8"))
        if u["unit_id"] in truth:
            # strip the L#### prefixes back to raw text
            raw = re.sub(r"(?m)^L\d+:\s?", "", u["section_numbered"])
            feats[u["unit_id"]] = features(raw)
    keys = list(next(iter(feats.values())).keys())
    bib = [u for u in feats if truth[u]]
    notb = [u for u in feats if not truth[u]]
    print(f"labelled: {len(bib)} bibliography / {len(notb)} not  (features measured on raw section text)\n")
    rows = []
    for k in keys:
        pv = [feats[u][k] for u in bib]
        nv = [feats[u][k] for u in notb]
        a = auc(pv, nv)
        sep = abs(a - 0.5) * 2  # 0..1
        rows.append((sep, a, k, sum(pv) / len(pv), sum(nv) / len(nv)))
    rows.sort(reverse=True)
    print(f"{'feature':<26}{'AUC':>6}{'sep':>6}{'mean|bib':>11}{'mean|not':>11}")
    for sep, a, k, mb, mn in rows:
        flag = " <== strong" if sep >= 0.5 else (" <- useful" if sep >= 0.3 else "")
        print(f"{k:<26}{a:>6.2f}{sep:>6.2f}{mb:>11.3f}{mn:>11.3f}{flag}")
    json.dump([{"feature": k, "auc": a, "sep": sep, "mean_bib": mb, "mean_not": mn}
               for sep, a, k, mb, mn in rows], open(f"{HERE}/feature_separability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
