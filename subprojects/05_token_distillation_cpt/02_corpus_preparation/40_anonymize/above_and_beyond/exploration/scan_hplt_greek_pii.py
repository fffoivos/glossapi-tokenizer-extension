#!/usr/bin/env python3
"""DIMENSION-2 empirical scan: Greek private-PII that HPLT's generic
email/phone/IP detector MISSES.

Streams a bounded sample of HPLT ell_Grek_ge8_no_mt_clean60 docs, applies
checksum-validated Greek structured-ID detectors + generic email/phone/IP,
and tabulates per-type prevalence. Read-only; bounded memory (iter_batches).

NOT a production detector. Investigative instrument for the recall sweep.
"""
import argparse, glob, json, re, sys, random
from collections import Counter, defaultdict

# ---------- Greek-aware character classes ----------
# Greek + Latin homoglyph letter families (the 14 Greek glyphs with Latin
# lookalikes used on plates / ID cards): Α Β Ε Ζ Η Ι Κ Μ Ν Ο Ρ Τ Υ Χ
GR = "ΑΒΕΖΗΙΚΜΝΟΡΤΥΧ"
LA = "ABEZHIKMNOPTYX"  # note Ρ->P, Χ->X, Υ->Y, Η->H, Ν->N, etc.
PLATE_LETTERS = "[" + GR + LA + "]"

# ---------- generic (what HPLT covers) ----------
RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
# Greek phone: landline 2xxxxxxxxx, mobile 69xxxxxxxx, 10 digits, opt +30/0030
RE_PHONE = re.compile(r"(?<![\d.])(?:\+?30|0030)?[\s\-.]?((?:2\d|69)\d{2}[\s\-.]?\d{6}|(?:2\d|69)\d{8})(?![\d])")

# ---------- Greek structured IDs (what HPLT MISSES) ----------
# ΑΦΜ: 9 digits, often labelled. Capture both labelled + bare-9-digit (gate by checksum).
RE_AFM_LABEL = re.compile(r"(?:Α\.?Φ\.?Μ\.?|ΑΦΜ|Αριθμ(?:ός|\.)?\s+Φορολογικού\s+Μητρώου)\s*[:\-]?\s*(\d{9})\b", re.IGNORECASE)
RE_NINE = re.compile(r"(?<!\d)(\d{9})(?!\d)")
# ΑΜΚΑ: 11 digits, first 6 = DDMMYY. Often labelled.
RE_AMKA_LABEL = re.compile(r"(?:Α\.?Μ\.?Κ\.?Α\.?|ΑΜΚΑ)\s*[:\-]?\s*(\d{11})\b", re.IGNORECASE)
RE_ELEVEN = re.compile(r"(?<!\d)(\d{11})(?!\d)")
# IBAN GR: GR + 25 alnum (27 total). Allow internal spaces in groups of 4.
RE_IBAN = re.compile(r"\bGR\d{2}(?:\s?[A-Z0-9]{4}){5}\s?[A-Z0-9]{3}\b")
RE_IBAN_NOSPACE = re.compile(r"\bGR\d{25}\b")
# ΑΔΤ (ID card): old = letter + (- or space) + 6 digits; accept Greek/Latin letters
RE_ADT = re.compile(r"(?:Α\.?Δ\.?Τ\.?|ΑΔΤ|Δ\.?Α\.?Τ\.?|ταυτότητ\w*|Δελτίο\s+Ταυτότητας)\s*[:\-]?\s*(" + PLATE_LETTERS + r"{1,2})[\-\s]?(\d{6})\b", re.IGNORECASE)
RE_ADT_BARE = re.compile(r"\b(" + PLATE_LETTERS + r")[\-\s]?(\d{6})\b")
# License plate: 3 letters + 4 digits (Greek/Latin homoglyphs), opt dash/space
RE_PLATE = re.compile(r"\b(" + PLATE_LETTERS + r"{3})[\-\s]?(\d{4})\b")
# Greek postal code: 5 digits, often "ΤΚ 12345" or "123 45"
RE_TK = re.compile(r"(?:Τ\.?Κ\.?|ταχ\.?\s*κωδ\w*|τ\.?\s*κ\.?)\s*[:\-]?\s*(\d{3}\s?\d{2})\b", re.IGNORECASE)
# Greek-format street address: οδός/λεωφόρος/πλατεία + name + number
RE_ADDRESS = re.compile(r"\b(?:οδ(?:ός|ού|ό)|λεωφ(?:όρος|όρου|\.)?|λεωφ|πλατεία|πλ\.|Λεωφ\.)\s+[Α-ΩA-Zά-ωa-z][\wά-ώΆ-Ώ.'\-]*(?:\s+[Α-ΩA-Zά-ωa-z][\wά-ώΆ-Ώ.'\-]*){0,3}\s+(?:αρ(?:ιθ|\.)?\.?\s*)?\d{1,4}\b", re.IGNORECASE)
# Contact-block cues
RE_TILEFONO = re.compile(r"(?:τηλ(?:έφωνο|\.|εφ)?|κιν(?:ητό|\.)?|fax|φαξ|email|e-mail|ηλεκτρ\w*\s+δι\w*)\s*[:\-]", re.IGNORECASE)
# Honorific name cue (person, contact-gated)
RE_HONORIFIC = re.compile(r"\b(?:κ\.|κος|κα|κα\.|Κος|Κα|Δρ\.?|Δρ|καθ\.?|Καθηγητ\w*|Δ/ντ\w*)\s+[Α-ΩΆΈΉΊΌΎΏ][α-ωά-ώ]+(?:\s+[Α-ΩΆΈΉΊΌΎΏ][α-ωά-ώ]+){0,2}")

# ---------- checksums ----------
def afm_valid(s):
    if len(s) != 9 or not s.isdigit():
        return False
    if s == "000000000":
        return False
    d = [int(c) for c in s]
    total = sum(d[i] * (2 ** (8 - i)) for i in range(8))
    check = (total % 11) % 10
    return check == d[8]

def luhn_valid(s):
    digs = [int(c) for c in s]
    odd = digs[-1::-2]
    even = digs[-2::-2]
    total = sum(odd) + sum(sum(divmod(d * 2, 10)) for d in even)
    return total % 10 == 0

def amka_valid(s):
    if len(s) != 11 or not s.isdigit():
        return False
    dd, mm, yy = int(s[0:2]), int(s[2:4]), int(s[4:6])
    if not (1 <= dd <= 31 and 1 <= mm <= 12):
        return False
    return luhn_valid(s)

def iban_gr_valid(s):
    s = re.sub(r"\s", "", s)
    if not re.fullmatch(r"GR\d{25}", s):
        return False
    rearr = s[4:] + s[:4]
    num = "".join(str(int(c, 36)) for c in rearr)
    return int(num) % 97 == 1

# ---------- main scan ----------
def scan_text(text, counters, examples, ctx):
    # generic (HPLT-covered)
    emails = RE_EMAIL.findall(text)
    ips = [m for m in RE_IP.findall(text) if all(0 <= int(o) <= 255 for o in m.split("."))]
    phones = RE_PHONE.findall(text)
    counters["email"] += len(emails)
    counters["ip"] += len(ips)
    counters["phone"] += len(phones)

    # ΑΦΜ
    afm_hits = []
    for m in RE_AFM_LABEL.finditer(text):
        s = m.group(1)
        afm_hits.append((s, afm_valid(s), True))
    for m in RE_NINE.finditer(text):
        s = m.group(1)
        if afm_valid(s):
            afm_hits.append((s, True, False))
    for s, valid, labelled in afm_hits:
        if valid:
            counters["afm_valid"] += 1
            if labelled:
                counters["afm_labelled"] += 1
            if len(examples["afm"]) < 12:
                examples["afm"].append(ctx(text, s))

    # ΑΜΚΑ
    for m in RE_AMKA_LABEL.finditer(text):
        s = m.group(1)
        counters["amka_labelled"] += 1
        if amka_valid(s):
            counters["amka_valid"] += 1
            if len(examples["amka"]) < 12:
                examples["amka"].append(ctx(text, s))
    for m in RE_ELEVEN.finditer(text):
        s = m.group(1)
        if amka_valid(s):
            counters["amka_valid_bare"] += 1
            if len(examples["amka_bare"]) < 8:
                examples["amka_bare"].append(ctx(text, s))

    # IBAN
    for rgx in (RE_IBAN, RE_IBAN_NOSPACE):
        for m in rgx.finditer(text):
            s = m.group(0)
            counters["iban_format"] += 1
            if iban_gr_valid(s):
                counters["iban_valid"] += 1
                if len(examples["iban"]) < 12:
                    examples["iban"].append(ctx(text, s))

    # ΑΔΤ labelled
    for m in RE_ADT.finditer(text):
        counters["adt_labelled"] += 1
        if len(examples["adt"]) < 12:
            examples["adt"].append(ctx(text, m.group(0)))

    # license plate (format only — no checksum exists)
    for m in RE_PLATE.finditer(text):
        counters["plate_format"] += 1
        if len(examples["plate"]) < 12:
            examples["plate"].append(ctx(text, m.group(0)))

    # ΤΚ labelled
    for m in RE_TK.finditer(text):
        counters["tk_labelled"] += 1
        if len(examples["tk"]) < 8:
            examples["tk"].append(ctx(text, m.group(0)))

    # address
    addr = list(RE_ADDRESS.finditer(text))
    counters["address"] += len(addr)
    for m in addr[:1]:
        if len(examples["address"]) < 14:
            examples["address"].append(ctx(text, m.group(0)))

    # contact-block cue density
    contact_cues = len(RE_TILEFONO.findall(text))
    counters["contact_cue"] += contact_cues

    # honorific name
    hon = list(RE_HONORIFIC.finditer(text))
    counters["honorific_name"] += len(hon)
    for m in hon[:1]:
        if len(examples["honorific"]) < 12:
            examples["honorific"].append(ctx(text, m.group(0)))

    # CONTACT CLUSTER: phone/email AND (address OR honorific OR afm) in same doc
    has_generic = bool(emails or phones)
    has_greek = bool(afm_hits) or bool(addr) or bool(hon) or counters  # placeholder
    if has_generic and (addr or hon):
        counters["cluster_generic+name/addr"] += 1
        if len(examples["cluster"]) < 16:
            snippet = (emails[:1] + [p for p in phones[:1]])
            examples["cluster"].append({"sample": snippet, "ctx": ctx(text, (emails or phones)[0] if (emails or phones) else "")})


def make_ctx(window=90):
    def ctx(text, needle):
        if not needle:
            return {"needle": needle, "passage": ""}
        i = text.find(needle if isinstance(needle, str) else "")
        if i < 0:
            i = 0
            seg = text[:window * 2]
            return {"needle": needle, "passage": seg.replace("\n", " ")}
        a = max(0, i - window)
        b = min(len(text), i + len(needle) + window)
        passage = text[a:i] + "<match>" + needle + "</match>" + text[i + len(needle):b]
        return {"needle": needle, "passage": passage.replace("\n", " ")}
    return ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-docs", type=int, default=60000)
    ap.add_argument("--files", type=int, default=6)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import pyarrow.parquet as pq
    files = sorted(glob.glob("/home/foivos/data/glossapi_work/hf_release_publish/data/HPLT__ell_Grek_ge8_no_mt.*.parquet"))
    random.seed(13)
    random.shuffle(files)
    files = files[:args.files]

    counters = Counter()
    examples = defaultdict(list)
    ctx = make_ctx()
    seen = 0
    total_chars = 0

    for f in files:
        if seen >= args.max_docs:
            break
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=args.batch, columns=["text"]):
            texts = batch.column("text").to_pylist()
            for t in texts:
                if not t:
                    continue
                seen += 1
                total_chars += len(t)
                scan_text(t, counters, examples, ctx)
                if seen >= args.max_docs:
                    break
            if seen >= args.max_docs:
                break

    result = {
        "docs_scanned": seen,
        "files_used": [f.split("/")[-1] for f in files],
        "total_chars": total_chars,
        "counters": dict(counters),
        "per_1k_docs": {k: round(v / seen * 1000, 3) for k, v in counters.items()} if seen else {},
        "examples": {k: v for k, v in examples.items()},
    }
    out = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(out)
    print(out)


if __name__ == "__main__":
    main()
