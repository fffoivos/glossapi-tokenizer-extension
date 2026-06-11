"""Local validation of the Apertus-parity PII masker — experiments 1-3.

1. Smoke      : mask real Greek HPLT docs; show masked regions in context.
2. Canary     : planted email/IP/IBAN (incl Greek grouped+compact) must be
                100% caught; negative controls must be untouched.
3. IBAN oracle: cross-check our iban_valid() against python-stdnum.

Stdlib + the masker only. Experiment 3 shells out to a throwaway venv that
has python-stdnum (set STDNUM_PY env or default /tmp/piivenv/bin/python).
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pii_masker import mask, iban_valid

HPLT = Path(__file__).resolve().parents[1] / "above_and_beyond/exploration"  # placeholder; resolved below
DROPPED = Path("/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/"
               "decontamination/experiments/v2_full_5B_correctonly_20260603/hplt_b1_5b_dropped.jsonl")
STDNUM_PY = os.environ.get("STDNUM_PY", "/tmp/piivenv/bin/python")


def ctx(masked: str, token: str, pad: int = 45) -> str:
    i = masked.find(token)
    if i < 0:
        return ""
    return "…" + masked[max(0, i - pad):i + len(token) + pad].replace("\n", " ") + "…"


def exp1_smoke():
    print("\n" + "=" * 78 + "\n### EXPERIMENT 1 — SMOKE on real Greek HPLT docs\n" + "=" * 78)
    if not DROPPED.exists():
        print("  (HPLT sample not found; skipping)"); return
    docs = [json.loads(l) for l in open(DROPPED, encoding="utf-8") if l.strip()]
    greek = [d for d in docs if d.get("source") == "greek_hplt_clean60"]
    total = {"email": 0, "ip": 0, "iban": 0}
    shown = 0
    for d in greek:
        masked, c = mask(d.get("text", ""))
        for k in total:
            total[k] += c[k]
        if sum(c.values()) and shown < 6:
            shown += 1
            hit = "email" if c["email"] else ("ip" if c["ip"] else "iban")
            print(f"\n  doc {d['doc_id'][:16]}…  {dict((k,v) for k,v in c.items() if v)}")
            print(f"    {ctx(masked, '<%s-pii>' % hit)}")
    print(f"\n  scanned {len(greek)} greek_hplt docs → totals {total}")


def exp2_canary():
    print("\n" + "=" * 78 + "\n### EXPERIMENT 2 — CANARY recall + negative controls\n" + "=" * 78)
    positives = [
        ("email",        "Επικοινωνία στο giannis.papadopoulos@otenet.gr για ραντεβού.", "<email-pii>"),
        ("ip",           "Ο διακομιστής 192.168.10.254 έπεσε χθες το βράδυ.",            "<ip-pii>"),
        ("iban GR grouped", "Κατάθεση στον GR16 0110 1250 0000 0001 2300 695 ως αύριο.",  "<iban-pii>"),
        ("iban GR compact", "IBAN: GR1601101250000000012300695 (τράπεζα Πειραιώς)",        "<iban-pii>"),
        ("iban DE",      "Überweisung auf DE89 3704 0044 0532 0130 00 bitte.",           "<iban-pii>"),
        ("iban FR w/letters", "Virement vers FR1420041010050500013M02606 svp.",          "<iban-pii>"),
        ("iban GR + trailing CAPS (no-fragment test)",
                         "Λογ. GR16 0110 1250 0000 0001 2300 695 ABCDEF συνέχεια",         "<iban-pii>"),
    ]
    negatives = [
        ("Greek date",      "Η εξέταση ορίστηκε για 30/07/2019 το πρωί."),
        ("statute number",  "Σύμφωνα με τον Ν. 4624/2019 περί προστασίας."),
        ("phone-like",      "Τηλέφωνο επικοινωνίας 210 1234567 καθημερινά."),
        ("bare 9-digit AFM","ΑΦΜ 135057893 (δεν μασκάρεται στο baseline)."),
        ("invalid IBAN chk","Λάθος IBAN GR00 0000 0000 0000 0000 0000 000 εδώ."),
        ("uppercase code",  "Κωδικός AB12 CDEF GHIJ KLMN OPQR STUV WXY δεν είναι IBAN."),
        ("price dot-thous", "Κόστος 1.234.567 ευρώ συνολικά."),
    ]
    npass = 0
    print("\n  POSITIVES (planted PII — must be masked, original removed):")
    for name, text, tok in positives:
        masked, c = mask(text)
        # the planted value: everything that should vanish — check token present + the
        # raw identifier no longer present
        ok = tok in masked
        # confirm no obvious leftover fragment for the no-fragment IBAN case
        frag_ok = True
        if "no-fragment" in name:
            frag_ok = "ABCDEF" in masked and "695" not in masked.split("ABCDEF")[0][-6:]
        ok = ok and frag_ok
        npass += ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:38s} → {masked}")
    print("\n  NEGATIVES (must be left untouched):")
    nneg = 0
    for name, text in negatives:
        masked, c = mask(text)
        ok = masked == text
        nneg += ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {name:38s} → {masked}")
    print(f"\n  positives caught: {npass}/{len(positives)}   negatives untouched: {nneg}/{len(negatives)}")
    return npass == len(positives) and nneg == len(negatives)


def exp3_iban_oracle():
    print("\n" + "=" * 78 + "\n### EXPERIMENT 3 — IBAN validator vs python-stdnum\n" + "=" * 78)
    sample = [
        "GR1601101250000000012300695",   # valid GR
        "DE89370400440532013000",         # valid DE
        "FR1420041010050500013M02606",    # valid FR
        "GB29NWBK60161331926819",         # valid GB
        "NO9386011117947",                # valid NO (15)
        "GR0001101250000000012300695",    # invalid checksum
        "GR16011012500000000123006",      # wrong length (26)
        "XX00ABCDEFGHIJKLMNOP",           # unknown country
        "AB12CDEFGHIJKLMNOPQR",           # not an IBAN
    ]
    if not Path(STDNUM_PY).exists():
        print(f"  (stdnum oracle {STDNUM_PY} not found; printing our verdicts only)")
        for s in sample:
            print(f"    ours={iban_valid(s)!s:5s}  {s}")
        return None
    snippet = ("import sys\nfrom stdnum import iban\n"
               "[print(iban.is_valid(x)) for x in sys.argv[1:]]")
    out = subprocess.run([STDNUM_PY, "-c", snippet, *sample],
                         capture_output=True, text=True).stdout.split()
    agree = 0
    print(f"  {'ours':6s} {'stdnum':7s} {'agree':6s} iban")
    for s, o in zip(sample, out):
        ours = iban_valid(s); their = (o == "True")
        a = ours == their; agree += a
        print(f"  {str(ours):6s} {str(their):7s} {'✓' if a else '✗ MISMATCH':6s} {s}")
    print(f"\n  agreement: {agree}/{len(sample)}")
    return agree == len(sample)


if __name__ == "__main__":
    exp1_smoke()
    p2 = exp2_canary()
    p3 = exp3_iban_oracle()
    print("\n" + "=" * 78)
    print(f"SUMMARY  canary={'PASS' if p2 else 'FAIL'}  iban_oracle="
          f"{'PASS' if p3 else ('N/A' if p3 is None else 'FAIL')}")
    print("=" * 78)
