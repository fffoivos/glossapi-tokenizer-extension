#!/usr/bin/env python3
"""HPLT confident-only residue clean (the E001 subset of 02_corpus_preparation/
10_clean_hplt) applied to a JSONL stream.

Scope = the deterministic, zero-FP residue removal only: strip Unicode control
(Cc except tab/newline/CR), format/zero-width (Cf), private-use (Co), surrogate
(Cs) characters and the replacement char U+FFFD. This is E001 — the confident,
non-destructive subset. The broader destructive overlay (drop_doc/split_doc) is
NOT applied (its precision/FN gates are not met). Safe on any text, so it runs
over the whole mixed stream. JSONL in → JSONL out, multiprocess.
"""
import argparse, json, sys, unicodedata
from multiprocessing import Pool

_KEEP_CC = {"\t", "\n", "\r"}

def _bad(ch):
    if ch in _KEEP_CC:
        return False
    if ch == "�":
        return True
    return unicodedata.category(ch) in ("Cc", "Cf", "Co", "Cs")

# precompute a translation table over the BMP control/format ranges is not
# enough (Co/Cs span planes); use a per-char filter but short-circuit ASCII.
def clean_text(t):
    if not t:
        return t, 0
    if t.isascii():
        # fast path: only Cc controls possible in ASCII
        if not any(ord(c) < 32 and c not in _KEEP_CC or ord(c) == 127 for c in t):
            return t, 0
    out = []
    removed = 0
    for c in t:
        if c.isascii():
            if (ord(c) < 32 and c not in _KEEP_CC) or ord(c) == 127:
                removed += 1; continue
            out.append(c)
        elif _bad(c):
            removed += 1
        else:
            out.append(c)
    return "".join(out), removed

def _proc(line):
    line = line.rstrip("\n")
    if not line:
        return None
    try:
        d = json.loads(line)
    except Exception:
        return line  # pass through non-JSON unchanged
    key = "text" if "text" in d else ("content" if "content" in d else None)
    if key:
        d[key], r = clean_text(d[key])
    return json.dumps(d, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    n = 0
    with open(a.input, encoding="utf-8") as fin, open(a.output, "w", encoding="utf-8") as fout, \
            Pool(a.workers) as pool:
        for res in pool.imap(_proc, fin, chunksize=2000):
            if res is not None:
                fout.write(res + "\n"); n += 1
                if n % 1_000_000 == 0:
                    print(f"  cleaned {n:,}", file=sys.stderr, flush=True)
    print(f"DONE hplt_clean (E001): {n:,} docs -> {a.output}", flush=True)

if __name__ == "__main__":
    main()
