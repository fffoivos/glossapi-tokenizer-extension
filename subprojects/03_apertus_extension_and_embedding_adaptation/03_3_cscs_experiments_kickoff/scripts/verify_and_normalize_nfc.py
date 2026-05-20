"""V9 helper: verify (and optionally normalize) NFC form in a parquet corpus.

Why this exists
---------------
Mistral-Nemo tekken v3 has `normalizer: null` — the tokenizer does NOT itself
normalize. Combining-mark (NFD-form) polytonic Greek tokenizes very differently
from precomposed (NFC) polytonic Greek. Concrete example from our 153,600 ship
bundle:

    "καὶ"  NFC  → [148480]                        (single polytonic-block token)
    "καὶ"  NFD  → [131139, 1204, 1128]            (3 tokens, base + byte-fallback)

If even a small fraction of the training corpus is NFD, those positions train
through base + byte-fallback paths instead of the new polytonic vocab. The
sampled HPLT slice on home shows 500/500 NFC; finepdfs-edu shows 181/201 NFC
with 16 combining marks in the leftover docs. Practical posture: corpus is
mostly NFC-clean from upstream, but not guaranteed. This script does the
verification and gives an idempotent normalize-pass to enforce.

Usage
-----
    # quick verification (sample 200 docs by default; no writes)
    python3 verify_and_normalize_nfc.py --check <input.parquet>

    # full audit (all docs; no writes)
    python3 verify_and_normalize_nfc.py --check --all <input.parquet>

    # normalize to NFC in place to a new parquet
    python3 verify_and_normalize_nfc.py --normalize <input.parquet> --out <output.parquet>
"""
from __future__ import annotations
import argparse
import sys
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def classify_doc(txt: str) -> tuple[str, int, int]:
    """Return (form_label, n_precomposed_polytonic, n_combining_marks).

    form_label ∈ {"NFC", "NFD", "mixed", "no-combining"}.
    """
    if not isinstance(txt, str) or not txt:
        return ("no-combining", 0, 0)
    nfc = unicodedata.normalize("NFC", txt)
    nfd = unicodedata.normalize("NFD", txt)
    n_pre = sum(1 for c in txt if 0x1F00 <= ord(c) <= 0x1FFF)
    n_combining = sum(1 for c in txt if 0x0300 <= ord(c) <= 0x036F)
    if txt == nfc and txt == nfd:
        label = "no-combining"
    elif txt == nfc and txt != nfd:
        label = "NFC"
    elif txt == nfd and txt != nfc:
        label = "NFD"
    else:
        label = "mixed"
    return (label, n_pre, n_combining)


def find_text_column(pf: pq.ParquetFile) -> str:
    cols = [f.name for f in pf.schema_arrow]
    for c in ("text", "content", "raw_content"):
        if c in cols:
            return c
    raise SystemExit(f"no text-like column in parquet (cols: {cols})")


def check(path: Path, all_docs: bool = False, max_docs: int = 200) -> int:
    pf = pq.ParquetFile(path)
    text_col = find_text_column(pf)
    counts = {"NFC": 0, "NFD": 0, "mixed": 0, "no-combining": 0}
    n_pre_total = 0
    n_combining_total = 0
    n_total = 0
    for batch in pf.iter_batches(batch_size=500, columns=[text_col]):
        for txt in batch.column(text_col).to_pylist():
            n_total += 1
            if not all_docs and n_total > max_docs:
                break
            label, n_pre, n_combining = classify_doc(txt)
            counts[label] += 1
            n_pre_total += n_pre
            n_combining_total += n_combining
        if not all_docs and n_total > max_docs:
            break
    print(f"File: {path.name}")
    print(f"  text column: {text_col!r}")
    print(f"  docs sampled: {n_total:,}")
    print(f"  NFC: {counts['NFC']:>6}    NFD: {counts['NFD']:>4}    mixed: {counts['mixed']:>4}    no-combining: {counts['no-combining']:>6}")
    print(f"  precomposed polytonic chars (NFC-form indicator): {n_pre_total:,}")
    print(f"  combining marks (NFD-form indicator):              {n_combining_total:,}")
    if n_combining_total > 0:
        leak = (counts["NFD"] + counts["mixed"]) / max(n_total, 1)
        print(f"  ⚠  some content is NFD or mixed — leak rate: {leak:.2%}")
        return 1
    print("  ✓ all sampled content is NFC-clean (no combining marks)")
    return 0


def normalize(path: Path, out: Path) -> int:
    pf = pq.ParquetFile(path)
    text_col = find_text_column(pf)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out, pf.schema_arrow, compression="zstd")
    n_total, n_changed = 0, 0
    for batch in pf.iter_batches(batch_size=5_000):
        text_idx = batch.schema.get_field_index(text_col)
        original = batch.column(text_col).to_pylist()
        normalized = []
        for txt in original:
            if not isinstance(txt, str):
                normalized.append(txt)
                continue
            nfc = unicodedata.normalize("NFC", txt)
            normalized.append(nfc)
            if nfc != txt:
                n_changed += 1
            n_total += 1
        # rebuild batch with normalized text
        cols = list(batch.columns)
        cols[text_idx] = pa.array(normalized, type=batch.schema.field(text_col).type)
        new_batch = pa.RecordBatch.from_arrays(cols, schema=batch.schema)
        writer.write_batch(new_batch)
    writer.close()
    print(f"Wrote {out}")
    print(f"  docs processed: {n_total:,}")
    print(f"  docs normalized (text changed): {n_changed:,}  ({n_changed/max(n_total,1):.2%})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)

    p_check = sub.add_parser("check", help="report NFC fraction without writing")
    p_check.add_argument("input", type=Path)
    p_check.add_argument("--all", action="store_true", help="scan all docs (default: 200)")
    p_check.add_argument("--max-docs", type=int, default=200)

    p_norm = sub.add_parser("normalize", help="write NFC-normalized copy")
    p_norm.add_argument("input", type=Path)
    p_norm.add_argument("--out", type=Path, required=True)

    # backward-compatible flag forms
    for shortflag in ("--check", "--normalize"):
        ap.add_argument(shortflag, dest="legacy", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.action == "check":
        return check(args.input, all_docs=args.all, max_docs=args.max_docs)
    elif args.action == "normalize":
        return normalize(args.input, args.out)
    return 1


if __name__ == "__main__":
    sys.exit(main())
