"""Aggregator — simplified per user direction 2026-05-13.

Reads all per-language histogram .npy files from staging/<worker_idx>/ and emits
the minimal artifact set:

  outputs/histogram_matrix.npz       — (N_langs, 131072) int64 raw counts + canonical_keys index
  outputs/lang_metadata.json         — per canonical key: ISO/script/sources/sample-size/vocab-fired-counts
  outputs/token_metadata.parquet     — 131,072 rows: decoded_string + script flags + structural flags + totals
  outputs/zero_sum_keys.json         — canonical keys whose histogram is all-zeros (re-run targeting)

Deliberately NOT produced (downstream investigation per user direction):
  - primary_lang, primary_lang_prob, signature_entropy, confidence_flag, top5_*

Reviewer-flagged bugs fixed:
  #1 staging path: now rglob's across <staging>/<worker_idx>/ subdirs.
  #5 alignment:    canonical_keys append happens AFTER shape validation.
  #2/#6/#7:        moot — we don't compute the derived layer the reviewer was concerned about.

Use:
  python3 aggregate.py \\
    --staging /path/to/vocab_lang_attribution/staging \\
    --lang-map /path/to/scripts/lang_code_map.json \\
    --out /path/to/vocab_lang_attribution/outputs
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

import numpy as np


SCRIPT_RANGES = {
    "has_latin_basic":    [(0x0041, 0x005A), (0x0061, 0x007A)],
    "has_latin_extended": [(0x00C0, 0x024F), (0x1E00, 0x1EFF)],
    "has_greek_mono":     [(0x0370, 0x03FF)],
    "has_greek_poly":     [(0x1F00, 0x1FFF)],
    "has_cyrillic":       [(0x0400, 0x04FF)],
    "has_cyrillic_ext":   [(0x0500, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)],
    "has_arabic":         [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "has_hebrew":         [(0x0590, 0x05FF), (0xFB1D, 0xFB4F)],
    "has_devanagari":     [(0x0900, 0x097F)],
    "has_han":            [(0x4E00, 0x9FFF), (0x3400, 0x4DBF)],
    "has_hiragana":       [(0x3040, 0x309F)],
    "has_katakana":       [(0x30A0, 0x30FF)],
    "has_hangul":         [(0xAC00, 0xD7AF), (0x1100, 0x11FF), (0x3130, 0x318F)],
    "has_thai":           [(0x0E00, 0x0E7F)],
    "has_lao":            [(0x0E80, 0x0EFF)],
    "has_georgian":       [(0x10A0, 0x10FF), (0x2D00, 0x2D2F)],
    "has_khmer":          [(0x1780, 0x17FF)],
    "has_myanmar":        [(0x1000, 0x109F), (0xAA60, 0xAA7F)],
    "has_ethiopic":       [(0x1200, 0x137F)],
    "has_tibetan":        [(0x0F00, 0x0FFF)],
    "has_armenian":       [(0x0530, 0x058F), (0xFB13, 0xFB17)],
}


def in_ranges(ch, ranges):
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in ranges)


def bytes_to_unicode_decoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(cs, bs))


def decode_token_string(raw, decoder):
    try:
        bs = bytes(decoder[c] for c in raw)
    except KeyError:
        return raw, False
    if bs and bs[0:1] == b" ":
        bs = bs[1:]
    try:
        return bs.decode("utf-8"), False
    except UnicodeDecodeError:
        return bs.decode("utf-8", errors="replace"), True


def build_token_metadata(tokenizer_repo="swiss-ai/Apertus-8B-2509"):
    import polars as pl
    from tokenizers import Tokenizer
    tok = Tokenizer.from_pretrained(tokenizer_repo)
    decoder = bytes_to_unicode_decoder()
    vocab = tok.get_vocab()
    id_to_str = [""] * tok.get_vocab_size()
    for s, i in vocab.items():
        id_to_str[i] = s

    decoded = []
    is_frag = []
    for s in id_to_str:
        d, frag = decode_token_string(s, decoder)
        decoded.append(d)
        is_frag.append(frag)

    rows = []
    for tok_id, dec in enumerate(decoded):
        row = {
            "token_id": tok_id,
            "decoded_string": dec,
            "byte_length": len(dec.encode("utf-8", errors="replace")),
            "is_byte_fragment": is_frag[tok_id],
        }
        for flag, ranges in SCRIPT_RANGES.items():
            row[flag] = any(in_ranges(c, ranges) for c in dec)
        row["is_pure_ascii"]      = bool(dec) and all(ord(c) < 128 for c in dec)
        row["is_pure_digits"]     = bool(dec) and dec.strip().isdigit()
        row["is_pure_whitespace"] = bool(dec) and all(c.isspace() for c in dec)
        if dec:
            row["is_structural_only"] = all(unicodedata.category(c)[0] in ("P", "S", "Z") for c in dec)
        else:
            row["is_structural_only"] = False
        row["is_special"] = dec.startswith("<") and dec.endswith(">") and len(dec) <= 32
        rows.append(row)

    return pl.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True, help="Root dir containing <worker_idx>/ subdirs with *.npy + *.summary.json")
    ap.add_argument("--lang-map", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    staging = Path(args.staging)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    lang_map = json.load(open(args.lang_map))

    # 1. Find all per-key histograms across all worker subdirs (FIX for reviewer #1)
    npy_files = sorted(staging.rglob("*.npy"))
    print(f"[aggregate] found {len(npy_files)} .npy files under {staging}")
    if not npy_files:
        sys.exit("no inputs found — aborting")

    # 2. Load + validate shape BEFORE appending the canonical-key (FIX for reviewer #5)
    canonical_keys = []
    summaries = {}
    histograms = []
    bad_shape = []
    zero_sum_keys = []

    for f in npy_files:
        ck = f.stem
        try:
            h = np.load(f)
        except Exception as e:
            print(f"  WARN load failed {ck}: {e}")
            bad_shape.append([ck, "load_error", str(e)])
            continue
        if h.shape != (131072,):
            print(f"  WARN {ck}: shape {h.shape} != (131072,)")
            bad_shape.append([ck, "shape", str(h.shape)])
            continue
        if h.dtype != np.int64:
            h = h.astype(np.int64)
        canonical_keys.append(ck)
        histograms.append(h)
        if int(h.sum()) == 0:
            zero_sum_keys.append(ck)
        sf = f.with_suffix(".summary.json")
        if sf.exists():
            try:
                summaries[ck] = json.loads(sf.read_text())
            except Exception:
                pass

    if not histograms:
        sys.exit("no valid histograms — aborting")

    H = np.stack(histograms, axis=0)
    print(f"[aggregate] H shape: {H.shape}, dtype: {H.dtype}")
    print(f"[aggregate] sum: {H.sum():,} tokens across all langs")
    print(f"[aggregate] zero-sum canonical keys: {len(zero_sum_keys)}")
    if bad_shape:
        print(f"[aggregate] bad-shape/load-error keys: {len(bad_shape)}")

    # 3. Save raw matrix + canonical-key index
    np.savez_compressed(out / "histogram_matrix.npz",
                        H=H,
                        canonical_keys=np.array(canonical_keys))
    sz = (out / "histogram_matrix.npz").stat().st_size
    print(f"[aggregate] wrote histogram_matrix.npz ({sz/1e9:.2f} GB compressed)")

    # 4. lang_metadata.json — merge canonical map + worker summary
    lang_meta_out = {}
    for i, ck in enumerate(canonical_keys):
        base = lang_map.get(ck, {})
        s = summaries.get(ck, {})
        lang_meta_out[ck] = {
            "row_index": i,
            "iso_639_3": base.get("iso_639_3"),
            "script_iso15924": base.get("script_iso15924"),
            "name": base.get("name"),
            "family": base.get("family"),
            "sources_in_map": list(base.get("sources", {}).keys()),
            "sources_contributed": s.get("sources_used", []),
            "sample_tokens_total": int(H[i].sum()),
            "vocab_entries_fired": int((H[i] > 0).sum()),
            "vocab_entries_fired_geq_10":  int((H[i] >= 10).sum()),
            "vocab_entries_fired_geq_100": int((H[i] >= 100).sum()),
            "wall_seconds": s.get("wall_seconds", 0),
        }
    (out / "lang_metadata.json").write_text(json.dumps(lang_meta_out, ensure_ascii=False, indent=2))
    print(f"[aggregate] wrote lang_metadata.json ({len(lang_meta_out)} entries)")

    # 5. token_metadata.parquet
    import polars as pl
    print("[aggregate] building token_metadata.parquet …")
    tok_meta = build_token_metadata()
    total_per_tok = H.sum(axis=0).astype(np.int64)
    langs_per_tok = (H > 0).sum(axis=0).astype(np.int32)
    langs_per_tok_geq10 = (H >= 10).sum(axis=0).astype(np.int32)
    tok_meta = tok_meta.with_columns([
        pl.Series("total_count_all_langs",         total_per_tok),
        pl.Series("total_langs_with_any_count",    langs_per_tok),
        pl.Series("total_langs_with_count_geq_10", langs_per_tok_geq10),
    ])
    tok_meta.write_parquet(out / "token_metadata.parquet")
    sz = (out / 'token_metadata.parquet').stat().st_size
    print(f"[aggregate] wrote token_metadata.parquet ({sz/1e6:.1f} MB)")

    # 6. zero-sum + bad-shape + missing report
    (out / "zero_sum_keys.json").write_text(json.dumps({
        "zero_sum_keys": zero_sum_keys,
        "bad_shape_keys": bad_shape,
        "n_canonical_total_in_map": len(lang_map),
        "n_aggregated": len(canonical_keys),
        "n_missing": len(lang_map) - len(canonical_keys),
        "missing_keys": sorted(set(lang_map.keys()) - set(canonical_keys)),
    }, indent=2))
    print(f"[aggregate] wrote zero_sum_keys.json")
    print("[aggregate] DONE.")


if __name__ == "__main__":
    main()
