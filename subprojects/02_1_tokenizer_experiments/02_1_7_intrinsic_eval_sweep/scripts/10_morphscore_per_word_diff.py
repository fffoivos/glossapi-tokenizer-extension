"""DIAGNOSTIC-ONLY (one-off): re-run MorphScore on two cutoffs and diff
per-word results.

Not part of run_all.sh. Was used once to explain the MorphScore recall
discontinuity at cutoff 10,240 (and indirectly the 17,408 milestone
referenced in REPORT.md / CHOSEN_CUTOFF.md). Output of that one-off
run lives at artifacts/morphscore_per_word_diff_8k_vs_10k.csv.

Goal: explain the discontinuity at cutoff 10,240. Run for the 8,192 and
10,240 variants with return_df=True; find words whose contribution to
recall/precision changed between cutoffs; look at the actual word splits.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

SSP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
)
sys.path.insert(0, str(SSP / "vendor/tokenizer-intrinsic-evals"))

# Same monkey-patch as 07
from morphscore import morphscore as _ms

def _patched_encode_text(tokenizer, text, add_special_tokens=False):
    enc = tokenizer(text, add_special_tokens=add_special_tokens)
    ids = None
    try: ids = enc["input_ids"]
    except (KeyError, TypeError): pass
    if ids is None and hasattr(enc, "input_ids"): ids = enc.input_ids
    if ids is None and hasattr(enc, "ids"): ids = enc.ids
    if ids is None and isinstance(enc, list): ids = enc
    if ids is None: raise ValueError(f"format: {type(enc)} - {enc}")
    if isinstance(ids, list) and ids and isinstance(ids[0], list): return ids[0]
    return list(ids)
_ms.encode_text = _patched_encode_text

from morphscore.morphscore import MorphScore
from transformers import AutoTokenizer

DATA_DIR = SSP / "vendor/tokenizer-intrinsic-evals/morphscore_data"
VARIANTS = {
    "add_8192":  SSP / "variants/c3_added_8192",
    "add_10240": SSP / "variants/c3_added_10240",
}


def run_with_df(name: str, path: Path):
    tok = AutoTokenizer.from_pretrained(str(path))
    morph = MorphScore(language_subset=["ell_Grek"], by_split=False,
                       freq_scale=True, exclude_single_tok=False,
                       data_dir=str(DATA_DIR))
    result, df = morph.eval(tok, return_df=True)
    return result, df


def main() -> None:
    rs = {}
    dfs = {}
    for name, path in VARIANTS.items():
        print(f"Running MorphScore for {name}...")
        r, df = run_with_df(name, path)
        rs[name] = r
        dfs[name] = df
        ms = r.get("ell_Grek", r)
        print(f"  recall={ms.get('morphscore_recall'):.4f}  "
              f"precision={ms.get('morphscore_precision'):.4f}  n={len(df)}")
    print()
    print("=== column shapes ===")
    for k, df in dfs.items():
        print(f"  {k}: cols={list(df.columns)}, rows={len(df)}")
    print()
    # Diff
    df_low = dfs["add_8192"].copy()
    df_hi = dfs["add_10240"].copy()
    # Try common merge key
    key_candidates = ["wordform", "word", "token_ids", "lemma"]
    key = next((k for k in key_candidates if k in df_low.columns), None)
    if key is None:
        print("ERR: no obvious merge key; cols are:", list(df_low.columns))
        return
    print(f"merging on '{key}'")
    merged = df_low.merge(df_hi, on=key, how="outer", suffixes=("_8k", "_10k"))
    # Find numeric columns common to both
    candidate_metrics = [c for c in df_low.columns
                         if pd.api.types.is_numeric_dtype(df_low[c])
                         and c in df_hi.columns and c != key]
    print(f"per-word numeric metrics: {candidate_metrics}")
    # For each metric, find the words with the biggest delta
    for metric in candidate_metrics:
        c_low, c_hi = f"{metric}_8k", f"{metric}_10k"
        if c_low not in merged.columns or c_hi not in merged.columns:
            continue
        merged[f"{metric}_delta"] = merged[c_hi] - merged[c_low]
        d = merged[merged[f"{metric}_delta"].abs() > 1e-9].sort_values(f"{metric}_delta")
        if d.empty: continue
        print()
        print(f"=== words where {metric} CHANGED most (10k vs 8k) ===")
        cols_show = [key, "stem_8k" if "stem_8k" in d.columns else None,
                     "preceding_part_8k", "following_part_8k",
                     c_low, c_hi, f"{metric}_delta"]
        cols_show = [c for c in cols_show if c is not None and c in d.columns]
        print("Worst (recall/precision went DOWN at 10k):")
        print(d.head(10)[cols_show].to_string(index=False))
        print()
        print("Best (recall/precision went UP at 10k):")
        print(d.tail(10)[cols_show].to_string(index=False))
    # Write merged frame
    out = SSP / "artifacts/morphscore_per_word_diff_8k_vs_10k.csv"
    merged.to_csv(out, index=False)
    print(f"\nfull per-word diff written to {out}")


if __name__ == "__main__":
    main()
