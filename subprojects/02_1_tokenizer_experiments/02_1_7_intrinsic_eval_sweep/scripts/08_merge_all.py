"""Merge ALL data sources into the long-format frame:
  - TokEval results (from 04_aggregate.py — Apertus-55 multilingual + Greek)
  - In-house held-out fertility from gcloud (13 raw + 2 curated × 4-8 slices)
  - MorphScore Greek (15 variants)

Output: artifacts/results_merged.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SSP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
)
ART = SSP / "artifacts"
OUT = ART / "results_merged.parquet"

# Mapping from our_suite metric names → standardized + tier
OUR_SUITE_METRIC_TIERS = {
    "chars_per_token": "T1",
    "tokens_per_byte": "T1",
    "bytes_per_token": "T1",
    "greek_word_space_fertility": "T1",  # the headline Greek metric
    "tokens_per_greek_word_in_context": "T1",
    "single_token_greek_word_share": "T3",
    "added_token_rate": "T1",
    "eval_added_vocab_utilization_rate": "T1",  # **the** curated-utilization
    "eval_unused_added_tokens": "T3",
    "vocab_utilization_rate": "T1",
    "unk_rate": "T3",
    "byte_fallback_rate": "T3",
    "tokens_per_100_chars": "T3",
    "unique_token_ids": "T3",
    "unique_added_token_ids": "T3",
    "added_token_count": "T3",
}

VARIANT_RENAME = {"apertus_base": "apertus_base"}
for _n in [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
           11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456,
           20480, 21504, 22528, 23552, 24576, 25600]:
    VARIANT_RENAME[f"c3_added_{_n}"] = f"add_{_n}"
for _n in [11264, 12288, 15360, 17408, 20480, 25600]:
    VARIANT_RENAME[f"c3_added_{_n}_curated"] = f"add_{_n}_curated"


def load_our_suite() -> pd.DataFrame:
    """In-house Greek held-out metrics from gcloud."""
    dfs = []
    for fn in ("metrics_by_slice.csv", "curated_metrics_by_slice.csv", "extended_metrics_by_slice.csv"):
        p = ART / "our_suite_raw_gcloud" / fn
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        return pd.DataFrame()
    raw = pd.concat(dfs, ignore_index=True)
    # Normalize column → metric rows
    id_cols = ["tokenizer", "slice", "kind"]
    metric_cols = [c for c in raw.columns if c not in id_cols + ["docs", "chars", "utf8_bytes", "greek_words", "tokens", "elapsed_seconds", "chars_per_second"]]
    long = raw.melt(id_vars=["tokenizer", "slice"],
                    value_vars=metric_cols,
                    var_name="metric", value_name="value")
    long = long[pd.to_numeric(long["value"], errors="coerce").notna()]
    long["value"] = long["value"].astype(float)
    long["variant_id"] = long["tokenizer"].map(VARIANT_RENAME).fillna(long["tokenizer"])
    long["source"] = "our_suite_02_1_3"
    long["language"] = "ell_Grek"  # all in-house held-outs are Greek
    long["tier"] = long["metric"].map(OUR_SUITE_METRIC_TIERS).fillna("T3")
    long["curated"] = long["variant_id"].str.endswith("_curated")
    long["added_tokens"] = long["variant_id"].str.replace("_curated", "").str.replace("add_", "").str.replace("apertus_base", "0").astype(int)
    return long[["variant_id", "added_tokens", "curated", "language", "slice",
                 "metric", "value", "source", "tier"]]


def load_morphscore() -> pd.DataFrame:
    p = ART / "morphscore_greek_results.json"
    if not p.exists():
        return pd.DataFrame()
    data = json.loads(p.read_text())
    rows = []
    for r in data:
        m = r["metrics"]
        # MorphScore returns metrics keyed by language code; with one
        # language we may have either {ell_Grek: {...}} or flat dict
        if "ell_Grek" in m:
            metrics = m["ell_Grek"]
        else:
            metrics = m
        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            tier = "T2" if "morphscore" in metric_name else "T3"
            rows.append({
                "variant_id": r["variant_id"],
                "added_tokens": r["added_tokens"],
                "curated": r["curated"],
                "language": "ell_Grek",
                "slice": "morphscore_ud",
                "metric": metric_name,
                "value": float(value),
                "source": "morphscore",
                "tier": tier,
            })
    return pd.DataFrame(rows)


def load_tokeval() -> pd.DataFrame:
    p = ART / "results.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def main() -> None:
    parts = []
    tev = load_tokeval()
    if not tev.empty:
        print(f"tokeval        : {len(tev):,} rows")
        parts.append(tev)
    ours = load_our_suite()
    if not ours.empty:
        print(f"02_1_3 harness : {len(ours):,} rows")
        parts.append(ours)
    morph = load_morphscore()
    if not morph.empty:
        print(f"morphscore     : {len(morph):,} rows")
        parts.append(morph)
    df = pd.concat(parts, ignore_index=True, sort=False)
    df.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(df):,} total rows)")
    print(f"  variants: {sorted(df.variant_id.unique())}")
    print(f"  sources : {sorted(df.source.unique())}")
    print(f"  slices  : {sorted(df.slice.unique())}")


if __name__ == "__main__":
    main()
