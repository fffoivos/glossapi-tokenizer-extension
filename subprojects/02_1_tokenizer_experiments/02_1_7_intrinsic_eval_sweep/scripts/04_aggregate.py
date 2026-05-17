"""Aggregate TokEval raw outputs into one long-format frame.

Schema discovered from analysis_results.json:

  metric_name:
    per_tokenizer:
      <tokenizer>:
        global: {mean, median, std, count}        # corpus aggregate
        per_language: { <lang>: {mean,...} }      # per-language breakdown
    per_language:                                 # some metrics also have this
      <tokenizer>: { <lang>: value_or_dict }
    metadata: {...}

Output schema (artifacts/results.parquet):
  variant_id    str
  added_tokens  int
  curated       bool
  language      str    # 'global' for corpus-aggregate
  slice         str    # the eval-set name (flores_plus_55_lines / _words)
  metric        str
  value         float
  source        str    # tokeval-lines / tokeval-words
  tier          str    # T1 / T2 / T3
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SSP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
)
TOKEVAL_RAW = SSP / "artifacts/tokeval_raw"
OUT_PARQUET = SSP / "artifacts/results.parquet"
MANIFEST_OUT = SSP / "manifests/per_cutoff_metrics.json"
TOKENIZERS_META = SSP / "configs/cutoff_sweep_tokenizers_meta.json"

# Mapping from TokEval metric name → tier
# Primary subkey for metrics whose `global` value is a dict-of-scalars rather
# than a single number — this lets us refer to the metric by its bare name
# in plots/reports while still keeping every subkey row in the parquet.
PRIMARY_SUBKEY = {
    "tokenizer_fairness_gini": "gini_coefficient",
    "renyi_efficiency": "renyi_2.5",
    "utf8_token_integrity": "completeness_rate",
    "utf8_char_split": "char_split_rate",
    "vocabulary_utilization": "utilization",
}

TIER_MAP = {
    "fertility": "T1",
    "compression_rate": "T1",
    "vocabulary_utilization": "T1",
    "tokenizer_fairness_gini": "T1",
    "renyi_efficiency": "T2",
    "utf8_token_integrity": "T2",
    "utf8_char_split": "T2",
    "lorenz_curve_data": "T1",
    "token_length": "T3",
    "type_token_ratio": "T3",
    "avg_tokens_per_line": "T3",
    "reconstruction_fidelity": "T3",
    "encoding_speed": "T3",
    "bigram_entropy": "T3",
    "unigram_distribution_metrics": "T3",
    "three_digit_boundary_alignment": "T3",
    "digit_split_variability": "T3",
    "numeric_magnitude_consistency": "T3",
    "operator_isolation_rate": "T3",
    "ast_boundary_alignment": "T3",
    "identifier_fragmentation": "T3",
    "indentation_consistency": "T3",
}


def _flatten_value(v):
    """If v is a dict with 'mean', return v['mean']. Else if numeric, return v.
    Otherwise yield (sub_key, sub_value) pairs.
    """
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        yield "value", float(v)
    elif isinstance(v, dict):
        if "mean" in v and isinstance(v["mean"], (int, float)):
            yield "mean", float(v["mean"])
            if "median" in v: yield "median", float(v["median"])
            if "std" in v: yield "std", float(v["std"])
        else:
            for k, sub in v.items():
                if isinstance(sub, (int, float)) and not isinstance(sub, bool):
                    yield k, float(sub)


def _emit_value(rows, tok, lang, slice_id, metric, value, source, tier):
    """Given a (possibly nested dict) value, emit one row per scalar leaf.
    Naming convention: if value is `{mean, median, std}`, emit base metric =
    mean and "<metric>__median" / "<metric>__std" as Tier-3 detail. If
    value is a dict of named scalars (e.g. `{gini_coefficient: 0.11, ...}`),
    emit one row per scalar with name `<metric>__<key>` (and choose the
    "primary" subkey to ALSO be emitted as the bare `<metric>` for plot
    convenience — see PRIMARY_SUBKEY below).
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append({
            "variant_id": tok, "language": lang, "slice": slice_id,
            "metric": metric, "value": float(value),
            "source": source, "tier": tier,
        })
        return
    if not isinstance(value, dict):
        return
    # mean/median/std (per-document distribution stats)
    if "mean" in value and isinstance(value["mean"], (int, float)):
        rows.append({
            "variant_id": tok, "language": lang, "slice": slice_id,
            "metric": metric, "value": float(value["mean"]),
            "source": source, "tier": tier,
        })
        for sub in ("median", "std"):
            if sub in value and isinstance(value[sub], (int, float)):
                rows.append({
                    "variant_id": tok, "language": lang, "slice": slice_id,
                    "metric": f"{metric}__{sub}", "value": float(value[sub]),
                    "source": source, "tier": "T3",
                })
        return
    # Rich-dict layout (named scalar leaves)
    for sub, sv in value.items():
        if isinstance(sv, (int, float)) and not isinstance(sv, bool):
            sub_metric = f"{metric}__{sub}"
            rows.append({
                "variant_id": tok, "language": lang, "slice": slice_id,
                "metric": sub_metric, "value": float(sv),
                "source": source, "tier": tier,
            })
            # If this is the canonical subkey for the metric, also emit the
            # bare metric name (lets the report plot work without per-metric
            # suffix lookup)
            if sub == PRIMARY_SUBKEY.get(metric):
                rows.append({
                    "variant_id": tok, "language": lang, "slice": slice_id,
                    "metric": metric, "value": float(sv),
                    "source": source, "tier": tier,
                })


def parse_results(path: Path, source: str, slice_id: str, rows: list) -> None:
    if not path.exists():
        return
    data = json.loads(path.read_text())
    for metric_name, mblock in data.items():
        if not isinstance(mblock, dict):
            continue
        tier = TIER_MAP.get(metric_name, "T3")
        per_tok = mblock.get("per_tokenizer", {})
        for tok, tval in per_tok.items():
            if isinstance(tval, (int, float)) and not isinstance(tval, bool):
                _emit_value(rows, tok, "global", slice_id, metric_name,
                            tval, source, tier)
                continue
            if not isinstance(tval, dict):
                continue
            # Global aggregate
            if "global" in tval:
                _emit_value(rows, tok, "global", slice_id, metric_name,
                            tval["global"], source, tier)
            # Per-language
            pl = tval.get("per_language")
            if isinstance(pl, dict):
                for lang, lv in pl.items():
                    _emit_value(rows, tok, lang, slice_id, metric_name,
                                lv, source, tier)
            # Direct scalar siblings under per_tokenizer[tok]
            for k, v in tval.items():
                if k in ("global", "per_language"):
                    continue
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    rows.append({
                        "variant_id": tok, "language": "global",
                        "slice": slice_id, "metric": f"{metric_name}__{k}",
                        "value": float(v), "source": source, "tier": tier,
                    })
        # Sibling per_language block
        pl2 = mblock.get("per_language", {})
        if isinstance(pl2, dict):
            for tok, langs in pl2.items():
                if not isinstance(langs, dict):
                    continue
                for lang, lv in langs.items():
                    _emit_value(rows, tok, lang, slice_id, metric_name,
                                lv, source, tier)


def main() -> None:
    rows: list[dict] = []
    # Prefer v2 (extended sweep up to 25,600) if present, else v1 (up to 12k)
    v2 = SSP / "artifacts/tokeval_raw_v2"
    v1 = TOKEVAL_RAW
    root = v2 if (v2 / "job1_tfg_apertus55/analysis_results.json").exists() else v1
    parse_results(
        root / "job1_tfg_apertus55/analysis_results.json",
        source="tokeval-lines", slice_id="flores_plus_55", rows=rows,
    )
    parse_results(
        root / "job2_perlang_apertus55_words/analysis_results.json",
        source="tokeval-words", slice_id="flores_plus_55", rows=rows,
    )

    if not rows:
        print("ERROR: no rows aggregated")
        return

    meta = {t["name"]: t for t in
            json.loads(TOKENIZERS_META.read_text())["tokenizers"]}
    valid_tokenizers = set(meta.keys())
    # Filter out non-tokenizer variant_ids (sibling per_language blocks
    # often nest by language code or sub-metric name, not tokenizer name)
    rows = [r for r in rows if r["variant_id"] in valid_tokenizers]
    for r in rows:
        m = meta.get(r["variant_id"], {})
        r["added_tokens"] = m.get("added_tokens", -1)
        r["curated"] = m.get("curated", False)

    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET}  ({len(df):,} rows)")
    print(f"  variants : {df.variant_id.nunique()}")
    print(f"  metrics  : {df.metric.nunique()}")
    print(f"  languages: {df.language.nunique()}")
    print(f"  sources  : {sorted(df.source.unique())}")
    print(f"  tiers    : {dict(df.groupby('tier').size())}")

    # Slim canonical manifest: T1 + T2 only
    canonical = df[df.tier.isin(["T1", "T2"])].copy()
    summary: dict = {}
    for _, row in canonical.iterrows():
        key = f"{row.variant_id}|{row.language}|{row.slice}|{row.metric}|{row.source}"
        summary[key] = row.value
    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    tk_commit = (SSP / "manifests/tokeval_commit.txt").read_text().strip().splitlines()[-1] if (SSP / "manifests/tokeval_commit.txt").exists() else "unknown"
    MANIFEST_OUT.write_text(json.dumps({
        "tokeval_commit": tk_commit,
        "tier_T1_T2_rows": int(len(canonical)),
        "all_rows": int(len(df)),
        "variants": sorted(df.variant_id.unique().tolist()),
        "languages_sample": sorted(df.language.unique().tolist())[:20],
        "metrics": sorted(df.metric.unique().tolist()),
        "sources": sorted(df.source.unique().tolist()),
        "values": summary,
    }, indent=2, ensure_ascii=False))
    print(f"wrote {MANIFEST_OUT}  (T1+T2 = {len(canonical):,} canonical rows)")


if __name__ == "__main__":
    main()
