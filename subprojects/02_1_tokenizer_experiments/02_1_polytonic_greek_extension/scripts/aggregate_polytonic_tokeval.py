#!/usr/bin/env python3
"""Aggregate TokEval JSON outputs for the polytonic cutoff sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

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
}


def emit_value(rows: list[dict[str, object]], tok: str, lang: str, slice_id: str, metric: str, value: object, source: str, tier: str) -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        rows.append(
            {
                "variant_id": tok,
                "language": lang,
                "slice": slice_id,
                "metric": metric,
                "value": float(value),
                "source": source,
                "tier": tier,
            }
        )
        return
    if not isinstance(value, dict):
        return
    if "mean" in value and isinstance(value["mean"], (int, float)):
        emit_value(rows, tok, lang, slice_id, metric, value["mean"], source, tier)
        for sub in ("median", "std"):
            if isinstance(value.get(sub), (int, float)):
                emit_value(rows, tok, lang, slice_id, f"{metric}__{sub}", value[sub], source, "T3")
        return
    for sub, sub_value in value.items():
        if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
            sub_metric = f"{metric}__{sub}"
            emit_value(rows, tok, lang, slice_id, sub_metric, sub_value, source, tier)
            if sub == PRIMARY_SUBKEY.get(metric):
                emit_value(rows, tok, lang, slice_id, metric, sub_value, source, tier)


def parse_results(path: Path, source: str, slice_id: str, rows: list[dict[str, object]]) -> None:
    if not path.exists():
        print(f"warning: missing {path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    for metric_name, metric_block in data.items():
        if not isinstance(metric_block, dict):
            continue
        tier = TIER_MAP.get(metric_name, "T3")
        for tok, tok_value in metric_block.get("per_tokenizer", {}).items():
            if isinstance(tok_value, (int, float)) and not isinstance(tok_value, bool):
                emit_value(rows, tok, "global", slice_id, metric_name, tok_value, source, tier)
                continue
            if not isinstance(tok_value, dict):
                continue
            if "global" in tok_value:
                emit_value(rows, tok, "global", slice_id, metric_name, tok_value["global"], source, tier)
            per_lang = tok_value.get("per_language")
            if isinstance(per_lang, dict):
                for lang, lang_value in per_lang.items():
                    emit_value(rows, tok, lang, slice_id, metric_name, lang_value, source, tier)
        sibling_per_lang = metric_block.get("per_language")
        if isinstance(sibling_per_lang, dict):
            for tok, langs in sibling_per_lang.items():
                if not isinstance(langs, dict):
                    continue
                for lang, lang_value in langs.items():
                    emit_value(rows, tok, lang, slice_id, metric_name, lang_value, source, tier)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokeval-raw-dir", type=Path, required=True)
    parser.add_argument("--tokenizers-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    parse_results(
        args.tokeval_raw_dir / "job1_tfg_apertus55/analysis_results.json",
        source="tokeval-lines",
        slice_id="flores_plus_55",
        rows=rows,
    )
    parse_results(
        args.tokeval_raw_dir / "job2_perlang_apertus55_words/analysis_results.json",
        source="tokeval-words",
        slice_id="flores_plus_55",
        rows=rows,
    )
    parse_results(
        args.tokeval_raw_dir / "job3_greek_only_words/analysis_results.json",
        source="tokeval-words",
        slice_id="flores_plus_ell_Grek",
        rows=rows,
    )
    meta = {item["name"]: item for item in json.loads(args.tokenizers_meta.read_text(encoding="utf-8"))["tokenizers"]}
    rows = [row for row in rows if row["variant_id"] in meta]
    for row in rows:
        item = meta[row["variant_id"]]
        row["polytonic_added_count"] = item["polytonic_added_count"]
        row["final_vocab_size"] = item["final_vocab_size"]
        row["tokenizer_sha256"] = item.get("tokenizer_sha256")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(args.output_dir / "tokeval_metrics.parquet", index=False)
        df.to_csv(args.output_dir / "tokeval_metrics.csv", index=False)
    write_json(args.output_dir / "tokeval_metrics.json", rows)
    write_json(
        args.output_dir / "summary.json",
        {
            "rows": len(rows),
            "variants": sorted(df["variant_id"].unique().tolist()) if not df.empty else [],
            "metrics": sorted(df["metric"].unique().tolist()) if not df.empty else [],
            "languages": sorted(df["language"].unique().tolist()) if not df.empty else [],
            "sources": sorted(df["source"].unique().tolist()) if not df.empty else [],
        },
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
