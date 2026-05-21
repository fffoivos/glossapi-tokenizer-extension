#!/usr/bin/env python3
"""Coordinator step 09: aggregate per-pair joins → per_hf_pool_doc_overlap +
per_hf_pool_source_actionable + summary_matrix + REPORT.md.

**Scope (per review r4)**: this audit measures the **HF source pool** vs
Apertus, NOT the exact C3 sampled mix. Output column names reflect this.

Per the audit's tier rules (PLAN §3.6):
- strict_exact whole-doc match → overlap_ratio = 1.0
- relaxed_exact whole-doc match → 1.0
- near whole-doc match (Jaccard ≥ 0.85) → 1.0
- sentence-level matches → matched_chars/total_doc_chars (deferred to a
  separate run)

Tier thresholds emit the THREE sensitivity grid points (strict / default / lenient).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

SUB = Path(__file__).resolve().parents[2]
RUN_ID = (SUB / "manifests/CURRENT_RUN_ID").read_text().strip()
ART = SUB / "artifacts" / RUN_ID
SRC = ART / "sources"
OVL = ART / "overlap"
REPORT = SUB / f"REPORT_{RUN_ID}.md"
HOLDOUT_PATH = ART / "holdout_contamination.parquet"
NEAR_FAILED_MARKER = ART / "near_overlap_FAILED.txt"

SENSITIVITY_GRID = [
    {"name": "strict",  "drop_ge": 0.10, "partial_ge": 0.02},
    {"name": "default", "drop_ge": 0.30, "partial_ge": 0.05},
    {"name": "lenient", "drop_ge": 0.50, "partial_ge": 0.10},
]


def tier_for(ratio: float, drop_ge: float, partial_ge: float) -> str:
    if ratio >= drop_ge:
        return "drop"
    if ratio >= partial_ge:
        return "partial"
    return "trace"


def main() -> int:
    overlap_rows: list[dict] = []
    for stage_dir, stage in (("strict_exact", "strict_exact"),
                              ("relaxed_exact", "relaxed_exact"),
                              ("near", "near")):
        d = OVL / stage_dir
        if not d.exists():
            continue
        for pf in sorted(d.glob("*.parquet")):
            apertus_src = pf.stem.split("_x_")[0]
            df = pl.read_parquet(pf)
            if df.height == 0:
                continue
            c_key_col = "c_doc_key" if "c_doc_key" in df.columns else "hf_pool_doc_key"
            for row in df.iter_rows(named=True):
                overlap_rows.append({
                    "hf_pool_doc_key": row[c_key_col],
                    "hf_pool_source_dataset": row.get("c_source_dataset") or row.get("hf_pool_source_dataset"),
                    "apertus_source": apertus_src,
                    "stage": stage,
                    "overlap_ratio": 1.0,
                    "estimated_jaccard": row.get("estimated_jaccard"),
                })
    if not overlap_rows:
        print("[report] no overlap rows found; writing minimal report")
        REPORT.write_text(_minimal_report_text(skipped_holdout=not HOLDOUT_PATH.exists()))
        return 0

    overlap = pl.DataFrame(overlap_rows, schema={
        "hf_pool_doc_key": pl.Utf8,
        "hf_pool_source_dataset": pl.Utf8,
        "apertus_source": pl.Utf8,
        "stage": pl.Utf8,
        "overlap_ratio": pl.Float64,
        "estimated_jaccard": pl.Float64,
    })
    print(f"[report] {overlap.height} raw overlap rows across all pairs")

    per_doc = overlap.group_by(["hf_pool_doc_key", "hf_pool_source_dataset"]).agg(
        pl.col("overlap_ratio").max().alias("overlap_ratio"),
        pl.col("apertus_source").first().alias("best_match_a_source"),
        pl.col("stage").first().alias("best_match_stage"),
        pl.col("estimated_jaccard").max().alias("best_estimated_jaccard"),
    )

    sens_blocks: dict[str, pl.DataFrame] = {}
    for s in SENSITIVITY_GRID:
        tiered = per_doc.with_columns(
            pl.col("overlap_ratio").map_elements(
                lambda r, dg=s["drop_ge"], pg=s["partial_ge"]: tier_for(r, dg, pg),
                return_dtype=pl.Utf8,
            ).alias(f"tier_{s['name']}")
        )
        sens_blocks[s["name"]] = tiered

    combined = per_doc
    for name, df in sens_blocks.items():
        combined = combined.with_columns(df[f"tier_{name}"])
    combined.write_parquet(ART / "per_hf_pool_doc_overlap.parquet", compression="zstd")
    print(f"[report] per_hf_pool_doc_overlap.parquet: {combined.height:,} matched docs")

    # Per-source actionable. The source tables contain hashes/metadata rather
    # than raw text, so this is expected to fit on the joins worker.
    c_files = sorted(SRC.glob("hf_source_pool_*.parquet"))
    universe = pl.concat([pl.read_parquet(p, columns=["doc_key", "source_dataset", "text_length"])
                          for p in c_files], how="vertical") if c_files else pl.DataFrame()
    print(f"[report] HF source pool universe: {universe.height:,} docs across {len(c_files)} sources")

    actionable_rows = []
    if universe.height:
        for (src_name,), src_group in universe.group_by("source_dataset"):
            tiered = sens_blocks["default"].filter(pl.col("hf_pool_source_dataset") == src_name)
            seen = set(tiered.filter(pl.col("tier_default") == "drop")["hf_pool_doc_key"].to_list())
            partial = set(tiered.filter(pl.col("tier_default") == "partial")["hf_pool_doc_key"].to_list())
            all_docs = set(src_group["doc_key"].to_list())
            fresh = all_docs - seen - partial
            fresh_chars = int(src_group.filter(pl.col("doc_key").is_in(list(fresh)))["text_length"].sum() or 0)
            fresh_share = (len(fresh) / max(len(all_docs), 1))
            actionable_rows.append({
                "hf_pool_source": src_name,
                "total_rows": len(all_docs),
                "seen_rows": len(seen),
                "partial_rows": len(partial),
                "fresh_rows": len(fresh),
                "fresh_chars": fresh_chars,
                "fresh_share": fresh_share,
                "recommended_action": (
                    "include_full" if fresh_share > 0.7
                    else "include_half_weight" if fresh_share > 0.3
                    else "replay_only"
                ),
            })
        pl.DataFrame(actionable_rows).write_parquet(
            ART / "per_hf_pool_source_actionable.parquet", compression="zstd")
        print(pl.DataFrame(actionable_rows))

    # Summary matrix.
    rows = []
    for stage_dir, stage in (("strict_exact", "strict_exact"),
                              ("relaxed_exact", "relaxed_exact"),
                              ("near", "near")):
        d = OVL / stage_dir
        if not d.exists():
            continue
        for pf in sorted(d.glob("*.parquet")):
            df = pl.read_parquet(pf)
            rows.append({"stage": stage, "pair": pf.stem, "match_rows": df.height})
    if rows:
        pl.DataFrame(rows).write_parquet(ART / "summary_matrix.parquet", compression="zstd")

    # REPORT.md.
    holdout_present = HOLDOUT_PATH.exists()
    holdout_rows = 0
    if holdout_present:
        try:
            holdout_rows = pl.read_parquet(HOLDOUT_PATH).height
        except Exception:
            holdout_rows = 0

    lines = [
        f"# Dedup audit REPORT — {RUN_ID}",
        "",
        f"## Scope",
        "",
        f"This audit measures **HF source-pool overlap with Apertus pretraining**, "
        f"NOT the exact 1:1 sampled C3 training mix. Per PLAN §2 / review r4 reframing: "
        f"the C3 mix was sampled from this HF source pool; the load-bearing CPT-replay "
        f"recipe would need a separate `c3_exact_mix_overlap` audit run that reads the "
        f"C3 mix manifest (currently held on the TERMINATED gcloud instance).",
        "",
        f"**Greek diacritic policy**: `preserve` (USER DECISION 2026-05-18).",
        f"**HF source-pool docs**: {universe.height:,}",
        f"**Matched HF source-pool docs**: {combined.height:,}",
        f"**Apertus sources audited**: FW2-HQ ell_Grek, Clean-Wikipedia el, EuroParl Greek (20 bitexts), EuroBlocks Greek",
        f"**Long-context phase**: excluded (FineWeb-Long + Institutional Books not measured).",
        f"**Held-out contamination check**: {'INCLUDED — see holdout_contamination.parquet (' + str(holdout_rows) + ' rows)' if holdout_present else '**SKIPPED** — no `manifests/run_<RUN_ID>/holdout_doc_ids.parquet` was provided. This report **does NOT** verify C3 val/test integrity. Re-run step 08 with a holdout doc-id list to add eval-contamination coverage.'}",
        f"**Near-dup (MinHash LSH) overlap**: " + (
            "**FAILED — step 07 exited nonzero (likely OOM). Exact-overlap results above are unaffected; re-run step 07 on a larger joins worker (`JW_MACHINE=c4-highmem-64`) to add near-dup coverage.**"
            if NEAR_FAILED_MARKER.exists()
            else "included (per-pair counts below)."
        ),
        "",
        "## Per-HF-pool-source actionable (default sensitivity)",
        "",
        "| hf_pool_source | total | seen | partial | fresh | fresh_share | recommendation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in actionable_rows:
        lines.append(f"| {r['hf_pool_source']} | {r['total_rows']:,} | {r['seen_rows']:,} | "
                     f"{r['partial_rows']:,} | {r['fresh_rows']:,} | "
                     f"{r['fresh_share']:.3f} | {r['recommended_action']} |")

    lines += [
        "",
        "## Sensitivity grid",
        "",
        "| setting | drop ≥ | partial ≥ | total drop | total partial | total fresh |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for s in SENSITIVITY_GRID:
        tiered = sens_blocks[s["name"]]
        total_drop = (tiered[f"tier_{s['name']}"] == "drop").sum()
        total_partial = (tiered[f"tier_{s['name']}"] == "partial").sum()
        total_fresh = universe.height - total_drop - total_partial
        lines.append(f"| {s['name']} | {s['drop_ge']:.2f} | {s['partial_ge']:.2f} | "
                     f"{total_drop:,} | {total_partial:,} | {total_fresh:,} |")

    lines += [
        "",
        "## Per-pair summary matrix",
        "",
        "| stage | pair | match_rows |",
        "|---|---|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['stage']} | {r['pair']} | {r['match_rows']:,} |")

    lines += [
        "",
        "## Methodology pins",
        "",
        "- `text_dedup.py` commit: `9a6b039`; file-hash `6b9bfdb0bd9923349c348f80866c472101ab8fcf`",
        "- `greek_diacritic_policy`: `preserve`",
        "- 128-perm MinHash, token 5-shingles, Jaccard ≥ 0.85, skip-short-doc < 20 tokens",
        "- Workers used **canonical** `text_dedup.hash_bytes` (full 64-char blake3 hex) and `text_dedup.minhash_signature` — no reimplementation",
        "",
        "## What this audit does NOT cover",
        "",
        "- Exact C3 sampled mix overlap (load-bearing for CPT replay recipe) — separate run needed",
        "- Long-context phase Greek (FineWeb-Long + Institutional Books) — separate run needed",
        f"- {'Held-out C3 val/test contamination — separate run needed (this report had no holdout list)' if not holdout_present else 'N/A'}",
        *([
            "- **Near-dup (MinHash LSH) overlap — step 07 FAILED in this run.** Above sensitivity-grid / actionable numbers exclude near-dup matches. Re-run on `JW_MACHINE=c4-highmem-64` to add."
        ] if NEAR_FAILED_MARKER.exists() else []),
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"[report] wrote {REPORT}")
    return 0


def _minimal_report_text(skipped_holdout: bool) -> str:
    return (
        f"# Dedup audit REPORT — {RUN_ID}\n\n"
        f"No overlap rows produced. Either workers found no matches (suspicious — investigate) "
        f"or the join inputs were empty.\n\n"
        f"Holdout check {'SKIPPED' if skipped_holdout else 'attempted'}.\n"
    )


if __name__ == "__main__":
    sys.exit(main())
