"""Render the final REPORT.md from results_merged.parquet.

Sections:
  - Headline (4-metric combined plot + table)
  - Tier 1 paper-aligned on Apertus-55 FLORES+ (the "paper-comparable" view)
  - Tier 1 in-domain Greek on real held-outs (the "actual cutoff signal")
  - Tier 2 (MorphScore + Rényi + UTF-8)
  - Curated-arm delta
  - Methodology delta (TokEval vs 02_1_3 on Greek)
  - Reproduction
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SSP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
)
PARQUET = SSP / "artifacts/results_merged.parquet"
PLOTS_DIR = SSP / "artifacts/plots"
REPORT_PATH = SSP / "REPORT.md"

VARIANT_ORDER = ["apertus_base"]
for _n in [1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192, 9216, 10240,
           11264, 12288, 13312, 14336, 15360, 16384, 17408, 18432, 19456,
           20480, 21504, 22528, 23552, 24576, 25600]:
    VARIANT_ORDER.append(f"add_{_n}")
    if _n in (11264, 12288, 15360, 17408, 20480, 25600):
        VARIANT_ORDER.append(f"add_{_n}_curated")


def md_table(df, float_fmt="{:.4f}"):
    if df is None or df.empty:
        return "_(no data)_"
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, bool):
                cells.append("✓" if v else "")
            elif isinstance(v, float):
                if pd.isna(v):
                    cells.append("")
                else:
                    cells.append(float_fmt.format(v))
            elif pd.isna(v):
                cells.append("")
            else:
                cells.append(str(v))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + body)


def render_combined_4metric_held_out(df: pd.DataFrame) -> None:
    """The headline 4-metric plot, now using HELD-OUT Greek fertility
    (not FLORES+) as the Greek signal."""
    METRICS = [
        # metric, source, slice (None=any), language, lower_better, label, color
        ("greek_word_space_fertility", "our_suite_02_1_3", "C3_val",  "ell_Grek", True,
         "Greek fertility on C3_val (in-domain held-out)", "#d62728"),
        ("tokenizer_fairness_gini",    "tokeval-lines",   None,       "global",   True,
         "TFG on Apertus-55 (multilingual fairness)",       "#1f77b4"),
        ("morphscore_recall",          "morphscore",      "morphscore_ud", "ell_Grek", False,
         "MorphScore recall (Greek)",                       "#2ca02c"),
        ("eval_added_vocab_utilization_rate", "our_suite_02_1_3", "C3_val", "ell_Grek", False,
         "Added-vocab utilization on C3_val",               "#9467bd"),
    ]
    rows = []
    cur_rows = []
    for metric, source, slice_filt, lang, lower_better, label, color in METRICS:
        sub = df[(df.metric == metric) & (df.language == lang) & (df.source == source)]
        if slice_filt:
            sub = sub[sub.slice == slice_filt]
        sub = sub.sort_values("added_tokens")
        if sub.empty:
            print(f"  ! no data for {label}")
            continue
        baseline_row = sub[sub.added_tokens == 0]
        if baseline_row.empty:
            baseline = sub.iloc[0].value
        else:
            baseline = baseline_row.value.iloc[0]
        for _, r in sub.iterrows():
            pct = (r.value - baseline) / abs(baseline) * 100 if baseline != 0 else 0
            if lower_better:
                pct = -pct
            d = {
                "added_tokens": r.added_tokens, "metric_label": label,
                "raw_value": r.value, "pct_improvement": pct,
                "curated": r.curated,
            }
            (cur_rows if r.curated else rows).append(d)

    if not rows:
        return False
    wide = pd.DataFrame(rows)
    cur = pd.DataFrame(cur_rows)

    fig = plt.figure(figsize=(11, 9))
    ax_pct = fig.add_axes([0.08, 0.43, 0.88, 0.50])
    for metric, source, slice_filt, lang, lower_better, label, color in METRICS:
        s = wide[wide.metric_label == label].sort_values("added_tokens")
        if s.empty: continue
        ax_pct.plot(s.added_tokens, s.pct_improvement,
                    marker="o", color=color, label=label, linewidth=2)
        c = cur[cur.metric_label == label] if not cur.empty else cur
        if not c.empty:
            ax_pct.scatter(c.added_tokens, c.pct_improvement,
                           marker="s", color=color, edgecolor="black", s=70, zorder=5)
    ax_pct.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_pct.set_xlabel("added tokens")
    ax_pct.set_ylabel("improvement over apertus_base (%)\n↑ = the cutoff helps this metric")
    ax_pct.set_title("4-metric cutoff curve — in-domain Greek (C3_val) + multilingual fairness\n"
                     "(squares = curated twins; signs flipped on lower=better metrics)")
    ax_pct.grid(True, alpha=0.3)
    ax_pct.legend(loc="upper left", fontsize=9, framealpha=0.95)
    # Annotate final value
    for metric, source, slice_filt, lang, lower_better, label, color in METRICS:
        s = wide[(wide.metric_label == label) & (wide.added_tokens == 12288)]
        if not s.empty:
            y = s.pct_improvement.iloc[0]
            ax_pct.annotate(f"{y:+.2f}%", xy=(12288, y),
                            xytext=(8, 0), textcoords="offset points",
                            color=color, fontsize=9, fontweight="bold", va="center")

    # 2×2 raw value mini-plots
    for i, (metric, source, slice_filt, lang, lower_better, label, color) in enumerate(METRICS):
        row, col = divmod(i, 2)
        ax = fig.add_axes([0.07 + col * 0.47, 0.05 + (1 - row) * 0.18, 0.41, 0.13])
        s = wide[wide.metric_label == label].sort_values("added_tokens")
        if s.empty: continue
        ax.plot(s.added_tokens, s.raw_value, marker="o", color=color)
        c = cur[cur.metric_label == label] if not cur.empty else cur
        if not c.empty:
            ax.scatter(c.added_tokens, c.raw_value,
                       marker="s", color=color, edgecolor="black", s=50, zorder=5)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, "↓ better" if lower_better else "↑ better",
                transform=ax.transAxes, fontsize=7, va="top",
                color="dimgray", style="italic")

    out = PLOTS_DIR / "headline_4metric_in_domain.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return True


def render_per_slice_fertility(df: pd.DataFrame) -> None:
    """Show Greek fertility on each in-domain slice — cutoff curve per slice."""
    sub = df[(df.metric == "greek_word_space_fertility") &
             (df.source == "our_suite_02_1_3") &
             (df.language == "ell_Grek")].sort_values("added_tokens")
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    for slice_id, grp in sub.groupby("slice"):
        raw = grp[~grp.curated].sort_values("added_tokens")
        cur = grp[grp.curated].sort_values("added_tokens")
        line, = ax.plot(raw.added_tokens, raw.value, marker="o",
                        label=slice_id, alpha=0.85)
        if not cur.empty:
            ax.scatter(cur.added_tokens, cur.value, marker="s",
                       color=line.get_color(), edgecolor="black",
                       s=60, zorder=5)
    ax.set_xlabel("added tokens")
    ax.set_ylabel("Greek word fertility (tokens/word) — ↓ better")
    ax.set_title("Greek fertility per in-domain slice (02_1_3 harness on gcloud)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = PLOTS_DIR / "fertility_per_in_domain_slice.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def render_morphscore(df: pd.DataFrame) -> None:
    sub = df[df.source == "morphscore"].sort_values("added_tokens")
    if sub.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric, label, lower in zip(
        axes,
        ["morphscore_recall", "morphscore_precision", "mean_token_char_ratio"],
        ["recall (↑)", "precision (↑)", "token-char ratio (↓ over-seg.)"],
        [False, False, True],
    ):
        s = sub[sub.metric == metric]
        if s.empty: continue
        raw = s[~s.curated].sort_values("added_tokens")
        cur = s[s.curated].sort_values("added_tokens")
        ax.plot(raw.added_tokens, raw.value, marker="o", color="#2ca02c")
        if not cur.empty:
            ax.scatter(cur.added_tokens, cur.value, marker="s",
                       color="#2ca02c", edgecolor="black", s=70, zorder=5)
        ax.set_xlabel("added tokens")
        ax.set_ylabel(label)
        ax.set_title(metric)
        ax.grid(True, alpha=0.3)
    fig.suptitle("MorphScore Greek (catherinearnett/morphscore UD-derived, n=693 after filtering)",
                 fontsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "morphscore_greek.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def pivot_global(df, metric, source, slice_id=None, lang="global"):
    sub = df[(df.metric == metric) & (df.source == source) & (df.language == lang)].copy()
    if slice_id:
        sub = sub[sub.slice == slice_id]
    if sub.empty:
        return pd.DataFrame()
    sub["order"] = sub.variant_id.map({v: i for i, v in enumerate(VARIANT_ORDER)})
    return sub.sort_values("order")[["variant_id", "added_tokens", "curated", "value"]]


def main() -> None:
    df = pd.read_parquet(PARQUET)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Plots
    render_combined_4metric_held_out(df)
    render_per_slice_fertility(df)
    render_morphscore(df)

    # Report
    md = []
    a = md.append
    a("# 02_1_7 Intrinsic Eval Sweep — REPORT.md (final)")
    a("")
    a("**Sweep date**: 2026-05-17 (extended 0 → 25,600 added tokens at 1k step).  ")
    n_raw = df[~df.curated].variant_id.nunique() - 1  # minus apertus_base
    n_cur = df[df.curated].variant_id.nunique()
    a(f"**Tokenizers**: {df.variant_id.nunique()} (apertus_base + {n_raw} raw cutoff variants + {n_cur} curated twins).  ")
    a(f"**Rows in merged parquet**: {len(df):,}.  ")
    a("**Sources**: tokeval-lines (Apertus-55), tokeval-words (Apertus-55), "
      "our_suite_02_1_3 (in-domain Greek held-outs on gcloud), "
      "morphscore (catherinearnett/morphscore Greek UD).  ")
    a("**TokEval commit**: "
      f"`{(SSP / 'manifests/tokeval_commit.txt').read_text().splitlines()[-1] if (SSP / 'manifests/tokeval_commit.txt').exists() else 'unknown'}`.")
    a("")

    a("## Headline plot — 4 metrics on one chart, in-domain")
    a("")
    a("Greek fertility on C3_val (actual in-domain held-out), TFG on")
    a("Apertus-55 multilingual, MorphScore Greek recall, and added-vocab")
    a("utilization on C3_val. All four shown as % improvement over")
    a("`apertus_base`, with the sign flipped on lower-is-better metrics so")
    a("**up always means the cutoff is doing something good**.")
    a("")
    a("![headline](artifacts/plots/headline_4metric_in_domain.png)")
    a("")

    a("## Tier 1 — in-domain Greek (02_1_3 harness on gcloud)")
    a("")
    a("This is the **decision-relevant** axis: actual Greek text from C2/C3")
    a("training-distribution held-outs (anti-joined, cleaned) plus 4 latest")
    a("glossAPI HF datasets. Not FLORES+. The 02_1_3 harness computes Greek-")
    a("specific metrics including space-fertility (the historical fertility")
    a("number this project has been tracking).")
    a("")
    a("### Greek word fertility per held-out slice")
    a("")
    sub = df[(df.metric == "greek_word_space_fertility") &
             (df.source == "our_suite_02_1_3")]
    if not sub.empty:
        tab = (
            sub.pivot_table(index=["variant_id", "added_tokens", "curated"],
                            columns="slice", values="value", aggfunc="mean")
              .reset_index()
              .assign(_order=lambda d: d.variant_id.map(
                  {v: i for i, v in enumerate(VARIANT_ORDER)}))
              .sort_values("_order").drop(columns="_order")
        )
        a(md_table(tab))
    a("")
    a("![per-slice-fertility](artifacts/plots/fertility_per_in_domain_slice.png)")
    a("")

    a("### Added-vocab utilization rate per slice")
    a("")
    a("**The cutoff signal for diminishing returns**: when this drops, the")
    a("added tokens stop firing on in-domain text.")
    a("")
    sub = df[(df.metric == "eval_added_vocab_utilization_rate") &
             (df.source == "our_suite_02_1_3")]
    if not sub.empty:
        tab = (
            sub.pivot_table(index=["variant_id", "added_tokens", "curated"],
                            columns="slice", values="value", aggfunc="mean")
              .reset_index()
              .assign(_order=lambda d: d.variant_id.map(
                  {v: i for i, v in enumerate(VARIANT_ORDER)}))
              .sort_values("_order").drop(columns="_order")
        )
        a(md_table(tab))
    a("")

    a("### Chars-per-token (compression) per slice")
    a("")
    sub = df[(df.metric == "chars_per_token") &
             (df.source == "our_suite_02_1_3")]
    if not sub.empty:
        tab = (
            sub.pivot_table(index=["variant_id", "added_tokens", "curated"],
                            columns="slice", values="value", aggfunc="mean")
              .reset_index()
              .assign(_order=lambda d: d.variant_id.map(
                  {v: i for i, v in enumerate(VARIANT_ORDER)}))
              .sort_values("_order").drop(columns="_order")
        )
        a(md_table(tab))
    a("")

    a("## Tier 1 — paper-aligned (TokEval on Apertus-55 FLORES+)")
    a("")
    a("These match the eval surface area the Apertus paper §2.2 uses; cite")
    a("for cross-paper comparison (Meister 2025 / Foroutan et al. 2025a).")
    a("")
    a("### TFG (Apertus-55 multilingual fairness)")
    a("")
    a("**Lower = more fair.** U-shape with minimum at ~4k — the only metric")
    a("with a clear non-monotonic optimum on this run.")
    a("")
    a(md_table(pivot_global(df, "tokenizer_fairness_gini", "tokeval-lines")))
    a("")

    a("### Fertility (FLORES+ Greek, words config)")
    a("")
    a("Same shape as the in-domain fertility above but on cleaner narrower")
    a("data — values are lower (clean Greek is easier to tokenize).")
    a("")
    a(md_table(pivot_global(df, "fertility", "tokeval-words",
                            slice_id="flores_plus_55", lang="ell_Grek")))
    a("")

    a("## Tier 2 — multi-criteria supplement")
    a("")
    a("### MorphScore Greek (catherinearnett/morphscore)")
    a("")
    a("21,428 Greek words from UD Greek-GDT with gold morpheme boundaries;")
    a("`n=693` after the freq-scale + length filter. **Recall** = fraction of")
    a("gold morpheme boundaries the tokenizer hits; **precision** = fraction")
    a("of tokenizer boundaries that hit a real morpheme boundary; **token-")
    a("char ratio** = mean tokens-per-character (lower = less over-segmenting).")
    a("")
    sub = df[df.source == "morphscore"]
    if not sub.empty:
        tab = (
            sub.pivot_table(index=["variant_id", "added_tokens", "curated"],
                            columns="metric", values="value", aggfunc="mean")
              .reset_index()
              .assign(_order=lambda d: d.variant_id.map(
                  {v: i for i, v in enumerate(VARIANT_ORDER)}))
              .sort_values("_order").drop(columns="_order")
        )
        a(md_table(tab))
    a("")
    a("![morphscore](artifacts/plots/morphscore_greek.png)")
    a("")
    a("**Reading**: recall is essentially flat (~0.69) across all cutoffs —")
    a("the added vocabulary does not appreciably improve morpheme-boundary")
    a("alignment. Precision improves marginally (+0.5 % at 11k). The")
    a("**token-char ratio drops 29 %** (0.49 → 0.35), confirming the added")
    a("vocab makes Greek encoding more compact without dramatically improving")
    a("morphological structure capture. Useful nuance the 4 paper metrics")
    a("don't see.")
    a("")

    a("### Rényi-2.5 efficiency (Apertus-55 global)")
    a("")
    a(md_table(pivot_global(df, "renyi_efficiency", "tokeval-lines")))
    a("")

    a("### UTF-8 completeness (Apertus-55 global)")
    a("")
    a(md_table(pivot_global(df, "utf8_token_integrity", "tokeval-lines")))
    a("")

    a("## Curated-arm delta (consumes 02_1_5 removal_list)")
    a("")
    a("Curated twins remove 39 (at 11,264) / 44 (at 12,288) tokens after")
    a("merge-graph validation. **In every metric tested, the delta vs the")
    a("un-curated variant is essentially zero** — 39/44 token removal is free.")
    a("")
    for n in [11264, 12288]:
        a(f"### Cutoff {n:,}")
        a("")
        rows = []
        for metric, source, lang in [
            ("greek_word_space_fertility", "our_suite_02_1_3", "ell_Grek"),
            ("chars_per_token", "our_suite_02_1_3", "ell_Grek"),
            ("eval_added_vocab_utilization_rate", "our_suite_02_1_3", "ell_Grek"),
            ("tokenizer_fairness_gini", "tokeval-lines", "global"),
            ("renyi_efficiency", "tokeval-lines", "global"),
            ("utf8_token_integrity", "tokeval-lines", "global"),
            ("morphscore_recall", "morphscore", "ell_Grek"),
            ("morphscore_precision", "morphscore", "ell_Grek"),
            ("mean_token_char_ratio", "morphscore", "ell_Grek"),
        ]:
            raw_sub = df[(df.metric == metric) & (df.language == lang) &
                         (df.source == source) & (df.variant_id == f"add_{n}")]
            cur_sub = df[(df.metric == metric) & (df.language == lang) &
                         (df.source == source) & (df.variant_id == f"add_{n}_curated")]
            if raw_sub.empty or cur_sub.empty:
                continue
            raw_mean = raw_sub.value.mean()
            cur_mean = cur_sub.value.mean()
            rows.append({
                "metric": metric, "source": source,
                "raw": raw_mean, "curated": cur_mean,
                "Δ": cur_mean - raw_mean,
            })
        a(md_table(pd.DataFrame(rows)))
        a("")

    a("## Methodology delta — TokEval-fertility vs 02_1_3-fertility on Greek")
    a("")
    a("Same 15 tokenizers, same target (Greek), different eval sets and")
    a("different normalizations. **TokEval = FLORES+ ell_Grek, words config.**")
    a("**02_1_3 = our in-domain held-outs, greek_word_space_fertility.**")
    a("")
    a("Use this to decide which fertility number to cite — TokEval for paper")
    a("comparability, 02_1_3 for in-domain decision-making.")
    a("")
    tev = (df[(df.metric == "fertility") & (df.language == "ell_Grek") &
              (df.source == "tokeval-words")]
           [["variant_id", "added_tokens", "value"]]
           .rename(columns={"value": "TokEval (FLORES+ Greek)"}))
    ours = (df[(df.metric == "greek_word_space_fertility") &
               (df.source == "our_suite_02_1_3") & (df.slice == "C3_val")]
            [["variant_id", "added_tokens", "value"]]
            .rename(columns={"value": "02_1_3 (C3_val)"}))
    merged = tev.merge(ours, on=["variant_id", "added_tokens"], how="outer")
    merged["_order"] = merged.variant_id.map({v: i for i, v in enumerate(VARIANT_ORDER)})
    merged = merged.sort_values("_order").drop(columns="_order")
    a(md_table(merged))
    a("")

    a("## Reproduction")
    a("")
    a("```bash")
    a("cd subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep")
    a("$VENV/python scripts/01_build_variants_inline.py    # local 13 variants")
    a("$VENV/python scripts/01b_build_curated_variants.py  # 2 curated twins")
    a("$VENV/python scripts/02_prep_eval_configs.py")
    a("bash scripts/03a_run_tokeval.sh                     # TokEval 2 jobs (~30 min)")
    a("# gcloud: resume → run 02_1_3 harness for our 4 in-house slices → suspend")
    a("$VENV/python scripts/07_morphscore_greek.py         # MorphScore Greek")
    a("$VENV/python scripts/08_merge_all.py                # merge all 4 sources")
    a("$VENV/python scripts/09_render_final_report.py      # this file + plots")
    a("```")

    REPORT_PATH.write_text("\n".join(md) + "\n")
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
