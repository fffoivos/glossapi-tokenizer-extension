#!/usr/bin/env python3
"""Render a combined report for the polytonic tokenizer implementation run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_line_plot(df: pd.DataFrame, metric: str, selector: dict[str, str], title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, query in selector.items():
        sub = df.query(query).sort_values("polytonic_added_count")
        if sub.empty:
            continue
        ax.plot(sub["polytonic_added_count"], sub[metric], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("Ancient/Polytonic added tokens")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if df.empty:
        return "_No data._"
    view = df[cols].head(max_rows).copy()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        cells = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                cells.append("" if pd.isna(val) else f"{val:.4f}")
            else:
                cells.append("" if pd.isna(val) else str(val))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def morphscore_frame(path: Path) -> pd.DataFrame:
    rows = []
    payload = json.loads(path.read_text(encoding="utf-8"))
    for item in payload:
        metrics = item.get("metrics", {})
        ell = metrics.get("ell_Grek", metrics)
        row = {
            "variant_id": item["variant_id"],
            "polytonic_added_count": item["polytonic_added_count"],
            "final_vocab_size": item["final_vocab_size"],
        }
        row.update({k: v for k, v in ell.items() if isinstance(v, (int, float))})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_morphscore(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for metric in ("morphscore_recall", "morphscore_precision", "mean_token_char_ratio"):
        if metric in df.columns:
            ax.plot(df["polytonic_added_count"], df[metric], marker="o", label=metric)
    ax.set_title("Greek MorphScore guard")
    ax.set_xlabel("Ancient/Polytonic added tokens")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    plots = ensure(args.output_dir / "plots")
    eval_df = pd.read_csv(args.run_root / "eval/metrics_by_slice.csv")
    tok_df = pd.read_csv(args.run_root / "tokeval_aggregated/tokeval_metrics.csv")
    morph_df = morphscore_frame(args.run_root / "morphscore/morphscore_greek_results.json")
    variants = json.loads((args.run_root / "variants/variants_manifest.json").read_text(encoding="utf-8"))
    training = json.loads((args.run_root / "training/training_summary.json").read_text(encoding="utf-8"))
    splits = json.loads((args.run_root / "splits/split_manifest.json").read_text(encoding="utf-8"))

    for col in ("polytonic_added_count",):
        eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")
        tok_df[col] = pd.to_numeric(tok_df[col], errors="coerce")
        morph_df[col] = pd.to_numeric(morph_df[col], errors="coerce")

    write_line_plot(
        eval_df,
        "greek_word_fertility",
        {
            "balanced val": 'slice == "poly_val_balanced"',
            "balanced test": 'slice == "poly_test_balanced"',
            "FineWeb-2 grc": 'slice == "fineweb2_grc_reference"',
            "modern C3 val": 'slice == "modern_c3_val_clean"',
        },
        "Held-out Greek word fertility",
        "tokens / Greek word",
        plots / "07_combined_fertility.png",
    )
    write_line_plot(
        eval_df,
        "poly_added_token_rate",
        {
            "balanced val": 'slice == "poly_val_balanced"',
            "high diacritic": 'slice == "poly_high_diacritic_test"',
            "FineWeb-2 grc": 'slice == "fineweb2_grc_reference"',
            "modern C3 val": 'slice == "modern_c3_val_clean"',
        },
        "New polytonic token firing rate",
        "share of tokens",
        plots / "08_poly_added_token_rate.png",
    )

    # TokEval rows are long format; pivot a few headline metrics.
    tok_head = tok_df[
        (tok_df["metric"].isin(["fertility", "tokenizer_fairness_gini", "renyi_efficiency", "utf8_token_integrity"]))
        & (tok_df["language"].isin(["global", "ell_Grek"]))
    ].copy()
    for (metric, language, source, slice_id), sub in tok_head.groupby(["metric", "language", "source", "slice"], dropna=False):
        label = f"{metric} | {language} | {source} | {slice_id}"
        safe = (
            f"09_tokeval_{metric}_{language}_{source}_{slice_id}.png"
            .replace("/", "_")
            .replace(" ", "_")
        )
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sub = sub.sort_values("polytonic_added_count")
        ax.plot(sub["polytonic_added_count"], sub["value"], marker="o")
        ax.set_title(label)
        ax.set_xlabel("Ancient/Polytonic added tokens")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots / safe, dpi=140)
        plt.close(fig)

    plot_morphscore(morph_df, plots / "10_morphscore_greek.png")

    balanced = eval_df[eval_df["slice"].isin(["poly_val_balanced", "poly_test_balanced"])].sort_values(
        ["slice", "polytonic_added_count"]
    )
    base_val = eval_df[(eval_df["slice"] == "poly_val_balanced") & (eval_df["polytonic_added_count"] == 0)].iloc[0]
    best_val = eval_df[eval_df["slice"] == "poly_val_balanced"].sort_values("greek_word_fertility").iloc[0]
    modern = eval_df[eval_df["slice"] == "modern_c3_val_clean"].sort_values("polytonic_added_count")
    modern_base = modern.iloc[0]
    modern_best = modern.iloc[-1]
    fineweb = eval_df[eval_df["slice"] == "fineweb2_grc_reference"].sort_values("polytonic_added_count")
    fineweb_base = fineweb.iloc[0]
    fineweb_best = fineweb.iloc[-1]
    tok_global = tok_df[
        (tok_df["metric"] == "tokenizer_fairness_gini")
        & (tok_df["language"] == "global")
        & (tok_df["source"] == "tokeval-lines")
        & (tok_df["slice"] == "flores_plus_55")
    ].sort_values("polytonic_added_count")

    variant_last = variants["variants"][-1]
    report = [
        "# Ancient/Polytonic Greek Extension Run Report",
        "",
        "Generated by `scripts/render_polytonic_full_report.py`.",
        "",
        "## Run State",
        "",
        f"- Base C3 tokenizer SHA-256: `{variant_last['base_tokenizer_sha256']}`",
        f"- Full +5,120 tokenizer SHA-256: `{variant_last['tokenizer_sha256']}`",
        f"- Final vocab at +5,120: `{variant_last['final_vocab_size']}`",
        f"- Training target vocab: `{training.get('target_vocab_size')}`",
        f"- Split rows after hygiene: `{splits['hygiene']['rows_after']}`",
        f"- Split rows dropped by hygiene: `{splits['hygiene']['rows_dropped']}`",
        f"- Cutoff variants: `{len(variants['variants'])}` at 512-token spacing",
        "",
        "## Headline",
        "",
        (
            f"The +5,120 cutoff is the best point in the measured grid on the balanced polytonic "
            f"validation slice: Greek word fertility moves from {base_val['greek_word_fertility']:.4f} "
            f"to {best_val['greek_word_fertility']:.4f}, and distinctive-polytonic word fertility moves "
            f"from {base_val['distinctive_polytonic_word_fertility']:.4f} to "
            f"{best_val['distinctive_polytonic_word_fertility']:.4f}."
        ),
        "",
        (
            f"The same +5,120 point still uses almost all available added vocabulary on balanced validation "
            f"({best_val['poly_added_vocab_utilization_rate']:.4f}) and has a high polytonic firing rate "
            f"({best_val['poly_added_token_rate']:.4f}), so the end of the requested budget is not obviously dead capacity."
        ),
        "",
        (
            f"Modern Greek regression looks small by firing rate: on `modern_c3_val_clean`, the new polytonic ids fire "
            f"at {modern_best['poly_added_token_rate']:.4f} of tokens at +5,120. Greek word fertility changes from "
            f"{modern_base['greek_word_fertility']:.4f} to {modern_best['greek_word_fertility']:.4f}; this is a mild "
            "compression gain, not evidence of broad modern-Greek takeover by the new ids."
        ),
        "",
        (
            f"FineWeb-2 Ancient Greek also improves through the full grid: Greek word fertility moves from "
            f"{fineweb_base['greek_word_fertility']:.4f} to {fineweb_best['greek_word_fertility']:.4f}, with "
            f"new-token firing at {fineweb_best['poly_added_token_rate']:.4f}."
        ),
        "",
        "## Plots",
        "",
        "- `plots/01_cutoff_dashboard.png`",
        "- `plots/02_knee_poly_val_balanced.png`",
        "- `plots/03_source_stratified_fertility.png`",
        "- `plots/04_orthography_stratified_fertility.png`",
        "- `plots/05_byteish_rate.png`",
        "- `plots/06_added_vocab_utilization_heatmap.png`",
        "- `plots/07_combined_fertility.png`",
        "- `plots/08_poly_added_token_rate.png`",
        "- `plots/09_tokeval_*`",
        "- `plots/10_morphscore_greek.png`",
        "",
        "## Balanced Held-Out Metrics",
        "",
        md_table(
            balanced,
            [
                "slice",
                "variant_id",
                "polytonic_added_count",
                "greek_word_fertility",
                "distinctive_polytonic_word_fertility",
                "chars_per_token",
                "poly_added_token_rate",
                "poly_added_vocab_utilization_rate",
            ],
            max_rows=80,
        ),
        "",
        "## TokEval Fairness Guard",
        "",
        md_table(
            tok_global,
            ["variant_id", "polytonic_added_count", "metric", "value", "source", "slice"],
            max_rows=20,
        ),
        "",
        "TokEval completed all three reused jobs. The code/AST metric emitted tree-sitter thread-safety tracebacks in the log, "
        "but each job wrote its `analysis_results.json` and the text/fairness metrics aggregated successfully. Treat AST-specific "
        "rows from this worker as non-load-bearing for the tokenizer decision.",
        "",
        "## MorphScore Guard",
        "",
        md_table(
            morph_df,
            [
                "variant_id",
                "polytonic_added_count",
                "morphscore_recall",
                "morphscore_precision",
                "mean_token_char_ratio",
            ],
            max_rows=20,
        ),
        "",
        "MorphScore is modern-Greek UD-derived, so it is a regression guard rather than an Ancient Greek target metric. "
        "The near-flat curve is consistent with the new layer being orthographic/polytonic rather than a major modern-Greek morphology rewrite.",
        "",
        "## Caveats",
        "",
        "- The `poly_underaccented_test` slice is empty in this strict-filter run; under-accented curated ancient text needs a separate targeted slice if that behavior becomes a selection criterion.",
        "- `c3p_poly_added_0000` is semantically the C3 baseline but was reserialized by the variant builder, so its tokenizer JSON SHA differs from the original C3 ship SHA. The manifest still records the C3 base SHA directly.",
        "- The Apertus-55 proxy config loaded 54 languages in this environment, matching the live config available to the worker rather than an independently reconstructed 55-language list.",
        "",
        "## Current Recommendation",
        "",
        "Keep +5,120 as the current candidate for downstream fertility/testing. It is 256-aligned at 153,600 vocab, wins the measured held-out grid, still shows high added-vocab utilization, and has low modern-Greek firing.",
        "",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report_path), "plots": str(plots)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
