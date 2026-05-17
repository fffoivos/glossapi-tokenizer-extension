#!/usr/bin/env python3
"""Render plots and a compact report for the polytonic cutoff sweep."""
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


def metric_line(df: pd.DataFrame, metric: str, slices: list[str], title: str, ylabel: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for slice_name in slices:
        sub = df[(df["slice"] == slice_name) & df[metric].notna()].sort_values("polytonic_added_count")
        if sub.empty:
            continue
        ax.plot(sub["polytonic_added_count"], sub[metric], marker="o", label=slice_name)
    ax.set_title(title)
    ax.set_xlabel("Ancient/Polytonic added tokens")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def dashboard(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        ("greek_word_fertility", "poly_val_balanced", "Balanced polytonic fertility", "tokens / Greek word"),
        ("distinctive_polytonic_word_fertility", "poly_high_diacritic_test", "High-polytonic fertility", "tokens / polytonic word"),
        ("byteish_token_rate", "poly_val_balanced", "Byte-ish token rate", "share of tokens"),
        ("poly_added_vocab_utilization_rate", "poly_val_balanced", "Poly added-vocab utilization", "used / added"),
        ("greek_word_fertility", "modern_c3_val_clean", "Modern Greek regression", "tokens / Greek word"),
        ("roundtrip_exact_doc_rate", "poly_val_balanced", "Roundtrip exact docs", "share of docs"),
    ]
    for ax, (metric, slice_name, title, ylabel) in zip(axes.flat, panels):
        sub = df[(df["slice"] == slice_name) & df[metric].notna()].sort_values("polytonic_added_count")
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.plot(sub["polytonic_added_count"], sub[metric], marker="o")
        ax.set_title(title)
        ax.set_xlabel("added")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Ancient/Polytonic Greek cutoff dashboard", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def knee_plot(df: pd.DataFrame, out: Path, slice_name: str = "poly_val_balanced") -> dict[str, object]:
    sub = df[(df["slice"] == slice_name) & df["greek_word_fertility"].notna()].sort_values("polytonic_added_count")
    if sub.empty:
        return {"slice": slice_name, "available": False}
    base = float(sub[sub["polytonic_added_count"] == 0]["greek_word_fertility"].iloc[0])
    best = float(sub["greek_word_fertility"].min())
    max_gain = base - best
    sub = sub.copy()
    sub["gain_pct_of_max"] = ((base - sub["greek_word_fertility"]) / max_gain * 100) if max_gain else 0.0
    sub["marginal_gain"] = -sub["greek_word_fertility"].diff()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(sub["polytonic_added_count"], sub["gain_pct_of_max"], marker="o")
    ax.set_title(f"Knee analysis on {slice_name}")
    ax.set_xlabel("Ancient/Polytonic added tokens")
    ax.set_ylabel("% of observed max fertility gain")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return {
        "slice": slice_name,
        "available": True,
        "base_fertility": base,
        "best_fertility": best,
        "max_gain": max_gain,
    }


def utilization_heatmap(df: pd.DataFrame, out: Path) -> None:
    sub = df[df["poly_added_vocab_utilization_rate"].notna()]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="slice", columns="polytonic_added_count", values="poly_added_vocab_utilization_rate", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(pivot))))
    im = ax.imshow(pivot.fillna(0).values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Added Ancient/Polytonic vocab utilization by slice")
    fig.colorbar(im, ax=ax, label="used / available")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._"
    view = df[cols].head(max_rows).copy()
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in view.iterrows():
        cells = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else f"{value:.4f}")
            else:
                cells.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots = ensure(args.output_dir / "plots")
    df = pd.read_csv(args.metrics_csv)
    df["polytonic_added_count"] = pd.to_numeric(df["polytonic_added_count"], errors="coerce")

    dashboard(df, plots / "01_cutoff_dashboard.png")
    knee = knee_plot(df, plots / "02_knee_poly_val_balanced.png")
    metric_line(
        df,
        "greek_word_fertility",
        [s for s in df["slice"].dropna().unique() if s.startswith("poly_") and s.endswith("_test")],
        "Source/held-out Greek fertility",
        "tokens / Greek word",
        plots / "03_source_stratified_fertility.png",
    )
    metric_line(
        df,
        "distinctive_polytonic_word_fertility",
        ["poly_high_diacritic_test", "poly_val_balanced", "poly_underaccented_test"],
        "Orthography-stratified fertility",
        "tokens / target word",
        plots / "04_orthography_stratified_fertility.png",
    )
    metric_line(
        df,
        "byteish_token_rate",
        ["poly_val_balanced", "poly_test_balanced", "poly_high_diacritic_test", "fineweb2_grc_reference"],
        "Byte-ish token rate",
        "share of tokens",
        plots / "05_byteish_rate.png",
    )
    utilization_heatmap(df, plots / "06_added_vocab_utilization_heatmap.png")

    headline = df[df["slice"].isin(["poly_val_balanced", "poly_test_balanced"])].sort_values(["slice", "polytonic_added_count"])
    report = [
        "# Ancient/Polytonic Greek Tokenizer Sweep",
        "",
        "Generated by `scripts/render_polytonic_eval_report.py`.",
        "",
        "## Plots",
        "",
        "- `plots/01_cutoff_dashboard.png`",
        "- `plots/02_knee_poly_val_balanced.png`",
        "- `plots/03_source_stratified_fertility.png`",
        "- `plots/04_orthography_stratified_fertility.png`",
        "- `plots/05_byteish_rate.png`",
        "- `plots/06_added_vocab_utilization_heatmap.png`",
        "",
        "## Knee Summary",
        "",
        "```json",
        json.dumps(knee, indent=2),
        "```",
        "",
        "## Balanced Held-Out Metrics",
        "",
        md_table(
            headline,
            [
                "slice",
                "variant_id",
                "polytonic_added_count",
                "greek_word_fertility",
                "distinctive_polytonic_word_fertility",
                "byteish_token_rate",
                "poly_added_vocab_utilization_rate",
            ],
            max_rows=80,
        ),
        "",
    ]
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.report_path), "plots": str(plots)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
