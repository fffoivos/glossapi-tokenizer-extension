"""4-metric combined plot extended to full 0-25.6k sweep.

Uses results_merged.parquet (which should now contain the wider data).
Same layout as 06_combined_4metric_plot.py but with:
  - x-axis up to 25,600 (full C3)
  - vertical annotations at the two MorphScore discontinuities (cutoff
    10,240 and 17,408)
  - all 5 curated twins highlighted as squares
"""
from __future__ import annotations

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
OUT = SSP / "artifacts/plots/extended_4metric_to_25k.png"

# (metric, source, slice, lang, lower_better, label, color)
METRICS = [
    ("greek_word_space_fertility", "our_suite_02_1_3", "C3_val",        "ell_Grek", True,  "Greek fertility on C3_val (in-domain held-out)", "#d62728"),
    ("tokenizer_fairness_gini",    "tokeval-lines",   "flores_plus_55", "global",   True,  "TFG on Apertus-55 (multilingual fairness)",       "#1f77b4"),
    ("morphscore_recall",          "morphscore",      "morphscore_ud",  "ell_Grek", False, "MorphScore recall (Greek)",                       "#2ca02c"),
    ("eval_added_vocab_utilization_rate", "our_suite_02_1_3", "C3_val", "ell_Grek", False, "Added-vocab utilization on C3_val",               "#9467bd"),
]


def main() -> None:
    df = pd.read_parquet(PARQUET)
    rows, cur_rows = [], []
    for metric, source, slice_filt, lang, lower_better, label, color in METRICS:
        sub = df[(df.metric == metric) & (df.language == lang) & (df.source == source)]
        if slice_filt:
            sub = sub[sub.slice == slice_filt]
        sub = sub.sort_values("added_tokens")
        if sub.empty: continue
        base = sub[sub.added_tokens == 0]
        baseline = base.value.iloc[0] if not base.empty else sub.iloc[0].value
        for _, r in sub.iterrows():
            pct = (r.value - baseline) / abs(baseline) * 100 if baseline != 0 else 0
            if lower_better: pct = -pct
            d = {"added_tokens": r.added_tokens, "metric_label": label,
                 "raw_value": r.value, "pct_improvement": pct, "curated": r.curated}
            (cur_rows if r.curated else rows).append(d)
    wide = pd.DataFrame(rows); cur = pd.DataFrame(cur_rows)

    fig = plt.figure(figsize=(13, 9))
    ax_pct = fig.add_axes([0.07, 0.45, 0.90, 0.48])
    for metric, source, slice_filt, lang, lower_better, label, color in METRICS:
        s = wide[wide.metric_label == label].sort_values("added_tokens")
        if s.empty: continue
        ax_pct.plot(s.added_tokens, s.pct_improvement, marker="o", color=color,
                    label=label, linewidth=2, markersize=5)
        c = cur[cur.metric_label == label] if not cur.empty else cur
        if not c.empty:
            ax_pct.scatter(c.added_tokens, c.pct_improvement, marker="s", color=color,
                           edgecolor="black", s=80, zorder=5)
    # MorphScore discontinuity markers
    ax_pct.axvline(10240, color="green", linestyle="--", alpha=0.35,
                   label="MorphScore stem-batch milestone #1 (~10k)")
    ax_pct.axvline(17408, color="green", linestyle=":", alpha=0.35,
                   label="MorphScore stem-batch milestone #2 (~17k)")
    ax_pct.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_pct.set_xlabel("added tokens")
    ax_pct.set_ylabel("improvement over apertus_base (%)\n↑ = the cutoff helps this metric")
    ax_pct.set_title("Extended 4-metric sweep (0 → 25,600) — in-domain Greek + multilingual fairness\n"
                     "(squares = curated twins; dashed green = MorphScore stem-batch milestones)")
    ax_pct.grid(True, alpha=0.3)
    ax_pct.legend(loc="upper left", fontsize=9, framealpha=0.95)
    for metric, source, slice_filt, lang, lower_better, label, color in METRICS:
        s = wide[(wide.metric_label == label) & (wide.added_tokens == 25600)]
        if not s.empty:
            y = s.pct_improvement.iloc[0]
            ax_pct.annotate(f"{y:+.1f}%", xy=(25600, y), xytext=(10, 0),
                            textcoords="offset points", color=color,
                            fontsize=9, fontweight="bold", va="center")

    for i, (metric, source, slice_filt, lang, lower_better, label, color) in enumerate(METRICS):
        row, col = divmod(i, 2)
        ax = fig.add_axes([0.06 + col * 0.48, 0.06 + (1 - row) * 0.16, 0.42, 0.13])
        s = wide[wide.metric_label == label].sort_values("added_tokens")
        if s.empty: continue
        ax.plot(s.added_tokens, s.raw_value, marker="o", color=color, markersize=4)
        c = cur[cur.metric_label == label] if not cur.empty else cur
        if not c.empty:
            ax.scatter(c.added_tokens, c.raw_value, marker="s", color=color,
                       edgecolor="black", s=50, zorder=5)
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, "↓ better" if lower_better else "↑ better",
                transform=ax.transAxes, fontsize=7, va="top",
                color="dimgray", style="italic")

    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
