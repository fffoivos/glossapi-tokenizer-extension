"""Plot the knee-analysis decomposition of the Greek-fertility curve.

Two-panel figure:
  Top    : fertility curve + Hill-1 fit + asymptote + three knee candidates
  Bottom : marginal Δ% per added 1k + the 1% floor line

Knee criteria computed:
  - Kneedle (max distance from chord)
  - Half-marginal rule
  - 1% per added 1k floor
plus the smooth Hill-1 asymptote (the "Greek fertility floor" the curve is
approaching).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SSP = Path(
    "/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/"
    "02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep"
)
PARQUET = SSP / "artifacts/results_merged.parquet"
OUT = SSP / "artifacts/plots/knee_analysis.png"


def hill_1(x, a, b, c):
    return a / (1 + x / b) + c


def main() -> None:
    df = pd.read_parquet(PARQUET)
    sub = df[(df.metric == "greek_word_space_fertility") &
             (df.source == "our_suite_02_1_3") & (df.slice == "C3_val") &
             (~df.curated)].sort_values("added_tokens")
    x = sub.added_tokens.values
    y = sub.value.values

    # Hill-1 fit
    popt, _ = curve_fit(hill_1, x, y, p0=[1.0, 5000, 1.0], maxfev=5000)
    a, b, c = popt
    x_dense = np.linspace(x.min(), x.max(), 400)
    y_fit = hill_1(x_dense, *popt)

    # Kneedle (max distance below chord on normalized axes)
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y.max() - y) / (y.max() - y.min())  # invert + normalize so up=better
    dist = yn - xn
    kneedle_i = int(np.argmax(dist))
    kneedle_x = x[kneedle_i]

    # Half-marginal
    start_drel = (y[0] - y[1]) / y[0] * 100
    half_x = None
    for i in range(1, len(x)):
        drel = (y[i - 1] - y[i]) / y[i - 1] * 100
        if drel < start_drel / 2:
            half_x = x[i]; break

    # 1%-per-1k floor
    one_pct_x = None
    for i in range(1, len(x)):
        drel = (y[i - 1] - y[i]) / y[i - 1] * 100
        if drel < 1.0:
            one_pct_x = x[i]; break

    # Marginal Δ% per added 1k (for bottom panel)
    margin_x = x[1:]
    margin_y = np.array([(y[i - 1] - y[i]) / y[i - 1] * 100
                         for i in range(1, len(x))])

    # 80% of theoretical max improvement
    target_y = y[0] - 0.8 * (y[0] - c)
    eighty_pct_x = None
    for i in range(len(x)):
        if y[i] <= target_y:
            eighty_pct_x = x[i]; break

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9),
                                    gridspec_kw={"height_ratios": [2, 1.2]},
                                    sharex=True)

    # Top: curve + fit + asymptote + knee markers
    ax1.scatter(x, y, color="#d62728", s=40, zorder=5, label="measured (C3_val)")
    ax1.plot(x_dense, y_fit, color="#d62728", alpha=0.4, linewidth=2,
             label=f"Hill-1 fit  y = {a:.3f}/(1 + x/{b:.0f}) + {c:.3f}")
    ax1.axhline(c, color="black", linestyle=":", linewidth=1.5, alpha=0.7,
                label=f"asymptote y∞ = {c:.3f} (Greek fertility floor)")
    ax1.axhline(target_y, color="gray", linestyle=":", linewidth=1, alpha=0.6,
                label=f"80% of achievable: y = {target_y:.3f}")

    knee_colors = {"Kneedle": "#1f77b4", "half-marginal": "#2ca02c",
                   "1%-per-1k": "#9467bd", "80% achievable": "#ff7f0e"}
    knees = [("Kneedle", kneedle_x), ("half-marginal", half_x),
             ("1%-per-1k", one_pct_x), ("80% achievable", eighty_pct_x)]
    for label, kx in knees:
        if kx is None: continue
        ky = y[list(x).index(kx)]
        ax1.axvline(kx, color=knee_colors[label], linestyle="--",
                    alpha=0.55, linewidth=1.5)
        ax1.annotate(f"{label}\nx = {kx:,}", xy=(kx, ky),
                     xytext=(8, 12), textcoords="offset points",
                     fontsize=8, color=knee_colors[label], fontweight="bold")
    ax1.set_ylabel("Greek word fertility (tokens/word, C3_val)  ↓ better")
    ax1.set_title("Where is the knee? — Greek fertility decay on the held-out + 4 knee criteria")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=9)

    # Bottom: marginal Δ% per 1k + the 1% line
    ax2.bar(margin_x, margin_y, width=900, color="#d62728", alpha=0.6,
            edgecolor="black", linewidth=0.5)
    ax2.axhline(1.0, color="#9467bd", linestyle="--", linewidth=1.5, alpha=0.7,
                label="1% per added 1k floor")
    ax2.axhline(start_drel / 2, color="#2ca02c", linestyle="--", linewidth=1.5,
                alpha=0.7, label=f"half-marginal threshold ({start_drel/2:.1f}%)")
    for label, kx in knees:
        if kx is None: continue
        ax2.axvline(kx, color=knee_colors[label], linestyle="--",
                    alpha=0.55, linewidth=1.5)
    ax2.set_xlabel("added tokens")
    ax2.set_ylabel("marginal Δ% per added 1k\n(% of previous fertility eliminated)")
    ax2.set_title("Marginal returns — same axis as above")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")

    print()
    print("Knee summary:")
    print(f"  asymptote        : {c:.4f} (max achievable)")
    print(f"  Kneedle elbow    : {kneedle_x:>6,}  (fertility {y[kneedle_i]:.4f},"
          f" {(y[0]-y[kneedle_i])/(y[0]-c)*100:.1f}% of achievable)")
    print(f"  half-marginal    : {half_x:>6,}  (fertility {y[list(x).index(half_x)]:.4f})")
    print(f"  1%-per-1k floor  : {one_pct_x:>6,}  (fertility {y[list(x).index(one_pct_x)]:.4f},"
          f" {(y[0]-y[list(x).index(one_pct_x)])/(y[0]-c)*100:.1f}% of achievable)")
    print(f"  80% of achievable: {eighty_pct_x:>6,}  (fertility {y[list(x).index(eighty_pct_x)]:.4f})")
    print(f"  full sweep max   : {x[-1]:>6,}  (fertility {y[-1]:.4f},"
          f" {(y[0]-y[-1])/(y[0]-c)*100:.1f}% of achievable)")


if __name__ == "__main__":
    main()
