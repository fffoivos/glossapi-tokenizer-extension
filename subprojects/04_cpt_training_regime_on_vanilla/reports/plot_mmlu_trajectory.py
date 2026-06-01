"""Plot Greek + English MMLU trajectories for the 04 Vanilla CPT 5 B run.

Data sources
------------
- Greek MMLU (per-checkpoint point estimate + 95 % CI):
    reports/v4_bootstrap_cis_native_mcq.json
        models[iter-119-Vanilla-0.5B   ].per_task.greekmmlu
        models[iter-238-Vanilla-1B     ].per_task.greekmmlu
        models[iter-477-Vanilla-2B     ].per_task.greekmmlu
        models[iter-834-Vanilla-3.5B   ].per_task.greekmmlu
        models[iter-1192-Vanilla-5B    ].per_task.greekmmlu
        models[Apertus-Base                       ].per_task.greekmmlu   (Path A)
        models[Apertus-Base-matched-Path-B-perturbed].per_task.greekmmlu (Path-B-perturbed)
        models[bakeoff-Vanilla-2B / 3.5B / 5B     ].per_task.greekmmlu

- English MMLU (lm-eval 'mmlu' aggregate, n=14042):
    eval_data_cache_5b/iter_<I>/retention/results.json
        results['mmlu']['acc,none']  +  results['mmlu']['acc_stderr,none']
    (CI approximated as +/- 1.96 * stderr; not bootstrap. Documented in caption.)

Checkpoint mapping (consumed tokens):
    iter 119 -> 0.499 B
    iter 238 -> 0.998 B
    iter 477 -> 2.001 B
    iter 834 -> 3.498 B
    iter 1192 -> 5.000 B

Output: plot_mmlu_trajectory.png next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path(__file__).resolve().parent
CI_JSON = REPORTS / "v4_bootstrap_cis_native_mcq.json"
PATH_A_CI_JSON = REPORTS / "v4_workspace_path_a" / "path_a_probe_bootstrap_cis.json"
EVAL = REPORTS / "eval_data_cache_5b"
OUT = REPORTS / "plot_mmlu_trajectory.png"

CHECKPOINTS = [
    ("0000119", 119, 0.499, "0.5B"),
    ("0000238", 238, 0.998, "1B"),
    ("0000477", 477, 2.001, "2B"),
    ("0000834", 834, 3.498, "3.5B"),
    ("0001192", 1192, 5.000, "5B"),
]

# Path A geometry probe — single 0.5 B point. PATH_A_GEOMETRY_PROBE_PLAN.md.
# GreekMMLU CI from reports/v4_workspace_path_a/path_a_probe_bootstrap_cis.json
# (V4 v3 methodology). English MMLU point from iter 119 retention; stderr
# approximated as binomial sqrt(p(1-p)/n) on n=14042 (same convention as
# the right-panel +/- 1.96*stderr ribbons).
import math
PATH_A_PROBE_GREEK_POINT = 0.5427489177489178
PATH_A_PROBE_GREEK_LO95 = 0.5354136604136605
PATH_A_PROBE_GREEK_HI95 = 0.5505125661375662
PATH_A_PROBE_ENG_MMLU = 0.5977068793619142
PATH_A_PROBE_ENG_N = 14042
PATH_A_PROBE_ENG_STDERR = math.sqrt(
    PATH_A_PROBE_ENG_MMLU * (1 - PATH_A_PROBE_ENG_MMLU) / PATH_A_PROBE_ENG_N
)
PATH_A_PROBE_TOKENS_B = 0.499

WARMUP_END_B = 1.204  # iter 287 per training log
WARMUP_END_ITER = 287

ITER_TO_MODEL_KEY = {
    119: "iter-119-Vanilla-0.5B",
    238: "iter-238-Vanilla-1B",
    477: "iter-477-Vanilla-2B",
    834: "iter-834-Vanilla-3.5B",
    1192: "iter-1192-Vanilla-5B",
}


def load_greek_mmlu_cis() -> dict:
    with CI_JSON.open() as f:
        d = json.load(f)
    return d["models"]


def load_english_mmlu() -> list[tuple[int, float, float, float]]:
    """Return list of (iter, tokens_b, mmlu_acc, mmlu_stderr) per checkpoint."""
    out = []
    for it_str, it, tok, _label in CHECKPOINTS:
        with (EVAL / f"iter_{it_str}" / "retention" / "results.json").open() as f:
            d = json.load(f)
        e = d["results"]["mmlu"]
        out.append((it, tok, e["acc,none"], e["acc_stderr,none"]))
    return out


def main() -> None:
    models = load_greek_mmlu_cis()

    greek_pts = []
    for _, it, tok, label in CHECKPOINTS:
        m = models[ITER_TO_MODEL_KEY[it]]["per_task"]["greekmmlu"]
        greek_pts.append((tok, label, m["point"], m["lo_95"], m["hi_95"]))

    apertus_a = models["Apertus-Base"]["per_task"]["greekmmlu"]
    apertus_b_perturbed = models["Apertus-Base-matched-Path-B-perturbed"][
        "per_task"
    ]["greekmmlu"]
    bakeoff_2b = models["bakeoff-Vanilla-2B"]["per_task"]["greekmmlu"]
    bakeoff_35b = models["bakeoff-Vanilla-3.5B"]["per_task"]["greekmmlu"]
    bakeoff_5b = models["bakeoff-Vanilla-5B"]["per_task"]["greekmmlu"]
    bakeoff_pts = [
        (2.0, bakeoff_2b),
        (3.5, bakeoff_35b),
        (5.0, bakeoff_5b),
    ]

    eng = load_english_mmlu()

    fig, (ax_g, ax_e) = plt.subplots(
        nrows=1, ncols=2, figsize=(15, 6.5), sharex=True
    )

    # ---------------- Greek MMLU (left panel) ----------------
    xs = [p[0] for p in greek_pts]
    ys = [p[2] for p in greek_pts]
    lo = [p[3] for p in greek_pts]
    hi = [p[4] for p in greek_pts]
    yerr_lo = [y - l for y, l in zip(ys, lo)]
    yerr_hi = [h - y for h, y in zip(hi, ys)]
    ax_g.errorbar(
        xs,
        ys,
        yerr=[yerr_lo, yerr_hi],
        fmt="o-",
        color="C0",
        lw=2.2,
        ms=9,
        capsize=4,
        label="04 Vanilla CPT (Path B) — GreekMMLU",
    )
    for tok, label, y, _l, _h in greek_pts:
        ax_g.annotate(
            f"{y:.4f}",
            (tok, y),
            textcoords="offset points",
            xytext=(8, 9),
            fontsize=8.5,
            color="C0",
        )

    # Apertus-Base Path A horizontal band + line
    ax_g.axhspan(
        apertus_a["lo_95"],
        apertus_a["hi_95"],
        alpha=0.13,
        color="seagreen",
        label=(
            f"Apertus-Base Path A [{apertus_a['lo_95']:.4f}, "
            f"{apertus_a['hi_95']:.4f}]"
        ),
    )
    ax_g.axhline(apertus_a["point"], ls=":", color="seagreen", lw=1.4, alpha=0.85)

    # Apertus-Base matched-Path-B-perturbed (diagnostic bookend)
    ax_g.axhspan(
        apertus_b_perturbed["lo_95"],
        apertus_b_perturbed["hi_95"],
        alpha=0.10,
        color="purple",
        label=(
            f"Apertus-Base matched-Path-B-perturbed "
            f"[{apertus_b_perturbed['lo_95']:.4f}, "
            f"{apertus_b_perturbed['hi_95']:.4f}]"
        ),
    )
    ax_g.axhline(
        apertus_b_perturbed["point"], ls=":", color="purple", lw=1.2, alpha=0.7
    )

    # Bakeoff Vanilla — at matched token marks (2/3.5/5 B)
    bxs = [p[0] for p in bakeoff_pts]
    bys = [p[1]["point"] for p in bakeoff_pts]
    blo = [p[1]["lo_95"] for p in bakeoff_pts]
    bhi = [p[1]["hi_95"] for p in bakeoff_pts]
    byerr_lo = [y - l for y, l in zip(bys, blo)]
    byerr_hi = [h - y for h, y in zip(bhi, bys)]
    ax_g.errorbar(
        bxs,
        bys,
        yerr=[byerr_lo, byerr_hi],
        fmt="s--",
        color="C3",
        lw=1.8,
        ms=8,
        capsize=4,
        label="Bakeoff Vanilla (Path B, original regime)",
    )
    for x, y in zip(bxs, bys):
        ax_g.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=8.5,
            color="C3",
        )

    # Path A geometry probe — single point at 0.5 B with bootstrap CI.
    path_a_yerr = [
        [PATH_A_PROBE_GREEK_POINT - PATH_A_PROBE_GREEK_LO95],
        [PATH_A_PROBE_GREEK_HI95 - PATH_A_PROBE_GREEK_POINT],
    ]
    ax_g.errorbar(
        [PATH_A_PROBE_TOKENS_B],
        [PATH_A_PROBE_GREEK_POINT],
        yerr=path_a_yerr,
        fmt="*",
        color="goldenrod",
        markersize=20,
        markeredgecolor="black",
        markeredgewidth=0.8,
        capsize=5,
        elinewidth=2.0,
        ecolor="goldenrod",
        zorder=10,
        label=(
            "Path A geometry probe @ 0.5 B "
            "(rope=12M, max_pos=65536, llama3 scaling)"
        ),
    )
    ax_g.annotate(
        f"{PATH_A_PROBE_GREEK_POINT:.4f}",
        (PATH_A_PROBE_TOKENS_B, PATH_A_PROBE_GREEK_POINT),
        textcoords="offset points",
        xytext=(10, 14),
        fontsize=8.5,
        color="darkgoldenrod",
        fontweight="bold",
    )

    ax_g.axvline(WARMUP_END_B, ls="--", color="orange", lw=1.4, alpha=0.7)
    ax_g.text(
        WARMUP_END_B + 0.08,
        0.43,
        f"warmup end\n(iter {WARMUP_END_ITER}, {WARMUP_END_B:.2f} B)",
        fontsize=8.5,
        color="darkorange",
    )

    ax_g.set_xlabel("CPT tokens (B)")
    ax_g.set_ylabel("GreekMMLU accuracy")
    ax_g.set_title(
        "GreekMMLU trajectory\n"
        "(V4 v3 bootstrap CIs, 1 000 resamples, per-task item-level)",
        fontsize=11,
    )
    ax_g.set_xlim(-0.25, 5.5)
    ax_g.set_ylim(0.42, 0.58)
    ax_g.grid(True, alpha=0.3)
    ax_g.legend(loc="lower right", fontsize=8.5)

    # ---------------- English MMLU (right panel) ----------------
    e_xs = [r[1] for r in eng]
    e_ys = [r[2] for r in eng]
    e_err = [1.96 * r[3] for r in eng]
    ax_e.errorbar(
        e_xs,
        e_ys,
        yerr=e_err,
        fmt="o-",
        color="C2",
        lw=2.2,
        ms=9,
        capsize=4,
        label="04 Vanilla CPT — English MMLU (lm-eval)",
    )
    for x, y in zip(e_xs, e_ys):
        ax_e.annotate(
            f"{y:.4f}",
            (x, y),
            textcoords="offset points",
            xytext=(8, 9),
            fontsize=8.5,
            color="C2",
        )

    # Path A geometry probe — single point at 0.5 B with binomial-stderr CI
    # (same convention as the rest of this panel — no bootstrap available
    # for English MMLU lm-eval task).
    ax_e.errorbar(
        [PATH_A_PROBE_TOKENS_B],
        [PATH_A_PROBE_ENG_MMLU],
        yerr=[1.96 * PATH_A_PROBE_ENG_STDERR],
        fmt="*",
        color="goldenrod",
        markersize=20,
        markeredgecolor="black",
        markeredgewidth=0.8,
        capsize=5,
        elinewidth=2.0,
        ecolor="goldenrod",
        zorder=10,
        label=(
            "Path A geometry probe @ 0.5 B "
            "(rope=12M, max_pos=65536, llama3 scaling)"
        ),
    )
    ax_e.annotate(
        f"{PATH_A_PROBE_ENG_MMLU:.4f}",
        (PATH_A_PROBE_TOKENS_B, PATH_A_PROBE_ENG_MMLU),
        textcoords="offset points",
        xytext=(10, 14),
        fontsize=8.5,
        color="darkgoldenrod",
        fontweight="bold",
    )

    ax_e.axvline(WARMUP_END_B, ls="--", color="orange", lw=1.4, alpha=0.7)
    ax_e.text(
        WARMUP_END_B + 0.08,
        0.545,
        f"warmup end\n(iter {WARMUP_END_ITER}, {WARMUP_END_B:.2f} B)",
        fontsize=8.5,
        color="darkorange",
    )

    ax_e.set_xlabel("CPT tokens (B)")
    ax_e.set_ylabel("English MMLU accuracy (n=14 042)")
    ax_e.set_title(
        "English MMLU trajectory\n"
        "(approximate +/- 1.96*stderr ribbons; no bootstrap available)",
        fontsize=11,
    )
    ax_e.set_xlim(-0.25, 5.5)
    ax_e.set_ylim(0.54, 0.61)
    ax_e.grid(True, alpha=0.3)
    ax_e.legend(loc="lower right", fontsize=9)

    fig.suptitle(
        "MMLU trajectories — 04 Vanilla CPT 5 B run on Path B (Apertus-faithful regime)\n"
        "+ Path A geometry probe @ 0.5 B (gold star, 2026-05-31)",
        fontsize=12,
        y=1.02,
    )

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
