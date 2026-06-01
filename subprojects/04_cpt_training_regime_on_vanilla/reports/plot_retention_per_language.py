"""Plot per-language retention trajectories (EN / FR / DE / RU) for the 04 Vanilla CPT 5 B run.

Y axes (two tasks per language):
    - global_mmlu_<lang> : macro-mean over 6 sub-domain accuracies
                           (business / humanities / medical / other /
                            social_sciences / stem)
    - xnli_<lang>        : accuracy on XNLI <lang> aggregate
                           (n=2490 per language)

Reference horizontal lines: **matched-config Apertus-Base Path B** (= the
true pre-CPT initial state of our run; weights loaded under our training
geometry rope_theta=500K / max_pos=4096 BEFORE any CPT updates). Source:
/capstor/.../eval_apertus_base_matched_rope500k_seq4096/retention/.

X axis: consumed CPT tokens (B). The trajectory now starts at tokens=0
(the matched-config base) and runs through the 5 checkpoint marks; iter
119 is the second point, not the baseline.

Data sources
------------
- reports/eval_data_cache_5b/iter_<I>/retention/results.json :
      results['global_mmlu_<lang>_<sub>']['acc,none']  for sub in 6 sub-domains
      results['xnli_<lang>']['acc,none']
- Matched-config baseline (constants below, pulled 2026-05-30) from
  eval_apertus_base_matched_rope500k_seq4096/retention/results_*.json.

Note: global_mmlu_ru is NOT in the lm-eval suite as configured; for RU we
plot xnli_ru only (the MMLU panel is empty for that subplot).

Checkpoint -> tokens (B):
    iter   0 -> 0.000 B  (matched-config Apertus-Base = our actual init)
    iter 119 -> 0.499 B
    iter 238 -> 0.998 B
    iter 477 -> 2.001 B
    iter 834 -> 3.498 B
    iter 1192 -> 5.000 B

Output: plot_retention_per_language.png next to this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPORTS = Path(__file__).resolve().parent
EVAL = REPORTS / "eval_data_cache_5b"
OUT = REPORTS / "plot_retention_per_language.png"

CHECKPOINTS = [
    ("0000119", 119, 0.499),
    ("0000238", 238, 0.998),
    ("0000477", 477, 2.001),
    ("0000834", 834, 3.498),
    ("0001192", 1192, 5.000),
]
LANGS = ["en", "fr", "de", "ru"]
SUBDOMAINS = [
    "business",
    "humanities",
    "medical",
    "other",
    "social_sciences",
    "stem",
]
LANG_TITLES = {"en": "English", "fr": "French", "de": "German", "ru": "Russian"}

# Matched-config Apertus-Base Path B = our true pre-CPT init state
# (weights loaded under our training geometry rope=500K / max_pos=4096 with
# NO training updates). Pulled 2026-05-30 from
# /capstor/.../eval_apertus_base_matched_rope500k_seq4096/retention/results_*.json.
MATCHED_BASE = {
    "global_mmlu_en": 0.6025,
    "global_mmlu_fr": 0.5425,
    "global_mmlu_de": 0.5725,
    # global_mmlu_ru: absent from the lm-eval suite (same as iter 119+)
    "xnli_en": 0.5112,
    "xnli_fr": 0.4859,
    "xnli_de": 0.4968,
    "xnli_ru": 0.4884,
}


def load_one(it_str: str) -> dict:
    with (EVAL / f"iter_{it_str}" / "retention" / "results.json").open() as f:
        return json.load(f)["results"]


def compute_lang_metrics() -> dict:
    """Returns {lang: {'tokens': [...], 'global_mmlu': [..|None], 'xnli': [...]}}.

    The first entry per series is the matched-config Apertus-Base = our actual
    pre-CPT init state (tokens=0). Then the 5 checkpoint marks follow.
    """
    series: dict[str, dict[str, list]] = {
        L: {"tokens": [0.0], "global_mmlu": [], "xnli": []} for L in LANGS
    }
    # Prepend matched-config base at tokens=0.
    for L in LANGS:
        if L == "ru":
            series[L]["global_mmlu"].append(None)
        else:
            series[L]["global_mmlu"].append(MATCHED_BASE[f"global_mmlu_{L}"])
        series[L]["xnli"].append(MATCHED_BASE[f"xnli_{L}"])
    # Then the trained-checkpoint marks.
    for it_str, _it, tok in CHECKPOINTS:
        r = load_one(it_str)
        for L in LANGS:
            series[L]["tokens"].append(tok)
            if L == "ru":
                series[L]["global_mmlu"].append(None)
            else:
                accs = [
                    r[f"global_mmlu_{L}_{s}"]["acc,none"] for s in SUBDOMAINS
                ]
                series[L]["global_mmlu"].append(sum(accs) / len(accs))
            series[L]["xnli"].append(r[f"xnli_{L}"]["acc,none"])
    return series


def main() -> None:
    series = compute_lang_metrics()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True)

    for ax, L in zip(axes.flat, LANGS):
        tokens = series[L]["tokens"]
        gmmlu = series[L]["global_mmlu"]
        xnli = series[L]["xnli"]

        # global_mmlu line (skip RU because all-None)
        if any(v is not None for v in gmmlu):
            ax.plot(
                tokens,
                gmmlu,
                "o-",
                color="C0",
                lw=2.2,
                ms=8,
                label=f"global_mmlu_{L} (macro-mean of 6 sub-domains)",
            )
            ax.axhline(
                MATCHED_BASE[f"global_mmlu_{L}"],
                ls=":",
                color="C0",
                alpha=0.55,
                lw=1.2,
                label=f"global_mmlu_{L} @ matched-config base (true init)",
            )
            for x, y in zip(tokens, gmmlu):
                ax.annotate(
                    f"{y:.3f}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(6, 8),
                    fontsize=8,
                    color="C0",
                )
        else:
            ax.text(
                0.5,
                0.72,
                "global_mmlu_ru NOT in eval suite\n(panel shows xnli_ru only)",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.5,
                color="dimgray",
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.7),
            )

        # xnli line (always available)
        ax.plot(
            tokens,
            xnli,
            "s-",
            color="C3",
            lw=2.0,
            ms=8,
            label=f"xnli_{L} (n=2490)",
        )
        ax.axhline(
            MATCHED_BASE[f"xnli_{L}"],
            ls=":",
            color="C3",
            alpha=0.55,
            lw=1.2,
            label=f"xnli_{L} @ matched-config base (true init)",
        )
        for x, y in zip(tokens, xnli):
            ax.annotate(
                f"{y:.3f}",
                (x, y),
                textcoords="offset points",
                xytext=(6, -14),
                fontsize=8,
                color="C3",
            )

        # Delta vs true init (matched-config Apertus-Base) callouts.
        end_gmmlu = gmmlu[-1] if gmmlu[-1] is not None else None
        end_xnli = xnli[-1]
        delta_lines = []
        if end_gmmlu is not None:
            base_gmmlu = MATCHED_BASE[f"global_mmlu_{L}"]
            delta_lines.append(
                f"Δ global_mmlu_{L} vs init: {(end_gmmlu - base_gmmlu) * 100:+.2f} pp"
            )
        base_xnli = MATCHED_BASE[f"xnli_{L}"]
        delta_lines.append(
            f"Δ xnli_{L} vs init: {(end_xnli - base_xnli) * 100:+.2f} pp"
        )
        ax.text(
            0.02,
            0.04,
            "\n".join(delta_lines),
            transform=ax.transAxes,
            fontsize=9,
            color="black",
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.8),
        )

        ax.set_title(f"{LANG_TITLES[L]} retention", fontsize=12)
        ax.set_xlabel("CPT tokens (B)")
        ax.set_ylabel("accuracy")
        ax.set_xlim(-0.25, 5.5)
        ax.set_ylim(0.40, 0.70)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Per-language retention — 04 Vanilla CPT 5 B run\n"
        "(global_mmlu macro-mean of 6 sub-domains + xnli aggregate; "
        "dotted lines = matched-config Apertus-Base = our true pre-CPT init; "
        "trajectory starts at tokens=0)",
        fontsize=12,
        y=1.00,
    )

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
