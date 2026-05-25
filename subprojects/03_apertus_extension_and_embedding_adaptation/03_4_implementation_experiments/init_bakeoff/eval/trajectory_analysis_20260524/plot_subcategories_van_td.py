"""Per-sub-category Vanilla-vs-TD trajectories for the two Greek MMLU-style benchmarks:
- `global_mmlu_full_el` — 4 high-level categories (humanities, social_sciences, stem, other) + aggregate.
- `include_base_44_greek_few_shot_en` — 7 INCLUDE-44 subjects + aggregate.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "per_iter_results"
PLOTS = Path(__file__).resolve().parent / "plots"
PLOTS.mkdir(exist_ok=True)

ARMS = ["vanilla", "td"]
ITERS = [130, 260, 325, 390, 455, 476, 585, 715, 834]  # TD skips 325
TOK_PER_ITER = 1024 * 4096

V4 = json.loads(Path("/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/results.json").read_text())["results"]


def get_acc(blob, task):
    v = blob.get(task)
    if v is None:
        return None
    return v.get("acc,none")


data = {arm: {} for arm in ARMS}
for arm in ARMS:
    for it in ITERS:
        p = ROOT / f"{arm}_iter{it}.json"
        if p.exists():
            data[arm][it] = json.loads(p.read_text())["results"]

COLORS = {"vanilla": "#1f77b4", "td": "#2ca02c"}
MARKERS = {"vanilla": "o", "td": "D"}

# ============================================================
# PLOT A: global_mmlu_full_el — 4 category rollups + aggregate
# ============================================================
GMMLU_PANELS = [
    ("global_mmlu_full_el", "Aggregate (4 categories)"),
    ("global_mmlu_full_el_humanities", "Humanities"),
    ("global_mmlu_full_el_social_sciences", "Social Sciences"),
    ("global_mmlu_full_el_stem", "STEM"),
    ("global_mmlu_full_el_other", "Other"),
]

fig, axes = plt.subplots(1, 5, figsize=(24, 5))
print(f"{'task':<42}{'V4-HF':>10}{'van130':>10}{'van834':>10}{'td130':>10}{'td834':>10}{'dvan':>10}{'dtd':>10}{'best834':>10}")
print("-" * 132)
for ax, (task, label) in zip(axes, GMMLU_PANELS):
    plot_data = {}
    for arm in ARMS:
        xs, ys = [], []
        for it in ITERS:
            if it in data[arm]:
                v = get_acc(data[arm][it], task)
                if v is not None:
                    xs.append(it * TOK_PER_ITER / 1e9)
                    ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], linewidth=2.4, markersize=9, label=arm)
            plot_data[arm] = (xs, ys)
    v4 = get_acc(V4, task)
    if v4 is not None:
        ax.axhline(v4, color="black", linestyle="--", alpha=0.45, label=f"V4-HF = {v4:.3f}")
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel("Accuracy")
    ax.set_title(label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # numeric summary
    van_xs, van_ys = plot_data.get("vanilla", ([], []))
    td_xs, td_ys = plot_data.get("td", ([], []))
    van_130 = van_ys[0] if van_ys else None
    van_476 = van_ys[-1] if van_ys else None
    td_130 = td_ys[0] if td_ys else None
    td_476 = td_ys[-1] if td_ys else None
    d_v = (van_476 - van_130) if (van_130 is not None and van_476 is not None) else None
    d_t = (td_476 - td_130) if (td_130 is not None and td_476 is not None) else None
    best = "vanilla" if (van_476 or 0) > (td_476 or 0) else "td"
    def f(x): return f"{x:.4f}" if x is not None else "  n/a"
    def fpp(x): return f"{x*100:+.2f} pp" if x is not None else "  n/a"
    print(f"{task:<42}{f(v4):>10}{f(van_130):>10}{f(van_476):>10}{f(td_130):>10}{f(td_476):>10}{fpp(d_v):>10}{fpp(d_t):>10}{best:>10}")

plt.suptitle("global_mmlu_full_el - Greek MMLU by category, Vanilla vs TD over 0.55 to 3.5 B tokens", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / "global_mmlu_full_el_subcategories_van_td.png", dpi=120, bbox_inches="tight")
print(f"\nsaved plots/global_mmlu_full_el_subcategories_van_td.png\n")

# ============================================================
# PLOT B: include_base_44_greek_few_shot_en — 7 subjects + agg
# ============================================================
INC44_PANELS = [
    ("include_base_44_greek_few_shot_en", "Aggregate (7 subjects)"),
    ("include_base_44_greek_few_shot_en_arts_humanities", "Arts / Humanities"),
    ("include_base_44_greek_few_shot_en_business_commerce", "Business / Commerce"),
    ("include_base_44_greek_few_shot_en_health_oriented_education", "Health-Oriented Education"),
    ("include_base_44_greek_few_shot_en_medical_license", "Medical License"),
    ("include_base_44_greek_few_shot_en_professional_certification", "Professional Certification"),
    ("include_base_44_greek_few_shot_en_social_science", "Social Science"),
    ("include_base_44_greek_few_shot_en_stem", "STEM"),
]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
print(f"{'task':<58}{'V4-HF':>10}{'van130':>10}{'van834':>10}{'td130':>10}{'td834':>10}{'dvan':>12}{'dtd':>12}{'best834':>10}")
print("-" * 154)
for ax, (task, label) in zip(axes.flat, INC44_PANELS):
    plot_data = {}
    for arm in ARMS:
        xs, ys = [], []
        for it in ITERS:
            if it in data[arm]:
                v = get_acc(data[arm][it], task)
                if v is not None:
                    xs.append(it * TOK_PER_ITER / 1e9)
                    ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], linewidth=2.4, markersize=9, label=arm)
            plot_data[arm] = (xs, ys)
    v4 = get_acc(V4, task)
    if v4 is not None:
        ax.axhline(v4, color="black", linestyle="--", alpha=0.45, label=f"V4-HF = {v4:.3f}")
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel("Accuracy")
    ax.set_title(label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    van_xs, van_ys = plot_data.get("vanilla", ([], []))
    td_xs, td_ys = plot_data.get("td", ([], []))
    van_130 = van_ys[0] if van_ys else None
    van_476 = van_ys[-1] if van_ys else None
    td_130 = td_ys[0] if td_ys else None
    td_476 = td_ys[-1] if td_ys else None
    d_v = (van_476 - van_130) if (van_130 is not None and van_476 is not None) else None
    d_t = (td_476 - td_130) if (td_130 is not None and td_476 is not None) else None
    best = "vanilla" if (van_476 or 0) > (td_476 or 0) else "td"
    def f(x): return f"{x:.4f}" if x is not None else "  n/a"
    def fpp(x): return f"{x*100:+.2f} pp" if x is not None else "  n/a"
    print(f"{task:<58}{f(v4):>10}{f(van_130):>10}{f(van_476):>10}{f(td_130):>10}{f(td_476):>10}{fpp(d_v):>12}{fpp(d_t):>12}{best:>10}")

plt.suptitle("include_base_44_greek_few_shot_en — native-Greek INCLUDE-44 by subject, Vanilla vs TD", fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS / "include_base_44_greek_subjects_van_td.png", dpi=120, bbox_inches="tight")
print(f"\nsaved plots/include_base_44_greek_subjects_van_td.png")
