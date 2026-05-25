"""Vanilla vs TD only — focused 2-arm plots and gap analysis."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "per_iter_results"
ARMS = ["vanilla", "td"]
ITERS = [130, 260, 325, 390, 455, 476, 585, 715, 834]  # TD missing 325
TOK_PER_ITER = 1024 * 4096

V4 = json.loads(Path("/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/results.json").read_text())["results"]


def get_metric(blob, task, prefer_norm=False):
    if task not in blob:
        return None
    v = blob[task]
    if task == "xquad_el":
        return v.get("f1,none")
    if prefer_norm and "acc_norm,none" in v:
        return v.get("acc_norm,none")
    return v.get("acc,none")


TASKS = [
    ("mmlu", False, "EN_ret"),
    ("hellaswag", True, "EN_ret"),
    ("arc_easy", True, "EN_ret"),
    ("arc_challenge", True, "EN_ret"),
    ("piqa", True, "EN_ret"),
    ("winogrande", False, "EN_ret"),
    ("global_mmlu", False, "Multi"),
    ("xcopa", False, "Multi"),
    ("xnli", False, "Multi"),
    ("global_mmlu_full_el", False, "Greek"),
    ("include_base_44_greek_few_shot_en", False, "Greek"),
    ("belebele_ell_Grek", False, "Greek"),
    ("arc_challenge_mt_el", True, "Greek"),
    ("xnli_el", False, "Greek"),
    ("xquad_el", False, "Greek"),
    ("global_piqa_completions_ell_grek", True, "Greek"),
]

data = {arm: {} for arm in ARMS}
for arm in ARMS:
    for it in ITERS:
        p = ROOT / f"{arm}_iter{it}.json"
        if p.exists():
            data[arm][it] = json.loads(p.read_text())["results"]


def arm_group_avg(arm, it, group):
    if it not in data[arm]:
        return None
    vals = [get_metric(data[arm][it], t, pn) for t, pn, g in TASKS if g == group]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# ---------- Plot 1: three-panel group-averaged with TD-vs-Vanilla overlays ----------
groups = {"EN_ret": "English retention", "Multi": "Multilingual", "Greek": "Greek slice"}
group_tasks = {g: [t for t, _, gg in TASKS if gg == g] for g in groups}
COLORS = {"vanilla": "#1f77b4", "td": "#2ca02c"}
MARKERS = {"vanilla": "o", "td": "D"}

fig, axes = plt.subplots(3, 1, figsize=(10, 14))
for ax, (group, label) in zip(axes, groups.items()):
    # Faded per-task lines so the structure is visible behind the bold aggregate
    for task in group_tasks[group]:
        pn = next(p for t, p, g in TASKS if t == task)
        for arm in ARMS:
            xs, ys = [], []
            for it in ITERS:
                if it in data[arm]:
                    v = get_metric(data[arm][it], task, pn)
                    if v is not None:
                        xs.append(it * TOK_PER_ITER / 1e9)
                        ys.append(v)
            if xs:
                ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], alpha=0.25, linewidth=0.9, markersize=3.5)

    # Bold group-average lines
    for arm in ARMS:
        xs, ys = [], []
        for it in ITERS:
            v = arm_group_avg(arm, it, group)
            if v is not None:
                xs.append(it * TOK_PER_ITER / 1e9)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], linewidth=3.0, markersize=11, label=f"{arm} (group avg)")
            # Slope annotation (full window)
            xa = np.array(xs)
            ya = np.array(ys)
            slope = np.polyfit(xa, ya, 1)[0]
            ax.text(xs[-1] + 0.02, ys[-1], f"  slope = {slope*1000:+.1f} m.p./B", color=COLORS[arm], fontsize=9, va="center")

    # V4-HF reference at iter 0
    v4_vals = [get_metric(V4, t, pn) for t, pn, gg in TASKS if gg == group]
    v4_vals = [v for v in v4_vals if v is not None]
    if v4_vals:
        v4_avg = np.mean(v4_vals)
        ax.axhline(v4_avg, color="black", linestyle="--", alpha=0.4, label=f"V4-HF avg = {v4_avg:.3f}")

    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel(f"{label} — group avg")
    ax.set_title(f"{label}: Vanilla vs TD per-arm trajectory")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "plots" / "trajectories_van_td.png", dpi=120)
print("saved plots/trajectories_van_td.png")

# ---------- Plot 2: per-task 8-panel (7 Greek + English MMLU) ----------
PLOT_TASKS = [
    ("global_mmlu_full_el", False, "Greek MMLU"),
    ("include_base_44_greek_few_shot_en", False, "INCLUDE-44 Greek"),
    ("belebele_ell_Grek", False, "Belebele Greek"),
    ("arc_challenge_mt_el", True, "ARC-Challenge MT-el"),
    ("xnli_el", False, "XNLI Greek"),
    ("xquad_el", False, "XQuAD Greek (f1)"),
    ("global_piqa_completions_ell_grek", True, "PIQA Greek"),
    ("mmlu", False, "English MMLU"),
]

fig, axes = plt.subplots(2, 4, figsize=(22, 10))
for ax, (task, pn, label) in zip(axes.flat, PLOT_TASKS):
    for arm in ARMS:
        xs, ys = [], []
        for it in ITERS:
            if it in data[arm]:
                v = get_metric(data[arm][it], task, pn)
                if v is not None:
                    xs.append(it * TOK_PER_ITER / 1e9)
                    ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], linewidth=2.4, markersize=9, label=arm)

    v4 = get_metric(V4, task, pn)
    if v4 is not None:
        ax.axhline(v4, color="black", linestyle="--", alpha=0.45, label=f"V4-HF = {v4:.3f}")

    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel("Accuracy / f1")
    ax.set_title(label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "plots" / "trajectories_per_task_van_td.png", dpi=120)
print("saved plots/trajectories_per_task_van_td.png")


# ---------- Plot 3: TD-minus-Vanilla gap on Greek aggregate, with linear extrapolation ----------
xs_v, ys_v = [], []
xs_t, ys_t = [], []
for it in ITERS:
    v = arm_group_avg("vanilla", it, "Greek")
    t = arm_group_avg("td", it, "Greek")
    if v is not None:
        xs_v.append(it * TOK_PER_ITER / 1e9); ys_v.append(v)
    if t is not None:
        xs_t.append(it * TOK_PER_ITER / 1e9); ys_t.append(t)

# Compute gap at common iters
common_iters = [it for it in ITERS if it in data["vanilla"] and it in data["td"]]
xs_g = [it * TOK_PER_ITER / 1e9 for it in common_iters]
ys_g = [arm_group_avg("vanilla", it, "Greek") - arm_group_avg("td", it, "Greek") for it in common_iters]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left panel: two trajectories with linear fits extrapolated to 5B
ax = axes[0]
x_v_arr = np.array(xs_v); y_v_arr = np.array(ys_v)
x_t_arr = np.array(xs_t); y_t_arr = np.array(ys_t)
# Mid-window fit (iter 130-390 = 0.55-1.64 B)
mask_v = x_v_arr <= 1.7
mask_t = x_t_arr <= 1.7
slope_v, intercept_v = np.polyfit(x_v_arr[mask_v], y_v_arr[mask_v], 1)
slope_t, intercept_t = np.polyfit(x_t_arr[mask_t], y_t_arr[mask_t], 1)

x_ext = np.linspace(0.5, 5.0, 50)
ax.plot(xs_v, ys_v, "o-", color=COLORS["vanilla"], linewidth=3, markersize=10, label=f"vanilla (obs)")
ax.plot(x_ext, intercept_v + slope_v * x_ext, "--", color=COLORS["vanilla"], alpha=0.5, label=f"vanilla mid-fit ({slope_v*1000:+.1f} m.p./B)")
ax.plot(xs_t, ys_t, "D-", color=COLORS["td"], linewidth=3, markersize=10, label=f"td (obs)")
ax.plot(x_ext, intercept_t + slope_t * x_ext, "--", color=COLORS["td"], alpha=0.5, label=f"td mid-fit ({slope_t*1000:+.1f} m.p./B)")

# Crossover marker
if slope_t > slope_v:
    x_cross = (intercept_v - intercept_t) / (slope_t - slope_v)
    y_cross = intercept_v + slope_v * x_cross
    if 2.0 < x_cross < 5.0:
        ax.plot(x_cross, y_cross, "k*", markersize=18, label=f"linear crossover ~{x_cross:.1f} B")
        ax.axvline(x_cross, color="black", alpha=0.3, linestyle=":")

ax.axvline(2.0, color="gray", alpha=0.4, linestyle="-", label="2 B bakeoff budget")
ax.axvline(3.5, color="gray", alpha=0.35, linestyle=":", label="3.5 B continuation")
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("Greek aggregate (mean of 7 Greek tasks)")
ax.set_title("Vanilla vs TD on Greek aggregate, with mid-window linear extrapolation")
ax.legend(loc="best", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.3, 5.2)

# Right panel: the gap (vanilla - td) over time
ax2 = axes[1]
ax2.plot(xs_g, np.array(ys_g) * 100, "ko-", linewidth=2, markersize=10, label="vanilla − td gap (Greek agg)")
ax2.axhline(0, color="black", linestyle="-", alpha=0.5)
ax2.fill_between(xs_g, 0, np.array(ys_g) * 100, where=(np.array(ys_g) > 0), color="#1f77b4", alpha=0.15, label="vanilla ahead")
ax2.fill_between(xs_g, 0, np.array(ys_g) * 100, where=(np.array(ys_g) < 0), color="#2ca02c", alpha=0.15, label="td ahead")
# Linear extrapolation of the gap
gap_arr = np.array(ys_g) * 100
gap_xs = np.array(xs_g)
gap_slope, gap_intercept = np.polyfit(gap_xs, gap_arr, 1)
ext_xs = np.linspace(0.5, 5.0, 50)
ax2.plot(ext_xs, gap_intercept + gap_slope * ext_xs, "k--", alpha=0.4, label=f"linear gap fit ({gap_slope*10:+.2f} pp/B)")
ax2.axvline(2.0, color="gray", alpha=0.4, linestyle="-", label="2 B bakeoff budget")
ax2.axvline(3.5, color="gray", alpha=0.35, linestyle=":", label="3.5 B continuation")
# Where does the linear gap hit zero?
if gap_slope < 0:
    x_zero = -gap_intercept / gap_slope
    if 2.0 < x_zero < 5.0:
        ax2.axvline(x_zero, color="black", alpha=0.3, linestyle=":", label=f"gap → 0 at ~{x_zero:.1f} B")
ax2.set_xlabel("Tokens consumed (B)")
ax2.set_ylabel("Greek-aggregate gap (p.p., vanilla − td)")
ax2.set_title("Vanilla-TD gap on Greek aggregate")
ax2.legend(loc="best", fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0.3, 5.2)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "plots" / "trajectories_van_td_gap.png", dpi=120)
print("saved plots/trajectories_van_td_gap.png")

# Numerical summary
print("\n=== Vanilla vs TD: group-averaged trajectory + slopes ===\n")
print(f"{'iter':>6}{'tokens':>10}{'van_EN':>10}{'td_EN':>10}{'van_Multi':>12}{'td_Multi':>12}{'van_Gr':>10}{'td_Gr':>10}{'Gr_gap':>10}")
for it in ITERS:
    line = f"{it:>6}{it * TOK_PER_ITER / 1e9:>10.3f}"
    for group in ("EN_ret", "Multi", "Greek"):
        for arm in ("vanilla", "td"):
            v = arm_group_avg(arm, it, group)
            line += f"{v:>10.4f}" if v is not None else "       n/a"
            if group == "Greek" and arm == "td":
                vg = arm_group_avg("vanilla", it, "Greek")
                if v is not None and vg is not None:
                    line += f"{(vg - v)*100:>+10.2f}"
                else:
                    line += "       n/a"
    print(line)

print()
print("=== Slopes on group aggregates (10^-3 acc per B-token) ===")
for group in ("EN_ret", "Multi", "Greek"):
    line = f"{groups[group]:<22}"
    for arm in ("vanilla", "td"):
        xs, ys = [], []
        for it in ITERS:
            v = arm_group_avg(arm, it, group)
            if v is not None:
                xs.append(it * TOK_PER_ITER / 1e9)
                ys.append(v)
        xs_a, ys_a = np.array(xs), np.array(ys)
        s_full = np.polyfit(xs_a, ys_a, 1)[0]
        mask = xs_a <= 1.7
        s_mid = np.polyfit(xs_a[mask], ys_a[mask], 1)[0]
        line += f"  {arm}: full={s_full*1000:+.2f}, mid={s_mid*1000:+.2f}  "
    print(line)
