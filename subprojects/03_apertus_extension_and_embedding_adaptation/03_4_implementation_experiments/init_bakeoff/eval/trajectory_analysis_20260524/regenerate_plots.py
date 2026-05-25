"""Robust slope analysis — full trajectory + tail, with LR-cooldown caveat noted."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "per_iter_results"
ARMS = ["vanilla", "retok", "centroid", "td"]
ITERS = [130, 260, 325, 390, 455, 476, 585, 715, 834]
TOK_PER_ITER = 1024 * 4096

V4 = json.loads(Path("/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/v4_baseline_corrected_20260521/results.json").read_text())["results"]

def get_metric(blob, task, prefer_norm=False):
    if task not in blob: return None
    v = blob[task]
    if task == "xquad_el":
        return v.get("f1,none")
    if prefer_norm and "acc_norm,none" in v: return v.get("acc_norm,none")
    return v.get("acc,none")

# task, prefer_norm, group
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
    if it not in data[arm]: return None
    vals = [get_metric(data[arm][it], t, pn) for t, pn, g in TASKS if g == group]
    vals = [v for v in vals if v is not None]
    return np.mean(vals) if vals else None

# Group-averaged trajectories
groups = {"EN_ret": "English retention", "Multi": "Multilingual", "Greek": "Greek slice"}
group_traj = {}
for arm in ARMS:
    group_traj[arm] = {}
    for group in groups:
        xs, ys = [], []
        for it in ITERS:
            v = arm_group_avg(arm, it, group)
            if v is not None:
                xs.append(it * TOK_PER_ITER / 1e9)
                ys.append(v)
        group_traj[arm][group] = (xs, ys)

print("=== Group-averaged metric vs tokens ===\n")
for arm in ARMS:
    print(f"-- {arm} --")
    for group in groups:
        xs, ys = group_traj[arm][group]
        if xs:
            traj_str = " → ".join(f"{y:.4f}@{x:.2f}B" for x, y in zip(xs, ys))
            print(f"  {group:<8}: {traj_str}")
    print()

# Compute slopes with three windows for each group
print("=== Slopes by window (10^-3 acc per B-token) ===\n")
print(f"{'group':<8}{'arm':<10}{'full(130-476)':>16}{'mid(130-390)':>14}{'tail(390-476)':>16}")
for group in groups:
    for arm in ARMS:
        xs, ys = group_traj[arm][group]
        if len(xs) < 4: continue
        xs, ys = np.array(xs), np.array(ys)
        # full
        s_full = np.polyfit(xs, ys, 1)[0]
        # mid (130-390 only)
        mask_mid = xs <= 1.7
        s_mid = np.polyfit(xs[mask_mid], ys[mask_mid], 1)[0] if mask_mid.sum() >= 2 else None
        # tail (390-476)
        mask_tail = xs >= 1.5
        s_tail = np.polyfit(xs[mask_tail], ys[mask_tail], 1)[0] if mask_tail.sum() >= 2 else None
        def fmt(s): return f"{s*1000:+.2f}" if s is not None else "n/a"
        print(f"{group:<8}{arm:<10}{fmt(s_full):>16}{fmt(s_mid):>14}{fmt(s_tail):>16}")
    print()

# Tail-only slopes are contaminated by WSD 1-sqrt cooldown — LR drops from peak 1.5e-5 to 1.5e-6 across the run
# Mid-window slope is the most informative for "is the arm still learning"
print("\n=== Linear extrapolation: TD vs Vanilla on Greek aggregate, using MID-WINDOW slope ===")
xs_v, ys_v = group_traj["vanilla"]["Greek"]
xs_t, ys_t = group_traj["td"]["Greek"]
xs_v, ys_v = np.array(xs_v), np.array(ys_v)
xs_t, ys_t = np.array(xs_t), np.array(ys_t)
mask = xs_v <= 1.7
slope_v_mid = np.polyfit(xs_v[mask], ys_v[mask], 1)[0]
mask_t = xs_t <= 1.7
slope_t_mid = np.polyfit(xs_t[mask_t], ys_t[mask_t], 1)[0]
final_x = xs_v[-1]
final_v = ys_v[-1]
final_t = ys_t[-1]
print(f"  At {final_x:.1f}B: vanilla = {final_v:.4f}, td = {final_t:.4f}  (gap = {final_v-final_t:+.4f})")
print(f"  Mid-window slopes: vanilla = {slope_v_mid*1000:+.3f} m.p./B, td = {slope_t_mid*1000:+.3f} m.p./B")
if slope_t_mid > slope_v_mid:
    dt = (final_v - final_t) / (slope_t_mid - slope_v_mid)
    target = final_x + dt
    print(f"  -> TD slope > Vanilla. Linear crossover at ~{target:.1f}B tokens (delta = {dt:.1f}B beyond current {final_x:.1f}B)")
else:
    print(f"  -> TD slope < Vanilla. Linear extrapolation does NOT predict TD overtakes Vanilla on Greek aggregate.")

print("\n=== Full-window slopes (more stable) for the same comparison ===")
s_v_full = np.polyfit(xs_v, ys_v, 1)[0]
s_t_full = np.polyfit(xs_t, ys_t, 1)[0]
print(f"  vanilla = {s_v_full*1000:+.3f}, td = {s_t_full*1000:+.3f} m.p./B")
if s_t_full > s_v_full:
    dt = (final_v - final_t) / (s_t_full - s_v_full)
    target = final_x + dt
    print(f"  -> Linear crossover at ~{target:.1f}B tokens")
else:
    print(f"  -> No crossover predicted")

# Now plot the four key Greek tasks individually
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
GREEK_TASKS = [
    ("global_mmlu_full_el", False, "Greek MMLU"),
    ("include_base_44_greek_few_shot_en", False, "INCLUDE-44 Greek"),
    ("belebele_ell_Grek", False, "Belebele Greek"),
    ("arc_challenge_mt_el", True, "ARC-Challenge mt el"),
    ("xnli_el", False, "XNLI el"),
    ("xquad_el", False, "XQuAD el (f1)"),
    ("global_piqa_completions_ell_grek", True, "PIQA-Greek"),
]
EN_TASKS = [
    ("mmlu", False, "MMLU (English)"),
]
PLOT_TASKS = GREEK_TASKS + EN_TASKS
COLORS = {"vanilla": "tab:blue", "retok": "tab:orange", "centroid": "tab:red", "td": "tab:green"}
MARKERS = {"vanilla": "o", "retok": "s", "centroid": "^", "td": "D"}
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
            ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], linewidth=2.0, markersize=8, label=arm)
    # V4-HF reference
    v4 = get_metric(V4, task, pn)
    if v4 is not None:
        ax.axhline(v4, color="black", linestyle="--", alpha=0.4, label=f"V4-HF = {v4:.3f}")
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel("Accuracy / f1")
    ax.set_title(label)
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / "plots" / "trajectories_per_task.png", dpi=110)
print(f"\nsaved {Path(__file__).resolve().parent / 'plots' / 'trajectories_per_task.png'}")
