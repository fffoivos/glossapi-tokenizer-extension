"""Tokenizer-fair intrinsic trajectories per arm (BPC, NLL/char, NLL/word).
LOWER IS BETTER for every metric in this script."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent / "per_iter_results" / "intrinsic"
PLOTS = Path(__file__).resolve().parent / "plots"
TOK_PER_ITER = 1024 * 4096

ARMS = ["vanilla", "retok", "centroid", "td"]
ITERS = [65, 130, 195, 260, 325, 390, 455, 476, 585, 715, 834]
COLORS = {"vanilla": "#1f77b4", "retok": "#ff7f0e", "centroid": "#d62728", "td": "#2ca02c"}
MARKERS = {"vanilla": "o", "retok": "s", "centroid": "^", "td": "D"}

data = {arm: {} for arm in ARMS}
for arm in ARMS:
    for it in ITERS:
        p = ROOT / f"{arm}_iter{it:03d}_fair.json"
        if p.exists():
            data[arm][it] = json.loads(p.read_text())["global"]

METRICS = [
    ("bpc_bits_per_byte", "BPC (bits / byte) ↓"),
    ("nll_per_char", "NLL / char ↓"),
    ("nll_per_word", "NLL / word ↓"),
    ("nll_per_token", "NLL / token ↓"),
    ("chars_per_token", "chars / token ↑"),
    ("tokens_per_word", "tokens / word ↓"),
]


def fetch(arm, it, key):
    return data[arm].get(it, {}).get(key)


# ----- Plot 1: 4-arm trajectory across all 6 metrics -----
fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for ax, (key, label) in zip(axes.flat, METRICS):
    for arm in ARMS:
        xs, ys = [], []
        for it in ITERS:
            v = fetch(arm, it, key)
            if v is not None:
                xs.append(it * TOK_PER_ITER / 1e9)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], linewidth=2.4, markersize=9, label=arm)
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel(label)
    ax.set_title(label.replace(" ↓", " (lower better)").replace(" ↑", " (higher better)"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("Tokenizer-fair intrinsic metrics on heldout 500-doc Greek slice (all available arms, iter 65 to 834)", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS / "intrinsic_trajectories.png", dpi=120, bbox_inches="tight")
print("saved plots/intrinsic_trajectories.png")

# ----- Plot 2: Vanilla vs TD only, BPC and NLL/char -----
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, key in zip(axes, ["bpc_bits_per_byte", "nll_per_char", "nll_per_word"]):
    for arm in ("vanilla", "td"):
        xs, ys = [], []
        for it in ITERS:
            v = fetch(arm, it, key)
            if v is not None:
                xs.append(it * TOK_PER_ITER / 1e9)
                ys.append(v)
        if xs:
            ax.plot(xs, ys, marker=MARKERS[arm], color=COLORS[arm], linewidth=2.8, markersize=11, label=arm)
            xa = np.array(xs); ya = np.array(ys)
            if len(xa) >= 2:
                slope = np.polyfit(xa, ya, 1)[0]
                ax.text(xs[-1] + 0.02, ys[-1], f"  {slope*1000:+.1f} m/B", color=COLORS[arm], fontsize=9, va="center")
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel(key.replace("_", " "))
    ax.set_title(f"{key.replace('_', ' ')} (lower=better)")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
plt.suptitle("Vanilla vs TD — Greek language modelling quality on heldout 500-doc slice", fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(PLOTS / "intrinsic_van_td.png", dpi=120, bbox_inches="tight")
print("saved plots/intrinsic_van_td.png")

# Numeric table
print()
print("=== BPC trajectory (bits/byte, LOWER IS BETTER) ===")
print(f"{'iter':>6}{'tokens':>10}{'vanilla':>10}{'retok':>10}{'centroid':>10}{'td':>10}")
for it in ITERS:
    line = f"{it:>6}{it * TOK_PER_ITER / 1e9:>10.3f}"
    for arm in ARMS:
        v = fetch(arm, it, "bpc_bits_per_byte")
        line += f"{v:>10.4f}" if v is not None else f"{'n/a':>10}"
    print(line)

print()
print("=== NLL / char (LOWER IS BETTER) ===")
print(f"{'iter':>6}{'tokens':>10}{'vanilla':>10}{'retok':>10}{'centroid':>10}{'td':>10}")
for it in ITERS:
    line = f"{it:>6}{it * TOK_PER_ITER / 1e9:>10.3f}"
    for arm in ARMS:
        v = fetch(arm, it, "nll_per_char")
        line += f"{v:>10.4f}" if v is not None else f"{'n/a':>10}"
    print(line)

print()
print("=== Full-window slopes on intrinsic metrics (Δ per B-token) ===")
for key, label in METRICS:
    print(f"\n{label}:")
    for arm in ARMS:
        xs, ys = [], []
        for it in ITERS:
            v = fetch(arm, it, key)
            if v is not None:
                xs.append(it * TOK_PER_ITER / 1e9); ys.append(v)
        if len(xs) >= 2:
            slope = np.polyfit(xs, ys, 1)[0]
            direction = "↓ better" if slope < 0 else "↑ worse" if "↓" in label else ("↑ better" if slope > 0 else "↓ worse")
            print(f"  {arm:<10} slope = {slope:+.5f} / B   ({direction})")
