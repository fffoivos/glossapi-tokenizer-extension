"""Parse Megatron iteration logs for the four bakeoff arms.

Raw `lm loss` is per-target-token CE in nats. It is dense and useful for
within-arm health, but it is not tokenizer-fair across Vanilla vs extended
vocab arms. Newer patched logs may also contain dense tokenizer-fair fields:
`bpb`, `bpt`, `base_loss`, `new_loss`, and `n_new`. When those exist, this
script emits BPB and base/new diagnostic plots; otherwise it keeps the raw-LM
plots explicitly labeled as diagnostic only.

Each .out file under per_iter_results/training_logs/ contains lines like:
  iteration   123/   476 | consumed samples: ... | consumed tokens: 1.234B | ... | lm loss: 1.234567E+00 | bpb: 0.5701 | bpt: 4.103 | base_loss: ... | new_loss: ... | n_new: 482 |

We parse those, dedupe by (arm, iteration), and plot one trajectory per arm.
"""
import csv
import math
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "per_iter_results" / "training_logs"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

# Map filename → arm
ARM_FROM_PREFIX = [
    ("bakeoff_vanilla", "vanilla"),
    ("bakeoff_resume2_vanilla", "vanilla"),
    ("3p5_vanilla", "vanilla"),
    ("5b_vanilla", "vanilla"),
    ("bakeoff_retok", "retok"),
    ("bakeoff_resume2_retok", "retok"),
    ("3p5_retok", "retok"),
    ("bakeoff_centroid", "centroid"),
    ("bakeoff_resume2_centroid", "centroid"),
    ("td_l11_2b", "td"),
    ("td_l11_2b_resume", "td"),
    ("3p5_td_layer11", "td"),
    ("5b_td_layer11", "td"),
]

FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?|nan|NaN|inf|Inf|-inf|-Inf"
ITER_RE = re.compile(r"iteration\s+(\d+)/\s*\d+\s+\|.*?consumed tokens:\s+([\d.]+)B")


def _float_or_none(value):
    if value is None:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _field(line, *names):
    for name in names:
        match = re.search(r"\b%s:\s*(%s)" % (re.escape(name), FLOAT_RE), line)
        if match:
            return _float_or_none(match.group(1))
    return None


def _int_field(line, name):
    match = re.search(r"\b%s:\s*(\d+)" % re.escape(name), line)
    return int(match.group(1)) if match else None

# Parse all logs into a deduped per-arm dict keyed by iteration
per_arm = {"vanilla": {}, "retok": {}, "centroid": {}, "td": {}}
for f in sorted(LOGS.glob("*.out")):
    arm = None
    for pfx, a in ARM_FROM_PREFIX:
        if f.name.startswith(pfx + "-") or f.name.startswith(pfx + "_"):
            arm = a
            break
    if arm is None:
        continue
    for line in f.read_text(errors="ignore").splitlines():
        m = ITER_RE.search(line)
        if not m:
            continue
        lm_loss = _field(line, "lm loss")
        if lm_loss is None:
            continue
        it = int(m.group(1))
        tokens = float(m.group(2))
        # last value wins (resume overwrites earlier partial)
        per_arm[arm][it] = {
            "tokens_b": tokens,
            "lm_loss": lm_loss,
            "bpb": _field(line, "bpb"),
            "bpt": _field(line, "bpt", "bytes_per_token_batch"),
            "base_loss": _field(line, "base_loss"),
            "new_loss": _field(line, "new_loss"),
            "n_new": _int_field(line, "n_new"),
            "source": f.name,
        }

# Quick sanity
print("=== rows parsed per arm ===")
for arm in per_arm:
    pts = per_arm[arm]
    print(f"  {arm:<10} {len(pts):>4} iters  (max iter = {max(pts) if pts else 'n/a'})")

# Emit CSV
CSV_PATH = LOGS / "training_loss_combined.csv"
with CSV_PATH.open("w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["arm", "iteration", "tokens_b", "lm_loss", "bpb", "bpt", "base_loss", "new_loss", "n_new", "source"])
    for arm, pts in per_arm.items():
        for it in sorted(pts):
            row = pts[it]
            w.writerow([
                arm,
                it,
                row["tokens_b"],
                row["lm_loss"],
                row["bpb"] if row["bpb"] is not None else "",
                row["bpt"] if row["bpt"] is not None else "",
                row["base_loss"] if row["base_loss"] is not None else "",
                row["new_loss"] if row["new_loss"] is not None else "",
                row["n_new"] if row["n_new"] is not None else "",
                row["source"],
            ])
print(f"\nwrote {CSV_PATH}")

COLORS = {"vanilla": "#1f77b4", "retok": "#ff7f0e", "centroid": "#d62728", "td": "#2ca02c"}

# ---------- Plot 1: 4-arm linear raw LM loss vs tokens ----------
fig, ax = plt.subplots(figsize=(13, 7))
for arm in ("vanilla", "retok", "centroid", "td"):
    pts = per_arm[arm]
    if not pts:
        continue
    xs = []
    ys = []
    for it in sorted(pts):
        xs.append(pts[it]["tokens_b"])
        ys.append(pts[it]["lm_loss"])
    ax.plot(xs, ys, color=COLORS[arm], linewidth=1.3, alpha=0.9, label=f"{arm} (max iter {max(pts)})")

# Mark major checkpoint boundaries
for x, label in [(2.0, "2.0B (bakeoff end)"), (3.5, "3.5B (continuation end)"), (4.25, "~4.25B (5B-run last seen)")]:
    ax.axvline(x, color="gray", linestyle=":", alpha=0.5)
    ax.text(x, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] else 5.5, label, rotation=90, fontsize=8, va="top", ha="right", color="gray")

ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("Raw per-token CE (nats)")
ax.set_title("Training raw LM loss vs tokens — diagnostic only, not tokenizer-fair")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "training_loss.png", dpi=120, bbox_inches="tight")
print(f"saved plots/training_loss.png")

# ---------- Plot 2: log-y zoom (post-warmup, iter > 20) ----------
fig, ax = plt.subplots(figsize=(13, 7))
for arm in ("vanilla", "retok", "centroid", "td"):
    pts = per_arm[arm]
    if not pts:
        continue
    xs, ys = [], []
    for it in sorted(pts):
        if it < 20:
            continue
        xs.append(pts[it]["tokens_b"])
        ys.append(pts[it]["lm_loss"])
    if xs:
        ax.plot(xs, ys, color=COLORS[arm], linewidth=1.3, alpha=0.9, label=f"{arm}")

for x, label in [(2.0, "2.0B"), (3.5, "3.5B")]:
    ax.axvline(x, color="gray", linestyle=":", alpha=0.5)
    ax.text(x, 0.99, label, rotation=90, fontsize=8, va="top", ha="right", color="gray", transform=ax.get_xaxis_transform())

ax.set_yscale("log")
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("Raw per-token CE (log scale)")
ax.set_title("Training raw LM loss — diagnostic only, post-warmup (iter > 20)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(PLOTS / "training_loss_logy.png", dpi=120, bbox_inches="tight")
print(f"saved plots/training_loss_logy.png")

# ---------- Plot 3: Vanilla vs TD only, linear scale, zoom on the meaningful range ----------
fig, ax = plt.subplots(figsize=(13, 7))
for arm in ("vanilla", "td"):
    pts = per_arm[arm]
    if not pts:
        continue
    xs, ys = [], []
    for it in sorted(pts):
        xs.append(pts[it]["tokens_b"])
        ys.append(pts[it]["lm_loss"])
    ax.plot(xs, ys, color=COLORS[arm], linewidth=1.5, alpha=0.9, label=f"{arm}")

for x, label in [(2.0, "2.0B (bakeoff end)"), (3.5, "3.5B (continuation end)")]:
    ax.axvline(x, color="gray", linestyle=":", alpha=0.5)
    ax.text(x, 0.99, label, rotation=90, fontsize=8, va="top", ha="right", color="gray", transform=ax.get_xaxis_transform())

ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("Raw per-token CE (nats)")
ax.set_title("Training raw LM loss — Vanilla vs TD layer11 (diagnostic only)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOTS / "training_loss_van_td.png", dpi=120, bbox_inches="tight")
print(f"saved plots/training_loss_van_td.png")

# ---------- Plot 4: measured dense BPB, when patched logs provide it ----------
has_bpb = any(any(row.get("bpb") is not None for row in pts.values()) for pts in per_arm.values())
if has_bpb:
    fig, ax = plt.subplots(figsize=(13, 7))
    for arm in ("vanilla", "retok", "centroid", "td"):
        pts = per_arm[arm]
        xs, ys = [], []
        for it in sorted(pts):
            if pts[it]["bpb"] is None:
                continue
            xs.append(pts[it]["tokens_b"])
            ys.append(pts[it]["bpb"])
        if xs:
            ax.plot(xs, ys, color=COLORS[arm], linewidth=1.4, alpha=0.9, label=arm)
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel("Measured batch BPB (bits / byte)  ↓")
    ax.set_title("Dense tokenizer-fair training BPB from Megatron logs")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "training_bpb.png", dpi=120, bbox_inches="tight")
    print("saved plots/training_bpb.png")

has_split = any(
    any(row.get("base_loss") is not None or row.get("new_loss") is not None for row in pts.values())
    for pts in per_arm.values()
)
if has_split:
    fig, ax = plt.subplots(figsize=(13, 7))
    for arm in ("vanilla", "retok", "centroid", "td"):
        pts = per_arm[arm]
        xs_base, ys_base, xs_new, ys_new = [], [], [], []
        for it in sorted(pts):
            row = pts[it]
            if row["base_loss"] is not None:
                xs_base.append(row["tokens_b"])
                ys_base.append(row["base_loss"])
            if row["new_loss"] is not None:
                xs_new.append(row["tokens_b"])
                ys_new.append(row["new_loss"])
        if xs_base:
            ax.plot(xs_base, ys_base, color=COLORS[arm], linewidth=1.1, alpha=0.75, label=f"{arm} base")
        if xs_new:
            ax.plot(xs_new, ys_new, color=COLORS[arm], linestyle="--", linewidth=1.2, alpha=0.9, label=f"{arm} new")
    ax.set_xlabel("Tokens consumed (B)")
    ax.set_ylabel("Per-token CE (nats)")
    ax.set_title("Base-vs-new target CE split from Megatron logs")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "training_base_new_loss.png", dpi=120, bbox_inches="tight")
    print("saved plots/training_base_new_loss.png")

# ---------- Summary table ----------
print("\n=== Final raw LM loss per arm + key checkpoints (diagnostic only) ===")
print(f"{'arm':<10}{'max iter':>10}{'tokens(B)':>12}{'final loss':>12}{'final bpb':>12}{'n_new':>10}{'loss@0.5B':>12}{'loss@1.0B':>12}{'loss@2.0B':>12}{'loss@3.5B':>12}")
for arm in ("vanilla", "retok", "centroid", "td"):
    pts = per_arm[arm]
    if not pts:
        continue
    sorted_iters = sorted(pts.keys())
    max_iter = sorted_iters[-1]
    final = pts[max_iter]
    # Closest-iter lookups for 0.5 / 1.0 / 2.0 / 3.5 B
    def near(target_tokens):
        best = None
        best_d = float("inf")
        for it in pts:
            tk = pts[it]["tokens_b"]
            d = abs(tk - target_tokens)
            if d < best_d:
                best_d = d
                best = pts[it]["lm_loss"]
        return best if best is not None else float("nan")
    final_bpb = final["bpb"] if final["bpb"] is not None else float("nan")
    n_new = final["n_new"] if final["n_new"] is not None else "n/a"
    print(f"{arm:<10}{max_iter:>10}{final['tokens_b']:>12.3f}{final['lm_loss']:>12.4f}{final_bpb:>12.4f}{str(n_new):>10}{near(0.5):>12.4f}{near(1.0):>12.4f}{near(2.0):>12.4f}{near(3.5):>12.4f}")
