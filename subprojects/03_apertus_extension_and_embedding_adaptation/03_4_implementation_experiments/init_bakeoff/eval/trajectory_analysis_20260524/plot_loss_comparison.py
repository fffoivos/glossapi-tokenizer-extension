"""Fair-loss-comparison plot for Vanilla vs TD (and all 4 arms).

Three panels per figure:
  (A) Raw training LM loss vs tokens — DENSE but UNFAIR across vocab sizes /
      tokenizer rates. The plot we already have; included here for contrast.
  (B) Heldout BPC vs tokens — SPARSE (eval checkpoints only) but FAIR.
      Tokenizer-invariant: per-byte log-prob on a fixed Greek heldout slice.
      This is the ground-truth TD-vs-Vanilla curve.
  (C) Dense BPB during training when patched Megatron logs provide `bpb`;
      otherwise a clearly labeled BPB proxy with eval-checkpoint heldout BPC
      markers overlaid as ground truth. The proxy divides per-token CE by a
      mix-weighted bytes/token (70% Greek heldout + 30% multilingual replay
      assumed at 3.5 bytes/token). APPROXIMATE — see caveats in this file.
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "per_iter_results" / "training_logs" / "training_loss_combined.csv"
INTRINSIC = ROOT / "per_iter_results" / "intrinsic"
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

ARMS = ["vanilla", "retok", "centroid", "td"]
COLORS = {"vanilla": "#1f77b4", "retok": "#ff7f0e", "centroid": "#d62728", "td": "#2ca02c"}
MARKERS = {"vanilla": "o", "retok": "s", "centroid": "^", "td": "D"}
TOK_PER_ITER = 1024 * 4096

# ---------- Mix-weighted bytes/token estimate ----------
# Apertus CPT v0.7 §6 mix: 70% Greek, 30% multilingual replay.
# - Greek side: measured on the 500-doc heldout (per-arm, from _fair.json)
# - Replay side: ~3.5 bytes/token, ~identical for Vanilla and TD because the
#   tokenizer extension only added Greek-bearing tokens. Caveat: this is a
#   literature estimate, not measured on our replay corpus.
GREEK_MIX = 0.70
REPLAY_MIX = 0.30
REPLAY_BPT = 3.5

# ---------- Load training loss curves ----------
train = {arm: {"tokens": [], "loss": [], "bpb": [], "bpt": [], "base_loss": [], "new_loss": [], "n_new": []} for arm in ARMS}


def maybe_float(value):
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def maybe_int(value):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


with CSV_PATH.open() as fp:
    r = csv.DictReader(fp)
    for row in r:
        arm = row["arm"]
        train[arm]["tokens"].append(float(row["tokens_b"]))
        train[arm]["loss"].append(float(row["lm_loss"]))
        train[arm]["bpb"].append(maybe_float(row.get("bpb")))
        train[arm]["bpt"].append(maybe_float(row.get("bpt")))
        train[arm]["base_loss"].append(maybe_float(row.get("base_loss")))
        train[arm]["new_loss"].append(maybe_float(row.get("new_loss")))
        train[arm]["n_new"].append(maybe_int(row.get("n_new")))

# ---------- Load heldout BPC (eval-checkpoint snapshots) ----------
heldout = {arm: {"iter": [], "tokens": [], "bpc": [], "bytes_per_token": [], "frac_trunc": []}
           for arm in ARMS}
all_iters = sorted({int(p.stem.split("_iter")[1].split("_fair")[0])
                    for p in INTRINSIC.glob("*_fair.json")})
for arm in ARMS:
    for it in all_iters:
        p = INTRINSIC / f"{arm}_iter{it:03d}_fair.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        g = d["global"]
        heldout[arm]["iter"].append(it)
        heldout[arm]["tokens"].append(it * TOK_PER_ITER / 1e9)
        heldout[arm]["bpc"].append(g["bpc_bits_per_byte"])
        heldout[arm]["bytes_per_token"].append(g["n_bytes"] / g["n_tokens"])
        heldout[arm]["frac_trunc"].append(d.get("truncation", {}).get("fraction_truncated"))

bpt_heldout = {arm: float(np.median(heldout[arm]["bytes_per_token"])) if heldout[arm]["bytes_per_token"] else None
               for arm in ARMS}
bpt_training = {arm: (GREEK_MIX * bpt_heldout[arm] + REPLAY_MIX * REPLAY_BPT) if bpt_heldout[arm] else None
                for arm in ARMS}
has_measured_bpb = any(any(v is not None for v in train[arm]["bpb"]) for arm in ARMS)


def dense_bpb_series(arm):
    tokens = np.array(train[arm]["tokens"])
    measured = np.array([np.nan if v is None else v for v in train[arm]["bpb"]], dtype=float)
    if np.isfinite(measured).any():
        return tokens, measured, "measured"
    if bpt_training[arm] is None:
        return tokens, np.array([], dtype=float), "missing"
    losses = np.array(train[arm]["loss"])
    return tokens, (losses / math.log(2)) / bpt_training[arm], "proxy"

# ============================================================================
# Figure 1 — Vanilla vs TD focused (the user's primary question)
# ============================================================================
SUB = ["vanilla", "td"]
fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

# Panel A — raw LM loss
ax = axes[0]
for arm in SUB:
    ax.plot(train[arm]["tokens"], train[arm]["loss"],
            color=COLORS[arm], linewidth=1.6, alpha=0.9, label=arm)
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("Per-token CE (nats)")
ax.set_title("(A) Raw LM loss — UNFAIR")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.04,
        "TD has 148,480 output classes,\nVanilla 131,072. TD tokens cover\nmore bytes. "
        "Per-token CE is NOT a fair\nhead-to-head.",
        transform=ax.transAxes, fontsize=8.5, va="bottom", color="#333",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.9))

# Panel B — heldout BPC at eval checkpoints
ax = axes[1]
for arm in SUB:
    ax.plot(heldout[arm]["tokens"], heldout[arm]["bpc"],
            color=COLORS[arm], marker=MARKERS[arm], linewidth=3.0, markersize=12, label=arm)
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("BPC (bits / byte)  ↓")
ax.set_title("(B) Heldout BPC — FAIR (canonical answer)")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.04,
        "Per-byte NLL on a fixed Greek\nheldout. Tokenizer-invariant.\n"
        "Directly measured at every\neval checkpoint.",
        transform=ax.transAxes, fontsize=8.5, va="bottom", color="#333",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.9))

# Panel C — measured dense BPB if present, otherwise approximate dense BPB
ax = axes[2]
for arm in SUB:
    tokens, bpb_dense, kind = dense_bpb_series(arm)
    if kind == "missing":
        continue
    finite = np.isfinite(bpb_dense)
    if not finite.any():
        continue
    label = f"{arm} measured" if kind == "measured" else f"{arm} proxy (÷ {bpt_training[arm]:.2f} B/tok)"
    alpha = 0.85 if kind == "measured" else 0.5
    ax.plot(tokens[finite], bpb_dense[finite], color=COLORS[arm], linewidth=1.0, alpha=alpha, label=label)
for arm in SUB:
    ax.scatter(heldout[arm]["tokens"], heldout[arm]["bpc"],
               color=COLORS[arm], marker=MARKERS[arm], s=130,
               edgecolor="black", linewidth=1.3, zorder=10,
               label=f"{arm} heldout (truth)")
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("BPB (bits / byte)  ↓")
ax.set_title("(C) Dense training BPB + heldout markers" if has_measured_bpb else "(C) Dense proxy + heldout markers — APPROXIMATE")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(True, alpha=0.3)
ax.text(0.02, 0.04,
        ("Lines: measured batch BPB from\ntraining logs when available.\nMarkers: heldout BPC truth."
         if has_measured_bpb else
         "Lines: training CE ÷ mix-weighted\nbytes/token (Greek 0.7 · heldout B/tok\n"
         "+ replay 0.3 · 3.5). Markers: ground\ntruth BPC. See caveats."),
        transform=ax.transAxes, fontsize=8.5, va="bottom", color="#333",
        bbox=dict(facecolor="white", edgecolor="#888", alpha=0.9))

plt.suptitle("Vanilla vs TD-layer11 — fair vs unfair loss comparison",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / "loss_comparison_van_td.png", dpi=120, bbox_inches="tight")
print(f"saved plots/loss_comparison_van_td.png")

# ============================================================================
# Figure 2 — All 4 arms, same three panels
# ============================================================================
fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))

ax = axes[0]
for arm in ARMS:
    ax.plot(train[arm]["tokens"], train[arm]["loss"],
            color=COLORS[arm], linewidth=1.2, alpha=0.85, label=arm)
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("Per-token CE (nats)")
ax.set_title("(A) Raw LM loss — UNFAIR")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[1]
for arm in ARMS:
    if not heldout[arm]["tokens"]:
        continue
    ax.plot(heldout[arm]["tokens"], heldout[arm]["bpc"],
            color=COLORS[arm], marker=MARKERS[arm], linewidth=2.5, markersize=10, label=arm)
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("BPC (bits / byte)  ↓")
ax.set_title("(B) Heldout BPC — FAIR")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes[2]
for arm in ARMS:
    tokens, bpb_dense, kind = dense_bpb_series(arm)
    if kind == "missing":
        continue
    finite = np.isfinite(bpb_dense)
    if not finite.any():
        continue
    ax.plot(tokens[finite], bpb_dense[finite], color=COLORS[arm], linewidth=0.9,
            alpha=0.85 if kind == "measured" else 0.5)
for arm in ARMS:
    ax.scatter(heldout[arm]["tokens"], heldout[arm]["bpc"],
               color=COLORS[arm], marker=MARKERS[arm], s=80,
               edgecolor="black", linewidth=1.0, zorder=10)
ax.set_xlabel("Tokens consumed (B)")
ax.set_ylabel("BPB (bits / byte)  ↓")
ax.set_title("(C) Dense training BPB + heldout markers" if has_measured_bpb else "(C) Dense proxy + heldout markers — APPROXIMATE")
ax.grid(True, alpha=0.3)

plt.suptitle("All 4 arms — fair vs unfair loss comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(PLOTS / "loss_comparison_4arm.png", dpi=120, bbox_inches="tight")
print(f"saved plots/loss_comparison_4arm.png")

# ============================================================================
# Numeric report
# ============================================================================
print()
print("=== Bytes-per-token per arm ===")
print(f"{'arm':<10}{'heldout B/tok':>16}{'training est':>16}")
for arm in ARMS:
    if bpt_heldout[arm] is None:
        continue
    print(f"{arm:<10}{bpt_heldout[arm]:>16.3f}{bpt_training[arm]:>16.3f}")

print()
print("=== Heldout BPC (lower = better, FAIR) ===")
header_iters = all_iters
print(f"{'iter':>6}{'tok(B)':>10}" + "".join(f"{a[:5]:>10}" for a in ARMS))
for it in header_iters:
    line = f"{it:>6}{it * TOK_PER_ITER / 1e9:>10.3f}"
    for arm in ARMS:
        if it in heldout[arm]["iter"]:
            idx = heldout[arm]["iter"].index(it)
            line += f"{heldout[arm]['bpc'][idx]:>10.4f}"
        else:
            line += f"{'n/a':>10}"
    print(line)

print()
print("=== Final raw LM loss vs final heldout BPC ===")
print(f"{'arm':<10}{'final LM loss':>16}{'final train BPB':>16}{'final BPC':>14}")
for arm in ARMS:
    if not train[arm]["loss"] or not heldout[arm]["bpc"]:
        continue
    latest_bpb = next((v for v in reversed(train[arm]["bpb"]) if v is not None), None)
    latest_bpb_str = f"{latest_bpb:.4f}" if latest_bpb is not None else "n/a"
    print(f"{arm:<10}{train[arm]['loss'][-1]:>16.4f}{latest_bpb_str:>16}{heldout[arm]['bpc'][-1]:>14.4f}")

if any(any(v is not None for v in train[arm]["n_new"]) for arm in ARMS):
    print()
    print("=== Latest base/new target training split ===")
    print(f"{'arm':<10}{'base_loss':>14}{'new_loss':>14}{'n_new':>12}")
    for arm in ARMS:
        idxs = [i for i, v in enumerate(train[arm]["base_loss"]) if v is not None or train[arm]["new_loss"][i] is not None]
        if not idxs:
            continue
        i = idxs[-1]
        base = train[arm]["base_loss"][i]
        new = train[arm]["new_loss"][i]
        n_new = train[arm]["n_new"][i]
        print(f"{arm:<10}{base if base is not None else 'n/a':>14}{new if new is not None else 'n/a':>14}{n_new if n_new is not None else 'n/a':>12}")

print()
print("=== Heldout truncation fraction per arm (latest eval) ===")
for arm in ARMS:
    if not heldout[arm]["frac_trunc"]:
        continue
    print(f"  {arm:<10} iter={heldout[arm]['iter'][-1]} frac_trunc={heldout[arm]['frac_trunc'][-1]}")
