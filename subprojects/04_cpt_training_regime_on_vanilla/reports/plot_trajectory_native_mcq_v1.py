"""Plot native Greek MCQ trajectory for the 04 Vanilla CPT regime experiment.

Data sourced from reports/v4_bootstrap_cis_native_mcq.json (v2 revision)
with iter 834 + iter 1192 endpoints added from the iter 1192 endpoint
workflow (CIs approximated for 3.5B + 5B pending V4 v3 re-emit).

1000-resample 95% percentile CIs, rng_seed=20260529, per-task item-level
+ paired bootstrap for V4 v2 entries.

Two panels:
  Top    : trajectory with 95% CI bands for 04-run + bakeoff Vanilla,
           horizontal CI band for Apertus-Base Path A, dotted line for
           matched-init floor, vertical warmup-end marker.
           iter 3.5B and iter 5B points shown as markers (CI bands
           pending V4 v3).
  Bottom : delta-vs-init for 04-run with bootstrap CIs as error bars.
           iter 3.5B + 5B deltas approximate (V4 v3 will replace).
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPORTS = Path(__file__).resolve().parent
OUT = REPORTS / "trajectory_native_mcq_with_cis.png"

# (label, tokens_B, point, lo_95, hi_95) — from V4 v2 artifact (init -> 2B).
# iter 3.5B + 5B from corrected aggregates; CIs approximated as iter-477
# CI shape (half-width ~0.018) centered on the point, marked dashed in
# the plot as "approximate until V4 v3 lands".
OUR_RUN = [
    ("init",   0.0, 0.4272, 0.4096, 0.4456),  # matched-config Path-B (= our iter 0)
    ("0.5B",   0.5, 0.4391, 0.4219, 0.4565),
    ("1.0B",   1.0, 0.4487, 0.4299, 0.4670),
    ("2.0B",   2.0, 0.4792, 0.4604, 0.4967),
]
OUR_RUN_NEW_NO_CI = [
    # CIs pending V4 v3; show as point markers only.
    ("3.5B",   3.5, 0.4790, None, None),
    ("5.0B",   5.0, 0.4973, None, None),
]
BAKEOFF = [
    ("2B",   2.0, 0.4327, 0.4144, 0.4502),
    ("3.5B", 3.5, 0.4370, 0.4185, 0.4545),
    ("5B",   5.0, 0.4305, 0.4134, 0.4485),
]
APERTUS_A = (0.4817, 0.4629, 0.4997)  # Path A point, CI
WARMUP_END_B = 1.2

# Delta vs init (Δ point, Δ lo, Δ hi) — from V4 v2 delta_table.
# All vs Apertus-Base-matched-Path-B-perturbed (= our init).
# iter 3.5B / 5B deltas approximate; CIs will land with V4 v3.
DELTAS_VS_INIT = [
    ("0.5B", 0.5,  0.0119,  -0.0040,  0.0274),  # iter-119 vs init (approx)
    ("1.0B", 1.0,  0.0215,   0.0056,  0.0371),  # iter-238 vs init (approx)
    ("2.0B", 2.0,  0.0519,   0.0355,  0.0676),  # iter-477 vs init — V4 explicit
    ("3.5B", 3.5,  0.0518,   None,    None),    # iter-834 vs init — V4 v3 pending
    ("5.0B", 5.0,  0.0701,   None,    None),    # iter-1192 vs init — V4 v3 pending
]

fig, (ax1, ax2) = plt.subplots(
    nrows=2, ncols=1, figsize=(11, 9), height_ratios=[2, 1], sharex=False
)

# ---------------- Top: trajectory ----------------

xs    = [d[1] for d in OUR_RUN]
ys    = [d[2] for d in OUR_RUN]
los   = [d[3] for d in OUR_RUN]
his   = [d[4] for d in OUR_RUN]

xs_new = [d[1] for d in OUR_RUN_NEW_NO_CI]
ys_new = [d[2] for d in OUR_RUN_NEW_NO_CI]

# Full line through all 6 points (init + 5 checkpoints).
xs_all = xs + xs_new
ys_all = ys + ys_new
ax1.plot(xs_all, ys_all, "-", color="C0", lw=2.2,
         label="04 Vanilla CPT (Path B, regime-fixed)")
# CI'd points with markers + CI band on first 4.
ax1.plot(xs, ys, "o", color="C0", ms=9)
ax1.fill_between(xs, los, his, alpha=0.18, color="C0")
# New points (no CI yet) — diamond markers + dashed CI placeholder.
ax1.plot(xs_new, ys_new, "D", color="C0", ms=9,
         markeredgecolor="white", markeredgewidth=1.4,
         label="04 Vanilla CPT — 3.5B + 5B (CI pending V4 v3)")

xs_b  = [d[1] for d in BAKEOFF]
ys_b  = [d[2] for d in BAKEOFF]
los_b = [d[3] for d in BAKEOFF]
his_b = [d[4] for d in BAKEOFF]
ax1.plot(xs_b, ys_b, "s--", color="C3", lw=2.0, ms=8,
         label="Bakeoff Vanilla (Path B, original regime)")
ax1.fill_between(xs_b, los_b, his_b, alpha=0.15, color="C3")

# Apertus-Base Path A reference band.
ax1.axhspan(APERTUS_A[1], APERTUS_A[2], alpha=0.13, color="seagreen",
            label=f"Apertus-Base Path A 95% CI [{APERTUS_A[1]:.4f}, {APERTUS_A[2]:.4f}]")
ax1.axhline(APERTUS_A[0], ls=":", color="seagreen", alpha=0.85, lw=1.4)

# Matched-init floor (= our init point at 0.4272).
ax1.axhline(OUR_RUN[0][2], ls=":", color="dimgray", alpha=0.6, lw=1.2)
ax1.text(5.05, OUR_RUN[0][2], "Vanilla at init\n(matched Path B)",
         fontsize=8, color="dimgray", va="center", ha="left")

# Warmup end vertical marker.
ax1.axvline(WARMUP_END_B, ls="--", color="orange", alpha=0.7, lw=1.4)
ax1.text(WARMUP_END_B + 0.05, 0.408, "warmup end\n(1.2 B)",
         fontsize=8, color="darkorange", va="bottom")

# Point labels for 04 run (all 6 points).
for _, x, y, _lo, _hi in OUR_RUN:
    ax1.annotate(f"{y:.4f}", (x, y),
                 textcoords="offset points", xytext=(7, 9), fontsize=8.5, color="C0")
for _, x, y, _lo, _hi in OUR_RUN_NEW_NO_CI:
    ax1.annotate(f"{y:.4f}", (x, y),
                 textcoords="offset points", xytext=(7, 9), fontsize=8.5,
                 color="C0", fontweight="bold")
for _, x, y, _lo, _hi in BAKEOFF:
    ax1.annotate(f"{y:.4f}", (x, y),
                 textcoords="offset points", xytext=(7, -14), fontsize=8.5, color="C3")

ax1.set_xlabel("CPT tokens (B)")
ax1.set_ylabel("Native Greek MCQ headline\n(3-task macro accuracy)")
ax1.set_title(
    "Native Greek MCQ trajectory — 04 Vanilla CPT vs bakeoff Vanilla vs Apertus-Base\n"
    "95% bootstrap CIs (1000 resamples, per-task item-level)",
    fontsize=12
)
ax1.legend(loc="lower right", fontsize=9.5)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.25, 5.5)
ax1.set_ylim(0.40, 0.52)

# ---------------- Bottom: delta vs init ----------------

# Split into CI'd and no-CI subsets.
DELTAS_CI = [d for d in DELTAS_VS_INIT if d[3] is not None]
DELTAS_NO_CI = [d for d in DELTAS_VS_INIT if d[3] is None]

# Connect line through ALL points.
all_xs = [d[1] for d in DELTAS_VS_INIT]
all_ys = [d[2] for d in DELTAS_VS_INIT]
ax2.plot(all_xs, all_ys, "-", color="C0", lw=2)

# Error-bar points (with CIs).
dxs_ci = [d[1] for d in DELTAS_CI]
dys_ci = [d[2] for d in DELTAS_CI]
dlo_ci = [d[2] - d[3] for d in DELTAS_CI]
dhi_ci = [d[4] - d[2] for d in DELTAS_CI]
ax2.errorbar(dxs_ci, dys_ci, yerr=[dlo_ci, dhi_ci],
             fmt="o", color="C0", ms=9, capsize=5,
             label="Δ vs Vanilla at init (V4 v2)")

# Pending points (no CI; mark with diamonds).
dxs_no = [d[1] for d in DELTAS_NO_CI]
dys_no = [d[2] for d in DELTAS_NO_CI]
ax2.plot(dxs_no, dys_no, "D", color="C0", ms=9,
         markeredgecolor="white", markeredgewidth=1.4,
         label="Δ vs init — V4 v3 pending")

ax2.axhline(0, ls="-", color="black", alpha=0.5, lw=0.8)
ax2.axvline(WARMUP_END_B, ls="--", color="orange", alpha=0.7, lw=1.4)

for _, x, y, lo, hi in DELTAS_CI:
    sig = "*" if lo > 0 else ""
    ax2.annotate(f"+{y * 100:.2f} pp{sig}\n[{lo * 100:+.2f}, {hi * 100:+.2f}]",
                 (x, y), textcoords="offset points",
                 xytext=(9, 0), fontsize=8.5, color="C0", va="center")
for _, x, y, _lo, _hi in DELTAS_NO_CI:
    ax2.annotate(f"+{y * 100:.2f} pp\n(CI pending)",
                 (x, y), textcoords="offset points",
                 xytext=(9, 0), fontsize=8.5, color="C0", va="center",
                 fontweight="bold")

ax2.set_xlabel("CPT tokens (B)")
ax2.set_ylabel("Δ headline\n(pp vs init)")
ax2.set_title("Δ from Vanilla at init (matched Path-B, perturbed) — 95% bootstrap CIs",
              fontsize=11)
ax2.set_xlim(-0.25, 5.5)
ax2.set_ylim(-0.01, 0.10)
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left", fontsize=9.5)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print(f"saved: {OUT}")
