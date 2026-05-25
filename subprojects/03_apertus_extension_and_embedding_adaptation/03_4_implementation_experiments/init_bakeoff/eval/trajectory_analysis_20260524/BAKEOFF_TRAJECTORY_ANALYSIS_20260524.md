# Bakeoff trajectory analysis — per-arm metric vs token count, slope, and TD-vs-Vanilla crossover projection

2026-05-24. Companion to [`bakeoff_1node_chain_20260522_005620_iter0000476_digest.md`](../live_summaries/bakeoff_1node_chain_20260522_005620_iter0000476_digest.md) and [`td_full25_layer11_2b_20260523T165038Z_iter0000476_digest.md`](../live_summaries/td_full25_layer11_2b_20260523T165038Z_iter0000476_digest.md). The iter-476 digests are snapshots; this doc is the *trajectory*.

Update: the 3.5B continuation is now complete. Use
[`CONTINUATION_3P5B_RESULTS_20260525.md`](CONTINUATION_3P5B_RESULTS_20260525.md)
for the current Vanilla / ReTok / TD result. This document remains the original
2B trajectory analysis and explains why the continuation was worth running.

## What this doc adds

The iter-476 digest answers "who wins at 2.0 B tokens?" — Vanilla, on aggregate Greek. This doc answers a different question: **what direction is each arm moving, and where would TD overtake Vanilla under linear extrapolation?**

The headline:
- **Vanilla's Greek aggregate is moving DOWN** during CPT (mid-window slope −17.9 m.p./B). It started at 0.467 (iter 130) and erodes to 0.441 (iter 476).
- **TD's Greek aggregate is moving UP** (mid-window slope +8.2 m.p./B). It started at 0.415 and climbs to 0.425.
- **Linear extrapolation: TD overtakes Vanilla at ~2.6-2.9 B tokens** — 30-45 % more compute than the 2 B bakeoff budget. The bakeoff stopped right before the crossover.

This doesn't change the production decision (Vanilla is still the right safe default at 2 B and beyond, given the 15 B production budget and the linear-extrapolation caveats below). It does suggest the bakeoff's 2 B budget was sitting on the crossover boundary, which is worth recording for the next iteration of this project.

## Artifacts in this directory

- [`plots/trajectories.png`](plots/trajectories.png) — three-panel group-averaged trajectory (English retention, Multilingual, Greek slice) per arm.
- [`plots/trajectories_per_task.png`](plots/trajectories_per_task.png) — eight per-task panels (the seven Greek tasks + English MMLU as reference).
- [`per_iter_results/*.json`](per_iter_results/) — raw lm-eval-harness `results.json` per arm × iter, 23 files (vanilla / retok / centroid at iter 130/260/325/390/455/476, TD at iter 130/260/390/455/476). Each ~590 KB. These are the gating data for every number below.
- [`regenerate_plots.py`](regenerate_plots.py) — self-contained reproduction script. Reads `per_iter_results/`, regenerates both PNGs, prints slopes + crossover projection.

## Group-averaged metric per arm at the six checkpoints

Average over the listed tasks per group (no weighting). Tokens consumed = iteration × 4.194304 M.

### English retention (mmlu, hellaswag, arc_easy, arc_challenge, piqa, winogrande)

| Iter | Tokens (B) | Vanilla | ReTok | Centroid | TD |
|---:|---:|---:|---:|---:|---:|
| 130 | 0.55 | **0.6966** | 0.6852 | **0.6990** | 0.6912 |
| 260 | 1.09 | 0.6844 | 0.6713 | 0.6890 | 0.6840 |
| 325 | 1.36 | 0.6867 | 0.6716 | 0.6853 | — |
| 390 | 1.64 | 0.6836 | 0.6749 | 0.6825 | 0.6838 |
| 455 | 1.91 | 0.6832 | 0.6750 | 0.6843 | 0.6831 |
| 476 | 2.00 | 0.6818 | 0.6750 | 0.6836 | 0.6828 |

### Multilingual (global_mmlu, xcopa, xnli)

| Iter | Tokens (B) | Vanilla | ReTok | Centroid | TD |
|---:|---:|---:|---:|---:|---:|
| 130 | 0.55 | 0.5034 | 0.4946 | 0.5025 | **0.5052** |
| 260 | 1.09 | 0.4900 | 0.4835 | 0.4897 | 0.4950 |
| 325 | 1.36 | 0.4896 | 0.4805 | 0.4916 | — |
| 390 | 1.64 | 0.4897 | 0.4843 | 0.4893 | 0.4900 |
| 455 | 1.91 | 0.4890 | 0.4869 | 0.4893 | 0.4898 |
| 476 | 2.00 | **0.4901** | 0.4873 | 0.4888 | 0.4899 |

### Greek slice (global_mmlu_full_el, include_base_44_greek_few_shot_en, belebele_ell_Grek, arc_challenge_mt_el, xnli_el, xquad_el f1, global_piqa_completions_ell_grek)

| Iter | Tokens (B) | Vanilla | ReTok | Centroid | TD |
|---:|---:|---:|---:|---:|---:|
| 130 | 0.55 | **0.4667** | 0.4034 | 0.2957 | 0.4147 |
| 260 | 1.09 | 0.4424 | 0.4111 | 0.2943 | 0.4286 |
| 325 | 1.36 | 0.4472 | 0.4085 | 0.2914 | — |
| 390 | 1.64 | 0.4472 | 0.4150 | 0.2956 | 0.4237 |
| 455 | 1.91 | 0.4447 | 0.4153 | 0.2938 | 0.4216 |
| 476 | 2.00 | **0.4409** | 0.4150 | 0.2927 | **0.4254** |

V4-HF baselines for reference: Greek avg = **0.5155** (mean of the seven Greek tasks at V4-HF). Every arm is below V4-HF at iter 476 by 7-22 p.p. on Greek.

## Slope analysis, three windows

`m.p./B` = milli-percentage points of accuracy per billion training tokens, linear fit over the group-averaged trajectory.

| Group | Arm | Full (130→476) | Mid (130→390) | Tail (390→476) |
|---|---|---:|---:|---:|
| EN retention | vanilla | −8.74 | −11.34 | −4.07 |
| EN retention | retok | −5.04 | −10.37 | +0.37 |
| EN retention | centroid | −10.23 | −15.35 | +3.99 |
| EN retention | **td** | **−5.05** | **−6.76** | **−2.74** |
| Multi | vanilla | −8.20 | −13.01 | +0.17 |
| Multi | retok | **−3.73** | −10.88 | +8.84 |
| Multi | centroid | −8.13 | −11.75 | −1.15 |
| Multi | td | −10.29 | −13.86 | −0.43 |
| **Greek** | **vanilla** | **−13.63** | **−17.91** | **−15.57** |
| **Greek** | **retok** | **+8.06** | **+9.30** | **+0.30** |
| Greek | centroid | −1.26 | −1.25 | −7.61 |
| **Greek** | **td** | **+4.21** | **+8.25** | **+1.88** |

**Caveats on the windows:**

- **Tail (390→476) is contaminated by WSD 1-sqrt cooldown.** LR went from peak 1.5e-5 → 1.5e-6 across the run; the 1-sqrt cooldown is concentrated in the last third. Tail-window slopes mostly say "all arms slow down at the very end" — that's an LR effect, not an init-method effect.
- **Mid (130→390) is the cleanest window for "is this arm still learning?".** LR was at ~6-10e-6 throughout — well above the cooldown floor. The mid-window slope is the most informative measurement of the arm's learning trajectory.
- **Full (130→476)** averages across the high-LR and cooldown regions; it's a compromise but more stable than the 3-point tail.

## TD vs Vanilla crossover projection on the Greek aggregate

Using both windows:

| Window | Vanilla slope | TD slope | Gap at 2.0 B | Linear crossover |
|---|---:|---:|---:|---|
| Mid (130→390) | −17.91 m.p./B | +8.25 m.p./B | 0.0155 | **~2.6 B tokens** |
| Full (130→476) | −13.63 m.p./B | +4.21 m.p./B | 0.0155 | **~2.9 B tokens** |

**Reading.** Vanilla and TD are diverging at ~18-26 m.p./B (the sum of magnitudes of their oppositely-signed slopes). At the current 1.6 p.p. gap, that gap closes in 0.6-0.9 B more tokens of training. **The bakeoff stopped 0.6-0.9 B tokens before TD's Greek aggregate would have caught up under linear extrapolation.**

### What this projection does NOT say

- It does not say TD's improvement rate is sustainable for arbitrary tokens. Vanilla's Greek loss curve likely *stabilizes* past 2 B (rate of forgetting decays as the model converges to the new mix); TD's gain likely *saturates* as the new-token rows converge. So "crossover at 2.9 B" is a lower bound on when divergence becomes visible, not a guarantee of indefinite TD lead.
- It does not say TD beats Vanilla on every Greek task. It says aggregate, mean-of-7-Greek-tasks. The per-task picture is mixed (see below).
- It does not invalidate the production decision. The decision was made for safety/predictability at the strict iter-476 snapshot, with the canonical 15 B-token Goldfish-loss production CPT. The trajectory analysis is a forward-looking signal for *the next iteration of this question*, not for changing the in-flight production launch.

## Per-task picture (which Greek tasks already favor TD?)

Reading off `trajectories_per_task.png` for the seven Greek tasks + English MMLU at iter 476, with the trajectory shape annotated:

| Task | TD476 | Vanilla476 | TD vs Vanilla | TD trajectory shape | Vanilla trajectory shape |
|---|---:|---:|---:|---|---|
| `global_mmlu_full_el` | 0.3859 | **0.4214** | −3.55 pp | gently rising | gently falling |
| `include_base_44_greek_few_shot_en` | 0.4040 | **0.4185** | −1.45 pp | rising | falling |
| **`belebele_ell_Grek`** | **0.5278** | 0.5133 | **+1.44 pp** | rising | flat/slightly rising |
| `arc_challenge_mt_el` (acc_norm) | 0.4036 | **0.4206** | −1.71 pp | flat | slowly falling |
| `xnli_el` | 0.3803 | **0.4020** | −2.17 pp | falling | falling faster |
| **`xquad_el` f1** | **0.3262** | 0.3101 | **+1.61 pp** | rising | falling |
| `global_piqa_completions_ell_grek` (acc_norm) | 0.5500 | **0.6000** | −5.00 pp | flat/noisy | noisy but slightly down |
| **English `mmlu`** | **0.5501** | 0.5340 | **+1.61 pp** | flat | flat |

**TD already wins three tasks** (Belebele Greek, XQuAD Greek f1, English MMLU) and is catching up on every other Greek task except `xnli_el` and `global_piqa_completions_ell_grek`, where both arms are noisy/declining. The reading-comprehension and extractive-QA Greek tasks favor TD; the reasoning-heavy Greek MC tasks still favor Vanilla but the gap is closing.

## Direct answer to the framing question

**Which has the best Greek-aggregate gradient?** ReTok at +8 to +9 m.p./B (mid-window). TD second at +4 to +8 m.p./B. Centroid is essentially flat at −1 m.p./B. **Vanilla is strongly negative at −14 to −18 m.p./B**, i.e., losing Greek performance during CPT despite Greek being 70 % of the diet. The retention paradox: Vanilla starts highest on Greek and erodes; TD/ReTok start lower and climb.

**Which has the best English-retention gradient?** TD at −2.7 to −6.8 m.p./B (least negative). Centroid's tail rebounds (+4.0) but its mid-window is the worst (−15.3). All four arms are degrading English in absolute terms; TD degrades least.

**Does TD seem promising to improve over Vanilla?** **Yes, on the slope signal.** Three convergent indicators:

1. **Slope signs are right.** TD's Greek slope is positive; Vanilla's is negative.
2. **Linear crossover at 2.6-2.9 B tokens** — 30-45 % beyond the bakeoff budget.
3. **TD already wins three benchmarks at iter 476**: Belebele Greek, XQuAD Greek f1, and English MMLU.

What the data does NOT support is claiming TD will *definitely* beat Vanilla at 15 B. The linear extrapolation is naive. The right strength of claim is: *the bakeoff at 2 B tokens is too short to fairly evaluate the extended-tokenizer path; the production decision for Vanilla is conservative-and-safe, but the next iteration of this project should explicitly budget a longer comparison (≥5 B per arm) before treating the Vanilla-vs-TD question as settled.*

## Reproducing this analysis

```bash
cd /home/foivos/Projects/glossapi-tokenizer-extension/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524
python3 regenerate_plots.py
```

The script reads `per_iter_results/*.json` plus the V4-HF baseline at `eval/v4_baseline_corrected_20260521/results.json`. Output: regenerates both PNGs, prints slopes and crossover projection. ~10 seconds runtime on home machine.
