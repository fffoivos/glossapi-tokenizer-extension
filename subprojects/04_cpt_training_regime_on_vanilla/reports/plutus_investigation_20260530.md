# Plutus QA diagnostic drop — iter 834 → iter 1192 investigation

**Date:** 2026-05-30
**Author:** automated analysis
**Scope:** 225-item Plutus QA diagnostic in the `04_vanilla_goldfish_5b` run
(eval root `eval_04_vanilla_goldfish_5b_20260528T112539Z`).

## 0. Trajectory summary

| Checkpoint | Tokens | Plutus point | Notes |
| --- | --- | --- | --- |
| iter 119 (Vanilla-0.5B) | 0.5 B | 0.4000 | baseline |
| iter 238 (Vanilla-1B)   | 1.0 B | 0.4133 | |
| iter 477 (Vanilla-2B)   | 2.0 B | 0.4889 | local peak |
| iter 834 (Vanilla-3.5B) | 3.5 B | 0.4889 | held steady |
| iter 1192 (Vanilla-5B)  | 5.0 B | **0.4356** | **−5.33 pp vs iter 834** |

## 1. Methodology

Item-level percentile bootstrap, identical to `v4_bootstrap_cis_native_mcq.json`
v2 methodology (`reports/v4_workspace/run_bootstrap_v2.py`):

- `N_RESAMPLES = 1000`
- `CI_LEVEL = 0.95`
- `RNG_SEED = 20260529`
- per-task item-level resampling within Plutus (n = 225)
- paired CIs use shared resample indices on the model pair → paired diff

Source predictions (pulled read-only from Clariden, deleted after analysis):

- `iter_0000477/native_mcq/Vanilla-2B_native_mcq_predictions.jsonl`
- `iter_0000834/native_mcq/Vanilla-3.5B_native_mcq_predictions.jsonl`
- `iter_0001192/native_mcq/Vanilla-5B_native_mcq_predictions.jsonl`
- `eval/native_greek_suite_20260526/mcq_all_checkpoints/chunk_1/Apertus-Base_/Apertus-Base_native_mcq_predictions.jsonl`
- `eval/native_greek_suite_20260526/mcq_all_checkpoints/chunk_2/Vanilla-5B_/Vanilla-5B_native_mcq_predictions.jsonl` (bakeoff-Vanilla-5B)
- `04_vanilla_cpt/eval_apertus_base_matched_rope500k_seq4096/native_mcq/Apertus-Base-matched-rope500k-seq4096_native_mcq_predictions.jsonl`

Plutus item alignment was verified across all six prediction files
(example_ids identical in identical order, n = 225).

Driver: `reports/v4_workspace_plutus/run_plutus_bootstrap.py`
(JSON output: `plutus_investigation_results.json`).

## 2. Marginal 95% CIs per checkpoint

| Checkpoint            | Point  | 95% CI               | n |
| ---                   | ---    | ---                  | --- |
| iter-477 (Vanilla-2B) | 0.4889 | [0.4222, 0.5556]     | 225 |
| iter-834 (Vanilla-3.5B) | 0.4889 | [0.4222, 0.5556]   | 225 |
| iter-1192 (Vanilla-5B) | 0.4356 | [0.3689, 0.4978]    | 225 |

All three marginal CIs **overlap heavily**. The marginal half-width is
≈6.7 pp at n = 225 — the noise floor is large relative to the 5.33 pp drop.

## 3. Paired 95% CIs (load-bearing analysis)

Paired bootstrap on the same 225 items (paired resample indices):

| A         | B        | Δ = A − B | 95% paired CI       | Outside zero? |
| ---       | ---      | ---       | ---                 | ---           |
| iter-1192 | iter-834 | **−0.0533** | **[−0.1111, +0.0044]** | **No (touches zero)** |
| iter-1192 | iter-477 | −0.0533   | [−0.1111, +0.0000]  | No (CI upper bound = 0.0000, sits exactly at boundary) |
| iter-834  | iter-477 | +0.0000   | [−0.0489, +0.0444]  | No (sanity check: deltas match across 14 wins / 14 losses, net zero) |

The iter-834 vs iter-477 paired CI confirms the methodology — both
checkpoints had Plutus point = 0.4889 with 28 item-level flips netting to
zero, and the paired CI is symmetric and includes zero.

For the **load-bearing pair (iter-1192 − iter-834)**, the paired 95% CI
is `[−0.111, +0.004]` — it **just touches zero on the upper tail**. With
1000 paired bootstrap draws this is statistically inconclusive at the
strict α = 0.05 level: we cannot reject "no change", but the point
estimate and the bulk of the bootstrap mass sit well below zero.

The iter-1192 vs iter-477 pair has paired CI `[−0.111, +0.000]`, also
touching zero from below.

## 4. Item-level diff (iter 834 → iter 1192)

- iter-834 right & iter-1192 wrong: **26 items** (the "drop")
- iter-834 wrong & iter-1192 right: 14 items (the "gain")
- net = −12 items / 225 = −5.33 pp

The same picture vs iter-477 (because iter-477 and iter-834 agree on
correctness for the 197 items that are stable):

- iter-477 right & iter-1192 wrong: 26
- iter-477 wrong & iter-1192 right: 14

### 4.1 Plutus has no rich task-type metadata

Plutus prediction records carry `metadata = {}` and a single `subject =
"finance"`. The only structural axis available is `num_choices`.
Distribution over the 225 items:

| num_choices | n items |
| --- | --- |
| 4 | 188 |
| 2 | 27 |
| 5 | 9 |
| 3 | 1 |

### 4.2 Drops concentrate in the 27 two-choice items

Of the 26 drops:

| num_choices | drops (834-right → 1192-wrong) | gains (834-wrong → 1192-right) | total items |
| --- | --- | --- | --- |
| 2 | **14** | 10 | 27 |
| 4 | 12 | 4 | 188 |
| 3 | 0 | 0 | 1 |
| 5 | 0 | 0 | 9 |

**14 out of 27 two-choice items flipped from correct (at 834) to wrong
(at 1192) — a 52% flip rate on the 2-choice subset.** The expected
random-pairwise-flip rate on a coin-flip task that doesn't move on
average is much lower; the 2-choice items are clearly a hot spot.

Drop-prediction offsets (`(num_choices, (pred_1192 − answer) mod
num_choices)`):

- (2, 1): 14  — every 2-choice drop is "wrong by 1", i.e. the model picked the alternative
- (4, 1): 9
- (4, 2): 1
- (4, 3): 2

So on 4-choice drops the model also overwhelmingly picks the
adjacent-index distractor (9/12), not a uniformly random wrong choice.
This is consistent with positional / token-frequency drift rather than a
content-level regression — but Plutus's flat metadata gives no
content-level features to test that against.

### 4.3 iter-1192 confidence on the drops is below average

Mean top-margin = `top1 avg_logprob − top2 avg_logprob` for iter-1192:

| Cohort | n | mean margin | median margin |
| --- | --- | --- | --- |
| All 225 Plutus items | 225 | 0.566 | 0.375 |
| 834-right → 1192-wrong drops | 26 | **0.183** | 0.125 |
| 834-wrong → 1192-right gains | 14 | 0.338 | 0.373 |

The drops are low-confidence in iter-1192 — top-1/top-2 margin is ≈3×
smaller than the population mean. This is structurally consistent with
"items the model was on the fence about, and a small representational
shift flipped them" rather than "a new systematic mistake direction".

### 4.4 Of the 26 drops, only 8 match iter-477's wrong prediction

For 26 drops, iter-1192's chosen index matches iter-477's chosen index
on **8** of them. The other 18 drops are *new* wrong answers — iter-1192
picked an index that neither iter-477 nor iter-834 picked. So this isn't
"unlearning back to the iter-477 prior"; it's a fresh, low-confidence
shuffle.

## 5. Bakeoff Vanilla cross-check

From `v4_bootstrap_cis_native_mcq.json` (already shipped marginal CIs):

| Bakeoff checkpoint | Plutus point | 95% CI |
| --- | --- | --- |
| bakeoff-Vanilla-2B (~2 B tokens) | 0.4044 | [0.3422, 0.4668] |
| bakeoff-Vanilla-3.5B             | 0.4222 | [0.3600, 0.4889] |
| bakeoff-Vanilla-5B               | 0.4400 | [0.3822, 0.4978] |

The bakeoff Vanilla showed **no Plutus instability** of the same kind —
it climbed monotonically 0.4044 → 0.4222 → 0.4400 over the same 2 B →
3.5 B → 5 B span. Step-to-step deltas are +1.78 pp and +1.78 pp,
well inside per-step noise.

So the iter-477 / iter-834 / iter-1192 trajectory in *this* run is
**not** mirroring the bakeoff: the bakeoff drifts gently up; this run
sat at 0.4889 for 1.5 B tokens then dropped 5.33 pp. The bakeoff 5 B
endpoint (0.4400) is statistically indistinguishable from iter-1192's
0.4356 — that's where the two trajectories meet.

Paired (same-item) CI for iter-1192 vs bakeoff-Vanilla-5B:

| A | B | Δ | 95% paired CI | Outside zero? |
| --- | --- | --- | --- | --- |
| iter-1192 | bakeoff-Vanilla-5B | −0.0044 | [−0.0711, +0.0578] | No |

So whatever this CPT did to Plutus by 5 B tokens, it lands on the same
Plutus accuracy as the bakeoff Vanilla 5 B endpoint, within noise. The
earlier 0.4889 plateau (iter-477, iter-834) was the *anomaly*, not the
5 B endpoint — the bakeoff never reached that level on Plutus.

## 6. vs-baseline verdict

Paired (same-item) CIs of iter-1192 against the three baselines:

| Comparison | Δ | 95% paired CI | Outside zero? |
| --- | --- | --- | --- |
| iter-1192 − Apertus-Base                    | **−0.0800** | [−0.1467, −0.0178] | **Yes, below zero** |
| iter-1192 − Apertus-Base-matched-rope500k   | **+0.0889** | [+0.0267, +0.1556] | **Yes, above zero** |
| iter-1192 − bakeoff-Vanilla-5B              | −0.0044 | [−0.0711, +0.0578] | No |

- **vs Apertus-Base Path A (0.5156):** iter-1192's 0.4356 is **a real
  regression** — paired CI excludes zero. The 5 B CPT endpoint is below
  the pre-training-tokenizer-only Apertus checkpoint on Plutus.
- **vs Apertus-Base matched-Path-B-perturbed (0.3467):** iter-1192 is
  **significantly above** — paired CI excludes zero. CPT is still doing
  better than the rope-perturbed matched config.
- **vs bakeoff-Vanilla-5B (0.4400):** statistically indistinguishable.

Marginal CI containment for the 0.4356 point:

- inside bakeoff-Vanilla-2B CI [0.342, 0.467]: yes
- inside bakeoff-Vanilla-3.5B CI [0.360, 0.489]: yes
- inside bakeoff-Vanilla-5B CI [0.382, 0.498]: yes
- inside iter-477 / iter-834 marginal CI [0.422, 0.556]: **no**, sits below the lower bound
- inside Apertus-Base CI [0.449, 0.582]: **no**, sits below the lower bound

## 7. Final read — signal vs noise vs ambiguous

**Ambiguous, leaning real-but-small.**

Calls:

1. **Paired iter-1192 vs iter-834 CI just touches zero** (`[−0.1111,
   +0.0044]`). At strict α = 0.05 we cannot reject "no change", but the
   bootstrap mass is heavily concentrated below zero and 95% of the
   draws are < +0.0044. This is the *load-bearing* statistic and it is
   marginally non-significant.
2. **iter-1192 vs Apertus-Base IS significant** (paired CI excludes
   zero, Δ = −0.08). Whatever the cause, the 5 B CPT endpoint is genuinely
   below Apertus-Base on Plutus.
3. **The 5 pp drop is concentrated in the 27 two-choice items** (14
   flips, 52% rate). That's a structural concentration, not a uniform
   accuracy slip. Combined with the low top1/top2 margins on the drops,
   the most parsimonious read is "small representational shift between
   3.5 B and 5 B that flips low-confidence binary items".
4. **The bakeoff doesn't validate the iter-477/iter-834 plateau** — the
   bakeoff Plutus trajectory has no such plateau; iter-1192's 0.4356 is
   right where the bakeoff was at 5 B (0.4400, paired Δ = −0.0044,
   indistinguishable). So the "iter-834 0.4889 was the lucky read" hypothesis
   is at least as well-supported by the cross-run data as "iter-1192 0.4356
   is the unlucky read".
5. **Plutus is diagnostic only** (`cpt-plan.md`). It is not in the
   headline. The 4-task headline drift driven by this 5 pp Plutus drop
   is ≈1.3 pp on the macro-mean — below headline noise floors at this
   n_items mix.

**Recommended posture:** treat the iter-1192 Plutus number as a slight
real regression vs Apertus-Base (significant) but a noise-level move
vs iter-834 (marginally non-significant). Do not redesign on the basis
of this single 225-item diagnostic point. If the next CPT endpoint shows
Plutus ≤ 0.44 again, that's the third independent confirmation that the
5 B CPT regime sits a few pp below Apertus-Base on Plutus and the drop
is signal; if it bounces back to ≥ 0.48, the iter-1192 read was a
two-choice-subset noise hit.

## 8. Artifacts

- `reports/v4_workspace_plutus/run_plutus_bootstrap.py` — driver
- `reports/v4_workspace_plutus/plutus_investigation_results.json` — full numeric results
- prediction JSONLs were pulled into `v4_workspace_plutus/` and **deleted** after analysis
