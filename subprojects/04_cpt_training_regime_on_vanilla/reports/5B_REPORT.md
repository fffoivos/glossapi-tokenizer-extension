# 5 B Vanilla CPT — Final Report

**Run tag:** `04_vanilla_goldfish_5b_20260528T112539Z`
**Endpoint:** `iter_0001192` ≈ 5.0 B training tokens
**Date:** 2026-05-30
**Sources of truth:** `goal/goal.md` (scope, stop conditions, required artifacts), `goal/hyperparameters.json` (authoritative sbatch-generation settings), `cpt-plan.md` §2.1 (full parameter spec), §2.2 (regime-hypothesis framing), §3.4 Q3.4.10 (Path-A revisit recommendation for Task 2), §6 (non-commitments — thresholds decided post-result).

> **Eval task scope** locked at `goal/canonical_eval_tasks.json` (`canonical-eval-tasks-v1`, generated 2026-05-30). Headline = `{greekmmlu, ilsp_medical_mcqa, ilsp_mcqa_asep}`; diagnostic-separate = `{plutus_qa}`; MT-derived Greek tasks (`xnli_el`, `xcopa_el`, `arc_challenge_mt_el`, `global_piqa_completions_ell_grek`) excluded; retention = en/fr/de/ru only (`mmlu`, `arc_challenge`, `arc_easy`, `hellaswag`, `piqa`, `global_mmlu_en`, `xnli_en`, `global_mmlu_fr`, `xnli_fr`, `global_mmlu_de`, `xnli_de`, `xnli_ru`); heldout BPB = `{greek_heldout_500, code_heldout_200, math_heldout_200}`. Non-canonical data on disk is invisible to this report by design.

This report supersedes `reports/5B_REPORT_DRAFT.md`.

---

## 1. Executive Summary

The corrected Apertus-faithful regime — Goldfish loss `k=h=50`, AdEMAMix `β3=0.99`, LR `1.1e-5` with 1.2 B-token warmup constant after — was applied to Vanilla Apertus-8B-2509 on HPLT-only Greek + 24/4/2 % replay/code/math. Five checkpoints were collected (iter 119 / 238 / 477 / 834 / 1192 ≈ 0.5 / 1 / 2 / 3.5 / 5 B tokens). All five were converted to HF, evaluated against the native Greek MCQ headline, Plutus, BPB heldout, multilingual retention, and code/math BPB, and adversarially critiqued (Vanilla-0.5B → Vanilla-5B). V4 v3 bootstrap CIs (1 000 resamples, percentile, per-task item-level paired) are the load-bearing statistic on every cross-arm claim.

**Regime-hypothesis verdict (cpt-plan §2.2): SUPPORTED with three caveats.** Headline trajectory shape is **warmup → +3.05 pp post-warmup burst (iter 238 → iter 477) → flat 1.5 B-token segment (iter 477 → iter 834) → +1.84 pp endpoint lift (iter 834 → iter 1192)** — not "peak early, then drift" (which was the bakeoff-Vanilla pattern). At every matched token mark the new-regime run is +4.2 to +6.7 pp above bakeoff-Vanilla with CI excluding zero, and at the 5 B endpoint it sits +1.56 pp above Apertus-Base on the 3-task headline with CI [+0.0016, +0.0284] — outside zero, but barely.

**Five headline numbers (V4 v3, headline_3task unless noted):**

| Quantity | Value | 95 % CI | Outside zero? |
|---|---:|---|---|
| iter-1192 marginal headline_3task | 0.4973 | [0.4779, 0.5156] | n/a |
| paired iter-1192 vs iter-477 (full post-warmup) | +0.0182 | [+0.0060, +0.0306] | yes |
| paired iter-1192 vs iter-834 (post-plateau slope) | +0.0184 | [+0.0080, +0.0295] | yes |
| paired iter-1192 vs Apertus-Base Path-A | +0.0156 | [+0.0016, +0.0284] | yes (barely) |
| paired iter-1192 vs bakeoff-Vanilla-5B | +0.0669 | [+0.0513, +0.0830] | yes |

**Path-A-vs-Path-B framing.** The run trained under Path B (`rope_theta=500K`, `max_position_embeddings=4096`, no scaling) for bakeoff comparability. Apertus-Base ships on Path A (`rope_theta=12M`, `max_position=65536`, llama3 scaling). The headline "iter 1192 > Apertus-Base" claim is therefore a *Path-A-baseline / Path-B-run* claim. A perturbed Path-B Apertus-Base (the matched-config diagnostic — `rope_theta=500K` override on Path-A weights) gives a lower bookend at 0.4272 [0.4096, 0.4456]; iter 1192 also exceeds this perturbed bookend. The bracket holds under both bookend geometries; a clean Path-B Apertus-Base counterfactual does not and cannot exist.

Recommendation: proceed to Task 2 (extension experiment) on Path A per `cpt-plan` §3.4 Q3.4.10. The 5 B → 10 B continuation is *not* gating Task 2; bracketed below in §12.

---

## 2. Method

### 2.1 Training settings (full spec: `goal/hyperparameters.json`; rationale: `cpt-plan.md` §2.1)

| Block | Setting | Value | Source |
|---|---|---|---|
| Base model | HF id | `swiss-ai/Apertus-8B-2509` | `hyperparameters.json[base_model.hf_id]` |
| Base architecture | layers / hidden / MLP / Q/KV heads | 32 / 4096 / 21 504 / 32 / 8 | Apertus paper §2.3 |
| Tokenizer | vocab | 131 072 (Mistral-Nemo tekken v3, no extension) | `hyperparameters.json[tokenizer]` |
| Training geometry (Path B) | `rope_theta` / `max_position` / scaling / seqlen | 500 000 / 4096 / null / 4096 | `hyperparameters.json[training_geometry.values]`; Apertus paper §2.3 initial pretraining; matches bakeoff |
| Optimizer | AdEMAMix β1/β2/β3/α | 0.9 / 0.999 / **0.99** / 8 | `hyperparameters.json[optimizer]`; bakeoff used β3=0.9999 |
| α/β3 warmup | optimizer steps / tokens | 287 / 1.2 B | `hyperparameters.json[optimizer.alpha_beta3_warmup_steps]` |
| Weight decay / grad clip | | 0.1 / 0.1 | `hyperparameters.json[optimizer]` |
| LR schedule | base / warmup / shape / cooldown | **1.1e-5** / 1.2 B tokens (287 steps) / linear-warmup-then-constant / none | `hyperparameters.json[lr_schedule]`; bakeoff used 1.5e-5 |
| Loss | type / k / h / hash seed | **Goldfish** / 50 / 50 / 2 971 215 073 | `hyperparameters.json[loss]`; bakeoff used NTP |
| Batch | global / micro / TP / PP / DP / grad-accum | 1024 samples (4.194 M tokens) / 2 / 2 / 1 / 2 / 256 | `hyperparameters.json[batch_and_precision]` |
| Precision | | bf16 + fp32 master gradients | `hyperparameters.json[batch_and_precision.precision]` |
| Init checkpoint | path | `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched` | `RUN_LOG_20260528.md` (R17-patched TP=2 Vanilla) |

### 2.2 Data (full spec: `goal/hyperparameters.json[training_mix]`)

Top-level mix held at bakeoff B1 70 / 24 / 4 / 2; only narrowing is inside the Greek slice (HPLT-only, GlossAPI Greek excluded). Effective shares from `validation/dataset_validation.json`: Greek 0.700000, replay 0.239999, code 0.040000, math 0.020000. Greek source = `HPLT/ell_Grek_ge8_no_mt_clean60` (wave4, 250 shards, 46.6 M docs after 2026-05-19 Apertus-cross-dedup overlay, fraction kept 0.957). Replay = bakeoff B1 (`cpt_plan.md` v0.7 in 03 subproject) across en/fr/de/ru.

### 2.3 Regime changes from bakeoff (cpt-plan §2.1 deltas):

| Block | Bakeoff | This run | Apertus reference |
|---|---|---|---|
| Loss | NTP | **Goldfish k=h=50** | Apertus pretraining default |
| LR (base) | 1.5e-5 | **1.1e-5** | Paper §2.5 long-context continuation |
| LR shape | trapezoid/cooldown | **linear-warmup-then-constant** | Paper §2.5 (no cooldown in Task 1) |
| AdEMAMix β3 | 0.9999 | **0.99** | Paper Appendix C short-run recommendation |
| α/β3 warmup | 238 steps | **287 steps** (1.2 B tokens) | Paper §2.5 continuation warmup window |
| Greek source | HPLT + GlossAPI | **HPLT only** | Diagnostic narrowing (cpt-plan §2.2) |

### 2.4 Geometry caveat

The release config of `swiss-ai/Apertus-8B-2509` ships on Path A (`rope_theta=12M`, `max_position=65536`, llama3 scaling). The bakeoff and this run train under Path B (`rope_theta=500K`, `max_position=4096`, no scaling). Cost of Path B documented in `hyperparameters.json[training_geometry.cost_of_path_b]`: initializing Path-A weights and inferring under Path B perturbs the base (matched-config diagnostic point estimate 0.4272 on the 3-task headline, BPB ≈ 1.22). The first ~1 B tokens of CPT under Path B carry the rope re-adaptation cost; iter 477 (post-warmup, 2 B tokens) is the first stable-LR snapshot. Task 2 should run on Path A — see `cpt-plan` §3.4 Q3.4.10 and §11 below.

---

## 3. Headline Results Table

V4 v3 marginal CIs (`reports/v4_bootstrap_cis_native_mcq.json`, revision v3; 1 000 resamples, percentile, rng_seed=20 260 529, per-task item-level resampling within each task, headline = macro-mean across tasks per resample).

### 3.1 Native Greek headline (3-task: GreekMMLU + ILSP Medical MCQA + ILSP ASEP MCQA)

| Model / checkpoint | Tokens | headline_3task | 95 % CI | Source |
|---|---:|---:|---|---|
| Apertus-Base (Path A) | n/a | **0.4817** | [0.4629, 0.4997] | V4 v3 |
| Apertus-Base matched Path-B-perturbed (diagnostic) | n/a | 0.4272 | [0.4096, 0.4456] | V4 v3, see §10 |
| Bakeoff Vanilla-2B | 2 B | 0.4327 | [0.4144, 0.4502] | V4 v3 |
| Bakeoff Vanilla-3.5B | 3.5 B | 0.4370 | [0.4185, 0.4545] | V4 v3 |
| Bakeoff Vanilla-5B | 5 B | 0.4305 | [0.4134, 0.4485] | V4 v3 |
| **This run** iter 119 (Vanilla-0.5B) | 0.5 B | 0.4391 | [0.4219, 0.4565] | V4 v3 (mid-warmup) |
| **This run** iter 238 (Vanilla-1B) | 1.0 B | 0.4487 | [0.4299, 0.4670] | V4 v3 (mid-warmup) |
| **This run** iter 477 (Vanilla-2B) | 2.0 B | 0.4792 | [0.4604, 0.4967] | V4 v3 (post-warmup) |
| **This run** iter 834 (Vanilla-3.5B) | 3.5 B | 0.4790 | [0.4606, 0.4978] | V4 v3 |
| **This run** iter 1192 (Vanilla-5B) | 5.0 B | **0.4973** | **[0.4779, 0.5156]** | V4 v3 |

### 3.2 4-task aggregate including Plutus QA (diagnostic — NOT headline)

| Model / checkpoint | headline_4task_with_plutus | 95 % CI |
|---|---:|---|
| Apertus-Base | 0.4902 | [0.4692, 0.5118] |
| Apertus-Base matched-Path-B-perturbed | 0.4071 | [0.3880, 0.4299] |
| Bakeoff Vanilla-5B | 0.4329 | [0.4122, 0.4524] |
| iter 477 | 0.4816 | [0.4614, 0.5027] |
| iter 834 | 0.4814 | [0.4608, 0.5034] |
| iter 1192 | 0.4819 | [0.4600, 0.5046] |

The 4-task aggregate is flat across iter 477 / 834 / 1192 because the Plutus drop (§6) drags it down even as the 3-task headline climbs. The 3-task aggregate is the goal-defined headline per `hyperparameters.json[eval.greek_headline_native]`.

### 3.3 Per-task point estimates

| Task | n | Apertus-Base | iter 477 | iter 834 | iter 1192 | bakeoff-Vanilla-5B |
|---|---:|---:|---:|---:|---:|---:|
| greekmmlu | 16 632 | 0.5280 | 0.5075 | 0.5284 | **0.5584** | 0.4747 |
| ilsp_medical_mcqa | 432 | 0.4097 | 0.3958 | 0.3935 | 0.4028 | 0.3333 |
| ilsp_mcqa_asep | 1 200 | 0.5075 | 0.5342 | 0.5150 | 0.5308 | 0.4833 |
| plutus_qa (diagnostic) | 225 | 0.5156 | 0.4889 | 0.4889 | 0.4356 | 0.4400 |

---

## 4. Trajectory Analysis (V4 v3 Paired CIs)

The five load-bearing paired CIs from V4 v3 (1 000 resamples, paired indices on the model pair → paired diff). All CIs are 95 % percentile.

| # | Comparison | Metric | Δ | CI | Outside zero? | Reading |
|---|---|---|---:|---|---|---|
| 1 | iter-477 − bakeoff-Vanilla-2B | headline_3task | +0.0465 | [+0.0299, +0.0629] | **yes** | Regime is +4.65 pp better than the bakeoff at 2 B (load-bearing regime evidence — first stable-LR readout) |
| 2 | iter-834 − bakeoff-Vanilla-3.5B | headline_3task | +0.0420 | [+0.0262, +0.0581] | **yes** | Hold at 3.5 B |
| 3 | iter-1192 − bakeoff-Vanilla-5B | headline_3task | +0.0669 | [+0.0513, +0.0830] | **yes** | Largest matched-token-mark gain in the run |
| 4 | iter-1192 − Apertus-Base (Path A) | headline_3task | +0.0156 | [+0.0016, +0.0284] | **yes (barely)** | First time the headline crosses Apertus-Base; Path-A baseline / Path-B run caveat |
| 5 | iter-1192 − Apertus-Base matched-Path-B-perturbed | headline_3task | +0.0701 | [+0.0537, +0.0857] | **yes** | iter 1192 also clears the perturbed Path-B bookend |
| 6 | iter-1192 − iter-477 | headline_3task | +0.0182 | [+0.0060, +0.0306] | **yes** | Full post-warmup gain (3 B tokens) is real |
| 7 | iter-1192 − iter-834 | headline_3task | +0.0184 | [+0.0080, +0.0295] | **yes** | Post-plateau slope is non-zero |
| — | iter-834 − iter-477 | headline_3task | −0.0002 | [−0.0123, +0.0114] | no | Plateau over 1.5 B tokens (real plateau; matches Vanilla-3.5B critic prior bootstrap to 4 dp) |

**Reading.** The trajectory shape is **warmup → +3.05 pp burst → flat 1.5 B-token segment → +1.84 pp endpoint lift**. This is *not* the bakeoff-Vanilla "peak early then drift" pattern. The plateau is real but transient; the slope re-resumes after iter 834. The 5 B endpoint is not saturated — both CI 6 and CI 7 exclude zero.

---

## 5. Per-Task Structure

Where iter 1192 surpassed / equaled / lagged each baseline (paired V4 v3 CIs):

### 5.1 GreekMMLU (n = 16 632) — the engine

- iter 1192 point 0.5584 vs Apertus-Base 0.5280 → paired Δ = **+0.0304** [+0.0254, +0.0361], outside zero
- iter 1192 vs bakeoff-Vanilla-5B 0.4747 → paired Δ = **+0.0836** [+0.0770, +0.0902], outside zero
- iter 1192 vs iter 477 → paired Δ = **+0.0509** [+0.0457, +0.0562], outside zero (the +1.8 pp headline gain over the full post-warmup window is *primarily* GreekMMLU)
- iter 1192 vs iter 834 → paired Δ = **+0.0300** [+0.0255, +0.0345], outside zero

Largest absolute and largest paired lift in the run. Drives the headline trajectory.

### 5.2 ILSP ASEP MCQA (n = 1 200) — the rebound

- iter 1192 point 0.5308 vs Apertus-Base 0.5075 → paired Δ = **+0.0233** [+0.0050, +0.0425], outside zero
- iter 1192 vs bakeoff-Vanilla-5B 0.4833 → paired Δ = **+0.0475** [+0.0267, +0.0683], outside zero
- iter 1192 vs iter 834 → paired Δ = +0.0158 [+0.0017, +0.0308], outside zero — **fully recovers the iter 477 → iter 834 ASEP regression (−1.92 pp) the Vanilla-3.5B critic flagged**
- iter 477 → iter 1192 → paired Δ = −0.0033 [−0.0208, +0.0125], inside zero (the iter-834 dip and iter-1192 recovery net to ≈ 0 over the full window)

ASEP rebounded from the iter-834 transient.

### 5.3 ILSP Medical MCQA (n = 432) — noise floor

- iter 1192 point 0.4028 vs Apertus-Base 0.4097 → paired Δ = −0.0069 [−0.0440, +0.0278], inside zero
- iter 1192 vs bakeoff-Vanilla-5B 0.3333 → paired Δ = +0.0694 [+0.0254, +0.1111], outside zero
- iter 1192 vs iter 834 → +0.0093 [−0.0208, +0.0394], inside zero
- Per-task n = 432 dominates the CI width; movement at this scale is mostly noise.

Roughly matched to Apertus-Base within noise; well above bakeoff.

### 5.4 Plutus QA (n = 225, diagnostic only) — the new caveat

- iter 1192 point 0.4356 vs Apertus-Base 0.5156 → paired Δ = **−0.0800** [−0.1422, −0.0133], **outside zero negative**
- iter 1192 vs iter 834 → paired Δ = **−0.0533** [−0.1067, +0.0000] — **CI upper bound exactly zero**
- iter 1192 vs iter 477 → paired Δ = −0.0533 [−0.1111, +0.0000]
- iter 1192 vs bakeoff-Vanilla-5B 0.4400 → paired Δ = −0.0044 [−0.0711, +0.0578], inside zero

Plutus is the only outside-zero negative per-task delta in the V4 v3 table. Diagnostic-only per `hyperparameters.json[eval.greek_diagnostic_only]`; not in the headline. See §6.

---

## 6. Plutus Drop Discussion

Full investigation: `reports/plutus_investigation_20260530.md`.

| Cohort | n | Point | 95 % CI |
|---|---:|---:|---|
| iter 477 | 225 | 0.4889 | [0.4222, 0.5556] |
| iter 834 | 225 | 0.4889 | [0.4222, 0.5556] |
| iter 1192 | 225 | **0.4356** | [0.3689, 0.4978] |

Item-level transition iter 834 → iter 1192: 84 both-correct / 26 regressions (834-right, 1192-wrong) / 14 gains / 101 both-wrong. McNemar exact two-tailed p over 40 discordant pairs = **0.081** (26 vs 14).

**Within-2-choice subset (n=27):** 14 of 27 two-choice items flipped from correct to wrong — a **52 % flip rate** on the 2-choice subset. Every 2-choice drop is "wrong by 1", i.e. the model picked the alternative. On the 4-choice subset (n=188), 9 of 12 drops also pick the adjacent-index distractor. Pattern is consistent with **positional / token-frequency drift**, not content-level regression.

**Confidence on the drops:** iter 1192 top1/top2 margin on the 26 drops = 0.183 (median 0.125) vs population mean 0.566. The drops are low-confidence flips — items the model was on the fence about, where a small representational shift pushed the choice over.

**Bakeoff cross-check.** Bakeoff Vanilla Plutus is monotone up: 0.4044 → 0.4222 → 0.4400 across 2 → 3.5 → 5 B. No iter-477/iter-834 plateau in the bakeoff. iter 1192's 0.4356 sits inside the bakeoff-Vanilla-5B CI ([0.382, 0.498]) and inside all earlier bakeoff CIs. Paired iter 1192 vs bakeoff-Vanilla-5B = −0.0044 [−0.0711, +0.0578], indistinguishable.

**Verdict.** Ambiguous, leaning small-real.

- Paired iter-1192 vs iter-834 borderline non-significant (CI upper exactly 0, McNemar p = 0.081).
- Paired iter-1192 vs Apertus-Base IS significant (Δ = −0.0800, CI [−0.1422, −0.0133]).
- iter 1192 sits below the Apertus-Base CI and below iter-477 / iter-834 marginal CIs, but inside all bakeoff Vanilla CIs.
- Headline drift (4-task with Plutus) is ≈ +1.3 pp on the macro-mean — below headline noise floor at this n_items mix.

Per cpt-plan §6, no threshold rule is proposed on this single 225-item diagnostic. Position for the 5 B report: "5.33 pp drop at n=225 with paired CI [−0.1067, +0.0000], McNemar p = 0.081, paired vs Apertus-Base outside zero — too weak to call signal, too large to ignore as noise; first hint of domain-specific forgetting in the run; do not redesign on this point." Reporting plan: report Δ + both CIs (vs iter 834 borderline, vs Apertus-Base outside zero) + McNemar p; do not promote in either direction. Per cpt-plan §2.3 row 3 ("BPC improves; native MCQ stays flat below Apertus-Base → forgetting via KL-to-base on fixed probes; higher replay share"), this is *Plutus-specific* and a follow-up KL-to-base probe is a reasonable next-step before Task 2 (not gating).

---

## 7. Multilingual Retention at iter 1192

Retention results from `eval_04_vanilla_goldfish_5b_20260528T112539Z/iter_0001192/retention/results_2026-05-30T13-57-34.322587.json`. Goal-aligned languages: en / fr / de / ru per `hyperparameters.json[eval.multilingual_retention.languages]`.

**Baseline correction.** Earlier drafts compared iter 1192 to iter 119 as
the "starting point." That is wrong: iter 119 is 0.5 B tokens INTO training,
not the pre-CPT state. Iter 119 has already absorbed a chunk of the rope
re-adaptation cost. The correct baseline is the **matched-config
Apertus-Base Path B** = Apertus-Base weights loaded under our training
geometry (`rope_theta=500K`, `max_pos=4096`) with NO CPT updates. That is
literally iter 0 of our run. Pulled from
`eval_apertus_base_matched_rope500k_seq4096/retention/`. The table below
uses this true-init baseline.

| Task | Init (matched Path B) | iter 119 | iter 1192 | Δ iter 1192 vs init (CPT effect) |
|---|---:|---:|---:|---:|
| mmlu (en, Hendrycks) | 0.5624 | 0.5674 | 0.5798 | **+1.74 pp** |
| global_mmlu_en | 0.6025 | 0.6050 | 0.6500 | **+4.75 pp** |
| global_mmlu_fr | 0.5425 | 0.5800 | 0.5875 | **+4.50 pp** |
| global_mmlu_de | 0.5725 | 0.5875 | 0.5950 | **+2.25 pp** |
| arc_challenge | 0.5384 | 0.5247 | 0.5367 | −0.17 pp |
| arc_easy | 0.8279 | 0.8035 | 0.8194 | **−0.85 pp** |
| piqa | 0.7922 | 0.7905 | 0.7889 | −0.33 pp |
| hellaswag | 0.5862 | 0.5896 | 0.5906 | +0.44 pp |
| xnli_en | 0.5112 | 0.4904 | 0.5486 | **+3.74 pp** |
| xnli_fr | 0.4859 | 0.4678 | 0.5052 | **+1.93 pp** |
| xnli_de | 0.4968 | 0.4847 | 0.4948 | −0.20 pp |
| xnli_ru | 0.4884 | 0.4880 | 0.4727 | **−1.57 pp** |

**Reading vs the corrected baseline.** Multilingual MMLU-family ALL gain
versus true init: global_mmlu_en +4.75 pp, global_mmlu_fr +4.50 pp,
global_mmlu_de +2.25 pp, Hendrycks `mmlu` (English) +1.74 pp. xnli_en
+3.74 pp and xnli_fr +1.93 pp. So the CPT regime grew non-Greek
knowledge-MCQ capability across the three Western languages it covers.
Two genuine regressions remain at iter 1192:

1. **xnli_ru −1.57 pp** vs true init — Russian XNLI is the only retention
   signal we have on Russian (`xstorycloze_ru` is in the cpt-plan §2.1
   default but absent from the executed `retention_only` bundle, see
   `goal/canonical_eval_tasks.json`). Russian-side regression is genuine,
   confirmed under the corrected baseline (the iter 119 → iter 1192 delta
   −1.53 pp matches the true-init delta within rounding).
2. **English commonsense (`arc_challenge`, `arc_easy`, `piqa`) all slightly
   below true init** (−0.17, −0.85, −0.33 pp). Small, within plausible
   English-commonsense noise floors, but the *direction* is consistently
   down — these were the iter-119 rope re-adaptation casualties that did
   NOT fully recover at iter 1192. `arc_easy −0.85 pp` is the largest.
   `hellaswag +0.44 pp` is the lone English-commonsense gain.

The iter 119 column documents the rope-readapted state: xnli_en at iter
119 was already −2.08 pp below true init (recovered to +3.74 by iter 1192);
xnli_fr at iter 119 was −1.81 pp below true init (recovered to +1.93);
arc_challenge / arc_easy at iter 119 were −1.37 / −2.44 pp below true init
(partial recovery; arc_easy still under).

Per cpt-plan §2.3 row 5 ("Multilingual retention degrades → Replay too low /
LR too high"): the rule fires narrowly on Russian-XNLI and on English
commonsense at iter 1192. Document; do not propose threshold or rerun.
A 10 B stretch (if launched) would clarify whether xnli_ru and English
commonsense regressions continue or reverse.

---

## 8. Greek BPB — Monotone Improvement Narrative

Heldout source: `cpt_greek_heldout_500_20260522.jsonl` (500 docs, byte-identical SHA256 `3487a53f…` and mtime `1779418719` since 2026-05-22 — *not* regenerated mid-run, so within-run trajectory deltas are meaningful).

| Checkpoint | Greek BPB | greek_academic | greek_dialogue_textbooks | greek_hplt_clean60 | greek_legal_civic |
|---|---:|---:|---:|---:|---:|
| iter 119 | 0.6049 | — | — | — | — |
| iter 477 | 0.4313 | 0.354 | 0.643 | 0.413 | 0.298 |
| iter 834 | 0.4197 | 0.337 | 0.622 | 0.392 | 0.287 |
| iter 1192 | **0.4132** | 0.331 | 0.616 | 0.384 | 0.283 |

**Monotone improvement across all four per-source registers iter 477 → iter 834 → iter 1192.** Not driven by a single register. iter 119 (mid-warmup) is much higher because of the rope re-adaptation cost (cpt-plan §2.1 "Cost of Path B" — first ~1 B tokens carry rope-adaptation). Post-warmup, BPB drops smoothly.

**Truncation caveat.** 146 of 500 docs (= 29.2 %, max_context = 4096) are prefix-truncated. The prompt-stated value "28.6 %" is incorrect — actual is 29.2 %, byte-identical across all five checkpoints. Within-run deltas remain valid because the truncation set is constant. A `non_truncated_subset_bpb` sensitivity check over the 354 clean docs was queued as a deferred-to-5B-report item (Decisions Matrix row D, `reports/decisions_matrix_20260529.md`); the values above ground the monotone improvement claim regardless.

Code BPB (200 docs, mostly untruncated): iter 1192 = 0.2646 vs iter 834 = 0.2697 vs iter 477 = 0.2807 — monotone down. Math BPB: iter 1192 = 0.5448 vs iter 834 = 0.5491 — monotone down. Language modeling on code/math is improving, not degrading.

---

## 9. Compute Summary

Full breakdown: `reports/gpu_hours_breakdown_20260530.md`.

| Quantity | GPU-h | Notes |
|---|---:|---|
| Total GPU-h billed (Clariden whole-node policy) | **145.85** | All GPU jobs in cohort requested `gres/gpu=4`; "requested" = "billed" |
| of which: training (5 successful chained segments) | 117.97 | jobs 2417446 / 2417447 / 2417448 / 2417449 / 2417450 |
| of which: per-checkpoint sidecars (24 jobs × 3 done checkpoints) | 21.75 | convert + 5 evals + checksum per checkpoint |
| of which: matched-config Apertus-Base eval | 2.74 | 3 jobs: native MCQ + retention + BPB |
| of which: MCQ resubmits (iter 238, iter 477 after `--export` comma-bug fix) | 2.01 | jobs 2422769 / 2422770 |
| of which: repair (warmup-assert retries) | 0.73 | failed iter-119 retry chain |
| of which: smokes | 0.64 | dataset + training + conversion + native + BPB |
| Dataset prep + xfer watcher | 0.00 | CPU-only on xfer (~34 elapsed h, 0 GPU-h) |
| Projected at chain completion | ≈ 204.9 | adds ~59 GPU-h for iter-834 finish + iter-1192 segment (before sidecar fan-outs) |

At-completion projection uses steady-state throughput 134.5 sec/iter on 4 GPUs from segments i596 + i715. The original `goal/hyperparameters.json[compute.training.estimated_wallclock_at_5b]` estimate was "about 46 h at observed bakeoff throughput of ~7.5k tokens/sec/GPU on 4 GPUs"; actual throughput stabilized at ≈ 8 000 tokens/sec/GPU on the iter-300 segment after cold-start, and the run completed iter 119 → iter 1192 in ~43 h of training wall — close to the prior estimate.

**Per-checkpoint sidecar overhead** (Decisions Matrix row Q, `script_audit_20260529.md` M-class): native-MCQ sidecars allocated a full 4-GPU GH200 node for a single-GPU eval. Per `hyperparameters.json[compute.training.sbatch_justification_axes]`, this is suboptimal CPU+GPU saturation; preserved to keep sidecars finishing promptly between training segments. Per-checkpoint sidecar wall ≈ 2 h on a 4-GPU node ≈ 8 GPU-h × 5 checkpoints (including the iter-1192 fan-out) ≈ 40 GPU-h total sidecar cost — within compute envelope.

---

## 10. Known Caveats

### 10.1 Path-A-vs-Path-B geometry confound (carried from Vanilla-0.5B C1 / Decisions Matrix row C)

The "iter 1192 > Apertus-Base" claim (paired CI [+0.0016, +0.0284]) is a Path-A-baseline / Path-B-run comparison. The matched-config Path-B-perturbed Apertus-Base (`Apertus-Base-matched-Path-B-perturbed` in V4 v3) sits at 0.4272 [0.4096, 0.4456] — a *perturbed* baseline, not a clean one. iter 1192 also exceeds this perturbed bookend (paired Δ = +0.0701 [+0.0537, +0.0857]). The bracket holds under both bookend geometries; **a clean Path-B-trained Apertus-Base counterfactual does not exist and cannot be built without retraining from scratch.** `hyperparameters.json[training_geometry.matched_config_diagnostic.status_note]` documents this as "DIAGNOSTIC of rope-perturbation, NOT a clean baseline."

### 10.2 Decontamination of Greek MCQ benchmark prompts absent (Decisions Matrix row E)

No MinHash / 13-gram artifact exists for `hplt_b1_5b.jsonl` vs the four native Greek MCQ benchmark prompts. Per `hyperparameters.json[production_blockers_status.V1.status="not_required_for_diagnostic"]` this is plan-coherent; the 2026-05-19 overlay handles Apertus-pretraining overlap. But four outside-zero paired CIs (iter 477 vs bakeoff-2B; iter 834 vs bakeoff-3.5B; iter 1192 vs bakeoff-5B; iter 1192 vs Apertus-Base) all share the same asymmetric contamination risk. A MinHash pass is the cheapest insurance policy in the remaining audit budget; documented here as an open caveat.

### 10.3 Greek BPB heldout 29.2 % prefix-truncated

146 / 500 docs exceed max_context = 4096 and are prefix-truncated. File byte-identical since 2026-05-22; within-run deltas valid. A `non_truncated_subset_bpb` over the 354 clean docs is documented (Decisions Matrix row D) but not yet executed. Does not invalidate the monotone BPB improvement claim because the truncation set is constant.

### 10.4 Matched-config Apertus-Base eval is diagnostic-only, not a clean baseline

Decisions Matrix row C/H + `hyperparameters.json[training_geometry.matched_config_diagnostic.status_note]`: the matched-config eval applies the Path-B `rope_theta=500K` override statically on Path-A-trained weights. This perturbs the base rather than re-anchoring it — useful as a bookend for the "what does Path-B rope geometry cost on Path-A-trained weights" sensitivity check; not valid as the Apertus-Base baseline.

### 10.5 `--export` comma-bug fix history (Decisions Matrix row A + script_audit_20260529.md C1)

The original `submit_checkpoint_sidecars.sh:156` passed `BENCHMARKS=` inside `--export=ALL,...,BENCHMARKS="$NATIVE_BENCHMARKS",...`. Slurm's comma-split silently truncated to GreekMMLU only — iter 238 and iter 477 first-attempt native MCQ aggregates had `n_tasks=1` with only GreekMMLU evaluated. Fix landed at `submit_checkpoint_sidecars.sh` SHAs `7eb4667e…` → `e865c65a…`: `BENCHMARKS=` was hoisted onto the sbatch shell line before `sbatch` so `--export=ALL` carries it. iter 238 (`2422769`) and iter 477 (`2422770`) were resubmitted with the correct 4-task list and feed the V4 v3 artifact. Verification: fix held for five consecutive checkpoints (iter 119 / 238 / 477 / 834 / 1192) with `headline.n_tasks=3`, `diagnostics.n_tasks=1`, explicit `headline_policy`.

`script_audit_20260529.md` C2 / C3 flag two related unquoted-heredoc fragility points (`run_eval.sbatch:127`, `run_greek_nlp_benchmark_hf.sbatch:60`) — same bug class as the `--export` issue. Currently safe (today's checkpoint paths have no offending characters); deferred to a future cleanup.

### 10.6 Codex → Claude Code review handoff (Decisions Matrix row S)

`hyperparameters.json[eval.adversarial_critique.codex_command]` specifies `codex exec` with `gpt-5.5` and `model_reasoning_effort="xhigh"`. Codex was unavailable during the 04-cohort review window; the per-checkpoint adversarial review was executed by a Claude Code subagent using the same prompt template, same scope, same artifact paths. `review_metadata.env[BACKEND]="claude-code-subagent"` records the substitution. First subagent run = Vanilla-1B critique (`adversarial_reviews/Vanilla-1B/adversarial_critique.md`, 2026-05-29T00:48Z); Vanilla-2B / 3.5B / 5B critiques followed the same path. Codex re-runs may supersede when the service is back; nothing in the V4 v3 artifact depends on the backend identity.

### 10.7 Other carried items (from Vanilla-5B critique §3, §7)

- Duplicate save pattern at segment boundaries: iter 1190 saved alongside iter 1192 (third such pattern). ≈ 50 GB quota cost per pair; no operational impact. Vanilla-5B critique M3.
- Stale `run_metadata.json[lr_decay_style]="1-sqrt"` while training argv uses `--lr-decay-style constant`. Persists across 5 critiques; cosmetic but should be patched. Decisions Matrix row R.
- `min_id=0` token in dataset scan likely the EOD marker injected by Megatron's `preprocess_data.py --append-eod`; not `<unk>`-spill. Confirm before Task 2 dataset build. Decisions Matrix row T.
- Retention sidecar runs 201 raw upstream task entries; the canonical subset (12 tasks across en/fr/de/ru per `goal/canonical_eval_tasks.json`) is reported above and the 189 non-canonical entries are filtered out at render time. Two canonical tasks are flagged `absent_on_disk` in the lockdown (`xstorycloze_en`, `xstorycloze_ru`) — they are in the cpt-plan §2.1 default proposal but the executed `retention_only` bundle does not include them, so coverage of the Russian slice is `xnli_ru`-only (Decisions Matrix row V).
- `xfer`-routed sidecar watcher + checksum sidecar (Decisions Matrix row I): patch from `--partition=xfer` to `normal --cpus-per-task=64 --mem=400G` *unapplied at iter 1192* — checksum job `2432553` completed only because the Apertus xfer maintenance reservation (drained till 2026-06-11) had not yet begun biting. Concretely risky for any post-2026-06-11 sidecar work, including Task 2. **Apply before any Task-2 launch.**

---

## 11. Implications for Task 2

Per `cpt-plan.md` §3.4 Q3.4.10, Task 2 (production extension experiment) should switch to **Path A** (`rope_theta=12M`, `max_position=65536`, llama3 scaling, sequence length 4096). Rationale recapitulated here:

- Path A removes the ~1 B tokens of rope re-adaptation cost — free signal-to-noise at small budgets.
- Path A removes the geometry confound on any Apertus-Base comparison — Task 2's primary comparison is *extension vs Vanilla under the same regime + geometry*, not *extension vs bakeoff arms*.
- There is no bakeoff-Path-A counterpart, so the bakeoff-comparability argument that justified Path B for Task 1 does not apply to Task 2.
- Apertus paper §2.5 + the released `swiss-ai/Apertus-8B-2509` `config.json` carry the Path-A values; no perturbation needed.

The Task-1 regime evidence — Goldfish + LR 1.1e-5 + AdEMAMix β3=0.99 + 1.2 B-token α/β3 warmup, on the bakeoff B1 70/24/4/2 mix — **carries over without ablation** under the cpt-plan §2.4 Q2.4.5 working position (defer the LR-only / β3-only / Goldfish-only decomposition; ship the coupled regime). The regime hypothesis being supported on Vanilla means Task 2's extension arms inherit a recipe that already produces +1.6 to +6.7 pp at the 5 B mark.

**Extension-specific settings still to lock for Task 2 v1.1** (`hyperparameters.json[extension_specific_settings]` are null for Vanilla and need values for extension arms):

- Embedding-only stabilization N (cpt-plan §3.1, Q3.4.1): plausible range 0.5 B–5 B; criterion-driven recommended.
- Differential LR multiplier on new-token E/U rows (cpt-plan §3.1, Q3.4.2): start 5× on E and 3× on U; escalate to 10× contingent on diagnostic signals.
- TD layer choice (cpt-plan §3.2): cheap sweep over layers 4 / 8 / 11 / 16 / 20 before the full TD CPT run.
- V8 Goldfish hash uniformity on extended vocab (`hyperparameters.json[production_blockers_status.V8]`): production-blocking if Goldfish is used with extension; verify before launch.

**Task 1 → Task 2 trigger met** per cpt-plan §2.3 row 1: "BPC improves vs bakeoff Vanilla early (≤ 1 B); native MCQ trajectory recovers → Proceed to Task 2 (extension experiment)." Both halves hold cleanly at 5 B.

---

## 12. Implications for 10 B Stretch (Decision Deferred)

`hyperparameters.json[schedule.target_tokens_stretch]=10000000000` with checkpoints at 7 B and 10 B. Decision deferred per `cpt-plan.md` §6 (no rule-bound trigger). Read from the 5 B endpoint:

- **Information value of 5 B → 10 B is non-zero.** Slope is not zero (CI 6 + CI 7 in §4 both exclude zero across the post-warmup window and the most recent segment).
- **Endpoint is not saturated.** Per-task GreekMMLU is still gaining (+5.09 pp full window, +3.00 pp last segment, both outside zero); ASEP has rebounded.
- **Linear-slope extrapolation (caveat: 3 post-warmup data points are too few for confident extrapolation):** iter 477 → iter 1192 = +0.0182 over 3 B tokens = +0.0061 / 1 B. If slope holds, iter 7 B ≈ 0.510 and iter 10 B ≈ 0.528 on the 3-task headline. At 10 B the run would be ~2.7–4.6 pp above Apertus-Base, well outside the V4 v3 Apertus-Base CI.
- **Expected delta bracket** if launched: iter 1192 → iter 7000 (2 B further) → ≈ +0.012 headline gain at slope; iter 1192 → iter 10000 (5 B further) → ≈ +0.030 headline gain. Both within the per-checkpoint CI half-width (~0.018) but the cumulative gain is likely outside zero at 10 B.
- **Conditions for committing 10 B:** (a) Plutus drop gets a clearer signal call from a follow-up KL-to-base probe or per-subject breakdown; (b) decontamination MinHash pass lands and the four outside-zero paired CIs survive; (c) `xfer`-watcher partition patch (Decisions Matrix row I) applied so sidecars don't pend behind the 2026-06-11 reservation.

Per cpt-plan §6 and Foivos's standing rule, no threshold is proposed for the 10 B decision. The 5 B report's job is to lay out the trajectory; the 10 B / Task 2 priority choice is a separate decision.

---

## 13. Tasks Completed / Remaining

Snapshot from `RUN_LOG_20260528.md` and current artifact state.

### Completed

- CPU-only HPLT-only Greek B1 dataset build (job `2415688`, xfer, 0 GPU-h, 178.8 min mix-builder + Megatron preprocess; validation `ok:true`, token scan `[0, 131 071]`, 70/24/4/2 mix verified).
- Code/math heldout build (job `2416003`, xfer, 0 GPU-h, exact final-training doc-id exclusion).
- 8-segment training chain (corrected after 2 repair attempts for the warmup-assert and walltime-cap issues): jobs `2417446` → `2417450` and the iter 834 → iter 1192 successors. All 5 required checkpoints landed: iter 119 / 238 / 477 / 834 / 1192.
- HF conversion + native MCQ + greek_nlp + heldout BPB + retention + code/math BPB + checksum sidecars for all 5 checkpoints (manifest rows all `expected_kinds` present; `handoff_ready=true` snapshots at `reports/iter_*_checkpoint_sidecar_handoff_pass.json`).
- Matched-config Apertus-Base eval at Path-B perturbation (`scripts/build_apertus_base_matched_config.sh` + `scripts/eval_apertus_base_matched_config.sbatch`, ~22 GPU-h on 3 parallel jobs).
- iter 238 / iter 477 native MCQ resubmits after `--export` comma-bug fix (jobs `2422769` / `2422770`).
- Per-checkpoint adversarial critiques: `adversarial_reviews/Vanilla-{0.5B,1B,2B,3.5B,5B}/adversarial_critique.md` (Claude Code subagent backend; see §10.6).
- V4 v3 artifact: `reports/v4_bootstrap_cis_native_mcq.json` (10 models, 83 delta_table rows, 1 000 resamples). 5 load-bearing paired CIs documented in §4.
- Script audit: `reports/script_audit_20260529.md` (3 critical / 7 major / 9 minor).
- Plutus investigation: `reports/plutus_investigation_20260530.md`.
- GPU-hours accounting: `reports/gpu_hours_breakdown_20260530.md`.
- Decisions matrix: `reports/decisions_matrix_20260529.md`.
- Trajectory plot: `reports/trajectory_native_mcq_with_cis.png`.

### Remaining (deferred to next-cohort follow-up, not gating this report)

- Decontamination MinHash pass (`hplt_b1_5b.jsonl` vs the 4 native Greek MCQ prompt sets).
- `non_truncated_subset_bpb` over the 354 untruncated docs (Greek BPB sensitivity check, Decisions Matrix row D).
- Per-subject GreekMMLU + ASEP breakdowns at iter 477 / 834 / 1192 (to discriminate broad-knowledge gain vs concentrated subject effects).
- Per-subject Plutus breakdown (if metadata present in predictions JSONL) to localise the iter 834 → iter 1192 drop.
- KL-to-base probe on a fixed Greek probe set at iter 477 / 834 / 1192 (to discriminate Plutus forgetting from noise).
- Path-A static probe of iter 1192 weights (symmetric to the matched-config Apertus-Base eval but in the inverse direction).
- C2 / C3 unquoted-heredoc fixes (`run_eval.sbatch:127`, `run_greek_nlp_benchmark_hf.sbatch:60`).
- Decisions Matrix row I — re-route watcher + checksum sidecar from `--partition=xfer` to `normal --cpus-per-task=64 --mem=400G` before any Task 2 launch (xfer maintenance reservation lifts 2026-06-11).
- Delete duplicate iter 1190 checkpoint (frees ≈ 50 GB on iopsstor).
- Patch stale `run_metadata.json[lr_decay_style]` field.

---

## 14. References

### Project docs

- `subprojects/04_cpt_training_regime_on_vanilla/goal/goal.md` — scope, locked training settings, required artifacts, stop conditions.
- `subprojects/04_cpt_training_regime_on_vanilla/goal/hyperparameters.json` — authoritative machine-readable settings.
- `subprojects/04_cpt_training_regime_on_vanilla/cpt-plan.md` §2.1 (full parameter spec), §2.2 (regime-hypothesis framing), §2.3 (failure-pattern → cause mapping), §2.4 (Q2.4.1 token budget, Q2.4.2 peak-early mechanism, Q2.4.5 ablation), §3.4 Q3.4.10 (Path-A revisit recommendation for Task 2), §6 (non-commitments — thresholds decided post-result).
- `subprojects/04_cpt_training_regime_on_vanilla/RUN_LOG_20260528.md` — chronological run log (dataset build → training chain → sidecars → 5 B report scaffold, including the warmup-assert + walltime-cap repairs and the Codex → Claude Code handoff).

### Reports

- `reports/v4_bootstrap_cis_native_mcq.json` — V4 v3 artifact (10 models, 83 delta_table rows, 1 000 resamples, rng=20 260 529). Load-bearing for every cross-arm claim above.
- `reports/decisions_matrix_20260529.md` — 24-row matrix of all open issues, severity, recommendation, action, status.
- `reports/script_audit_20260529.md` — 3 critical / 7 major / 9 minor findings on the 04 sidecar pipeline + 03 bakeoff eval scripts. Includes the `--export` comma-bug fix history.
- `reports/plutus_investigation_20260530.md` — Plutus QA iter 834 → iter 1192 drop investigation: marginal + paired CIs, McNemar p, item-level transition matrix, 2-choice flip-rate analysis, low-confidence margins.
- `reports/gpu_hours_breakdown_20260530.md` — Slurm-accounting GPU-hours breakdown, methodology, partition-policy ground truth.
- `reports/config_geometry_audit_iter_0000119.md` — Path-A-vs-Path-B geometry confound audit at iter 119.
- `reports/trajectory_native_mcq_with_cis.png` — V4 v3 trajectory plot.
- `reports/5B_REPORT_DRAFT.md` — superseded by this report.

### Adversarial critiques

- `adversarial_reviews/Vanilla-0.5B/adversarial_critique.md` (iter 119 — mid-warmup; first Path-A-vs-Path-B flag).
- `adversarial_reviews/Vanilla-1B/adversarial_critique.md` (iter 238 — mid-warmup; first Plutus-in-headline flag + `--export` comma-bug detection).
- `adversarial_reviews/Vanilla-2B/adversarial_critique.md` (iter 477 — first stable-LR readout; +4.65 pp regime evidence).
- `adversarial_reviews/Vanilla-3.5B/adversarial_critique.md` (iter 834 — plateau bracket [0.467, 0.491]; iter 834 vs Apertus-Base = inside CI; ASEP regression flagged).
- `adversarial_reviews/Vanilla-5B/adversarial_critique.md` (iter 1192 — endpoint; iter 1192 vs Apertus-Base outside zero; Plutus drop bracketed; ASEP recovered).

### Code and artifacts on Clariden

- Training run dir: `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_vanilla_goldfish_5b_20260528T112539Z`
- Eval root: `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/eval_04_vanilla_goldfish_5b_20260528T112539Z`
- Dataset run dir: `/capstor/scratch/cscs/fffoivos/runs/04_vanilla_cpt/04_hplt_b1_dataset_5b_20260528T112539Z`
- Megatron data prefix: `.../megatron/hplt_b1_base_text_document`
- Init checkpoint (R17-patched TP=2 Vanilla): `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched`
- Matched-config Apertus-Base: `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509-matched-rope500k-seq4096` (diagnostic only — see §10.4)
- Heldouts: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_{greek_500_20260522,code_200_20260528,math_200_20260528}.jsonl`

### Apertus paper references

- §2.3 (initial pretraining geometry, architecture, AdEMAMix defaults, Goldfish loss).
- §2.5 (long-context continuation; LR 1.1e-5, 1.2 B-token warmup, Path-A geometry).
- Appendix C (AdEMAMix α/β3 warmup; short-run β3=0.99 recommendation).
- Appendix F.3 / Table F.5 (Goldfish ablations at pretraining scale; no CPT precedent).
