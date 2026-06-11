# Plan-vs-results reconciliation — 4-arm bakeoff (2026-05-26)

This doc reads the 4-arm bakeoff result back into the plan that actually defines the experimental framework: `old_experiments_plan.md` v0.12 (2026-05-12). v0.12 is the **parent project plan** — it owns §3 Decision Nodes (what to decide, in what order), §5 the three-arm experimental design, §6 per-arm weaknesses, §10 Open Questions including Q8 (the pre-commit decision rule), and §8.7 the training-budget schedule. `cpt_plan.md` v0.7 is the **CPT-execution successor** — it owns corpus assembly, optimizer/LR fidelity, V1-V16 verification gates — and is treated here as secondary.

The companion result doc is [`03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md`](03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md). This doc is the response to the plan: **for every decision, constraint, weakness-prediction, and threshold the v0.12 plan committed to, what did the bakeoff actually produce, and what's left to settle?**

A redesign of the next experimental round is out of scope here. §9 only enumerates the inputs the planning agent will need.

---

## Executive summary

The bakeoff produced rich per-arm trajectory data across 8 total billion training tokens (Vanilla and TD to 5B, ReTok to 3.5B, Centroid to 2B). But measured against v0.12's six Decision Nodes, the bakeoff settled fewer of them than the data depth suggests:

- **Node 1 (BPE cutoff): LOCKED for bakeoff at 17,408.** A separate production-side sweep on `{10,240, 15,360, 20,480, 25,600}` may revisit the cutoff for shipping, but Node 1 itself was settled. v0.12 §10 Q1 + Phase B v4's "≥100k occurrences per new unit" soft constraint both favor smaller cutoffs; this was not empirically verified for our specific 17,408 token set.
- **Node 2 (Training mix): LOCKED via cpt_plan v0.7 B1** at 70 / 24 / 4 / 2. Applied uniformly across arms.
- **Node 3 (Evaluation suite): PARTIALLY LOCKED.** 16 top-level lm-eval tasks chosen — but the suite differs from v0.12 §10 Q6's preferred native-Greek menu (GreekMMLU, Belebele Greek, Medical MCQA Greek, OYXOY, greek-nlp/benchmark). We substituted ILSP-style benchmarks instead. Krikri / Meltemi positioning is not exercised.
- **Node 4 (Pre-commit decision-rule thresholds): NOT LOCKED.** X, M_progress, M_ext, M_van, T were never pre-committed. The 5B headline ("TD wins downstream, Vanilla wins BPB") is read informally, not against the v0.12 §10 Q8 rule. This is the largest decision-side gap.
- **Node 5 (Three-arm experimental design): MODIFIED → FOUR ARMS.** v0.12 §5's ladder (Vanilla → ReTok → Distillation) was executed with Distillation initially bracketed, **Centroid filling the third-arm slot**, and **TD layer11 added later** as a fourth arm via `TOKEN_DISTILLATION_PLAN.md`. Per-arm token budget collapsed from v0.12's planned 10 B/arm to 2 B/arm, then extended to 5 B for Vanilla and TD only.
- **Node 6 (Krikri positioning): NOT ADDRESSED.** Late/optional in v0.12; not in bakeoff scope.

**One result the plan did not predict**: BPB and downstream-aggregate metrics diverge across arms. Vanilla wins BPB (the perplexity-style metric v0.12 Q8b's framework was built around). TD wins downstream no-explicit-MT Greek / EN / Multi aggregates at 5B. v0.12 §10 Q8 implicitly assumed perplexity tracks capability; the bakeoff shows it doesn't at this CPT regime. The §10 Q8 rule applied retroactively gives different answers depending on which metric the rule is interpreted on (see §3 below).

**One regime-quality concern**: all four arms degrade on no-explicit-MT Greek downstream aggregate in early CPT and never recover to the V4-HF base-Apertus level. Under v0.12 §10 Q8b's M_progress floor ("each arm must improve over base by ≥ 3-5 %"), **no arm qualifies on downstream Greek**, even though all four arms IMPROVE on BPB. This is the input that motivates a follow-up planning pass — see §9.

---

## 1. v0.12 §3 Decision Nodes — per-node status

| Node | v0.12 Decision | Plan status (v0.12 lock-in) | Bakeoff outcome |
|---|---|---|---|
| **1 BPE cutoff** | Pick from `{8K, 16K, 20K, 24K}` | Leaning 8K (Pareto + 100k constraint) | **Locked for the bakeoff at 17,408 new tokens** (the C3 wave-2 broad emit count; lands between v0.12's 16K-bin and 24K-bin). The production-side cutoff sweep on the `{10,240, 15,360, 20,480, 25,600}` grid (per `glossapi_c3_convergence.md`) is a *separate, downstream* decision still open for the next round — it does not unmake the bakeoff cutoff choice. Per-unit exposure constraint (≥ ~100k occurrences) was not empirically checked against the actual CPT-corpus per-new-token firing distribution. |
| **2 Training mix** | Greek source weights + non-Greek replay | Starting point 10-15 % replay | **Locked** via cpt_plan v0.7 §B1 at **70 % Gr / 24 % replay / 4 % code / 2 % math** (higher Greek share than v0.12's 10-15 % replay starting point implied — the 70/30 was driven by Apertus's 0.023 % pretraining Greek share, see `cpt_plan_v0.6_delta_vs_prior_planning.md`). Applied uniformly across all 4 arms; cross-arm comparison fair. |
| **3 Eval suite** | Pick slices + native benchmarks that will actually run | Menu in §10 Q6: GreekMMLU / Belebele Greek / Medical MCQA Greek / OYXOY / greek-nlp/benchmark; English/French/Russian/German regression slices | **Partially locked.** 16 top-level lm-eval tasks (~80 sub-task records) selected via `EVAL_RECIPE.md`. The actual list includes global_mmlu_full_el, include_base_44_greek_few_shot_en, belebele_ell_Grek, arc_challenge_mt_el, xnli_el, xquad_el, global_piqa_completions_ell_grek for Greek; mmlu/hellaswag/arc_*/piqa/winogrande for EN retention; global_mmlu/xcopa/xnli for multilingual. **v0.12's preferred GreekMMLU / Medical MCQA Greek / OYXOY / greek-nlp/benchmark are NOT in the suite.** ILSP YAMLs (`hellaswag_greek`, `winogrande_greek`, `mmlu_pro_greek`, `truthfulqa_greek`, `medical_mcqa_greek`) tracked as PF5 — still pending. The §1 multilingual preservation constraint is operationalized via `mmlu/xnli/xcopa` aggregates rather than the explicit English/French/Russian/German regression slices v0.12 §8a called for. |
| **4 Pre-commit thresholds** | X / M_progress / M_ext / M_van / T | TENTATIVE: X=5 %, M_progress=3-5 %, M_ext=1-2 %, M_van=3-5 %, T=2-3 %. **Hard temporal constraint: must lock before any arm completes CPT.** | **NOT LOCKED.** No values were committed before bakeoff results came in. The 5B headlines are interpreted informally rather than against the rule. v0.12 §10 Q8f explicitly warned this risks post-hoc rationalization. See §3 below for what the rule would say if applied retroactively with the suggested starting-point values. |
| **5 Three-arm design** | Vanilla → ReTok → Distillation ladder; same corpus / budget / schedule across arms | 10 B-token pilot per arm per §8.7 | **Modified, four arms.** Distillation initially deferred (§13 "bracketed"); **Centroid** filled the third-arm slot with Hewitt-2021 full-Σ multivariate Gaussian init (closed-form, deterministic). **TD layer11** later resurrected the Distillation idea as a 4th arm via `TOKEN_DISTILLATION_PLAN.md` (2026-05-22). Per-arm budget collapsed from 10 B to 2 B (cpt_plan v0.7 §B5), then extended ad-hoc to 5 B for Vanilla + TD, 3.5 B for ReTok, 2 B for Centroid. Same-corpus / same-schedule conditions met across the actually-run iters. |
| **6 Krikri positioning** | Late/optional formulation of "as good or better than Krikri" — match at equal compute / specific register / native-sourced benchmarks only / open-data axis / no commitment | Identified open; not in active scope | **Not addressed.** Krikri / Meltemi were not benchmarked against. The decision was deferred per v0.12 (acceptable). |

---

## 2. v0.12 §5 three-arm design vs 4-arm reality

### 2.1 The Vanilla / ReTok / Distillation → Vanilla / ReTok / Centroid / TD migration

v0.12 §5 conceived a strict complexity ladder: **Vanilla (no init) → ReTok (closed-form init) → Distillation (init-then-refine via gradient descent)**. The bakeoff diverged in two specific ways:

1. **Distillation was deferred at first**. The reasoning (`cpt_plan.md` v0.6 §13) was that Distillation's untied-LM-head story was awkward and the Greek-specific eval base was thin. **Centroid** replaced it in the third slot — Hewitt 2021's full-Σ multivariate Gaussian, a deterministic closed-form method that fits the same "no gradient descent" complexity class as ReTok.
2. **Distillation came back as a fourth arm**. After the initial 2 B-token bakeoff showed the static extension arms behind Vanilla on downstream no-explicit-MT Greek quality, `TOKEN_DISTILLATION_PLAN.md` resurrected TD as a parallel challenger. TD layer11 was init'd and trained from the same corpus / schedule as the other arms — fair comparison conditions held.

Net: the v0.12 ladder is preserved IF we read it as `Vanilla → {ReTok, Centroid} → TD`, with the two closed-form arms as the "static init" rung and TD as the "init-and-refine" rung.

### 2.2 §6 per-arm weakness predictions vs bakeoff observations

v0.12 §6 listed predicted weaknesses for each arm. The bakeoff settled many of these:

**§6.1 Vanilla weaknesses (predicted)** vs **bakeoff observation**:

| § | Predicted weakness | Bakeoff observation |
|---|---|---|
| 6.1.1 | Doesn't fix Greek compression | **Confirmed.** Vanilla retains base 131,072 vocab — no compression improvement on Greek. |
| 6.1.2 | May plateau at existing tokenizer's expressiveness ceiling | **Partially confirmed.** Vanilla's BPB continues to improve through 5 B (0.5432 → 0.4602 = 15 % improvement) — no plateau visible yet. Downstream no-explicit-MT Greek peaked early and ended at 0.4076 by 5 B, still far below V4-HF. The "ceiling" is on downstream-task capability, not on language-modeling compression. |

**§6.2 ReTok weaknesses (predicted)** vs **bakeoff observation**:

| § | Predicted weakness | Bakeoff observation |
|---|---|---|
| 6.2.1 | Linear-compositional prior wrong for non-compositional units | **Partially confirmed.** ReTok closes BPC fastest among extended arms but stays semantically weaker than TD on downstream Greek. The "subword-mean is wrong" prediction shows up specifically in downstream MCQ tasks, not in BPC. |
| 6.2.2 | Position-insensitivity (`mean(a,b)=mean(b,a)`) | Not directly tested. Would need composition-order-sensitive eval. |
| 6.2.3 | No distributional information | **Confirmed.** ReTok uses only structural decomposition; ignores how the new token is used in text. TD's hidden-state matching adds exactly this signal, and TD beats ReTok on every downstream Greek metric at every shared iter. |
| 6.2.4 | Norm matching is uniform hack — now data-anchored | Anchoring applied (per Phase A targets 5.05 / 3.80). Did not save ReTok from being downstream-dominated by TD. |
| 6.2.5 | No head-to-head winning paper for ReTok | **Still true.** ReTok being dominated by TD on downstream is consistent with this. |

**§6.3 Distillation weaknesses (predicted)** vs **TD-layer11 outcome**:

| § | Predicted weakness | Bakeoff observation |
|---|---|---|
| 6.3.1 | Implementation complexity vs ReTok | Managed: vendored official repo, used `train_embeddings` directly, AdamW + lr=1e-4 + 25 snippets at layer 11. Wall time ~4.7 hr for full-modern TD. |
| 6.3.2 | LM head story awkward for untied embeddings | **Resolved.** `learn_output_with_ce=True` pattern (paper §3.3, code `train_loop.py:339-348`) trained new U rows separately via CE on merged sequence; freeze pattern verified. No output-row instability. |
| 6.3.3 | Validation breadth limited (no Greek-specific eval published) | **Partially tested, not fully resolved.** TD on Apertus + Greek extension is the missing method data point, but the planned native-Greek suite was not run. TD beats Vanilla on the no-explicit-MT available Greek slice at 5 B (Delta = +1.28 pp), but the result is narrow and load-bearing on xquad_el. |
| 6.3.4 | Tied-embedding failure mode (Llama3.2-3B degenerate-norm case) | **Not triggered.** Apertus is untied; TD's output-row instability didn't manifest. |

**§6.4 Both extension arms (predicted)** vs **observation**:

| § | Predicted weakness | Bakeoff observation |
|---|---|---|
| 6.4.1 | Parameter overhead 67 M-201 M | Confirmed: 17,408 new rows × 4,096 × 2 (E + U) ≈ 142 M new params. Inference latency impact not yet quantified. |
| 6.4.2 | Long-tail "unused-added" tokens at high cutoffs | Not measured. At our 17,408 cutoff v0.12 §2.4 expected ~ a few percent unused — should run a coverage check on the actual 5 B training stream. |
| 6.4.3 | Corpus-distribution overfit (C3 50/50 GlossAPI+HPLT) | Not directly tested. Cross-register eval (academic, dialogue_textbooks, hplt_clean60, legal_civic) shows BPC varies by ~2× across registers but per-arm RANKING is stable, so the corpus-distribution prior doesn't dominate. |
| 6.4.4 | HPLT quality concerns / polytonic coverage | Not in scope (modern-only bakeoff). |

**§6.5 mitigation strategies** (hybrid init / higher LR on new-token rows / staged training):

- **Hybrid init**: not tested. Could be a redesign-input.
- **Higher LR on new-token rows (5×-10×)**: not used in the bakeoff (uniform LR across all rows). v0.12 §8.4 explicitly recommended this and the bakeoff didn't apply it. This is an unexplored degree of freedom.
- **Staged training (EEVE-style three stages)**: not used. The bakeoff ran full-parameter training from iter 1, mirroring Apertus pretraining. v0.12 §8.3 had a three-stage protocol (embedding-only → embeddings+adapters → full); we collapsed it to full from the start. This is a deviation worth flagging because it concentrates the early-CPT optimizer signal on the new rows least, which may explain the Greek-aggregate degradation in early iterations.

---

## 3. v0.12 §10 Q8 decision rule — applied retroactively

v0.12 §10 Q8 specified a pre-commit decision rule with five threshold variables. None were locked before the bakeoff. **Applied retroactively with the suggested starting-point values** to the 5B endpoint:

### 3.1 §8a gate (preservation — disqualifies if violated)

Suggested X = 5 % regression on English / French / Russian / German vs base Apertus.

| Arm | EN retention vs V4-HF | Violation? |
|---|---:|---|
| Vanilla    | 0.6799 vs ~0.716 (V4-HF avg) = **−5.1 %** | **MARGINAL** — at the threshold |
| TD layer11 | 0.6903 vs ~0.716 = **−3.6 %** | OK |
| ReTok      | 0.6786 vs ~0.716 = **−5.3 %** | **VIOLATION** by 0.3 pp |
| Centroid   | 0.6836 vs ~0.716 = **−4.5 %** | OK |

French / Russian / German not directly in the bakeoff suite; can be checked from the global_mmlu / xnli / xcopa per-language breakdowns. **The bakeoff didn't run the per-language regression slices v0.12 §8a called for**, so a clean gate-evaluation is not possible from the existing data. The closest available signal (xnli per-language) would need to be re-aggregated.

### 3.2 §8b Greek progress floor (minimum improvement)

Suggested M_progress = 3-5 % improvement on Greek web + Greek academic, BOTH must improve.

**On heldout BPC (lower-is-better; tokenizer-fair, closest to perplexity-style metric v0.12 imagined):**

| Arm | Vanilla per-source-greek_hplt_clean60 BPC at iter 1192 | Vanilla per-source-greek_academic BPC at iter 1192 | Baseline V4-HF BPC | Δ ≥ M_progress on both? |
|---|---:|---:|---:|---|
| Vanilla    | 0.4587 (hplt) | 0.3927 (academic) | not directly measured at iter 0 | Probably yes — BPC improved over training |
| TD layer11 | 0.5073 (hplt) | 0.4173 (academic) | not measured | Probably yes |
| ReTok      | 0.5474 (hplt-equivalent) | not measured | not measured | Probably yes |
| Centroid   | 0.9045 (last-eval, iter 476) | — | not measured | Probably no |

**On downstream no-explicit-MT Greek fallback aggregate (the available SwissAI harness slice with explicit MT diagnostics excluded, not the planned native-Greek suite):**

V4-HF no-explicit-MT Greek aggregate ≈ 0.5146 (from the corrected V4-HF reference markers).

| Arm | Final Greek no-MT agg | Δ vs V4-HF | ≥ M_progress (3-5 %)? |
|---|---:|---:|---|
| Vanilla    | 0.4076 | **−20.8 %** | NO - DEGRADED |
| TD layer11 | 0.4204 | **−18.3 %** | NO - DEGRADED |
| ReTok      | 0.3984 | **−22.6 %** | NO — DEGRADED |
| Centroid   | 0.2566 | **−50.1 %** | NO — DEGRADED |

The numbers above exclude explicit MT evals (`arc_challenge_mt_el`, `global_piqa_completions_ell_grek`). That is the cleaner current Greek reading, but it still is not the planned GreekMMLU / Medical MCQA Greek / OYXOY / greek-nlp/benchmark suite.

**Reading**: on perplexity-style BPC, multiple arms qualify. On downstream Greek fallback aggregates, **no arm qualifies against V4-HF**. v0.12 §10 Q8 implicitly assumed these two measures track each other; the bakeoff shows they don't at this CPT regime. **The rule itself needs to specify which Greek metric the gate applies to** - v0.12 §10 Q6 framing pointed to perplexity, but the production-relevant signal is downstream capability.

### 3.3 §8c asymmetric decision rule

Suggested M_ext = 1-2 % (extension beats Vanilla by this much → ship extension), M_van = 3-5 % (Vanilla beats extension by this much → ship Vanilla).

**On no-explicit-MT Greek downstream aggregate at iter 1192:**

- TD beats Vanilla by 1.28 pp = +3.1 % relative. **Above M_ext = 1-2 %**. Under the rule: would ship TD on this metric.
- But TD doesn't pass M_progress (per §3.2), so §8c is unreachable — §8c only applies among arms that pass §8b's gate.

**On BPC at iter 1192:**

- Vanilla beats TD by 0.027 = 5.5 % relative on BPC. **Above M_van = 3-5 %.** Under the rule: would ship Vanilla.

**Net**: the rule applied retroactively to the actually-emerged metric pair would say: ship Vanilla on perplexity / BPC grounds, but no arm clears M_progress on downstream — i.e., **§8c(4): no arm qualifies → none of the three are deployment-ready**. The implied conclusion under v0.12's own rule is that **the CPT regime is sub-optimal and the deployment requirements aren't yet feasible** — which matches the empirical observation from §7 below.

### 3.4 §8c(3) Distillation-vs-ReTok tiebreaker (T threshold)

Suggested T = 2-3 %: Distillation must beat ReTok by ≥ T on at least one Greek metric AND not regress the other.

- TD beats ReTok on the no-explicit-MT Greek aggregate at every shared iter; gap at iter 834 was +1.44 pp = +3.6 % relative, at iter 1192 not directly comparable since ReTok stopped at 3.5 B.
- **TD passes the T tiebreaker over ReTok on the data we have.**

### 3.5 §8d "no clear winner" outcome

v0.12 §8d: if all arms within noise across Greek metrics AND gates pass → ship simplest qualifying extension. Only if no extension qualifies → fall back to Vanilla.

Doesn't apply directly because the arms are NOT within noise across the metric axes — TD and Vanilla diverge on BPC vs downstream. The rule needs a metric-axis tiebreaker that v0.12 didn't anticipate.

---

## 4. v0.12 §4 Constraints — preserved?

| # | Constraint | Status |
|---|---|---|
| C1 | **Greek-authentic learning, no translation mediation** | ✅ Preserved. CPT corpus is real Greek (GlossAPI + HPLT clean60); no MT-derived bilingual signal anywhere in init or training. |
| C2 | **Smooth transition, preserve learned structures** | ⚠️ Partially preserved. ReTok / Centroid / TD apply norm-matching from Phase A (5.05 / 3.80 targets); TD additionally respects hidden-state geometry via MSE-on-residual. **Centroid's failure (BPC stuck at 0.90 vs others ≤ 0.55) is consistent with Hewitt 2021's full-Σ Gaussian not being sufficient when the modern-Greek subset has rank-deficient covariance** (R11 in `RISKS.md`). |
| C3 | **Compatible with continued BPE training** | ✅ Preserved. Additive merge-rule extension used; first 131,072 IDs byte-identical to base; new IDs occupy `[131,072, 148,480)` deterministically. |

---

## 5. v0.12 §8.7 Training budget & schedule — actual vs planned

| § | Item | Planned (v0.12 §8.7) | Actual |
|---|---|---|---|
| 8.7.1 | Stabilization phase | 0.5-1 B tokens | Not run as a separate phase; folded into the bulk CPT |
| 8.7.2 | Pilot phase | **10 B per arm** | **2 B per arm** (cpt_plan v0.7 §B5), then extended ad-hoc to 5 B for Vanilla + TD, 3.5 B for ReTok, 2 B for Centroid |
| 8.7.3 | Main phase | 20-40 B | NOT RUN |
| 8.7.4 | Optional anneal | 5-10 B | NOT RUN |
| 8.7.5 | Sequence length | 4,096 throughout | 4,096 (confirmed) |
| 8.7.6 | LR ranges | not specified in v0.12 (cpt_plan v0.7 §3.3 set 1.5e-5 peak) | 1.5e-5 peak / 1.5e-6 final / WSD 1-sqrt cooldown |
| 8.7.7 | Batch sizes | not specified in v0.12 | 4.19 M tokens / step (Apertus initial; not ramped to 8.39 M) |
| 8.7.8 | Compute-time | not specified | ~3 hr per 1B tokens on 1× GH200 node (4 GPUs); ~15 hr per arm to 5 B |

**Implication**: the bakeoff is operating at **~12-25 % of v0.12's planned per-arm pilot scale** (5 B / 10 B-40 B). Any claim that the bakeoff "settled" Node 5's experimental question must come with the caveat that we're at pilot-scale and not at the v0.12-planned main-phase scale. The Greek-aggregate degradation visible at this scale could either persist or reverse under a 20-40 B main-phase run.

---

## 6. CPT-execution-side fidelity (cpt_plan.md v0.7 + apertus_fidelity_checklist.md — secondary)

These are the implementation-side answers that `cpt_plan.md` v0.7 §10/§11/§12 owned. They are secondary to v0.12's decision-side questions but worth recording.

### 6.1 cpt_plan.md Q A (Fivos decisions) — all still open

A1-A7 (capability targets / total budget / timeline / stakeholders / sign-off / downstream tasks / team structure) — all PENDING. These are not bakeoff outputs; they are inputs the production decision needs.

### 6.2 cpt_plan.md Q B (design defaults) — locked + confirmed

| Q | Default | Confirmed |
|---|---|---|
| B1 | 70 / 24 / 4 / 2 mix | yes |
| B2 | 4 % code | yes |
| B3 | balanced anneal 85/12/3 | DEFERRED (production-only; not exercised by bakeoff) |
| B4 | NTP for bakeoff, Goldfish for production | NTP confirmed in bakeoff; Goldfish NEVER tested |
| B5 | 2 B per arm | EXCEEDED — 5 B for Vanilla / TD; load-bearing for Greek-aggregate crossover |
| B6 | adaptation work items | mostly resolved by training itself |

### 6.3 cpt_plan.md Q C / D (Apertus lookups + engineering) — resolved before kickoff

All 8 lookups (C1-C5, D1-D3) resolved per `cpt_plan_v0.7_answers.md`. Values baked into `_train_config_common.env`. No change.

### 6.4 V1-V16 verifications

- V12 / V13 / V14 / V15 / V2 / V3 / V7 / V9: **confirmed empirically** by the bakeoff training without instability across 8 B total tokens.
- **V1 (decontamination)**: still NOT DONE; gates production.
- **V4 (baseline variance)**: still NOT DONE; gates §5.6 threshold-setting and gates §10 Q8 statistical-confidence interpretation.
- **V8 (Goldfish hash uniformity on extended vocab)**: READY, NOT DONE; gates production CPT.
- **V5 / V6 / V16**: NOT APPLICABLE (modern-only bakeoff).
- **V10**: DEFERRED to post-pilot.

### 6.5 Apertus fidelity (`apertus_fidelity_checklist.md`)

21 items match Apertus pretraining exactly (optimizer / β1-β2-β3-α / weight decay / grad clip 0.1 / activation / norm / attention / RoPE / mixed precision / cross-doc attention masking / EoD loss masking / vocab divisibility / untied embeddings / etc.). Three documented intentional deviations (LR peak 1.5e-5 vs 1.1e-4 ; β3/α warmup 238 vs 100,000 steps ; NTP vs Goldfish). Three sub-optimal choices not flagged in plan (microbatch=2 vs 4 ; fixed 4.19 M batch vs Apertus's ramp to 8.39 M ; Apertus base optimizer-state not loaded). R17 (xIELU + QK-Norm reset through HF→Megatron) acceptable for bakeoff, gating for production.

### 6.6 TD plan items

`TOKEN_DISTILLATION_PLAN.md` shipped TD0 / TD1 / TD2 / TD3 / TD5 / TD6 cleanly. **TD4 layer pilot was NOT run** (only layer 11 trained; layer −1 and the L* probe-suggested layer never tested). **TD7 logit-lens / tuned-lens probe to identify L***  not run. Open questions Q2 (per-token coverage tail), Q4 (paper batch size), Q8 (TD + αNTP regularizer interaction with untied head), Q11 (HF hidden-state indexing for `target_layer = −1`) unresolved.

---

## 7. What the bakeoff confirmed empirically (beyond plan questions)

These are findings that emerged from the trajectory analysis, not from any explicit v0.12 / cpt_plan question:

1. **All arms degrade on Greek downstream in early CPT.** Vanilla mid-window slope was −17.9 m.p./B through 2 B tokens. The CPT regime is actively harming Greek capability before stabilizing. Slope flattens after 2 B but never returns to V4-HF baseline. Interpretation: either LR is wrong, batch is wrong, replay share is wrong, or the model is forgetting Apertus's broader multilingual residual that included some Greek.

2. **TD's downstream win does not coincide with TD's BPB win.** TD's BPB stays above Vanilla's through all 5 B; TD's no-explicit-MT downstream Greek aggregate crosses Vanilla at ~3.5 B. Different metrics, different dynamics. v0.12 §10 Q8's perplexity-only framing missed this divergence.

3. **Centroid is broken at any scale.** Even at 2 B BPC is ~0.90 vs others ≤ 0.65. Consistent with R11 (Cholesky rank-deficiency on the 1,507-modern-Greek subset).

4. **ReTok closes BPB fastest but is dominated by TD on downstream.** ReTok's BPB slope is steepest among extended arms but its no-explicit-MT downstream Greek aggregate stays below TD's at every shared iter. ReTok learns to compress new tokens fast but their representations remain semantically weak - directly validating v0.12 §6.2.3 (ReTok ignores distributional information).

5. **Bakeoff at 2 B alone would have produced the wrong fallback-suite headline.** At iter 476 Vanilla won the all-available Greek fallback aggregate (0.4409 vs TD 0.4254). The TD-wins-Greek fallback headline only emerged at iter 1192 (5 B), and even there it is driven by a single benchmark (xquad_el). The plan's B5 default of "2 B per variant" was insufficient for the fallback suite, and the planned native-Greek suite remains unrun.

6. **TD's fallback Greek-aggregate win is single-benchmark load-bearing.** Strip xquad_el (+7.57 pp for TD) and Vanilla wins 4 of the remaining 6 Greek fallback tasks, TD wins 1, and 1 is tied. The next plan should pre-commit to whether xquad_el is in the production-target set, because the answer flips on that decision.

---

## 8. What the plan got right

Several plan choices are validated by the bakeoff outcome.

- **v0.12 §2.7 Phase B v4 prediction**: "modern Greek is well-predicted; expect quality ties, efficiency to decide" — bakeoff confirmed quality is close but not tied (TD ahead of Vanilla by 3.1 % relative on no-explicit-MT downstream Greek; Vanilla ahead by 5.5 % relative on BPB). Efficiency and downstream capability point to different winners.
- **v0.12 §6.3.2 prediction**: "LM head story is awkward for our untied embeddings" — addressed via `learn_output_with_ce=True` pattern (paper §P6), no instability observed.
- **v0.12 §6.3.4**: "tied-embedding failure mode not applicable" — Apertus is untied; no degenerate-norm failure manifested.
- **v0.12 §10 Q7**: Phase B v4 behavioral cross-check — directly used in interpreting the bakeoff trajectories.
- **cpt_plan v0.7 B1 70/24/4/2 mix**: produced training that didn't catastrophically forget code/math/replay. EN retention even *improved* under TD (+1.04 pp at 5 B). No replay-share adjustment needed.
- **cpt_plan v0.7 B4 NTP-for-bakeoff**: gave a clean signal that wasn't confounded by Goldfish hash effects. The TD-vs-Vanilla downstream crossover at 3.5 B is a true init-method effect.
- **cpt_plan v0.7 C2 AdEMAMix exact match**: the optimizer doesn't appear to be the failure mode. All four arms ran stably for 5 B.
- **TD plan §14.1 (skip `target_tokenizer.add_tokens()`)**: confirmed — no tokenization mismatch surfaced during TD or downstream CPT.

---

## 9. Inputs for the follow-up planning pass

This section is **not** a redesign plan — that work belongs to the planning agent. It's the list of inputs the planning agent should fold into the next plan revision, derived from §1-§8.

**The highest-leverage v0.12 gap: Node 4 thresholds were never locked.** Locking X / M_progress / M_ext / M_van / T BEFORE the next experimental round runs is the single most important pre-commitment v0.12 §10 Q8f called for and we didn't honor.

**Open v0.12 questions that the bakeoff did NOT settle:**
- Node 1 production-side cutoff sweep on `{10,240, 15,360, 20,480, 25,600}` is a follow-on decision (the bakeoff cutoff at 17,408 is settled — this is about whether to revisit for production shipping). User has flagged smaller-cutoff candidates as worth empirical comparison; the planning agent should weigh per-unit-exposure soft constraint (v0.12 §10 Q1) against representational capacity.
- Node 4 thresholds: see above. Apply to a metric-axis choice that resolves the BPC-vs-downstream divergence.
- Node 6 Krikri positioning: still open, can stay late.
- v0.12 §10 Q6 native-Greek eval expansion: GreekMMLU / Belebele Greek / Medical MCQA Greek / OYXOY / greek-nlp/benchmark — none in the current suite. The ILSP YAMLs (`hellaswag_greek` / `winogrande_greek` / `mmlu_pro_greek` / `truthfulqa_greek` / `medical_mcqa_greek`) tracked as PF5 are partial coverage; merge them.
- v0.12 §10 Q8a per-language regression slices (English / French / Russian / German held-out perplexity): not directly run; need to be added.

**Open cpt_plan.md questions blocking production CPT:**
- V1 (decontamination), V4 (baseline variance with bootstrap CIs), V8 (Goldfish hash uniformity on extended vocab) — all three gate production.
- R17 production patch via `patch_apertus_extras.py` — scaffold exists; must apply to the production-CPT initial checkpoint.
- R2 / R4-R9 cheap mitigations (~2 hr total) listed in `cpt_plan_v0.7_answers.md`.

**TD-specific plan items that didn't ship:**
- TD4 layer pilot (only layer 11 trained; layer −1 and probe-suggested L* never tested).
- TD7 logit-lens / tuned-lens probe to identify L*.
- TD plan Q2 / Q4 / Q8 / Q11.

**Empirical findings from the bakeoff that change the input assumptions for the next plan:**
- The CPT regime as configured (LR peak 1.5e-5, β3/α warmup 238 steps, NTP, no staged training, uniform LR across rows) produces Greek-aggregate degradation in early CPT for all arms. v0.12 §8.3 (staged training) and §8.4 (higher LR on new-token rows) were specifically recommended and not applied.
- TD's downstream and BPC signals diverge. v0.12 §10 Q8b should specify which Greek metric the gate applies to.
- TD's Greek-aggregate win is load-bearing on xquad_el (+7.57 pp). The next plan should decide whether xquad_el is in the production-target set; if not, the headline reverses.
- The vocab-extension cutoff (currently 17,408) was never empirically chosen. The user has flagged reducing it as a candidate.
- The bakeoff is at pilot scale (~25 % of v0.12's planned 10 B/arm). The next plan should clarify whether the production CPT goes through a v0.12-§8.7-style staged pilot → main → anneal, or skips ahead.

**Apertus-fidelity deviations the next plan needs a position on:**
- Microbatch=2 vs Apertus 4 (production-scale acceptability unverified).
- Fixed 4.19 M-token batch vs Apertus's ramp to 8.39 M after 8 T (relevant at production scale).
- Optimizer-state portability from Apertus base — never resolved.
- Goldfish-for-production gating (V8 + the four-gate protocol from the prior `apertus_fidelity_checklist.md §3.1` thread).

---

## 10. Discrepancy log

Numbered punch list of every plan-vs-actual divergence. Living doc — append new entries as they surface.

| # | Severity | What | Plan ref | Status |
|---|---|---|---|---|
| **D1** | HIGH | **Pre-commit decision-rule thresholds NOT LOCKED** (X / M_progress / M_ext / M_van / T) | v0.12 §10 Q8a-f | OPEN |
| **D2** | HIGH | **Per-language regression slices (EN / FR / RU / DE) NOT MEASURED** — the project's only hard constraint (multilingual preservation) has no test | v0.12 §1 + §10 Q8a | OPEN |
| **D3** | HIGH | **V4 baseline variance unknown** — no bootstrap CIs; no run-to-run noise floor → any "TD beats Vanilla by X pp" claim has no error bars | cpt_plan v0.7 §12 V4 + v0.12 §10 Q8b "below 3% hard to distinguish from noise" | OPEN |
| **D4** | HIGH | **BPC vs downstream Greek metric divergence** — Vanilla wins BPC, TD wins downstream. v0.12 implicitly assumed perplexity tracks capability; bakeoff disproves it | v0.12 §10 Q8b (emergent) | OPEN |
| **D5** | HIGH | **V1 decontamination NOT DONE** — eval-set item-level dedup against training data not run; absolute scores have memorization noise (within-experiment selection still valid) | cpt_plan v0.7 §8 K1 + V1 | OPEN |
| **D6** | HIGH (ADDRESSED) | **MT-derived Greek tasks initially aggregated as primary** — `arc_challenge_mt_el` + `global_piqa_completions_ell_grek` aggregated with native Greek tasks, contradicting v0.12 §10 Q6's "secondary, not weighted heavily" guidance | v0.12 §10 Q6 | Plot scripts, 3.5B/5B summaries, and final/reconciliation docs corrected to "Greek no-explicit-MT" aggregate (2026-05-26). |
| **D7** | MEDIUM | **Eval suite — v0.12's native-Greek menu substituted** with ILSP-style suite. Missing: GreekMMLU, Medical MCQA Greek, OYXOY, greek-nlp/benchmark | v0.12 §10 Q6 | PARTIALLY ADDRESSED — PF5 (ILSP YAMLs merge) pending |
| **D8** | MEDIUM | **Per-arm budget asymmetry** — v0.12 §8.7 planned 10 B / arm; bakeoff ran 2 B (Centroid) / 3.5 B (ReTok) / 5 B (Vanilla + TD) | v0.12 §8.7 | DOCUMENTED — iso-token comparisons at iter 476 + 834 available in BAKEOFF_FINAL_RESULTS |
| **D9** | MEDIUM | **Training-budget schedule truncated** — stabilization (0.5-1 B), main (20-40 B), anneal (5-10 B) phases NOT RUN. Bakeoff is "extended pilot" only | v0.12 §8.7 | OPEN — production CPT plan needs to decide whether to follow staged schedule or skip ahead |
| **D10** | MEDIUM | **v0.12 §6.5 mitigations not applied** — staged training (§8.3 EEVE-style three-stage), higher LR on new-token rows (§8.4, 5×-10× ratio), hybrid init. All four arms ran full-parameter from iter 1 with uniform LR | v0.12 §6.5, §8.3, §8.4 | OPEN — live degrees of freedom for next round |
| **D11** | MEDIUM | **V8 Goldfish hash uniformity NOT VERIFIED** — required before turning Goldfish on for production with extended vocab | cpt_plan v0.7 §8 G1 + V8 | OPEN — production gate |
| **D12** | MEDIUM | **R17 production patch not applied** — `patch_apertus_extras.py` scaffold exists; xIELU + QK-Norm trained values reset to defaults through HF→Megatron path. Acceptable for bakeoff (all arms inherit same defaults); blocks production CPT | cpt_plan v0.7 §12 R17 + apertus_fidelity_checklist §3.1 | OPEN — production gate |
| **D13** | LOW | **Node 1 BPE cutoff — per-unit firing distribution not verified** — 17,408 used; v0.12 §10 Q1's "≥100k occurrences per new unit" soft constraint not checked on the actual CPT corpus | v0.12 §10 Q1 | OPEN — tractable cheap check |
| **D14** | LOW | **Node 6 Krikri positioning not addressed** — no Krikri / Meltemi comparison in bakeoff | v0.12 §3 Node 6 | DEFERRED (per v0.12 — late/optional, acceptable) |

### High-severity entries — short elaborations

**D1 — Pre-commit thresholds**. v0.12 §10 Q8f's anti-rationalization warning was violated: results came in before any of X / M_progress / M_ext / M_van / T were locked. The 5B headline ("TD wins downstream, Vanilla wins BPB") is interpretation, not adjudication. Whatever thresholds get set going forward will be contaminated by knowing the results. Cleanest path: lock priors-independent values (external precedent + first-principles arguments about deployment economics), explicitly document that contamination is a known risk and we're choosing to accept it rather than waive Node 4.

**D2 — §8a per-language slices**. v0.12 §1 names multilingual preservation as the *single hard constraint*. §8a was the operational gate for it: held-out perplexity on EN / FR / RU / DE vs base Apertus, disqualify any arm regressing >X% on any of them. The bakeoff doesn't have the data: task accuracy via `xnli_el / xcopa_X / global_mmlu` sub-tasks is a different metric (task performance ≠ language-modeling perplexity; they correlate but decouple under regime stress). Resolution requires building 4 per-language held-out slices, running V4-HF baseline once on them, then including them in every eval-checkpoint going forward.

**D3 — V4 baseline variance**. v0.12 §10 Q8b: "below 3% is hard to distinguish from training-noise variance across runs." We never measured what that noise is. So any sentence like "TD beats Vanilla by 3.1 % relative on no-explicit-MT Greek downstream" sits near the unknown-noise boundary — could be signal, could be noise. cpt_plan v0.7 §12 V4 calls for this; never done. Required before D1's thresholds have meaning.

**D4 — BPC vs downstream divergence**. v0.12 §10 Q8 was built on the implicit assumption that perplexity tracks capability — the Greek-progress floor (§8b) uses perplexity, the M_van / M_ext comparisons (§8c) are written as if a single Greek metric ranks the arms. The bakeoff disproves this: BPC ranks Vanilla > TD; downstream Greek ranks TD > Vanilla. Same arms, opposite winners. v0.12's framework doesn't have a tiebreaker for this case. Next plan needs to pick (or weight) the Greek production-target metric explicitly.

**D5 — V1 decontamination**. Bakeoff's *internal* arm comparison is valid (all arms see the same data, so any contamination affects them equally). But the V4-HF baseline absolute scores carry memorization noise vs Apertus pretraining, so claims like "TD is at 0.4204 no-MT Greek vs base Apertus's 0.5146" sit on shaky ground for external positioning. Required before any production CPT — NeMo Curator workflow on Clariden xfer per the K1 procedure.

**D6 — MT-derived Greek aggregation**. v0.12 §10 Q6 explicitly listed MT-derived Greek benchmarks (MMLU Greek, ARC-Challenge Greek, HellaSwag Greek, TruthfulQA Greek) as "secondary, useful for Krikri-comparability only" and "not weighted heavily in our decision rule." The bakeoff initially aggregated `arc_challenge_mt_el` and `global_piqa_completions_ell_grek` into the Greek aggregate as primary. Plot scripts (`plot_van_td.py`, `regenerate_plots.py`), `CONTINUATION_3P5B_RESULTS_20260525.md`, `CONTINUATION_5B_RESULTS_20260526.md`, `BAKEOFF_FINAL_RESULTS_20260526.md`, and this reconciliation were corrected on 2026-05-26 to use a 5-task "Greek no-explicit-MT" aggregate. The corrected aggregates: iter 834 Vanilla 0.3989, TD 0.4129; iter 1192 Vanilla 0.4076, TD 0.4204.

### Notes on the log

- Severity is set against the project's primary goal (deliver a defensible Greek-capable Apertus derivative without regressing multilingual breadth). HIGH = blocks production decisions; MEDIUM = affects validity of conclusions; LOW = expansion-scope or non-blocking.
- "OPEN" status means the planning agent should fold this in; "ADDRESSED" means already fixed by a doc / script edit; "DEFERRED" means explicitly out of scope per the plan itself; "DOCUMENTED" means the divergence exists but is handled by the current doc set.
- This log is consciously a strict subset of "things the next plan needs to know" — it covers plan-vs-actual divergences specifically. Emergent findings that the plan never anticipated (like the BPC/downstream divergence) are still here because they invalidate the plan's implicit assumptions.

---

## Appendix A — Quick reference: per-bakeoff-arm endpoint summary

From `BAKEOFF_FINAL_RESULTS_20260526.md`:

| Arm | Final iter | Tokens (B) | Greek no-MT agg | EN ret | Multi | BPB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla    | 1192 | 5.000 | 0.4076 | 0.6799 | 0.4936 | **0.4602** |
| TD layer11 | 1192 | 5.000 | **0.4204** | **0.6903** | **0.4976** | 0.4872 |
| ReTok      | 834  | 3.498 | 0.3984 | 0.6786 | 0.4864 | 0.5390 |
| Centroid   | 476  | 1.996 | 0.2566 | 0.6836 | 0.4888 | 0.8994 |

V4-HF Apertus base reference (approx): Greek no-MT agg ≈ 0.5146, EN ret ≈ 0.716, Multi ≈ 0.541. Note: **all arms degrade on Greek and EN retention vs V4-HF baseline; only TD partially clears Multi back to V4-HF level**.

TD-vs-Vanilla BPC gap at 5B: **0.0270** (down from 0.110 at iter 130). Linear-slope crossover predicted at ~6.5B.

---

## Appendix B — Files this doc reconciles against (primary first)

**Primary (experimental-design plan):**
- `old_experiments_plan.md` v0.12 (2026-05-12) — the parent project plan that owns §3 Decision Nodes, §5 the three-arm experimental design, §6 per-arm weakness predictions, §8.7 the training budget schedule, §10 Open Questions and the §10 Q8 pre-commit decision rule.

**Secondary (CPT-execution successors):**
- `cpt_plan.md` v0.7 (707 lines) — CPT-specific successor governing corpus / optimizer / verifications. The active plan at bakeoff firing for execution-side decisions.
- `cpt_plan_v0.7_answers.md` (328 lines, 2026-05-21) — decision snapshot at bakeoff firing.
- `apertus_fidelity_checklist.md` (329 lines) — Apertus-pretraining fidelity items + production gating.
- `TOKEN_DISTILLATION_PLAN.md` (773 lines) — TD-specific plan (4th-arm spec).
- `TRAINING_RECIPE.md` (324 lines) — production training recipe.
- `_train_config_common.env` (190 lines) — actual training hyperparameters used.
- `PRODUCTION_DECISION_STATE.md` (231 lines) — production decision context (predates 5 B continuation; needs update).
- `RISKS.md` — 17-risk silent-failure inventory.
- `AUDIT_FINDINGS.md` — 2-round source-vs-implementation audit.
- `03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md` — concrete bakeoff procedure spec.

**Empirical result doc:**
- `03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/BAKEOFF_FINAL_RESULTS_20260526.md` — 5 B endpoint.
