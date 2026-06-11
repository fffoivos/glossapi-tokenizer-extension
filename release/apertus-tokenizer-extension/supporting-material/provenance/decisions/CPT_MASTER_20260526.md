# Apertus 8B Greek CPT — master synthesis (2026-05-26)

Synthesis of 10 source docs into one reference. Originals moved to [`_archive/synthesis_sources_20260526/`](_archive/synthesis_sources_20260526/). When this doc gets superseded, write a new `CPT_MASTER_<date>.md` and archive this one.

Sources synthesized: `old_experiments_plan.md` (v0.12), `cpt_plan.md` (v0.7), `cpt_plan_v0.7_answers.md`, `cpt_plan_v0.7_status.md`, `apertus_fidelity_checklist.md`, `PLAN_VS_RESULTS_RECONCILIATION_20260526.md`, `PRODUCTION_DECISION_STATE.md`, `ARTIFACTS_AND_HYDRATION.md`, `CLARIDEN_INVENTORY_20260524.md`, `collegues_Apertus_plan.md`.

---

## TL;DR

Continue-pretrain Apertus-8B-2509 on Greek without damaging its multilingual character. Tokenizer extension is complete (17,408 modern + 5,120 polytonic). A 4-arm init bakeoff (Vanilla / ReTok / Centroid / TD-layer11) trained on a 70 % Greek / 24 % replay / 4 % code / 2 % math mix. **Native-Greek suite update:** the later vetted native-Greek eval pass favors Vanilla among continued arms on the native MCQ headline, while Apertus-Base remains above all continued checkpoints. TD layer11 remains ahead on the older fallback downstream/retention bundle at 5 B (Greek-no-MT fallback, EN retention, Multilingual), but it is no longer correct to call TD the Greek-native winner. **Vanilla retains tokenizer-fair BPB** (gap 0.027 at 5 B, narrowing). Centroid is broken at any scale. ReTok is dominated by TD on older fallback downstream and trails TD/Vanilla on the native suite.

**Big caveat:** the v0.12 §10 Q8 pre-commit decision rule was never instantiated — none of the X / M_progress / M_ext / M_van / T thresholds were locked before results came in. So the bakeoff produced **data, not an adjudicated winner**. The current `PRODUCTION_DECISION_STATE` snapshot picks Vanilla as the safe 2 B-stage default, but that pick is partially superseded by the 5 B continuation and is not rule-bound. Production CPT is gated on three open Apertus-fidelity items (V1 decontamination, V8 Goldfish hash uniformity, R17 xIELU/QK-Norm patch) plus measurement infrastructure that was planned but never built (V4 baseline variance, per-language perplexity slices for §8a gate).

---

## 1. Mission and scope

### 1.1 The project goal

Continue-pretrain Apertus-8B-2509 on a curated Greek corpus to **deepen modern + polytonic Greek capabilities** while **preserving multilingual, code, and reasoning performance**. Tokenizer extension is complete and frozen at the 02 stage:

- +17,408 modern Greek tokens → vocab 148,480 (the "modern-only" ship bundle used in the bakeoff).
- +5,120 polytonic / ancient Greek tokens stacked on top → vocab 153,600 (the "composite" ship bundle, parked for a future polytonic specialization run).
- First 131,072 IDs are byte-identical to the Mistral-Nemo `tekken` v3 base tokenizer that Apertus inherited.

The 03 stage runs from extension-frozen → trained Greek-capable Apertus derivative.

### 1.2 The hard constraint

From `old_experiments_plan.md` v0.12 §1 and `cpt_plan.md` v0.7 §1: **preserve Apertus's multilingual character.** Apertus is a fully-open, fully-compliant, multilingual model with native support for ~1,800 languages. The project will not ship a derivative that has bought Greek improvements at the cost of meaningfully degrading other languages.

Operationalized as v0.12 §10 Q8a's gate: any arm regressing English / French / Russian / German held-out perplexity by more than X % vs base Apertus is disqualified regardless of Greek progress.

### 1.3 The colleague's origin framing

`collegues_Apertus_plan.md` is the project's original Greek-language framing by p-skarvelis. It argues Apertus's FineWeb2-HQ-based base provides high information density in English but lacks Greek-specific localization (legal, medical, literary, technical). It proposes ranking GlossAPI sources by perplexity (Apertus's existing knowledge gap), quality (correctness / coherence), and novelty (distance from FineWeb2-HQ content) before selecting CPT corpus subsets. The v0.6 / v0.7 cpt_plan iteration absorbed the framing but rejected the HF-Trainer scaffold p-skarvelis used in favor of swiss-ai's Megatron-LM stack — per the 2026-05-20 user directive that closest-to-Apertus-original-process is canonical.

### 1.4 Sub-subprojects (chronological)

| Dir | Role | State |
|---|---|---|
| `03_1_greek_embedding_diagnostic/` | Pre-extension diagnostic of Greek in Apertus E/U matrices. Produced norm targets 5.05 / 3.80 used by ReTok and Centroid arms. | Complete (v4, 2026-05-15) |
| `03_2_apertus_c3_dedup_audit/` | Document-level overlap between Apertus pretraining and C3 corpus. Produced the `apertus_overlap_drop_docs.parquet` overlay gating the CPT mix. | Complete (2026-05-19) |
| `03_3_cscs_experiments_kickoff/` | Planning bridge: v0.12 → v0.7 plan + CSCS auth + first Clariden submission readiness. No SLURM jobs ran here. | Complete |
| `03_4_implementation_experiments/` | The hands-on runs: 4-arm bakeoff, eval, trajectory analysis, production-CPT launcher (dry-run validated). | **Active** |

---

## 2. The two governing plans

The bakeoff was governed by **two plans, in two layers.** The split matters because the reconciliation in §5 below maps results back to v0.12 — which is the experimental-design layer — while v0.7 owns the execution-side recipe and verification gates.

### 2.1 v0.12 — `old_experiments_plan.md` (2026-05-12, primary)

**Owns the experimental framework**: what to test, how to decide, what to constrain. Structured around 6 Decision Nodes.

#### Decision Nodes

| Node | Decision | Status as of 5 B endpoint |
|---|---|---|
| **1 BPE cutoff** | Pick a cutoff from `{8K, 16K, 20K, 24K}` for the comparison | **Locked at 17,408** for the bakeoff (C3 wave-2 broad emit count). Production-side cutoff sweep on `{10,240, 15,360, 20,480, 25,600}` is a separate downstream decision still open. |
| **2 Training mix** | Greek source weights + non-Greek replay | **Locked at 70 / 24 / 4 / 2** via cpt_plan v0.7 §B1. Applied uniformly across arms. |
| **3 Evaluation suite** | Pick slices + native benchmarks that will actually run | **Partially locked.** 16 lm-eval tasks selected. v0.12 §10 Q6 native-Greek menu (GreekMMLU / Medical MCQA Greek / OYXOY / greek-nlp/benchmark) NOT included. Per-language regression slices for §8a gate NOT built. |
| **4 Pre-commit thresholds** | Lock X / M_progress / M_ext / M_van / T **before any arm completes CPT** (v0.12 §10 Q8f) | **NOT LOCKED.** Largest decision-side gap. |
| **5 Three-arm experimental design** | Run Vanilla → ReTok → Distillation ladder; same corpus / budget / schedule | **Modified → four arms.** Distillation initially deferred; Centroid filled third slot (Hewitt 2021 multivariate Gaussian); TD layer11 added later as 4th arm via `TOKEN_DISTILLATION_PLAN.md`. Per-arm budget collapsed from 10 B → 2 B, extended ad-hoc to 5 B for Vanilla + TD. |
| **6 Krikri positioning** | Optional late decision on "as good as Krikri" framing | **Not addressed.** No Krikri / Meltemi comparison in bakeoff. |

#### Hard constraints (§4)

1. **Greek-authentic learning, no translation mediation.** No WECHSEL / Trans-tokenization / OFA; no MT corpora; no bilingual-anchor methods. ✅ preserved.
2. **Smooth transition, preserve learned structures.** Norm-matched init using Phase A data (5.05 E / 3.80 U). ⚠️ partially preserved — Centroid's failure mode (BPC stuck at 0.90) suggests covariance preservation wasn't sufficient at the modern-Greek subset's rank.
3. **Compatible with continued BPE training.** Additive merge-rule extension; deterministic decomposition. ✅ preserved.

#### The §10 Q8 pre-commit decision rule

Five thresholds the bakeoff was supposed to be adjudicated against:

| Var | What it gates | Suggested value |
|---|---|---|
| **X** | Disqualify any arm regressing EN/FR/RU/DE perplexity by more than X % vs base Apertus | 5 % |
| **M_progress** | Each arm must improve Greek by ≥ M_progress % over base Apertus (both web and academic slices) to qualify | 3-5 % |
| **M_ext** | Extension arm ships over Vanilla only if it beats Vanilla by ≥ M_ext % on both Greek metrics | 1-2 % |
| **M_van** | Vanilla ships over extensions only if it beats them by ≥ M_van % on at least one Greek metric | 3-5 % |
| **T** | Distillation ships over ReTok only if it beats ReTok by ≥ T % on at least one Greek metric | 2-3 % |

**None were locked.** v0.12 §10 Q8f explicitly warned: *"Doing this with results visible risks post-hoc rationalization; the rule is only as honest as the pre-commitment."* The current 5 B headline ("TD wins downstream / Vanilla wins BPB") is interpretation, not adjudication.

#### Budget schedule (§8.7)

Planned phases:
1. Stabilization 0.5-1 B
2. **Pilot 10 B per arm**
3. Main 20-40 B
4. Optional anneal 5-10 B

Sequence length 4,096 throughout CPT.

Actual: pilot truncated to 2 B per arm (cpt_plan v0.7 §B5), then ad-hoc extended to 5 B for Vanilla + TD, 3.5 B for ReTok, 2 B for Centroid. Main + anneal NOT RUN. **Bakeoff is at ~25 % of v0.12's planned per-arm pilot scale.**

### 2.2 cpt_plan.md v0.7 (2026-05-20, CPT-execution successor)

**Owns the CPT recipe**: how to train cleanly. 14 sections; key contents:

| § | Topic | Key content |
|---|---|---|
| 1 | Objective | Same goal as v0.12 §1, with concrete vocab-extension numbers (17,408 modern + 5,120 polytonic) |
| 2 | Settled shape | Shuffled-mixture bulk + anneal tail; 70 % Greek / 30 % non-Greek from token 0; WSD LR; no replay of Apertus pretraining Greek |
| 3 | Curriculum | LR schedule (1.5e-5 peak, 1.5e-6 final, WSD with 1-sqrt cooldown, 40 M-token warmup); §3.1 polytonic exposure metrics for V5 |
| 4 | Replay design | 24 languages in 3 tiers (8 T1 + 11 T2 + 5 T3); FineWeb2-HQ for T1, FineWeb-2 for T2/T3; FineWeb-Edu Score-3 for English; StarCoderData (code); FineMath-3plus (math) |
| 5 | Init experiments | The 3-arm bakeoff spec (Vanilla / ReTok / Centroid). TD bracketed in v0.7 §13; returned via `TOKEN_DISTILLATION_PLAN.md` |
| 6 | Evaluation | Cadence, benchmark suite, §5.6 weighted-score selection criterion (still parameter-set placeholders pending V4 variance + Q A6 weights) |
| 7 | Tooling | Megatron-LM-Swiss-AI pinned at `c92402e3`; swiss-ai pretrain-code at `531cc8be`; swiss-ai lm-evaluation-harness |
| 8 | Apertus-specific adaptation | Gates G1 / K1 / H1 / I1 / I2 / J1 / B FOCUS / E1 |
| 9 | Production run shape | 15-20 B production CPT on bakeoff winner; specific shape pending Q A2 |
| 10 | Q A/B decisions | Pending Fivos input or design defaults (see §2.3 below) |
| 11 | Q C/D lookups | Apertus tech-report values + engineering pins (see §3.1 below) |
| 12 | V1-V16 verifications | Status legend per check |

### 2.3 Q A / Q B / Q C / Q D status (from cpt_plan_v0.7_answers.md)

#### Q A — Fivos decisions (all pending)

| # | Question | Status |
|---|---|---|
| A1 | Capability targets | DEFERRED |
| A2 | Total CPT token budget | PENDING (working: 15-20 B) |
| A3 | Compute timeline | PENDING |
| A4 | Stakeholders / consumers | PENDING (gates V1 decontamination scope) |
| A5 | Colleague sign-off | PENDING |
| A6 | Specific downstream tasks | PENDING (affects §5.6 weights) |
| A7 | Team structure | PENDING (soft) |

#### Q B — Design defaults (all locked in code)

| # | Question | Locked default |
|---|---|---|
| B1 | Outer mix split | 70 % Greek / 24 % replay / 4 % code / 2 % math |
| B2 | Code share | 4 % (StarCoderData) |
| B3 | Anneal composition | (d) balanced — 85 % Greek / 12 % replay / 3 % code (production-only) |
| B4 | Loss objective | **NTP for bakeoff, Goldfish for production** (gated on V8) |
| B5 | Init budget | 2 B per arm |
| B6 | §8 prioritization | G1 ready (production-only); K1 pending Q A4; H1 / I1 resolved; I2 / B FOCUS / E1 not in scope |

#### Q C — Apertus lookups (all resolved)

| # | Param | Apertus pretrain value | Bakeoff value |
|---|---|---|---|
| C1 | Peak LR | 1.1e-4 | **1.5e-5** (intentional CPT divergence) |
| C2 | β1 / β2 / β3 / α | 0.9 / 0.999 / 0.9999 / 8.0 | Exact match |
| C2 | weight_decay | 0.1 | Exact match |
| C2 | grad clip | 0.1 | Exact match |
| C2 | α / β3 warmup | 100,000 steps (~2.8 % of pretrain) | **238 steps** (50 % of 477-step bakeoff — intentional deviation) |
| C2 | init_method_std | 0.008944 | Exact match |
| C4 | Goldfish config | k = h = 50, hash table 1,000,003, seed 2971215073, prod-mod hash | (loaded but NOT fired in bakeoff) |
| C5 | Tokenizer base | Mistral-Nemo tekken v3, 131,072, no normalizer | First 131,072 IDs byte-identical |

#### Q D — Engineering lookups

| # | Status |
|---|---|
| D1 | RESOLVED — Megatron-LM-Swiss-AI `c92402e3` |
| D2 | PENDING — FineWeb-2 Tier 3 audit (cheap, non-blocking) |
| D3 | PENDING — Apertus intermediate checkpoints |

---

## 3. Apertus fidelity

From `apertus_fidelity_checklist.md` cross-checked against `_train_config_common.env`.

### 3.1 Confirmed exact matches with Apertus pretraining (21 items)

Optimizer (AdEMAMix β1 / β2 / β3 / α / weight decay 0.1 / grad clip 0.1 / init_method_std). Architecture (xIELU αp=αn=0.8 + β=0.5 / RMSNorm Pre-Norm / QK-Norm per-head before RoPE via apex / RoPE θ=500,000 with llama3 scaling 8.0). Training (sequence length 4,096 / TP=2 / bf16 + fp32 master grads / cross-doc attention masking ON / EoD loss masking ON / `--make-vocab-size-divisible-by 128`). Embedding (untied `tie_word_embeddings=False`; both E and U resize independently). FP8 explicitly rolled back per paper Appendix D.

### 3.2 Three intentional deviations

| # | Param | Apertus | Bakeoff | Reason |
|---|---|---|---|---|
| 1 | LR peak | 1.1e-4 | **1.5e-5** (≈ 14 %) | CPT operates near-converged; Llama-3 CPT and Aya use 10-20 % of pretrain peak |
| 2 | α / β3 warmup | 100,000 steps | **238 steps** (50 % of bakeoff horizon) | Apertus's 2.8 %-of-run policy collapses to ~14 steps at 477-step bakeoff scale; AdEMAMix paper cold-restart guidance favors a meaningful fraction of the new run |
| 3 | Loss objective | Goldfish (k=h=50) | **NTP** | Bakeoff isolates init effect; loss held constant across arms. Production restores Goldfish gated on V8 |

### 3.3 Three sub-optimal choices not flagged in plan

| # | Param | Apertus | Bakeoff |
|---|---|---|---|
| 4 | Microbatch | 4 | **2** (GH200 memory; global-batch tokens preserved at 4.19 M) |
| 5 | Global batch | ramped 4.19 M → 8.39 M after 8 T | **fixed at 4.19 M** (bakeoff too short for the ramp window) |
| 6 | Apertus optimizer state | available? | **NOT loaded** — bakeoff inits AdEMAMix state fresh; first 1-2 % of CPT acts as optimizer-state warmup |

### 3.4 Production gates (open)

These block production CPT submission, not the bakeoff:

- **R17 — xIELU + QK-Norm reset through HF→Megatron conversion.** `patch_apertus_extras.py` scaffold exists; must be applied to the production initial checkpoint. Acceptable for bakeoff (all arms inherit same defaults).
- **V8 — Goldfish hash uniformity across new tokens.** Q C4 unblocked the verification (k=h=50, hash 1,000,003, seed 2971215073, prod-mod). Procedure: tokenize sample, simulate hash, count masked positions per new ID, verify uniformity within ±2σ.
- **V1 — Eval-set decontamination.** Item-level dedup of clean-measurement benchmarks against training data. NeMo Curator workflow on Clariden xfer. ~1-3 days.
- **V4 — Run-to-run variance baseline on unmodified Apertus.** Full eval suite with bootstrap CIs. Required for §5.6 threshold-setting and for v0.12 §10 Q8 thresholds to have a noise floor.

---

## 4. Bakeoff results

### 4.1 Final endpoint per arm

| Arm | Final iter | Tokens (B) | Greek no-MT agg | EN ret | Multi | BPB ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla    | 1192 | 5.000 | 0.4076 | 0.6799 | 0.4936 | **0.4602** |
| TD layer11 | 1192 | 5.000 | **0.4204** | **0.6903** | **0.4976** | 0.4872 |
| ReTok      | 834  | 3.498 | 0.3984 | 0.6786 | 0.4864 | 0.5390 |
| Centroid   | 476  | 1.996 | 0.2566 | 0.6836 | 0.4888 | 0.8994 |

"Greek no-MT agg" = 5-task aggregate (Greek MMLU / INCLUDE-44 / Belebele / XNLI-el / XQuAD-el) excluding the two MT-derived diagnostics (arc_challenge_mt_el, global_piqa_completions_ell_grek) per v0.12 §10 Q6 — MT-derived benchmarks are "secondary, not weighted heavily in our decision rule."

V4-HF Apertus base reference: Greek agg ≈ 0.525, EN ret ≈ 0.716, Multi ≈ 0.541. **All arms degrade on Greek aggregate and EN retention vs base Apertus.**

### 4.2 Iso-token comparison at iter 476 (2 B, all 4 arms)

| Arm | Greek no-MT agg | EN ret | Multi | BPB ↓ |
|---|---:|---:|---:|---:|
| Vanilla    | **0.4131** | 0.6818 | **0.4901** | **0.4906** |
| TD layer11 | 0.4048 | 0.6828 | 0.4899 | 0.5311 |
| ReTok      | 0.3906 | 0.6750 | 0.4873 | 0.5739 |
| Centroid   | 0.2566 | **0.6836** | 0.4888 | 0.8994 |

At 2 B (the original bakeoff budget), **Vanilla was the safe default** — wins 3 of 4 metrics. This is the conclusion that landed in `PRODUCTION_DECISION_STATE.md`.

### 4.3 The 3.5 B → 5 B continuation flipped the downstream headline

| Aggregate | Vanilla @ 5B | TD @ 5B | Δ vs iter 834 |
|---|---:|---:|---|
| Greek no-MT | 0.4076 | **0.4204** | TD lead widened from +1.40 pp (iter 834) to +1.28 pp (iter 1192) |
| EN retention | 0.6799 | **0.6903** | TD lead widened from +0.83 pp to +1.04 pp |
| Multilingual | 0.4936 | **0.4976** | TD lead narrowed slightly but persists |
| BPB | **0.4602** | 0.4872 | Gap narrowed from 0.033 → **0.027** |

Linear-slope BPB crossover predicted at ~6.5 B tokens (possibly never under WSD cooldown attenuation).

### 4.4 BPC vs downstream — the metric divergence

Vanilla wins BPB (tokenizer-fair perplexity-equivalent). TD wins downstream MCQ + extractive QA. Same arms, opposite winners on different metric axes. v0.12 §10 Q8b's rule was written assuming perplexity tracks capability — the bakeoff disproves that at this CPT regime.

**TD's Greek win is load-bearing on xquad_el (+7.57 pp).** Strip xquad_el and the remaining 4 no-MT Greek tasks put Vanilla narrowly ahead. Whether xquad_el is in the production-target set determines whether the headline holds.

### 4.5 BPC trajectory (lower better)

| Iter | Tokens (B) | Vanilla | ReTok | Centroid | TD |
|---:|---:|---:|---:|---:|---:|
| 130 | 0.545 | 0.5432 | 0.7561 | 1.1318 | 0.6531 |
| 476 | 1.996 | **0.4906** | 0.5739 | **0.8994** | 0.5311 |
| 834 | 3.498 | 0.4724 | **0.5390** | — | 0.5054 |
| 1192 | 5.000 | **0.4602** | — | — | **0.4872** |

ReTok closes BPC fastest (~28 % improvement vs Vanilla's 15 %) but stays semantically weaker on downstream — validating v0.12 §6.2.3 (ReTok ignores distributional information).

---

## 5. Discrepancy log

From `PLAN_VS_RESULTS_RECONCILIATION_20260526.md` §10. The numbered punch list of every plan-vs-actual divergence.

### 5.1 HIGH severity (6 entries — block production)

| # | What | Plan ref | Status |
|---|---|---|---|
| D1 | **Pre-commit decision-rule thresholds NOT LOCKED** (X / M_progress / M_ext / M_van / T) | v0.12 §10 Q8a-f | OPEN |
| D2 | **Per-language regression slices (EN / FR / RU / DE) NOT MEASURED** — multilingual preservation hard constraint has no test | v0.12 §1 + §10 Q8a | OPEN |
| D3 | **V4 baseline variance unknown** — no bootstrap CIs; no run-to-run noise floor | cpt_plan v0.7 §12 V4 + v0.12 §10 Q8b | OPEN |
| D4 | **BPB vs downstream metric divergence** — Vanilla wins BPB, TD wins downstream. v0.12 didn't anticipate. | v0.12 §10 Q8b (emergent) | OPEN |
| D5 | **V1 decontamination NOT DONE** — eval-set item-level dedup against training data | cpt_plan v0.7 §8 K1 + V1 | OPEN |
| D6 | **MT-derived Greek tasks were initially aggregated as primary** | v0.12 §10 Q6 | Plot scripts + 3.5 B + 5 B docs corrected to "Greek no-MT" 5-task aggregate (2026-05-26) |

### 5.2 MEDIUM severity (6 entries — affect conclusions)

| # | What | Status |
|---|---|---|
| D7 | **v0.12 §10 Q6 native-Greek menu substituted** with ILSP-style suite | PARTIALLY ADDRESSED (PF5 pending) |
| D8 | **Per-arm budget asymmetry** — 2 B / 3.5 B / 5 B by arm | DOCUMENTED |
| D9 | **Training schedule truncated** — pilot only, no main / anneal phases | OPEN |
| D10 | **v0.12 §6.5 mitigations not applied** — staged training (§8.3), higher LR on new-token rows (§8.4), hybrid init | OPEN |
| D11 | **V8 Goldfish hash uniformity NOT VERIFIED** | OPEN — production gate |
| D12 | **R17 production patch not applied** | OPEN — production gate |

### 5.3 LOW severity (2 entries)

| # | What | Status |
|---|---|---|
| D13 | Node 1 BPE cutoff per-unit firing distribution not verified | OPEN — tractable |
| D14 | Node 6 Krikri positioning not addressed | DEFERRED per v0.12 |

### 5.4 v0.12 §10 Q8 applied retroactively (informational only)

Reading the rule against the 5 B endpoint with suggested starting-point thresholds:

- **§8a gate (preservation, X = 5 %)**: per-language regression slices weren't built; ReTok marginally fails on EN aggregate (-5.3 % vs V4-HF); Vanilla at -5.1 % is at the threshold. **Can't evaluate cleanly.**
- **§8b Greek progress floor (M_progress = 3-5 %)**: on BPB, multiple arms qualify (Vanilla improved 15 %, TD improved 25 %, ReTok 28 %, Centroid 20 %). **On downstream Greek aggregate, no arm qualifies — all degraded vs V4-HF by 15-44 %.**
- **§8c asymmetric decision rule (M_ext = 1-2 %, M_van = 3-5 %)**: TD beats Vanilla by 3.1 % on Greek no-MT aggregate, above M_ext. Vanilla beats TD by 5.5 % on BPB, above M_van. Different winners depending on the metric axis the rule reads.
- **§8c(4) "no arm qualifies"** if §8b fails on the downstream metric — implied conclusion is *"none of the four are deployment-ready, the CPT regime is sub-optimal"*. This matches the empirical Greek-aggregate degradation pattern.

The rule, even applied retroactively, doesn't pick a winner — it surfaces the ambiguity.

---

## 6. V1-V16 verification status

| # | Verification | Status | Notes |
|---|---|---|---|
| V1 | Decontamination scope (item-level dedup of clean-measurement benchmarks) | **NOT DONE** | Gates production. NeMo Curator workflow ~1-3 d on xfer. |
| V2 | Tokenizer extension forward pass | **CONFIRMED** | Empirically via training without NaN/inf at vocab 148,480. |
| V3 | Dataloader state preserved across resume | **CONFIRMED** | 3.5B + 5B continuations resumed cleanly. |
| V4 | Run-to-run variance baseline | **NOT DONE** | Gates §5.6 threshold-setting + v0.12 Q8 noise floor. |
| V5 | Polytonic-token concentration audit | **N/A** | Modern-only bakeoff. Required for polytonic Phase 2. |
| V6 | Accent-normalized dedup re-verification | **N/A** | Same as V5. |
| V7 | Replay-dataset acquisition | **CONFIRMED** | All pulled. |
| V8 | Goldfish hash uniformity on extended vocab | **READY, NOT DONE** | Production gate. Q C4 unblocked. |
| V9 | NFC normalization of training corpus | **CONFIRMED** | `normalize_nfc.sh` wrapper. |
| V10 | vLLM / SGLang compatibility | **DEFERRED** | Post-pilot. |
| V12 | Cross-document attention masking | **CONFIRMED** | `--reset-attention-mask --reset-position-ids`. |
| V13 | EoD loss masking | **CONFIRMED** | `--eod-mask-loss`. |
| V14 | BoD/EoD special-token preservation | **CONFIRMED** | All 1,000 added_tokens byte-identical. |
| V15 | xIELU per-layer scalars survive resize | **CONFIRMED EMPIRICALLY** | Mechanism audited LOW RISK + 5 B training stable. |
| V16 | Tokenizer byte-fallback for polytonic | **N/A** | Modern-only bakeoff. |

**Open and gating production: V1, V4, V8.**

---

## 7. Operational reference

### 7.1 What lives in git vs what lives on Clariden

**In git (this repo, the control plane):** corpus recipes + source-mix manifests + validation summaries + small JSON/CSV evidence files + Slurm launchers + dry-run submission plans + HF↔Megatron conversion scripts + R17/xIELU/QK-Norm verification reports + TD coverage/training/eval scripts + final eval digests + trajectory analysis + plots + handoff docs.

**NOT in git** (blocked by `.gitignore`): `.safetensors / .distcp / .bin / .idx / .pt / .pth / .ckpt / .gguf`; raw JSONL/parquet corpora; per-sample eval logs; full run dirs.

**On Clariden** (the execution copy + the multi-TB payloads): see §7.2.

### 7.2 Clariden filesystem map

User `fffoivos`, account `a0140`. Total project state ~6.9 TB.

| Filesystem | Path | Size | What |
|---|---|---:|---|
| capstor | `/capstor/scratch/cscs/fffoivos/runs/bakeoff/` | 5.1 TB | Bakeoff + TD training runs (checkpoints + tensorboard) |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/` | 654 GB | Raw + intermediate + final corpus |
| capstor | `/capstor/scratch/cscs/fffoivos/runs/eval/` | 480 GB | All eval outputs |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/repo/` | 273 GB | Mirrored repo |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/` | 168 GB | Init arms × {HF, Megatron, TP=2, TP=2 R17-patched} |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/token_distillation/` | 125 GB | TD prep / snippets / model outputs / R17-patched Megatron |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/code/` | 112 GB | Megatron-LM-Swiss-AI + pretrain-code |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/models/` | 16 GB | Apertus-8B-2509 base |
| iopsstor | `/iopsstor/scratch/cscs/fffoivos/tokenizers/` | 9.2 MB | Extended modern-only 148,480 ship bundle |

### 7.3 Most-used paths

| Need | Path |
|---|---|
| Production training data (.bin) | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document` |
| Production init checkpoint (Vanilla TP=2 R17-patched) | `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release` |
| TD challenger init checkpoint | `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/release` |
| Selected NFC corpus pool | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet` |
| Heldout Greek eval JSONL (500 docs) | `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/heldout/cpt_greek_heldout_500_20260522.jsonl` |
| Apertus base HF model (also teacher) | `/iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/` |
| Extended tokenizer | `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/` |
| Megatron-LM commit | `/iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI/` (`c92402e3`) |
| lm-eval runtime | `PYTHONPATH=/iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval` |

### 7.4 Production hydration check

Before launching production CPT, verify:

```bash
ssh clariden 'for p in \
  /iopsstor/scratch/cscs/fffoivos/models/apertus-8b-2509/config.json \
  /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480/tokenizer.json \
  /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.bin \
  /iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document.idx \
  /iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release \
  /iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI \
  /iopsstor/scratch/cscs/fffoivos/python_envs/lm_eval ; do
  if [ -e "$p" ]; then echo "OK  $p"; else echo "MISSING  $p"; fi
done'
```

All seven should print `OK`.

### 7.5 Loss measurement policy

Raw Megatron `lm loss` is per-target-token CE and is NOT comparable across the 131,072-token Vanilla arm and the 148,480-token extended arms. Treat raw `lm loss` as health-only telemetry unless all compared runs use the same tokenizer.

For cross-tokenizer loss signal, use **heldout checkpoint BPB** (bits-per-byte) from `compute_tokenizer_fair_metrics.py`. Older artifacts call this `BPC`; it is bits per UTF-8 byte. See `03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md` for the full rule.

---

## 8. Current production decision state

From `PRODUCTION_DECISION_STATE.md` (2 B-stage snapshot, partially superseded).

The 2 B bakeoff selected **Vanilla Apertus-8B-2509 with the base 131,072-token tokenizer** as the safe production default for the next 15-20 B Greek CPT run. Centroid eliminated. ReTok not selected as-is. TD layer11 was the strongest extended-tokenizer path but did not beat Vanilla on the aggregate 2 B Greek/preservation criteria.

**The 3.5 B + 5 B continuations changed the picture on downstream aggregates** — at 5 B, TD leads all three downstream aggregates. Vanilla still leads BPB. **No Node 4 thresholds were ever locked**, so the bakeoff is informational not adjudicated. The current production pick is the 2 B-stage conclusion that the 5 B data does not strongly support.

**Production CPT launcher** is dry-run validated at `03_4_implementation_experiments/init_bakeoff/production_cpt/submit_vanilla_base_15b_chain.sh`. It points at the Vanilla base-tokenizer NFC-safe Megatron prefix with Goldfish loss restored — but cannot fire until V8 + R17 + V1 gates clear.

---

## 9. What's needed for production

Block-list ordered by approximate dependency:

### Block 1 — Lock Node 4 thresholds (decision side)

Lock X / M_progress / M_ext / M_van / T BEFORE any new arm trains. Use priors independent of the existing bakeoff results (external precedent + first-principles arguments about deployment economics). Also pre-commit:

- Which Greek metric the §8b gate evaluates against (BPB, downstream aggregate, or weighted combination) — the BPB-vs-downstream divergence (D4) needs an explicit resolution.
- Whether xquad_el is in the production-target set — D5 +7.57 pp swing means the headline reverses on that decision.
- Exact list of per-language regression slices for §8a gate.

### Block 2 — Build measurement infrastructure

- **V4 baseline variance** — bootstrap CIs on V4-HF per-sample logs. Required for any "X beats Y by N pp" claim to have a noise floor. ~3-4 h on Clariden normal.
- **§8a per-language regression slices** — held-out FineWeb-2 EN/FR/RU/DE slices + V4-HF reference measurement on each. ~hours of eval work + corpus pull.
- **V1 decontamination** — NeMo Curator workflow against the chosen benchmarks (per Q A4 scope decision). ~1-3 days on Clariden xfer.
- **V5 polytonic firing audit** (optional, gates polytonic Phase 2 only) — per-new-token firing distribution on actual CPT corpus.

### Block 3 — Apply production patches

- **R17 patch** — apply `patch_apertus_extras.py` to production initial checkpoint. xIELU + QK-Norm trained values must survive HF→Megatron conversion. Scaffold exists; apply step pending.
- **V8 Goldfish hash uniformity** — tokenize CPT corpus sample with extended tokenizer, simulate Apertus's Goldfish (k=h=50, hash 1,000,003, seed 2971215073), count masked positions per new ID, verify uniformity within ±2σ. ~1 h once the corpus is built.

### Block 4 — Land cheap mitigations (cpt_plan v0.7 risk inventory)

~2 h total for the 7 cheap risk mitigations (R2 / R4 / R5 / R6 / R7 / R8 / R9 in archived `RISKS.md`).

### Block 5 — Decide outstanding plan items

- Q A2 — total CPT token budget (working assumption 15-20 B).
- Q A4 — stakeholders / benchmark set for V1 decontamination scope.
- Q A6 — specific downstream tasks for §5.6 weighting.
- D7 — PF5 ILSP YAMLs merge for native-Greek eval coverage (hellaswag_greek, winogrande_greek, mmlu_pro_greek, truthfulqa_greek, medical_mcqa_greek).
- Math 2 % bucket — keep at 2 % for production or drop in favor of more replay.
- Codeparrot vs StarCoderData — accept fallback or re-pull from BigCode.

---

## 10. Open items the planning agent owns

Beyond the gating blocks above, the planning agent's redesign should also fold in:

**Empirical findings from the bakeoff that change input assumptions:**

- The CPT regime as configured (LR 1.5e-5, β3/α warmup 238 steps, NTP, no staged training, uniform LR across rows) **degrades Greek aggregate in early CPT for all arms**. v0.12 §8.3 (staged training) and §8.4 (higher LR on new-token rows) were specifically recommended and not applied.
- TD's downstream and BPB signals diverge — the next plan must specify which is the production target metric, or how to combine them.
- The vocab cutoff at 17,408 was never empirically chosen against the open `{10K, 15K, 20K, 25K}` grid. User has flagged smaller cutoffs as worth investigating.
- The bakeoff is at ~25 % of v0.12's planned per-arm pilot scale. The next plan should clarify whether production CPT goes through the v0.12 §8.7 staged pilot → main → anneal or skips ahead.

**Apertus-fidelity deviations the next plan needs a position on:**

- Microbatch=2 (vs Apertus 4) — production-scale acceptability unverified.
- Fixed 4.19 M batch (vs Apertus's ramp to 8.39 M after 8 T) — relevant at production scale.
- Apertus optimizer-state portability — never resolved.
- Goldfish-for-production gating (V8 + the four-gate canary protocol).

---

## Sources synthesized

The 10 source docs this synthesis is built from are archived at [`_archive/synthesis_sources_20260526/`](_archive/synthesis_sources_20260526/). They remain accessible if you need any of the long-form content this synthesis abridged.

| Doc | Owns |
|---|---|
| `old_experiments_plan.md` v0.12 | Experimental-design parent plan (§3 Decision Nodes, §5 arms, §10 Q8 decision rule) |
| `cpt_plan.md` v0.7 | CPT-execution successor (corpus / optimizer / verifications) |
| `cpt_plan_v0.7_answers.md` | Q A/B/C/D decision snapshot at bakeoff firing |
| `cpt_plan_v0.7_status.md` | V1-V16 verification status |
| `apertus_fidelity_checklist.md` | Apertus-pretraining fidelity items + production gates |
| `PLAN_VS_RESULTS_RECONCILIATION_20260526.md` | Plan-vs-results reconciliation + 14-entry discrepancy log |
| `PRODUCTION_DECISION_STATE.md` | 2 B-stage production decision (with 5 B-supersedes banner) |
| `ARTIFACTS_AND_HYDRATION.md` | Repo ownership policy + hydration check |
| `CLARIDEN_INVENTORY_20260524.md` | Clariden filesystem map (~6.9 TB) |
| `collegues_Apertus_plan.md` | Original Greek-language project framing |

Also still in this directory and NOT archived (still actively used):
- `README.md` — directory index
- `TRAINING_RECIPE.md` — production training spec (referenced by `_train_config_common.env`)
- `TOKEN_DISTILLATION_PLAN.md` — TD-specific plan
- `RISKS.md` — 17-risk silent-failure inventory
- `TODO.md` — current active items

---

*Last updated 2026-05-26 from synthesis. Supersede with a new dated `CPT_MASTER_<date>.md` when state changes meaningfully.*
