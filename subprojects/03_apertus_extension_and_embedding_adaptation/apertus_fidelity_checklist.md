# Apertus Fidelity Checklist — Critical Architectural Characteristics Requiring Special CPT Attention

*2026-05-20. Companion to [`cpt_plan.md`](cpt_plan.md) v0.6.*

*Per user directive 2026-05-20: **closest-to-Apertus-original-process
is canonical.** p-skarvelis's HF-Trainer pipeline is an interesting
baseline but does not constrain our decisions. This doc enumerates
every Apertus architectural characteristic that requires special
attention during CPT and tracks our position on each.*

This is the **must-not-miss** list. Every item below is either (a) a
training-recipe choice that's non-default in HF-Trainer-style CPT,
(b) an artifact of Apertus's architecture that's easy to forget, or
(c) a state-portability concern when resuming from the Apertus base
checkpoint.

## Reading guide

| Status | Meaning |
|---|---|
| ✅ verified locally | We've checked this in our prior artifacts; reproducible. |
| 🟡 partially verified | Some part is checked; the rest needs CSCS-side execution. |
| 🔴 not yet verified | Item is real and we have to do it before kickoff. |
| 🔍 needs lookup | The fact we need is in the Apertus tech report or HF repo. |

## 1. Optimizer and training dynamics

### 1.1 AdEMAMix optimizer (not AdamW) 🔍

Apertus's pretraining optimizer is **AdEMAMix**, which adds a long-term EMA of gradients to the standard Adam framework. It's the under-recipe that, combined with §1.2 below, produces Apertus's frequency-independent per-token norm convergence.

**Why it matters for CPT:**
- AdamW (HF Trainer default) **breaks the convergence mechanism** even at the same nominal LR.
- AdEMAMix's effective updates per nominal step are larger than AdamW's, which is why v0.6 §3.3 corrects the CPT peak LR down to 1.5e-5 (vs the naïve "Llama-style" 3e-5).
- Optimizer-state portability when continuing from Apertus base checkpoint: ideally we load the AdEMAMix state, not initialize fresh. If state is not portable, the first 1-2 % of CPT acts as optimizer-state warmup — accepted but worth measuring.

**What we need:**
- β1, β2, α (long-term EMA decay), weight decay values. From tech report §2.3 / Appendix B.4. (Q C2 in v0.6 §11.)
- Megatron-LM-Swiss-AI fork that natively supports AdEMAMix (the swiss-ai fork has it; verify branch/commit per Q D1).

**Our position:** **MUST** use AdEMAMix via Megatron-LM-Swiss-AI. AdamW under any framework would silently violate Apertus-fidelity.

### 1.2 Gradient clipping at 0.1 🔴

Apertus uses **global-norm gradient clipping at 0.1** (10× tighter than the typical 1.0). The tech report explicitly states that with AdEMAMix, this clipping fires *basically every step*. Together with §1.1, §1.3, and §1.4 below, this is one of the four mechanisms producing Apertus's per-token norm convergence regime.

**Why it matters for CPT:**
- Aggressive clip caps the per-step movement of every parameter uniformly. This is what makes Greek's 0.023 % pretraining share produce norm-parity with English: clipping throttles high-frequency-token gradients to the same per-update budget as rare-token gradients.
- For our new-token rows: the §8.4 plan to apply higher LR (5×–10×) on new-token rows interacts non-trivially with this clip — the clip is computed on the global gradient norm, so any per-row LR multiplier still gets globally normalized down. The intended effect of the per-row higher LR may be muted in practice.
- **Recommend measuring**: in the bakeoff, log per-row update magnitudes on new-token rows vs base-token rows. If new-token rows aren't actually moving faster despite the LR multiplier, drop the multiplier.

**Our position:** **MUST** preserve 0.1 grad clip. Megatron-LM-Swiss-AI's config has this as the value Apertus pretraining used; don't change it.

### 1.3 Pre-Norm + RMSNorm ✅ (architectural, not a CPT choice)

Apertus uses Pre-Norm with RMSNorm: RMSNorm sits before attention and before MLP at every layer; the residual stream is RMS-normalized at every layer entry. Consequence: **the forward pass is scale-invariant in the input embedding** (up to the final LM-head logit-magnitude effect).

**Why it matters for CPT:**
- This is *why* norm-matched init (5.05 / 3.80 from Phase A) is a "do it but it's not behavior-defining for the residual path" choice. The forward pass doesn't directly depend on absolute embedding norm; only the LM-head logit magnitudes (§1.4) do.
- For Centroid init: the dispersion-matched Gaussian noise around the centroid produces rows in approximately the same norm regime as existing rows, but the absolute norm could drift. Norm-matching as a post-init step is still recommended.

**Our position:** ✅ accounted for in `experiments_plan.md` §8.2.

### 1.4 QK-Norm ✅ (architectural)

Apertus uses **QK-Norm** in attention: Q and K are normalized per head before the dot product. Removes the second-order coupling between embedding norm and attention-logit magnitude.

**Why it matters for CPT:**
- For init: combined with §1.3, **QK-Norm flattens the forward graph's sensitivity to embedding scale**. Per-row norm choice (5.05 / 3.80) controls the LM-head logit contribution and the residual-stream contribution differently; QK-Norm specifically removes the attention-side sensitivity.
- For Distillation (if revisited): the attention-matching objective needs Q-and-K-norm-aware extraction of attention targets, not just raw `score = Q @ K^T`. This is one of the reasons v0.6 §13 brackets Distillation.
- For internal-state-matching techniques generally: use `model.forward()` outputs, not reimplemented attention math. v0.6 §13 makes this explicit as a general principle.

**Our position:** ✅ for Centroid + ReTok arms (closed-form inits, no attention matching). Flagged in v0.6 §13 for any future technique.

### 1.5 WSD learning-rate schedule ✅

Apertus pretraining used WSD (warmup-stable-decay) with multiple stages. v0.6 §3.3 specifies CPT also uses WSD with brief re-warmup → plateau → linear decay aligned with anneal.

**Why it matters for CPT:**
- Cosine LR (p-skarvelis's choice) wastes the highest-quality anneal tokens on the early plateau and the lowest-quality late tokens on the steep decay endpoint. WSD flips this: the decay aligns with anneal, so the high-quality tail data sees decreasing LR.
- Re-warmup from low LR to the CPT peak (1-2 % of CPT tokens) is the standard pattern for CPT restart from a converged pretraining checkpoint.

**Our position:** ✅ accepted in v0.6 §3.3.

### 1.6 CPT peak LR 1.5e-5 ✅

Per Q C1, Apertus 8B pretraining peak LR = 1.1e-4 (tech report Table 2). CPT peak = 1/5 to 1/10 of that = **1.5e-5 default** (range 1.1e-5 to 2.2e-5).

**Note**: this is 2-5× lower than the Llama-style 3e-5 we might have anchored to without checking the tech report. AdEMAMix's larger effective updates explain the difference.

**Our position:** ✅ accepted in v0.6 §3.3.

### 1.7 AdEMAMix batch-size scaling 🔍

Apertus 8B's batch schedule started at 4.2M tokens/step and doubled to 8.4M during training. This is non-trivial because AdEMAMix's long-term EMA is sensitive to batch size.

**Why it matters for CPT:**
- We continue from Apertus's final pretraining checkpoint, which means the model expects an 8.4M-token batch. Setting CPT batch lower changes the optimizer dynamics. Setting it higher would too.
- p-skarvelis used `effective_global_batch_size: 256` × `max_seq_length: 2048` = 524k tokens — about 1/16 of Apertus's final batch.

**What we need:** confirm whether Apertus's final batch size is the target for CPT, or whether a smaller batch is acceptable for the bakeoff. Tech report §2.3 / Q D1 from the Megatron config.

**Our position:** 🔴 **needs decision before the bakeoff fires**. Likely OK to use a smaller batch for the 2 B-token bakeoff; for production CPT (15-20 B) we should match Apertus's batch size unless cost-prohibitive.

## 2. Activation and norm trainable parameters

### 2.1 xIELU activation with trainable per-layer scalars (αp, αn) 🔴 → V15

xIELU is Apertus's novel activation. It has **two trainable scalars per layer** — αp and αn — that the optimizer needs to track. These are NOT part of the embedding tables.

**Why it matters for CPT:**
- After calling `resize_token_embeddings()`, the optimizer's parameter list is rebuilt. **Verify the xIELU scalars are still in the optimizer's parameter list.** Easy to miss; resize_token_embeddings is documented for the embedding tables, not for "everything else."
- Param count sanity check: after resize, `model.num_parameters()` should = base + 184.5 M (for 22,528 new rows × 4,096 × 2). The xIELU scalars add zero. If the param count is off, something's wrong.

**Our position:** 🔴 V15 in v0.6 §12 — **must verify** post-resize. ~30 minutes on a debug allocation.

### 2.2 RMSNorm trainable scale weights ✅

Standard for any RMSNorm model. The CPT optimizer should be training these as part of the full-parameter pass. v0.6 §8 doesn't flag this as a separate item; we don't need to either.

## 3. Loss function and data masking

### 3.1 Goldfish loss with hash-based masking 🔴 → V8, G1 (gating for production)

Apertus uses **Goldfish loss**: a hash function chooses which target tokens to mask out from the loss during training, providing memorization suppression.

**Why it matters for CPT:**
- If the hash is **vocab-aware** (i.e., hash takes the token ID), extending the vocab from 131,072 to 148,480 (or 153,600) changes which positions get masked per new token. The masking distribution over new tokens may NOT be uniform.
- Specifically: if new-token IDs concentrate in a hash-distribution low-mass region, those tokens get rarely / never masked → over-trained relative to other tokens → unstable. Or the opposite — over-masked → undertrained.
- v0.6 §10 Q B4 default: **NTP for bakeoff (cleaner signal), Goldfish for production (matches Apertus's recipe).** This is right: the bakeoff is small and a confound from Goldfish-masking would hurt init-method discrimination; production fidelity needs Goldfish.

**What we need:**
- Q C4 lookup: Apertus's Goldfish hash function + masking rate from tech report §3.3.
- V8 verification: confirm uniform masking distribution across new tokens before turning Goldfish on for production CPT.

**Our position:**
- **Bakeoff: NTP only** ✅ (cleaner discrimination).
- **Production CPT: Goldfish gated on V8 passing**. If V8 reveals non-uniform masking, fix or fall back to NTP-only for production.

### 3.2 Strict cross-document attention masking 🔴 → V12

Apertus pretraining masks attention across document boundaries within a packed sequence. CPT dataloader must do the same; if not, sequences from different documents attend to each other → leak of training-dynamics variable.

**Why it matters for CPT:**
- Megatron-LM-Swiss-AI has this as a config flag (`reset_attention_mask` or equivalent). Setting it correctly matches Apertus pretraining; missing it changes the training dynamic.
- HF Trainer-style pipelines (p-skarvelis's setup) often DO NOT mask cross-document attention by default — this is a real divergence from Apertus, even if the loss curves look fine.

**Our position:** 🔴 **must verify** in Megatron-LM-Swiss-AI config before kickoff. ~1 hour.

### 3.3 EoD token loss masking 🔴 → V13

Apertus masks loss on EoD (end-of-document) positions. The model isn't asked to predict-the-next-document.

**Why it matters for CPT:**
- Same logic as §3.2. Megatron config flag; verify it's set; HF Trainer defaults are different.

**Our position:** 🔴 **must verify**. ~1 hour.

### 3.4 BoD/EoD special token IDs preserved 🟡 → V14

Apertus's special tokens have fixed IDs in the base tokenizer:
- `<unk>` = 0
- `<s>` (BoS / BoD) = 1
- `</s>` (EoS / EoD) = 2
- `<pad>` = 3
- IDs 4-999 = reserved Mistral specials

**Why it matters for CPT:**
- Tokenizer extension must NOT renumber these IDs. The first-1000-ids-preserved hard constraint from `GLOBAL_DECISIONS.md` is exactly this requirement.

**Our position:**
- ✅ **verified locally** in `scripts/build_and_verify_ship_tokenizer.py` for both ship bundles (148,480 and 153,600). All 1,000 added_tokens are byte-identical to Apertus base.
- 🟡 **HF↔Megatron roundtrip** still to verify on CSCS — confirm Megatron loads the extended tokenizer with the same special-token IDs at the same positions.

## 4. Embedding architecture

### 4.1 Untied embeddings (`tie_word_embeddings=False`) ✅

Both input embedding `E` and LM-head `U` are separate matrices, both [131,072, 4,096] in the base.

**Why it matters for CPT:**
- We resize both. We init both separately. Per-side norm targets (5.05 for E, 3.80 for U) from Phase A.
- HuggingFace `resize_token_embeddings()` handles both matrices — verified by `experiments_plan.md` §8.1 + our ship bundles.

**Our position:** ✅ explicit throughout our plan and `experiments_plan.md` §2.2.

### 4.2 Norm-matched init for new rows ✅

Per Phase A norm diagnostic (`runs/apertus_greek_diagnostic_20260511_v2/`), Greek tokens have median ‖E‖ = 5.05 and ‖U‖ = 3.80 — within 1 % of English. This is what makes "norm-match to existing Greek tokens" defensible: it's also matching English-baseline.

**Why it matters:** §1.3 (Pre-Norm) means the residual path doesn't depend on absolute embedding norm, but §1.4 (LM-head logit magnitude) does. Norm-matched U-rows produce logit magnitudes comparable to existing rows; under-norm-matched U-rows produce under-confident predictions, and the model burns its first training steps re-norming them instead of learning content.

**Our position:** ✅ in v0.6 §5.2 (applied to both E and U for both ReTok and Centroid arms).

### 4.3 Per-token cross-language norm convergence ✅ (mechanism understood)

The combination of §1.1 + §1.2 + §1.3 + §1.4 + logit saturation = Apertus's per-token-norm-converges-across-frequency property. This is documented in `docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`.

**Why it matters for CPT:**
- New-token rows will tend to converge to the same norm regime as existing rows under faithful Apertus training dynamics. **Norm-match init is therefore aligned with the equilibrium the training will pull rows toward.** Mismatched init would create transient instability while the rows re-norm.
- Implication: if our init is wrong, the loss curve will recover; but if our other mechanisms (AdEMAMix, 0.1 clip, etc.) are NOT faithful, the equilibrium itself shifts and we don't get the same norm-converge property.

**Our position:** ✅ understood. **This is the load-bearing reason to be faithful to AdEMAMix + 0.1 clip + Pre-Norm + QK-Norm**: they're not independent design choices, they're four mechanisms that together produce the norm-convergence regime. Breaking any of them changes the equilibrium.

## 5. Tokenizer

### 5.1 Mistral-Nemo `tekken` v3 byte-level BPE ✅ → V16 still pending

Apertus inherited Mistral-Nemo's tekken v3 byte-level BPE tokenizer. It was *never retrained on Apertus's pretrain corpus*; Apertus just trained the existing merges to convergence.

**Why it matters for CPT:**
- Our extension adds 17,408 (or 22,528 with polytonic) new tokens **on top of** the inherited tokenizer. The first 131,072 IDs must remain byte-identical to Mistral-Nemo tekken v3.
- ✅ **verified locally** for both ship bundles.
- 🔴 V16: byte-fallback collision check — confirm that none of our new polytonic tokens decode to byte sequences that would have been byte-fallback-tokenized by the base. If a polytonic token's bytes match a byte-fallback sequence, the same character could be tokenized two ways depending on which BPE merge fires first.

**Our position:** ✅ structural fidelity verified. 🔴 V16 pending. ~1 hour.

### 5.2 NFC normalization for training text 🔴 → V9, I1

Mixed NFC / NFD forms tokenize differently in tekken. Apertus pretraining likely used NFC (standard); our training text should too.

**Why it matters for CPT:**
- Polytonic Greek frequently appears as NFD (decomposed) in raw scrapes — base letter + combining accent. NFC composes these into single codepoints where defined.
- If our training text is mixed-form, the same word tokenizes differently across documents. The model has to learn both forms, which dilutes per-token effective exposure.

**Our position:** 🔴 V9 — normalize all training text to NFC before tokenization. ~2 hours. Apply during CPT corpus build on Clariden `xfer`.

## 6. Context length and positional encoding

### 6.1 RoPE-scaled context (Llama-3-style, factor 8.0) ✅

Apertus's `max_position_embeddings = 65,536` is achieved via RoPE scaling (factor 8.0, original 8,192). We train at seq=4,096 (well below the scaled max).

**Why it matters for CPT:**
- We keep seq=4,096 (matches Apertus pretraining). Out of scope for v0.6.
- If long-context capability becomes important downstream, that's a separate continuation step after CPT.

**Our position:** ✅ kept at 4,096 — explicitly stated in `experiments_plan.md` §8.7.

## 7. Data pipeline characteristics

### 7.1 Pretraining-Greek-not-replayed principle ✅

v0.6 §2: *"Old Apertus Greek pretraining data is not replayed."*

**Why it matters for CPT:**
- Apertus already trained on FineWeb-2-HQ Greek; replaying it teaches nothing new.
- The dedup audit (03_2) produced the apertus_overlap_drop_docs.parquet overlay that implements this principle: hard-drop any doc in our CPT pool that overlaps with Apertus pretraining.
- Implementation: download nanochat corpus → exclude via the overlay → internal dedup → final pool.

**Our position:** ✅ aligned with v0.6 + our [`CURRICULUM_AND_INIT_CORPUS.md`](03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md).

### 7.2 Replay maintains Apertus-side multilingual capability

This is the "preserve Apertus's multilingual character" hard constraint from `experiments_plan.md` §1, operationalized as the replay design in v0.6 §4.

**Why it matters for CPT (architectural angle):**
- Replay tokens go through the same training pipeline (AdEMAMix + 0.1 clip + Goldfish + xIELU). The clipping in particular means each replay token's gradient update is the same magnitude as a Greek token's. So 30 % replay means roughly 30 % of update budget is keeping non-Greek capability alive.
- v0.6's 70/30 split is more conservative than our prior 85/15 or p-skarvelis's 90/10. **Given the "closest to Apertus" directive, 70/30 is the right call** — it's the highest replay share in the converged-CPT literature for languages with comparable pretraining shares to Greek's 0.023 %.

**Our position:** **align with v0.6's 70/30 default.** Adjust only if V4 baseline + early bakeoff metrics show retention regression at <30 % replay.

## 8. Decontamination as Apertus-fidelity

### 8.1 Eval-set decontamination 🔴 → V1, K1 (must-do gating)

Apertus's compliance posture includes memorization suppression (Goldfish) and presumably training-set decontamination against standard evals. Our CPT pool must match this hygiene.

**Why it matters:**
- If our CPT pool contains GreekMMLU / Meltemi / Belebele / HumanEval / GSM8K / MMLU / ARC / HellaSwag test items, the resulting model's scores on those benchmarks are not honest. **Required for shipping a defensible model.**
- Greek-specific complication: contamination check must use multiple normalization views (raw, NFC, accent-normalized, monotonic-folded for detection only) because Greek text has more orthographic variants than English.

**Our position:** 🔴 **must run before any training** via NeMo Curator workflow. v0.6 K1; 3-5 days.

## 9. State portability when continuing from base checkpoint

### 9.1 Loading the Apertus base into the resized model 🔴 → V2

After `model.resize_token_embeddings(N)` where N > 131,072, loading the Apertus base checkpoint should preserve all base weights and leave only the new rows uninitialized (or carrying the init we chose).

**Why it matters:**
- HF and Megatron treat resize differently. Cross-system roundtrip (HF ship bundle → Megatron format → load checkpoint → resize → save → reload in HF for eval) needs verification.
- The `hfconverter` repo from swiss-ai's GitHub org is likely the right tool for HF↔Megatron conversion. Worth using.

**Our position:** 🔴 V2 + I2 — needs CSCS-side smoke test (resize tiny extension first, full extension second). ~4 hours on debug allocation.

### 9.2 AdEMAMix optimizer state portability 🔍

If the Apertus checkpoint includes optimizer state (the full long-term EMA momentum vectors), loading it gives us optimizer-state continuity. If not, the first ~1-2 % of CPT acts as optimizer-state warmup.

**What we need:** check the HF model repo for `optimizer.pt` or equivalent. The Apertus tech report should say whether optimizer state is published.

**Our position:** 🔴 **needs lookup**. If available, use it; if not, accept ~1-2 % warmup as a known cost.

## 10. Summary: what's gating kickoff

The items that MUST be resolved or verified before the bakeoff fires:

| # | Item | Effort | Where |
|---|---|---|---|
| 1 | AdEMAMix β1/β2/α/weight_decay lookup (Q C2) | <1 h | tech report §2.3 / Appendix B.4 |
| 2 | Apertus Megatron-LM-Swiss-AI fork branch/commit (Q D1) | <1 h | github.com/swiss-ai/Megatron-LM + pretrain-code |
| 3 | Goldfish hash + masking rate (Q C4) | <1 h | tech report §3.3 |
| 4 | Optimizer state availability (§9.2) | <1 h | HF Apertus model repo + tech report |
| 5 | V1 decontamination (K1) | 3-5 days | NeMo Curator on Clariden xfer |
| 6 | V12 cross-document attention masking | 1 h | Megatron config check |
| 7 | V13 EoD loss masking | 1 h | Megatron config check |
| 8 | V14 HF↔Megatron special-token roundtrip | 30 min | smoke test |
| 9 | V15 xIELU scalars in optimizer param list | 30 min | smoke test |
| 10 | V16 byte-fallback collision check | 1 h | local + smoke |
| 11 | V9 NFC normalization probe | 2 h | local |
| 12 | V8 Goldfish hash uniformity (gates production, not bakeoff) | 1 day | after V5 polytonic audit |

Total **<2 days of focused engineering** + **3-5 days for V1 decontamination** in parallel.

## 11. What changes in our prior planning

Given the "closest to Apertus" directive:

| Doc | Edit |
|---|---|
| `cpt_plan_v0.6_delta_vs_prior_planning.md` §5 (Recommended posture) | Flip "Framework" row from pragmatic split to **Megatron-LM-Swiss-AI throughout**. Flip "Replay split" from 80/20 compromise to **v0.6's 70/30**. Promote V12-V16 from "should" to **MUST**. |
| `03_4_implementation_experiments/ENVIRONMENT_AND_BENCHMARKS.md` | Demote `swiss-ai/apertus-finetuning-recipes` from "primary trunk" to "alternative for tiny experiments." Promote `swiss-ai/Megatron-LM` + `swiss-ai/pretrain-code` to canonical. |
| `03_3_cscs_experiments_kickoff/ANALYSIS.md` Review Checkpoint D | **Resolved**: Megatron-LM-Swiss-AI throughout. |
| `03_3_cscs_experiments_kickoff/CURRICULUM_AND_INIT_CORPUS.md` | Update 85/15 default to **70/30** per v0.6. |
| `03_3_cscs_experiments_kickoff/STORAGE_AND_EXISTING_WORK.md` §3.4 | Reframe p-skarvelis's HF-Trainer artifacts as **interesting baseline only** (not a scaffold we adopt). |
| `cpt_plan_v0.6_answers.md` Q A5 (sign-off) | Update — the divergence from p-skarvelis is now intentional, not a coordination question. |

Awaiting your go to apply these.
