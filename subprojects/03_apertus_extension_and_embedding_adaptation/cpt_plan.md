# Apertus 8B Greek CPT — Plan v0.7

**Status:** Draft (phrasing refactored to status checks; decontamination scope narrowed)
**Author:** Fivos (GlossAPI / Swiss AI Initiative)
**Date:** 2026-05-20
**Supersedes:** v0.6

**2026-05-24 decision overlay:** keep this plan as the design-space reference,
but use [`PRODUCTION_DECISION_STATE.md`](PRODUCTION_DECISION_STATE.md) for the
current post-bakeoff production path. The 2B bakeoff selected Vanilla/base
tokenizer as the safe default, and the bounded `td_full25_layer11` Token
Distillation challenger did not clear the aggregate downstream gate needed to
displace it.

Concrete launcher for the selected path:
[`03_4_implementation_experiments/init_bakeoff/production_cpt/`](03_4_implementation_experiments/init_bakeoff/production_cpt/).
It runs the base tokenizer on the NFC-safe bulk Megatron prefix with Goldfish
loss restored for production.

This version refactors §8 and §12 from prescriptive TODOs into status checks. The doc is a coordination artifact, not a list of work to do from scratch — items may already have been handled during prior work; the relevant question per item is "what's the current status." Decontamination scope (§8 K1, V1) also narrowed: the concern is verbatim test items in training data for benchmarks you want as clean measurement instruments, not blanket removal of on-topic Greek material.

---

## 1. Objective

Continue-pretrain Apertus 8B (base) on a curated Greek corpus to deepen the model's modern and polytonic Greek capabilities while preserving its multilingual, code, and reasoning performance.

Tokenizer extension is already complete: +17,408 modern Greek tokens and +5,120 ancient/polytonic Greek tokens. These artifacts are available, but the post-bakeoff production default is the base 131,072-token tokenizer; the bounded ReTok + Token Distillation challenger remained useful but did not prove the extended path for the next 15-20B CPT run. If both extension layers are activated, the new embedding rows add roughly **184.5M parameters** (22,528 × 4,096 × 2 for untied input + output) before optimizer state, or ~2.2% of the 8B base.

The model's intended downstream uses determine some of the design choices below; explicit capability targets are pending in §10 (Q A1) but deferred for now.

## 2. Settled shape

The curriculum has the shape of a shuffled-mixture bulk followed by an annealing tail. Replay of non-Greek data is present from token 0 at ~70% Greek / ~30% non-Greek (working default). Old Apertus Greek pretraining data is not replayed. Initialization is resolved by experiment: three closed-form variants for 1.5–2B tokens each. Dataloader state preserved across checkpoint boundary. LR schedule is WSD with decay aligned to anneal.

## 3. Curriculum: design space

### 3.1 Greek mixture (within the ~70% Greek portion of training)

Three weighting strategies: *quality-weighted upsampling*, *register-balanced*, *goal-driven*. The choice is value judgment; Sailor2's RegMix can resolve it empirically if compute permits proxy runs.

**Polytonic-token effective exposure metrics** (for V5):

| Exposure metric | Why it matters |
|---|---|
| Input occurrences per new token | Updates input embedding `E[T]` |
| Target occurrences per new token | Updates LM-head row `U[T]` |
| Target occurrences NOT masked by Goldfish | Actual output-learning opportunities |
| Distinct documents and sources | Avoids one corpus teaching all polytonic behavior |
| Register distribution | Catches register-specific overfitting |
| Frequency quantiles (p5/p25/p50/p95) | Average hides dead-tail tokens |
| Update/weight-norm ratio for `E[T]` and `U[T]` rows | Shows whether rows are actually learning |
| Cosine similarity matrix among new tokens, effective rank | Catches embedding collapse |

If polytonic tokens at p25 are getting <5k effective target occurrences at the planned 1B-token mark, upweight the polytonic-bearing bucket.

### 3.2 Anneal mixture (final ~10–20%)

Three published patterns: *high-quality narrow* (Llama 3), *quality-curated broad* (OLMo Dolmino), *goal-targeted*. Replay continues through anneal at reduced share (e.g., 30% → 15%). Length 10–20% of total budget.

### 3.3 LR schedule

WSD with brief re-warmup (1–2% of CPT tokens) from low LR to CPT peak, then plateau, then linear decay aligned with anneal.

**CPT peak LR:** Apertus 8B's pretraining peak LR was 1.1e-4 (tech report Table 2), lower than typical Llama-family 8B values because of AdEMAMix's effective-update characteristics. CPT peak should be 1.1e-5 to 2.2e-5; default 1.5e-5.

LR/anneal co-design: either WSD with explicit plateau over anneal window (decay only at the end), or weight averaging across anneal checkpoints à la Llama 3.

> **★ Apertus adaptation note (Goldfish hash interaction with extended vocab):** If the Goldfish hash is vocab-aware, extension from 131,072 to 148,480 changes per-token masking distribution. See §8 G1 and V8 for status check.

> **★ Apertus adaptation note (Megatron blended dataset for new-token-density warmup):** If implementing the optional density warmup, standard Megatron blended-dataset config doesn't natively support density-based sampling. See §8 E1.

### 3.4 References (curriculum)

Pythia curriculum study (2026); Llama 3 paper (Grattafiori et al. 2024); OLMo 2/3 (Allen AI); "How LR Decay Wastes Your Best Data" (2025); Ibrahim et al. (2024); Mid-Training survey (Oct 2025).

## 4. Replay: design space

### 4.1 Outer split

Published range 44–80% target / 56–20% replay. For Apertus-Greek deepening regime, middle of range — 65/35 to 75/25 defensible. Working default 70/30; Q B1.

### 4.2 Which languages — convergence framework

Four lenses: (1) Geographic, (2) Western EU / Swiss institutional, (3) Historical connection to Greece, (4) Script-system and major-corpus coverage.

| Language | Geo | West EU / Swiss | Historical | Major-corpus | Notes |
|---|---|---|---|---|---|
| Italian | ✓ | ✓ (Swiss official) | ✓ (Venetian, Latin descent) | | Strong triple |
| Turkish | ✓ | | ✓ (Ottoman period) | | Geographic primary |
| Russian | | | ✓ (Orthodox, Byzantine) | ✓ | Cyrillic; Byzantine inheritance |
| Bulgarian | ✓ | | ✓ (Orthodox; OCS) | | |
| Serbian | ✓ | | ✓ (Orthodox) | | |
| Arabic | ✓ (E. Med) | | ✓ (Islamic-period transmission) | ✓ | Strongest non-European hit |
| English | | ✓ | | ✓ | Unavoidable |
| French | | ✓ (Swiss official) | partial (Crusades, Frankokratia) | | |
| German | | ✓ (Swiss official) | partial (19c classical philology) | | |
| Latin | | ✓ (Western canon) | ✓ (antiquity, Catholic Church) | | Data-poor on modern web |
| Romanian | ✓ | | ✓ (Orthodox, Phanariot period) | | |
| Albanian | ✓ | | partial (Arvanites, Ottoman) | | Geographic priority |
| Macedonian | ✓ | | ✓ (Slavic, Orthodox) | | Small data volume |
| Spanish | | ✓ (large EU) | | ✓ | Convenient large language |
| Hebrew | ✓ (E. Med) | | ✓ (Septuagint, Sephardic) | | |
| Armenian | | | ✓ (Byzantine contact) | | Data-medium |
| Georgian | | | ✓ (Orthodox, Byzantine) | | Data-low |
| Ukrainian | | | ✓ (Orthodox, Byzantine via Kyivan Rus) | | |
| Chinese | | | | ✓ | Logographic; major corpus |
| Japanese | | | | ✓ | Different script |
| Persian / Farsi | | | ✓ (Hellenistic, Sufi-Greek contact) | ✓ | |
| Portuguese | | ✓ | | ✓ | Convenient large language |
| Dutch | | ✓ | | | EU coverage |
| Polish | | ✓ | | | EU coverage |

**Tier structure (weights pending Q C3 — Apertus per-language token shares):**

- *Tier 1, ~40–50% of replay slice:* English, French, German, Italian, Spanish, Russian, Arabic, Chinese.
- *Tier 2, ~35–45% of slice:* Turkish, Bulgarian, Serbian, Romanian, Hebrew, Portuguese, Polish, Dutch, Persian, Ukrainian, Japanese.
- *Tier 3, ~10–15% of slice:* Latin, Armenian, Georgian, Albanian, Macedonian.

**Honest note about Tier 3:** Replay can only preserve what the base has. For Tier 3 languages where Apertus had near-zero exposure, the small replay share is "doesn't hurt anything" rather than "actively maintains a capability."

### 4.3 Code share

Published range 0% (Sailor2) to 20% (SEA-LION v3). Q B2.

### 4.4 Replay sources (HuggingFace candidates)

| Component | Candidate dataset | Notes |
|---|---|---|
| Multilingual non-English (excl. Greek) | `HuggingFaceFW/fineweb-2` filtered to Tier 1–3 languages, OR `epfml/FineWeb2-HQ` | Filter out `ell_Grek` |
| English high-quality | `HuggingFaceFW/fineweb-edu` (Score-3 config) | Match Apertus's later-stage filtering |
| Code | `bigcode/starcoderdata` or `bigcode/the-stack-v2` | Match Apertus's StarCoder source |
| Math | `HuggingFaceTB/finemath` (`finemath-3plus`) | Apertus used FineMath CC subset |
| Cross-lingual parallel (optional) | OPUS Greek-English; OPUS Greek-Latin | Small but uniquely valuable for classical philology |

### 4.5 References (replay)

Sailor / Sailor2; Ibrahim et al. (2024); Conneau et al. (2020); EstLLM, Racka, SEA-LION v3, AMD Finnish playbook; "Reuse, Don't Retrain" (Parmar et al. 2024); RegMix; FineWeb2-HQ.

## 5. Initialization experiments

The completed 2B bakeoff compared three **closed-form** init methods. Token
Distillation remains bracketed out of that bakeoff, but is now a bounded ReTok
refinement challenger rather than an open-ended fourth method (§13 and
[`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md)).

| Variant | Description | Reference |
|---|---|---|
| Vanilla | Original Apertus tokenizer (vocab 131,072). No vocab extension. | Yuan et al. 2024, *LLaMA Beyond English*. Load-bearing baseline. |
| ReTok | Extended vocab (148,480). New row `E[T]` = norm-matched mean of base-tokenizer subpieces of `T`'s surface form. Same for `U[T]`. Uses per-token-specific subpiece information. | EEVE-Korean, Chinese LLaMA, FOCUS, Minixhofer et al. 2022. |
| Centroid | Extended vocab (148,480). New row `E[T]` = script-conditional centroid + Gaussian noise. Uses script-level distributional prior. Same procedure for `U[T]`. | Closely related to mean-of-existing-vocab strategies; per-script variant mostly novel. |

**Centroid init procedure:**

1. Identify Greek tokens in the base vocab. For each base-vocab token t, decode to surface form; flag as *modern Greek* if it contains U+0370–U+03FF characters, *polytonic* if it contains U+1F00–U+1FFF characters.
2. Compute per-script centroids in E and U: `E_centroid_modern = mean(E[t] for t in modern_set)`, same for polytonic; same in U. Also compute std of each set around its centroid.
3. For each new token T, classify by script; look up centroid; if both tags, average centroids. Fallback: if polytonic centroid is computed from fewer than ~50 base tokens, fall back to modern centroid for polytonic new tokens.
4. Initialize: `E[T] = E_centroid + ε`, `ε ~ Normal(0, σ_E)` with σ_E from step 2. Same in U.
5. Apply Phase A norm targets (5.05 / 3.80).

Cost: <1 minute of CPU work.

> **★ Apertus adaptation note (ReTok + FOCUS auxiliary vectors):** FOCUS uses fastText embeddings; fastText doesn't exist for polytonic. Default is plain ReTok subpiece-mean without FOCUS. See §8 B.

### 5.1 Primary intrinsic metrics: tokenizer-fair signals

Per-token PPL is **not comparable across Vanilla and extended-tokenizer variants**. Primary metrics:

| Metric | Why it matters |
|---|---|
| **Bits-per-byte (BPC)** | Cleanest cross-tokenizer comparison |
| **NLL per Unicode character** | More interpretable for Greek/polytonic |
| **NLL per word** | Human-facing language metric |
| **Tokens-per-word, chars-per-token, compression ratio** | Quantifies tokenizer efficiency |
| **STRR** | Whole-word preservation |
| **Throughput** for Greek and non-Greek | Net effect of extended softmax vs Greek savings |

> **★ Apertus adaptation note (BPC computation unit for polytonic Greek):** Choice of unit (UTF-8 bytes, Unicode code points NFC/NFD, grapheme clusters) affects reported numbers. NFC code points are the recommended unit; document the choice. See §8 H1.

Per-token-group PPL on new tokens remains a useful **secondary** signal within ReTok vs Centroid comparison.

### 5.2 Init procedure for E and U

The closed-form arms apply their respective procedures to both `E` and `U`
matrices independently. For the bounded Token Distillation challenger, Apertus's
untied `lm_head` is handled explicitly: hidden-state TD updates new input rows,
and a separate next-token CE path may update only new output rows.

### 5.3 New-token integration diagnostic suite

Read at every bakeoff checkpoint:

| Diagnostic | Failure it catches |
|---|---|
| Rank of correct new token in next-token logits | New token invisible |
| Aggregate probability mass on new Greek tokens | Under- or over-emitted |
| New-token entropy by register | Polytonic rows collapsed or avoided |
| Top-k substitutions between new token and old subpieces | Model still prefers old segmentation |
| Greedy generation new-token utilization rate | New rows exist but behaviorally dead |
| Embedding L2-norm distribution for new vs existing | Degenerate-subspace collapse |
| Cosine similarity matrix among new tokens, effective rank | Same-direction collapse |

### 5.4 Token budget per variant

1.5–2B tokens per variant. Below 1.5B, init-method gaps may be within noise. Total bakeoff: 4.5–6B tokens.

### 5.5 Phase 0 — Optional stress probes

Before production-faithful bakeoff, optionally run 50–200M-token stress probes (high new-token density, OCR-noisy polytonic, code-switching, replay-heavy slices) for bug discovery. **Not resumable to production.**

### 5.6 Selection criteria

**Hard gates** — a candidate fails if any of:

- English/core retention drops more than threshold (set per V4)
- Code retention drops more than threshold, if code is a release requirement
- New-token rows show collapse (cosine clustering, near-zero usage)
- Polytonic text gets worse than base on character-normalized loss
- Throughput/memory hit disproportionate to Greek compression gain
- **Language-ID drift:** model over-emits Greek in non-Greek prompts

**Selection score (weighted, applied to non-failing candidates).** Indicative — final weights tied to Q A1:

| Component | Weight |
|---|---|
| Greek held-out BPC/char-NLL across registers | 30–40% |
| GreekMMLU + Belebele Greek + Meltemi suite | 25–35% |
| Polytonic custom eval | 10–15% |
| Retention suite | 15–25% |
| Efficiency | 5–10% |

Selection uses **windowed averages** (last 3–5 checkpoints in 80–100% range of each budget) with **bootstrap CIs over evaluation samples**.

### 5.7 What the three-arm bakeoff tests

- **Vanilla vs (ReTok or Centroid):** does vocab extension justify its parameter overhead? The 2B result currently says no for production unless TD closes the gap.
- **ReTok vs Centroid:** does per-token subpiece info beat a script-level distributional prior? The 2B result eliminates Centroid.

### 5.8 Caveat on polytonic signal

At 2B tokens, polytonic embeddings may still be undertrained. Honest comparison is on modern Greek tokens.

## 6. Evaluation: cadence, benchmarks, metrics

### 6.1 Evaluation cadence and statistical methodology

*Evaluate frequently enough to catch regressions early.* Every 100M tokens in init experiments, every 500M in production.

*Account for downstream task instability.* Checkpoint-window averaging (Park et al. Oct 2025).

*Use bootstrap CIs over evaluation samples.* Most benchmark runs are deterministic, so "run 3×" doesn't establish variance.

*Distinguish trajectory metrics from selection metrics.* Loss, per-bucket PPL, BPC, §5.3 diagnostics — read at every checkpoint. Downstream benchmarks — less frequent, windowed averages.

**Checkpoint averaging scope:** within a single init's checkpoints only, never across init experiments. Averaged for measurement/release; raw for training continuation.

### 6.2 Greek benchmarks

**Core:** GreekMMLU (`dascim/GreekMMLU` — 21,805 native MCQs, centerpiece), Belebele (`facebook/belebele` ell_Grek), Meltemi eval suite, held-out per-register PPL.

**Custom evals to construct:**

| Eval | What it measures |
|---|---|
| Polytonic continuation | Stay in polytonic vs collapse to monotonic |
| Accent/diacritic accuracy | Character-level breathing marks, oxia/varia/perispomeni, iota subscript, diaeresis |
| Modern Greek morphology minimal pairs | Case, number, gender, tense, agreement |
| Greeklish → Greek robustness | Real users type Greeklish |
| Greek-English code-switching | Real assistant usage, technical Greek |
| Legal/EU style eval | Does Eurlex upweighting help without making the model wooden |
| Register preservation | Demotic, katharevousa, ecclesiastical, classical, academic, journalistic |
| **Language-ID drift** | Non-Greek prompts shouldn't get Greek responses |

Construction ~1–2 weeks. Polytonic continuation and language-ID drift highest priority.

**MultiLoKo:** multilingual local knowledge for cultural eval.

### 6.3 Retention benchmarks (non-Greek)

| Benchmark | Capability | Source |
|---|---|---|
| HellaSwag (English) | Commonsense | `Rowan/hellaswag` |
| ARC-Challenge | Scientific reasoning | `allenai/ai2_arc` |
| MMLU (English subset) | Knowledge | `cais/mmlu` |
| HumanEval | Code | `openai/openai_humaneval` |
| GSM8K | Math | `openai/gsm8k` |
| XNLI (Tier 1 languages) | Multilingual NLI | `facebook/xnli` |
| Belebele (Tier 1/2 non-Greek) | Multilingual reading | `facebook/belebele` |
| Language-ID drift on Tier 1 languages | Cross-lingual response language | Custom |

Threshold for "regression" set after V4 baseline.

### 6.4 Stability diagnostics

Per checkpoint: training/validation loss per bucket; BPC trajectory per register; full §5.3 diagnostic suite; update norms vs weight norms.

### 6.5 Evaluation tooling

- `EleutherAI/lm-evaluation-harness` — primary for standard benchmarks
- `huggingface/lighteval` — alternative
- **Inspect AI** (`inspect.aisi.org.uk`) — custom open-ended evals

### 6.6 References (evaluation)

Park et al. (Oct 2025); GreekMMLU paper (Zhang et al. 2026); Meltemi paper (Voukoutis et al. 2024); MultiLoKo (Schmidt et al. 2025); Inspect AI documentation.

## 7. Tooling and repositories

### 7.1 Training framework

**Megatron-LM.** Apertus was trained on a Swiss AI fork supporting xIELU + AdEMAMix; verify branch/commit (Q D1).

- `NVIDIA/Megatron-LM` upstream; ROCm fork (per AMD Finnish playbook); Swiss AI fork (verify)
- Alternative: TRL + Accelerate (`swiss-ai/apertus-finetuning-recipes`)

### 7.2 Reference repositories

- `swiss-ai/apertus-tech-report`
- `swiss-ai/apertus-finetuning-recipes`
- AMD ROCm CPT playbook
- Sailor2 cookbook
- EstLLM repository (when released)

### 7.3 Data tooling

- `huggingface/datasets`
- Megatron `tools/preprocess_data.py`
- `datatrove`
- **`NVIDIA-NeMo/Curator`** — GPU dedup; downstream task decontamination workflow
- `text-dedup` library
- **`nlpaueb/gr-nlp-toolkit`** — modern Greek POS, morphology, NER, Greeklish detection
- **CLTK** — pre-modern Greek lemmatization

Classical sources: First1KGreek (`OpenGreekAndLatin/First1KGreek`), Perseus.

### 7.4 Evaluation tooling

`EleutherAI/lm-evaluation-harness`, `huggingface/lighteval`, **Inspect AI**.

### 7.5 Serving stack

vLLM and SGLang officially supported by Apertus. Compatibility with vocab 148,480 — see V10.

---

## 8. Apertus-specific adaptation requirements

This section lists places where standard recipes may need Apertus-specific adaptation. Items are grouped by engineering effort. **This is a coordination artifact, not a TODO list** — many items may already have been handled during prior work, particularly tokenizer extension, OCR/normalization pipelines, and dedup. Each item has a corresponding status check in §12; the section here describes *what the adaptation would entail* if the status check reveals work is still needed.

### Moderate adaptation (days, if needed)

**Item B — ReTok with FOCUS-style auxiliary vectors for polytonic.** (Referenced in §5.) FOCUS uses fastText embeddings as the similarity space for subpiece selection. fastText embeddings don't typically exist at the polytonic level. If FOCUS-style refinement is desired, options are (a) monotonic-folded fastText as proxy (~2–3 days, loses polytonic-specific information) or (b) skip FOCUS for polytonic and use baseline subpiece mean (no work). Default is (b).

**Item E1 — Megatron blended dataset for new-token-density warmup.** (Referenced in §3.3.) Only relevant if implementing the optional new-token-density warmup. Megatron's standard blended-dataset config doesn't natively support density-based sampling. Easiest workaround: preprocess to bucket sequences by new-token density and weight at bucket level (~2–4 days).

**Item G1 — Goldfish hash uniformity with extended vocabulary.** (Referenced in §3.3.) If the Goldfish hash is vocab-aware, extension from 131,072 to 148,480 may change which positions get masked per token. Verification: tokenize a sample corpus with the extended tokenizer, count masked positions per new token, confirm distribution is uniform. Only relevant if retaining Goldfish for production (per Q B4). Status: V8.

### Verification-only (hours, if not yet handled)

**Item F1 — Checkpoint averaging scope (documentation discipline).** (Referenced in §6.1.) Within a single init's checkpoints only; never across different init experiments. Averaged for measurement/release; raw for training continuation. Discipline, not code.

**Item H1 — BPC computation unit choice for Greek.** (Referenced in §5.1.) NFC code points are the recommended unit. Status: V9-related — document the choice if not already documented.

**Item I1 — NFC normalization of training text.** Mixed NFC/NFD forms tokenize differently. Likely already handled given prior OCR/normalization work. Status: V9 — confirm.

**Item I2 — `resize_token_embeddings` with untied E and U.** (Referenced in §5.) Must handle both `embed_tokens` and `lm_head` matrices. Likely already verified as part of tokenizer extension work. Status: V2 — confirm.

**Item J1 — vLLM/SGLang compatibility with vocab 148,480.** (Referenced in §7.5.) Non-power-of-2 vocab may break kernel assumptions in serving systems. Defer if production serving isn't in immediate scope. Status: V10.

### Decontamination (scope-dependent)

**Item K1 — Decontamination scope for chosen measurement benchmarks.** (Reframed in v0.7.) The relevant question is *not* blanket removal of on-topic Greek material. Training on Greek academic prose, exam-prep style writing, Kallipos theses on subjects GreekMMLU tests — these are exactly on-task and what CPT is for.

The relevant question is whether the *specific test items* of benchmarks intended as clean measurement instruments are verbatim in training data. This matters because:

- If your model has memorized the literal MCQs of GreekMMLU public split, you can't compare it to Meltemi, Apertus base, or other CPT recipes — any score gap reflects memorization, not capability.
- For Belebele's reading comprehension, if the exact passage appears in training, the model is doing recall rather than reading.

Operational approach (if you want a clean measurement benchmark):

1. Identify which benchmarks you intend to use as clean measurements (relevant to Q A4 — depends on whether the model is for internal use or external comparison).
2. For those benchmarks, extract the test items (literal MCQ stems + options, or source passages).
3. Run item-level dedup against training data (MinHash + exact match).
4. Remove offending training documents (typically a tiny fraction; not the whole on-topic corpus).

Tooling: NeMo Curator's downstream task decontamination workflow targets exactly this. Effort: ~1–3 days for the full pipeline depending on benchmark count. Status: V1 — current state of decontamination work is unconfirmed.

Not required: removing all academic Greek, all exam-prep material, all on-topic prose.

---

## 9. Production run shape

After init experiments select a winner:

1. Continue from winning checkpoint (dataloader resumes at next token).
2. Bulk phase: same mixture and LR plateau as init experiments, until ~85% of total budget.
3. Anneal phase: switch to anneal mixture, begin WSD decay window, run to end of budget.
4. Optional: checkpoint averaging within the anneal window for the released model.
5. Optional: new-token-density warmup in first 100–300M tokens of production (Item E1).

Total budget pending Q A2.

---

## 10. Decisions pending (from Fivos)

### Q A1. Capability targets (deferred)

> **Response:** _deferred — placeholder defaults used throughout where this would otherwise gate decisions_

### Q A2. Total token budget for CPT (post-init)

> **Response:** _pending_

### Q A3. Compute timeline / deadline

> **Response:** _pending_

### Q A4. Stakeholders / downstream consumers

Determines decontamination scope (which benchmarks need to be clean for external comparison).

> **Response:** _pending_

### Q A5. Colleague sign-off on shuffled-bulk + annealing

> **Response:** _pending_

### Q A6. Specific downstream tasks

> **Response:** _pending_

### Q A7. Team structure

> **Response:** _pending_

### Q B1. Outer target/replay split

Default 70/30.

> **Response:** _pending_

### Q B2. Code share

Default (b) ~4%.

> **Response:** _pending_

### Q B3. Anneal composition priority

Default (d) balanced, shifted by Q A1.

> **Response:** _pending_

### Q B4. Loss objective for init bakeoff

Default: NTP for bakeoff, Goldfish for production.

> **Response:** _pending_

### Q B5. Init experiment budget per variant

Default 2B.

> **Response:** _pending_

### Q B6. Adaptation work prioritization

§8 items relevant before kickoff (depending on status checks in §12):
- Possibly need attention: G1 (Goldfish hash, if Goldfish retained), K1 (decontamination scope)
- Status-confirmable in hours: H1, I1, I2, J1
- Conditional / deferrable: B (FOCUS for polytonic), E1 (density warmup)

> **Response:** _pending_

---

## 11. Lookups pending

### Q C1. Apertus pretraining peak LR

**Resolved:** 1.1e-4 (tech report Table 2). CPT peak 1.1e-5 to 2.2e-5; default 1.5e-5.

### Q C2. Apertus optimizer hyperparameters

AdEMAMix β1, β2, α, weight decay. Section 2.3 / B.4.

> **Response:** _pending_

### Q C3. Apertus per-language token shares

Section 3. Gating for §4.2 Tier weights.

> **Response:** _pending_

### Q C4. Apertus Goldfish loss configuration

Token masking rate, hash function. Relevant for Q B4 and Item G1.

> **Response:** _pending_

### Q C5. Apertus tokenizer config

**Partially resolved:** byte-level BPE from Mistral-Nemo tekken v3. Specific extension compatibility — V16.

### Q D1. Apertus Megatron-LM fork

Organization, repository, branch/commit; xIELU + AdEMAMix + QK-Norm + Goldfish support.

> **Response:** _pending_

### Q D2. FineWeb-2 Tier 3 language audit

Token counts for `lat_Latn`, `hye_Armn`, `kat_Geor`, `sqi_Latn`, `mkd_Cyrl`. Under ~100M → "preservation aspiration."

> **Response:** _pending_

### Q D3. Apertus intermediate checkpoints

Available on HF branches. Useful for annealing-as-quality-meter.

> **Response:** _pending_

---

## 12. Status checks (verifications)

This is a status table, not a TODO list. Each item is a question about current state; items already handled during prior work simply need confirmation. The "if not yet handled" notes describe what the work would be if the status check reveals it open.

### V1. Decontamination scope and status

**Question:** For benchmarks intended as clean measurement instruments (e.g., GreekMMLU public split, Belebele Greek source passages), have the *specific test items* been confirmed absent from training data?

**Scope clarification:** This is about the literal test items (MCQ stems and options; reading-comprehension source passages), not about removing all on-topic Greek material. Training on Greek academic prose, exam-prep material, and Kallipos theses is on-task and desirable.

**If not yet handled:** NeMo Curator's downstream task decontamination workflow runs item-level dedup against eval sets. Effort: ~1–3 days. Output: small fraction of training docs flagged and removed.

**Dependency:** Q A4 (which benchmarks are intended for external/comparative measurement determines what needs to be clean).

> **Status:** _unconfirmed_

### V2. Tokenizer extension forward pass

**Question:** Has the extended model been confirmed to produce vocab-148480 logits, with new token IDs routing correctly through both `embed_tokens` and `lm_head`, and a forward pass on Greek input completing without error?

**If not yet handled:** ~4 hours. Smallest viable test: tokenize a sample of Greek text with the extended tokenizer; run forward pass; confirm logit shape and absence of nan/inf.

**Note:** Likely already verified as part of tokenizer extension work.

> **Status:** _unconfirmed_

### V3. Dataloader state preservation

**Question:** Is Megatron-LM configured to preserve dataloader state in checkpoints (so resumption continues at next token, not from token 0)?

**If not yet handled:** Megatron-LM default behavior, but worth confirming via the relevant config flag. Test if uncertain: stop at 100M tokens, resume from checkpoint, confirm next batch index.

> **Status:** _unconfirmed_

### V4. Run-to-run variance baseline (gating for bakeoff selection)

**Question:** Has the full eval suite been run on unmodified Apertus-8B base, with bootstrap CIs over evaluation samples (~1000 resamples) to establish per-benchmark variance?

**If not yet handled:** Gating — this is what sets the "stability failure" thresholds in §5.6 hard gates. Without it, "more than X% regression" doesn't have a defined threshold. Effort: ~1 day of eval runs + analysis.

> **Status:** _unconfirmed_

### V5. Polytonic token concentration

**Question:** Has the §3.1 effective-exposure audit been run under the proposed Greek mixture (input/target occurrences per new token, Goldfish-masked target occurrences, register distribution, frequency quantiles, update norms)?

**If not yet handled:** Run before launching bakeoff. If polytonic tokens at p25 are getting <5k effective target occurrences at 1B tokens, upweight the polytonic-bearing bucket.

> **Status:** _unconfirmed_

### V6. Accent-normalized dedup re-verification

**Question:** Has the existing dedup against `fffoivos/apertus-c3-dedup-audit-dedup-...` been re-verified under accent-normalized hashing (to catch polytonic/monotonic variants of the same passage that may have escaped standard MinHash)?

**Note:** Standard dedup likely already done. The question is whether accent-normalization was part of the hashing pipeline.

> **Status:** _unconfirmed_

### V7. Replay dataset acquisition

**Question:** Are all replay sources (§4.4) accessible, downloaded, and tokenizable with the extended Apertus tokenizer at expected throughput?

**If not yet handled:** Datasets to acquire: FineWeb-2 filtered to Tier 1–3, FineWeb2-HQ, FineWeb-Edu Score-3, StarCoder v2 (or The Stack v2), FineMath-3+, OPUS Greek-English (and optionally Greek-Latin).

> **Status:** _unconfirmed_

### V8. Goldfish hash uniformity across new tokens

**Question:** If Goldfish is retained for production (Q B4), is the hash output uniform across the new 22,528 tokens?

**If not yet handled:** Tokenize a sample corpus with the extended tokenizer; count masked positions per new token; confirm uniform distribution. If non-uniform, the hash may need a vocab-aware adjustment.

**Dependency:** Q B4 and Q C4 (Goldfish configuration details).

> **Status:** _unconfirmed_

### V9. NFC normalization of training corpus

**Question:** Is the training corpus normalized to NFC form (so identical-looking polytonic text doesn't tokenize differently depending on encoding)?

**Note:** Almost certainly handled in the existing OCR/normalization pipeline — confirm explicitly.

> **Status:** _unconfirmed_

### V10. vLLM and SGLang compatibility

**Question:** Does the extended-vocab Apertus checkpoint load in both vLLM and SGLang with correct logit shapes?

**If not yet handled:** ~2–4 hours per system. Defer if production serving isn't in immediate scope.

> **Status:** _unconfirmed_

### V12. Cross-document attention masking

**Question:** Is the Megatron-LM config flag for cross-document attention masking enabled (matching Apertus's pretraining-time convention)?

**Note:** Default in Megatron when reading Apertus-format data; confirm flag.

> **Status:** _unconfirmed_

### V13. EoD token loss masking

**Question:** Does the CPT dataloader mask loss on EoD positions (matching Apertus's pretraining)?

**Note:** Default if reading Apertus-format data through Megatron; confirm.

> **Status:** _unconfirmed_

### V14. BoD/EoD special token preservation

**Question:** Are BoD and EoD special tokens preserved at their original IDs in the extended tokenizer config (new Greek tokens slotted in after, not before)?

**Note:** Likely already verified during extension work.

> **Status:** _unconfirmed_

### V15. xIELU trainable scalars in optimizer

**Question:** After vocab extension, are xIELU's per-layer trainable αp and αn parameters still in the optimizer's parameter list?

**Check:** `extended_model.num_parameters() == base_num_params + 184.5M` (no new xIELU scalars added; existing ones still trainable).

> **Status:** _unconfirmed_

### V16. Tokenizer byte-fallback for new polytonic tokens

**Question:** For a few new polytonic tokens, do they tokenize cleanly via the new vocab entry rather than collapsing to byte-fallback sequences for the same Unicode characters?

**Note:** Apertus's base tokenizer is byte-level BPE (Mistral-Nemo tekken v3); verifying no collision for polytonic specifically.

> **Status:** _unconfirmed_

---

## 13. Out of scope for v0.7

- Long-context extension beyond Apertus's native 65,536
- Post-training (SFT / DPO / QRPO)
- Multi-stage CPT with a dedicated polytonic specialization run
- Embedding-only warmup as a separate init arm
- LoRA / PEFT variants
- Synthetic data generation for underrepresented registers
- Construction of a polytonic generation eval benchmark for external release (internal version in §6.2 is in scope)
- **Token Distillation (bracketed for the current three-arm bakeoff).** High Apertus-specific adaptation cost (untied E/U handling; QK-Norm/xIELU validation; layer choice). Conditions to revisit: if ReTok-vs-Centroid bakeoff is inconclusive. A parallel-ready follow-up plan now lives in [`TOKEN_DISTILLATION_PLAN.md`](TOKEN_DISTILLATION_PLAN.md); it treats TD as a ReTok refinement, not as a mid-run replacement for the live arms. Engineering details preserved in v0.5 changelog and §8 Item A history.
- **QK-Norm-aware techniques generally.** Any future technique involving matching internal model states (not just outputs) needs to account for QK-Norm. Use model.forward() outputs rather than reimplemented attention math.

## 14. Changelog

- **v0.7 (2026-05-20):** Refactored §8 and §12 from prescriptive TODOs into status checks. Each verification item is now framed as a question about current state with conditional notes on what the work would be if status reveals it open. Added intro to §8 explicitly noting it's a coordination artifact, not a TODO list. §8 Item K1 (decontamination) substantially narrowed in scope — concern is *specific verbatim test items* in training data for benchmarks intended as clean measurement instruments, not blanket removal of on-topic Greek material. V1 reframed accordingly. Q A1 (capability targets) marked as deferred with placeholder defaults used elsewhere.

- **v0.6 (2026-05-20):** Bracketed Token Distillation. Replaced with Centroid init as third closed-form arm. Corrected CPT peak LR guidance to 1.1e-5 to 2.2e-5 based on Apertus 8B pretraining peak. Added V12–V16 for cross-document attention masking, EoD loss masking, special token preservation, xIELU scalars, and tokenizer byte-fallback.

- **v0.5 (2026-05-20):** Incorporated reviewer feedback: BPC/char-NLL primary metric; LM-head calibration; new-token integration diagnostic suite; stress-probe phase; language-ID drift hard gate; bootstrap CIs; custom Greek evals; expanded polytonic exposure metrics; NeMo Curator + GR-NLP-TOOLKIT + CLTK + Inspect AI tooling; NTP-for-bakeoff default flip; decontamination as gating. Added §8 consolidated adaptation checklist.

- **v0.4 (2026-05-20):** Restructured §9 into Decisions / Lookups / Verifications with response fields.
- **v0.3 (2026-05-20):** Added §4.2.1 convergence-based language selection framework.
- **v0.2 (2026-05-20):** Restructured as design-space document with published-recipe comparisons.
- **v0.1 (2026-05-20):** Initial draft as a specific recipe.
