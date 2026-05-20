# Apertus 8B Greek CPT — Plan v0.6

**Status:** Draft (Distillation bracketed; Centroid init added; architecture mismatches incorporated)
**Author:** Fivos (GlossAPI / Swiss AI Initiative)
**Date:** 2026-05-20
**Supersedes:** v0.5

This version brackets Token Distillation (adaptation cost too high relative to expected benefit at bakeoff scale) and replaces it with **Centroid init**, a third closed-form approach. Also incorporates findings from a systematic pass through Apertus's architecture: corrected CPT peak LR guidance (much lower than initially defaulted), QK-Norm interaction note, and additional verification items for cross-document attention, EoD loss masking, special tokens, xIELU scalars, and tokenizer byte-fallback.

---

## 1. Objective

Continue-pretrain Apertus 8B (base) on a curated Greek corpus to deepen the model's modern and polytonic Greek capabilities while preserving its multilingual, code, and reasoning performance.

Tokenizer extension is already complete: +17,408 modern Greek tokens and +5,120 ancient/polytonic Greek tokens (vocabulary 131,072 → 148,480). The new embedding rows add roughly **184.5M parameters** (22,528 × 4,096 × 2 for untied input + output) before optimizer state, or ~2.2% of the 8B base.

The model's intended downstream uses determine some of the design choices below; explicit capability targets are pending in §10 (Q A1).

## 2. Settled shape

These principles are not open for revision.

The curriculum has the shape of a shuffled-mixture bulk followed by an annealing tail. The bulk is one undifferentiated stream of corpora at their target weights from token 0; the anneal is the final ~10–20% of training, where the mixture shifts to highest-priority subsets and the learning rate decays.

Replay of non-Greek data is present from token 0, not deferred. ~70% Greek / ~30% non-Greek replay is the working outer split (refined in §10).

Old Apertus Greek pretraining data is not replayed.

Initialization is resolved by experiment: three **closed-form** variants are trained for 1.5–2B tokens each under identical conditions; the winning checkpoint becomes the start of production CPT. Token Distillation was bracketed in v0.6 (see §5 and §13).

Dataloader state is preserved at the checkpoint boundary.

The LR schedule is WSD (warmup–stable–decay), with the decay window aligned to the anneal data window.

## 3. Curriculum: design space

### 3.1 Greek mixture (within the ~70% Greek portion of training)

Three weighting strategies: *quality-weighted upsampling*, *register-balanced*, *goal-driven*. The choice is value judgment; Sailor2's RegMix methodology can resolve it empirically if compute permits a proxy sweep.

**Polytonic-token effective exposure metrics** (for V5 audit):

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

Three published patterns: *high-quality narrow* (Llama 3), *quality-curated broad* (OLMo Dolmino), *goal-targeted*. Replay continues through anneal at reduced share (e.g., 30% → 15%). Length 10–20% of total budget is the standard range.

### 3.3 LR schedule

WSD with brief re-warmup (1–2% of CPT tokens) from low LR to CPT peak, then plateau, then linear decay aligned with anneal.

**CPT peak LR (corrected in v0.6):** Apertus 8B's pretraining peak LR was **1.1e-4** (verified in tech report Table 2), which is much lower than the Llama-family 8B values typically quoted (~3e-4). The lower nominal LR reflects AdEMAMix's larger effective updates per nominal step compared to AdamW. Translating to the standard 1/5 to 1/10 CPT range:

**CPT peak: 1.1e-5 to 2.2e-5**, with 1.5e-5 as a reasonable default.

This is 2–5× lower than the v0.5 starting suggestion (3e-5 to 5e-5), which was wrongly anchored to Llama-style nominal LRs. Q C1 in §11 is now mostly answered.

LR/anneal co-design per "How LR Decay Wastes Your Best Data" (2025): either WSD with explicit plateau over the anneal window (decay only at the very end), or weight averaging across anneal checkpoints à la Llama 3.

> **★ Apertus adaptation note (Goldfish hash interaction with extended vocabulary):** Apertus's Goldfish loss masks tokens via a hash function. If the hash is vocab-aware, extending the vocab from 131,072 to 148,480 changes which positions get masked per token. Verify uniform masking distribution across new tokens before training. See §8 item G1 and V8.

> **★ Apertus adaptation note (Megatron blended dataset for new-token-density warmup):** If implementing the optional new-token-density warmup, Megatron's standard blended-dataset config doesn't natively support density-based sampling. Preprocess to bucket sequences by density and weight at bucket level. See §8 item E1.

### 3.4 References (curriculum)

- Pythia curriculum study (2026); Llama 3 paper (Grattafiori et al. 2024); OLMo 2/3 (Allen AI); "How LR Decay Wastes Your Best Data" (2025); Ibrahim et al. (2024); Mid-Training survey (Oct 2025).

## 4. Replay: design space

### 4.1 Outer split (target/replay ratio)

Published range 44–80% target / 56–20% replay (Sailor2, AMD Finnish, SEA-LION v3, EstLLM, Racka). For Apertus-Greek deepening regime, middle of range — 65/35 to 75/25 defensible. Working default 70/30; final value Q B1.

### 4.2 Within-replay composition: which languages

Three meta-patterns: token-share-weighted, curated-subset uniform, skill/use-based. "Preserve most of 1800 languages" is not achievable.

#### 4.2.1 Convergence-based selection framework

Four lenses: (1) Geographic (Balkan/E. Mediterranean), (2) Western EU / Swiss institutional, (3) Historical connection to Greece (Latin, Slavic-Orthodox, Arabic transmission, Hebrew, Armenian/Georgian Byzantine, Persian Hellenistic), (4) Script-system and major-corpus coverage (logographic preservation; matches Apertus's distribution).

**Convergence table:**

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
| Albanian | ✓ | | partial (Arvanites, Ottoman period) | | Geographic priority |
| Macedonian | ✓ | | ✓ (Slavic, Orthodox) | | Small data volume |
| Spanish | | ✓ (large EU) | | ✓ | Convenient large language |
| Hebrew | ✓ (E. Med) | | ✓ (Septuagint, Sephardic) | | |
| Armenian | | | ✓ (Byzantine contact, early Christianity) | | Data-medium |
| Georgian | | | ✓ (Orthodox, Byzantine) | | Data-low |
| Ukrainian | | | ✓ (Orthodox, Byzantine via Kyivan Rus) | | |
| Chinese | | | | ✓ | Logographic; major corpus |
| Japanese | | | | ✓ | Different script |
| Persian / Farsi | | | ✓ (Hellenistic, Sufi-Greek contact) | ✓ | |
| Portuguese | | ✓ | | ✓ | Convenient large language |
| Dutch | | ✓ | | | EU coverage |
| Polish | | ✓ | | | EU coverage |

**Proposed tier structure (pending Apertus §3 verification for weights, Q C3):**

- *Tier 1, ~40–50% of replay slice:* English, French, German, Italian, Spanish, Russian, Arabic, Chinese.
- *Tier 2, ~35–45% of slice:* Turkish, Bulgarian, Serbian, Romanian, Hebrew, Portuguese, Polish, Dutch, Persian, Ukrainian, Japanese.
- *Tier 3, ~10–15% of slice (floor weight):* Latin, Armenian, Georgian, Albanian, Macedonian.

**Honest note about Tier 3:** Replay can only preserve what the base has. For Tier 3 languages where Apertus had near-zero exposure, the small replay share is "doesn't hurt anything" rather than "actively maintains a capability."

### 4.3 Code share

Published range 0% (Sailor2) to 20% (SEA-LION v3). Position depends on downstream consumer priorities. Q B2 in §10.

### 4.4 Replay sources (HuggingFace candidates)

| Component | Candidate dataset | Notes |
|---|---|---|
| Multilingual non-English (excl. Greek) | `HuggingFaceFW/fineweb-2` filtered to Tier 1–3 languages, OR `epfml/FineWeb2-HQ` | Filter out `ell_Grek` |
| English high-quality | `HuggingFaceFW/fineweb-edu` (Score-3 config) | Match Apertus's later-stage filtering |
| Code | `bigcode/starcoderdata` or `bigcode/the-stack-v2` | Match Apertus's StarCoder source |
| Math | `HuggingFaceTB/finemath` (`finemath-3plus`) | Apertus used FineMath CC subset |
| Cross-lingual parallel (optional) | OPUS Greek-English; OPUS Greek-Latin (`grc-lat`, `ell-lat`) | 1% parallel adds cross-lingual transfer; Greek-Latin small but uniquely valuable for classical philology |

### 4.5 References (replay)

Sailor / Sailor2 (curse of multilinguality); Ibrahim et al. (2024); Conneau et al. (2020); EstLLM (2026), Racka, SEA-LION v3, AMD Finnish playbook; "Reuse, Don't Retrain" (Parmar et al. 2024); RegMix (Liu et al. 2024); FineWeb2-HQ (EPFL ML 2025).

## 5. Initialization experiments

Three **closed-form** init methods evaluated under identical conditions for 1.5–2B tokens each. Token Distillation has been bracketed (see §13) due to high Apertus-specific adaptation cost.

| Variant | Description | Reference |
|---|---|---|
| Vanilla | Original Apertus tokenizer (vocab 131,072). No vocab extension. | Yuan et al. 2024, *LLaMA Beyond English*. Load-bearing baseline. |
| ReTok | Extended vocab (148,480). New row `E[T]` = norm-matched mean of base-tokenizer subpieces of `T`'s surface form. Same for `U[T]`. Uses per-token-specific subpiece information. | EEVE-Korean, Chinese LLaMA, FOCUS (Dobler & de Melo 2023), Minixhofer et al. 2022. |
| Centroid | Extended vocab (148,480). New row `E[T]` = script-conditional centroid + Gaussian noise. Uses script-level distributional prior rather than per-token subpiece info. Same procedure for `U[T]`. | Closely related to mean-of-existing-vocab strategies in extension literature; the per-script variant is mostly novel. |

**Centroid init procedure:**

1. **Identify Greek tokens in the base vocab.** For each base-vocab token t (IDs 0..131,071), decode to surface form; flag as *modern Greek* if it contains characters in U+0370–U+03FF, as *polytonic* if it contains characters in U+1F00–U+1FFF. A token may be in both sets.
2. **Compute per-script centroids in E and U.** `E_centroid_modern = mean(E[t] for t in modern_set)`, similarly for polytonic; same in U. Also compute the std (dispersion) of each set around its centroid.
3. **For each new token T**, classify by script (modern, polytonic, both). For each script tag, look up the centroid; if T has both tags, average the centroids. Fallback: if the polytonic centroid is unreliable (computed from fewer than ~50 base tokens), fall back to the modern centroid for polytonic new tokens.
4. **Initialize:** `E[T] = E_centroid + ε`, where `ε ~ Normal(0, σ_E)` with σ_E = the std observed in the centroid computation (matches existing-token dispersion). Same in U.
5. **Norm-match:** apply the Phase A per-group norm targets (5.05 / 3.80) to the resulting rows.

This is essentially a "warm prior" — new tokens start near the existing Greek-token cloud, with dispersion that lets them differentiate during training. Unlike ReTok, it doesn't use the new token's specific subpiece decomposition; unlike Distillation, no gradient descent is required at init time. Cost: <1 minute of CPU work.

> **★ Apertus adaptation note (ReTok + FOCUS auxiliary vectors):** FOCUS uses fastText embeddings as the auxiliary similarity space for subpiece selection. fastText embeddings don't typically exist at the polytonic level. Options: (a) use monotonic-folded versions as proxy (loses information), (b) skip FOCUS and use pure subpiece mean. Default is (b). See §8 item B.

### 5.1 Primary intrinsic metrics: tokenizer-fair signals

Per-token perplexity is **not comparable across Vanilla and extended-tokenizer variants** because tokenizers represent different amounts of Greek text per token.

Primary tokenizer-agnostic metrics:

| Metric | Why it matters |
|---|---|
| **Bits-per-byte (BPC)** | Cleanest cross-tokenizer comparison |
| **NLL per Unicode character** | More interpretable for Greek/polytonic |
| **NLL per word** on fixed raw validation text | Human-facing language metric |
| **Tokens-per-word, chars-per-token, compression ratio** | Quantifies tokenizer efficiency |
| **STRR (subword-token-retention rate)** | Whole-word preservation |
| **Throughput: Greek words/sec and non-Greek words/sec** | Net effect of extended softmax vs Greek token savings |

> **★ Apertus adaptation note (BPC computation for polytonic Greek):** UTF-8 polytonic Greek uses 2–3 bytes per character due to combining diacriticals. BPC numbers differ by unit choice (UTF-8 bytes, Unicode code points, grapheme clusters). Use NFC code points; document choice. See §8 item H1.

Per-token-group PPL on the 17k modern and 5k polytonic new tokens remains a useful **secondary** signal *within* ReTok vs Centroid comparison (both have extended vocab), but cannot compare Vanilla.

### 5.2 Init procedure for E and U

All three arms apply their respective procedures to both `E` and `U` matrices independently:

- Vanilla: nothing to do (no new rows).
- ReTok: norm-matched subpiece-mean init applied to both `E` and `U` separately.
- Centroid: script-conditional centroid + noise applied to both `E` and `U` separately.

This is simpler than the v0.5 plan, which had a separate LM-head calibration phase for the (now-bracketed) Distillation arm. With both extension arms being closed-form, no special U-row procedure is needed beyond applying the same closed-form init to U as to E.

### 5.3 New-token integration diagnostic suite

Before launching the main CPT, and at every bakeoff checkpoint:

| Diagnostic | Failure it catches |
|---|---|
| Rank of correct new token in next-token logits | New token invisible |
| Aggregate probability mass on new Greek tokens | Under- or over-emitted |
| New-token entropy by register | Polytonic rows collapsed or avoided |
| Top-k substitutions between new token and old subpieces | Model still prefers old segmentation |
| Greedy generation new-token utilization rate | New rows exist but are behaviorally dead |
| Embedding L2-norm distribution for new vs existing | Degenerate-subspace collapse |
| Cosine similarity matrix among new tokens, effective rank | Same-direction collapse |

### 5.4 Token budget per variant

1.5–2B tokens per variant. Below 1.5B, init-method gaps may be within noise (per Park et al. Oct 2025 instability). The 184.5M new parameter overhead means each extended-vocab run costs ~2.2% more wall-clock per step than Vanilla. Total bakeoff: 3 arms × 1.5–2B = 4.5–6B tokens.

### 5.5 Phase 0 — Optional stress probes

Before the production-faithful bakeoff, optionally run 50–200M-token stress probes under extreme conditions (high new-token density, OCR-noisy polytonic, code-switching, replay-heavy slices) for bug discovery. **Not resumable to production.**

### 5.6 Selection criteria

**Hard gates.** A candidate fails if:

- English/core retention drops more than threshold (set per V4)
- Code retention drops more than threshold, if code is a release requirement
- New-token rows show collapse (cosine clustering, near-zero usage)
- Polytonic text gets worse than base on character-normalized loss
- Throughput/memory hit is disproportionate to Greek compression gain
- **Language-ID drift:** model over-emits Greek in non-Greek prompts

**Selection score (weighted, applied to non-failing candidates).** Indicative — final weights tied to Q A1:

| Component | Weight |
|---|---|
| Greek held-out BPC/char-NLL across registers | 30–40% |
| GreekMMLU + Belebele Greek + Meltemi suite | 25–35% |
| Polytonic custom eval (§6.2) | 10–15% |
| Retention suite (§6.3) | 15–25% |
| Efficiency: Greek words/sec, memory, inference token savings | 5–10% |

Selection uses **windowed averages** across the last 3–5 checkpoints in the 80–100% range of each variant's budget, with **bootstrap CIs over evaluation samples**.

### 5.7 What the three-arm bakeoff tests

Now that all three arms are closed-form, the comparison is clean:

- **Vanilla vs (ReTok or Centroid):** does vocab extension justify its parameter overhead and complexity? Tests the *LLaMA Beyond English* finding.
- **ReTok vs Centroid:** does per-token subpiece-specific information beat a script-level distributional prior? If ReTok wins, subpiece info matters and is worth computing. If Centroid wins or ties, simpler is better and the recipe is genuinely cheaper. Either way the result is informative.

### 5.8 Caveat on polytonic signal

At 2B tokens, polytonic embeddings may still be undertrained because polytonic tokens are sparse. Expect smaller gaps between ReTok and Centroid on polytonic than on modern Greek. Honest comparison is on modern Greek tokens.

## 6. Evaluation: cadence, benchmarks, metrics

### 6.1 Evaluation cadence and statistical methodology

Three principles: *evaluate frequently enough to catch regressions early* (every 100M tokens in init experiments, every 500M in production); *account for downstream task instability* via checkpoint-window averaging (Park et al. Oct 2025); *use bootstrap CIs over evaluation samples* (most benchmark runs are deterministic, so "run 3×" doesn't establish variance).

For init bakeoff selection: average scores across checkpoints at 1.6B, 1.7B, 1.8B, 1.9B, 2.0B; report mean + bootstrap CI; select by composite score with hard retention gates.

**Distinguish trajectory metrics from selection metrics.** Loss, per-bucket PPL, BPC, embedding-norm distributions, the §5.3 diagnostic suite — read at every checkpoint. Downstream benchmarks — read less frequently and prefer windowed averages.

**Checkpoint averaging scope:** Use averaging WITHIN a single init's checkpoints for both selection and final released model. Do NOT average across different init experiments. For training continuation (resuming production from the bakeoff winner), use the **raw** checkpoint, not the averaged one — the averaged model has no corresponding optimizer state.

### 6.2 Greek benchmarks

**Core benchmarks:** GreekMMLU (`dascim/GreekMMLU` — 21,805 native-sourced MCQs, the right centerpiece); Belebele (`facebook/belebele` ell_Grek subset); Meltemi eval suite (backward compat with Greek-LLM literature); held-out per-register PPL on splits from each bucket.

**Custom evals to construct:**

| Eval | What it measures | Construction notes |
|---|---|---|
| Polytonic continuation | Can the model stay in polytonic Greek vs collapse to monotonic | ~200 polytonic prompts; measure polytonic-form output |
| Accent/diacritic accuracy | Character-level correctness of breathing marks, oxia/varia/perispomeni, iota subscript, diaeresis | Polytonic-aware comparator with `unicodedata.normalize` |
| Modern Greek morphology minimal pairs | Case, number, gender, tense, agreement | ~500 pairs from CLTK/GR-NLP-TOOLKIT outputs |
| Greeklish → Greek robustness | Real users type Greeklish | GR-NLP-TOOLKIT Greeklish detector for input |
| Greek-English code-switching | Real assistant usage and technical Greek | ~200 code-switched prompts; measure coherence |
| Legal/EU style eval | Does Eurlex upweighting help without making the model wooden | Held-out Eurlex prompts; accuracy + register |
| Register preservation | Demotic, katharevousa, ecclesiastical, classical, academic, journalistic | Per-register prompts with register-classifier eval |
| **Language-ID drift** | English/French/German prompts shouldn't get Greek responses | Cross-lingual prompts; language-ID classifier on completions |

Construction is ~1–2 weeks. Polytonic continuation and language-ID drift are highest priority.

**MultiLoKo** (Schmidt et al. 2025): multilingual local knowledge across 31 languages, useful for cultural eval.

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

Threshold for "regression" set after V4 baseline runs.

### 6.4 Stability diagnostics

Per checkpoint: training/validation loss per bucket; BPC trajectory per register; full §5.3 diagnostic suite; update norms vs weight norms for new `E` and `U` rows.

### 6.5 Evaluation tooling

- `EleutherAI/lm-evaluation-harness` — primary for standard benchmarks; sample-level logging enabled
- `huggingface/lighteval` — alternative; used by Meltemi
- **Inspect AI** (`inspect.aisi.org.uk`) — custom open-ended evals (polytonic continuation, language-ID drift, accent accuracy, register preservation)

Pick lm-eval-harness or lighteval as primary, use Inspect AI for custom.

### 6.6 References (evaluation)

Park et al. (Oct 2025) downstream task instability; GreekMMLU paper (Zhang et al. 2026); Meltemi paper (Voukoutis et al. 2024); MultiLoKo (Schmidt et al. 2025); Inspect AI documentation.

## 7. Tooling and repositories

### 7.1 Training framework

**Primary: Megatron-LM.** Apertus was trained on a Swiss AI fork supporting xIELU + AdEMAMix; verify branch/commit (Q D1).

- `NVIDIA/Megatron-LM` upstream; ROCm fork (per AMD Finnish playbook); Swiss AI fork (verify).
- Alternative: TRL + Accelerate (`swiss-ai/apertus-finetuning-recipes`) for smaller experiments.

### 7.2 Reference repositories

- `swiss-ai/apertus-tech-report` — hyperparameters, data pipeline, xIELU, AdEMAMix, Goldfish
- `swiss-ai/apertus-finetuning-recipes` — eval harness scaffolding
- AMD ROCm CPT playbook — closest published analog
- Sailor2 cookbook — most comprehensive multilingual-CPT methodology
- EstLLM repository — directly comparable, also done on Apertus 8B

### 7.3 Data tooling

- `huggingface/datasets` — `interleave_datasets` for mixing
- Megatron `tools/preprocess_data.py` — tokenization to mmap format
- `datatrove` (HuggingFace) — large-scale text processing
- **`NVIDIA-NeMo/Curator`** — GPU dedup; **downstream task decontamination** (used for V1)
- `text-dedup` library — MinHash/LSH with Greek Unicode normalization
- **`nlpaueb/gr-nlp-toolkit`** (GR-NLP-TOOLKIT) — modern Greek POS, morphology, NER, Greeklish detection
- **CLTK** (`cltk.org`) — pre-modern Greek lemmatization and analysis

Classical sources as anchors: First1KGreek (`OpenGreekAndLatin/First1KGreek`, CC-BY-SA-4.0), Perseus Digital Library (CC-BY-SA-4.0).

### 7.4 Evaluation tooling

`EleutherAI/lm-evaluation-harness`, `huggingface/lighteval`, **Inspect AI**. GreekMMLU custom task config per `dascim/GreekMMLU` card.

### 7.5 Serving stack

vLLM and SGLang officially supported by Apertus. Both must be tested with vocab 148,480 (V10).

---

## 8. Apertus-specific adaptation requirements

Consolidated checklist of places where standard recipes need Apertus-specific adaptation. Grouped by engineering effort. Reduced from v0.5 after bracketing Token Distillation (Items A and C removed).

### Moderate adaptation (days)

**Item B — ReTok with FOCUS-style auxiliary vectors for polytonic.** (§5.) FOCUS uses fastText embeddings; fastText doesn't exist for polytonic. Options: (a) monotonic-folded proxy, (b) skip FOCUS for polytonic (current default). *Effort: ~2–3 days for (a); none for (b).*

**Item E1 — Megatron blended dataset for new-token-density warmup.** (§3.3.) If implementing the optional density warmup, Megatron's standard config doesn't support density-based sampling. Preprocess to bucket sequences by density. *Effort: ~2–4 days.*

**Item G1 — Goldfish hash verification with extended vocabulary.** (§3.3.) If Goldfish hash is vocab-aware, extension changes masking distribution per token. Verify uniform masking; adjust hash if needed. *Effort: ~1 day for verification; more if fix needed.*

### Verification-only (hours)

**Item F1 — Checkpoint averaging scope.** (§6.1.) Average within a single init's checkpoints only; never across different init experiments. Use averaged checkpoints for measurement/release, raw checkpoints for training continuation. *Effort: documentation only.*

**Item H1 — BPC computation unit choice for Greek.** (§5.1.) Choose NFC code points; document. *Effort: utility function + documentation.*

**Item I1 — NFC normalization for training text.** Mixed NFC/NFD forms tokenize differently. Normalize all training text to NFC. *Effort: ~2 hours.*

**Item I2 — HuggingFace `resize_token_embeddings` with untied E and U.** (§5.) Resizes both `embed_tokens` and `lm_head` separately. Test smallest possible extension. Verify HF↔Megatron roundtrip preserves both. *Effort: ~4 hours.*

**Item J1 — vLLM/SGLang compatibility with vocab 148,480.** (§7.5.) Non-power-of-2 vocab may break kernel assumptions. Smoke test. *Effort: ~2–4 hours per system.*

### Decontamination as gating

**Item K1 — Decontamination as V1 (gating).** (§12.) Decontaminate training data against GreekMMLU public split, Meltemi tasks, Belebele Greek, MultiLoKo, HumanEval, GSM8K, MMLU, ARC, HellaSwag. Use multiple Greek normalization views (raw, NFC, accent-normalized, monotonic-folded for detection only). NeMo Curator workflow. *Effort: ~3–5 days for full pipeline.*

---

## 9. Production run shape

After init experiments select a winner:

1. Continue from winning checkpoint (dataloader resumes at next token).
2. Bulk phase: same mixture and LR plateau as init experiments, until ~85% of total budget.
3. Anneal phase: switch to anneal mixture; begin WSD decay window; run to end of budget.
4. Optional: checkpoint averaging WITHIN the anneal window for the released model.
5. Optional: new-token-density warmup in first 100–300M tokens of production (Item E1).

Total budget pending Q A2.

---

## 10. Decisions pending (from Fivos)

### Q A1. Capability targets

Options: (a) balanced register-aware Greek assistant, (b) academic/digital-humanities Greek, (c) polytonic-strong classical generator, (d) modern Greek conversational, (e) other.

> **Response:** _pending_

### Q A2. Total token budget for CPT (post-init)

Working range 10–20B.

> **Response:** _pending_

### Q A3. Compute timeline / deadline

> **Response:** _pending_

### Q A4. Stakeholders / downstream consumers

> **Response:** _pending_

### Q A5. Colleague sign-off on shuffled-bulk + annealing

> **Response:** _pending_

### Q A6. Specific downstream tasks

Translation, OCR post-correction (Anemi?), summarization, dialect detection, polytonic generation as first-class?

> **Response:** _pending_

### Q A7. Team structure

> **Response:** _pending_

### Q B1. Outer target/replay split

65/35 to 75/25 range. Default 70/30.

> **Response:** _pending_

### Q B2. Code share

(a) 0%, (b) ~4%, (c) 15–20%. Default (b).

> **Response:** _pending_

### Q B3. Anneal composition priority

(a) Academic/clean, (b) Literary, (c) Classical/polytonic, (d) Balanced. Default (d), shifted by Q A1.

> **Response:** _pending_

### Q B4. Loss objective for init bakeoff

Default: NTP for bakeoff (cleaner learning signal), Goldfish for production (consistency, memorization suppression).

> **Response:** _pending_

### Q B5. Init experiment budget per variant

1.5B or 2B. Default 2B (cleaner ReTok-vs-Centroid discrimination; 6B total bakeoff compute).

> **Response:** _pending_

### Q B6. Adaptation work prioritization

§8 adaptation items required before kickoff vs deferrable. Simpler now that Distillation is bracketed.

- Must-have: G1 (Goldfish hash), H1 (BPC units), I1 (NFC normalization), I2 (resize_token_embeddings), K1 (decontamination)
- Optional / deferrable: B (FOCUS for polytonic), E1 (density warmup), J1 (serving compatibility)

> **Response:** _pending_

---

## 11. Lookups pending (from Apertus tech report and other sources)

### Q C1. Apertus pretraining peak LR

**Answered in v0.6:** Apertus 8B peak LR was 1.1e-4 (Apertus tech report Table 2). CPT peak should be 1.1e-5 to 2.2e-5; default 1.5e-5. *Verify against original tech report Table 2 for confirmation.*

### Q C2. Apertus optimizer hyperparameters

AdEMAMix β1, β2, α (EMA decay), weight decay. Section 2.3 / B.4.

> **Response:** _pending_

### Q C3. Apertus per-language token shares

Section 3. Top ~30 languages by token share. **Gating for §4.2.1 Tier weights.**

> **Response:** _pending_

### Q C4. Apertus Goldfish loss configuration

Token masking rate, hash function. Relevant for Q B4 and Item G1.

> **Response:** _pending_

### Q C5. Apertus tokenizer config

Base tokenizer is byte-level BPE from Mistral-Nemo `tekken` v3 (confirmed in v0.6 search). Verify the extension you've built is compatible with this base — specifically byte-fallback behavior for new Greek tokens (V16).

> **Response:** _pending (mostly resolved by tech report)_

### Q D1. Apertus Megatron-LM fork

Organization, repository, branch/commit. Compatibility with xIELU + AdEMAMix + QK-Norm + Goldfish + AdEMAMix batch-size scaling.

> **Response:** _pending_

### Q D2. FineWeb-2 Tier 3 language audit

Token counts for `lat_Latn`, `hye_Armn`, `kat_Geor`, `sqi_Latn`, `mkd_Cyrl`. Under ~100M → "preservation aspiration."

> **Response:** _pending_

### Q D3. Apertus intermediate checkpoints

Available on different HF branches. Useful for annealing-as-quality-meter on Greek subcorpora à la Llama 3.

> **Response:** _pending_

---

## 12. Verifications pending (action items)

- [ ] **V1. Decontamination (gating).** All training data vs all eval sets. NeMo Curator. Run before any training. Highest priority. Per Item K1.
- [ ] **V2. Tokenizer extension forward pass test.** Per Item I2.
- [ ] **V3. Dataloader state preservation probe.** Stop/resume at 100M; confirm next batch is 100M+1.
- [ ] **V4. Run-to-run variance baseline + bootstrap CI calibration.** Full eval suite on Apertus-8B base. Bootstrap variance (1000 resamples). Sets stability-failure thresholds for §5.6 hard gates.
- [ ] **V5. Polytonic token concentration audit.** Per §3.1 expanded metrics including Goldfish-masked target occurrences.
- [ ] **V6. Dedup re-verification with accent-normalized hashing.** Polytonic vs monotonic of same passage.
- [ ] **V7. Replay dataset acquisition test.** Verify all replay sources accessible/tokenizable with extended Apertus tokenizer.
- [ ] **V8. Goldfish hash uniformity check.** Per Item G1.
- [ ] **V9. NFC normalization probe.** Per Item I1.
- [ ] **V10. vLLM and SGLang compatibility smoke test.** Per Item J1.
- [ ] **V11.** *(removed in v0.6 — LM-head calibration was specific to Distillation arm)*
- [ ] **V12. Cross-document attention masking preserved in CPT dataloader.** Apertus trained with strict document separation. Megatron handles this with the right config flag; verify it's set. *Effort: ~1 hour.*
- [ ] **V13. EoD token loss masking preserved.** Apertus masks loss on EoD positions during training. Verify CPT dataloader does the same. *Effort: ~1 hour.*
- [ ] **V14. BoD/EoD special tokens preserved through tokenizer extension.** Special tokens have fixed IDs in base; extension shouldn't have moved them. Verify config: BoD/EoD at original IDs, new Greek tokens slotted in after. *Effort: ~30 minutes.*
- [ ] **V15. xIELU trainable scalars in optimizer parameter list.** xIELU has trainable αp and αn per layer. After resize, confirm these are still in optimizer's parameter list. Parameter-count diff: extended_model.num_parameters() should = base_num_params + 184.5M + 0 (no new xIELU scalars added). *Effort: ~30 minutes.*
- [ ] **V16. Tokenizer byte-fallback sanity check for new tokens.** Tokenizer is byte-level BPE from Mistral-Nemo tekken v3. Verify a few new polytonic tokens don't collide with byte-fallback sequences for the same characters. *Effort: ~1 hour.*

---

## 13. Out of scope for v0.6

- Long-context extension beyond Apertus's native 65,536.
- Post-training (SFT / DPO / QRPO).
- Multi-stage CPT with a dedicated polytonic specialization run.
- Embedding-only warmup as a separate init arm.
- LoRA / PEFT variants.
- Synthetic data generation for underrepresented registers.
- Construction of a polytonic generation eval benchmark for external release (internal version in §6.2 is in scope).
- **Token Distillation (bracketed in v0.6).** Adaptation cost too high for Apertus-specific constraints: untied E/U requires separate LM-head calibration (~1–2 weeks); QK-Norm interaction requires careful handling of how attention targets are extracted; xIELU's gradient characteristics require validation; layer-choice sweep for the attention target adds compute; context selection across registers adds methodology work. Total: ~3 weeks vs ~1 week naive port, with uncertain benefit at the 1.5–2B bakeoff scale. **Conditions to revisit:** if the ReTok-vs-Centroid bakeoff is inconclusive (no clear winner, both significantly worse than Vanilla on retention, or both showing embedding collapse), Token Distillation may be worth implementing as a second-pass arm. If kept on the radar for later, the engineering effort details are preserved in the v0.5 changelog and §8 Item A.
- **QK-Norm-aware techniques generally.** Any future technique that involves matching internal model states (not just outputs) needs to account for QK-Norm in Apertus's attention layers. Use model.forward() outputs rather than reimplemented attention math.

## 14. Changelog

- **v0.6 (2026-05-20):** Bracketed Token Distillation (moved to §13 out of scope with conditions to revisit). Replaced with **Centroid init** as third closed-form arm. §5 rewritten: detailed Centroid procedure (per-script centroid identified by Unicode block membership, dispersion-matched Gaussian noise, applied symmetrically to E and U). §5.2 simplified (no separate LM-head calibration phase needed; both extension arms apply same closed-form to E and U). §5.7 added: explicit framing of what the three-arm bakeoff tests. Architecture findings incorporated:
  - §3.3: Corrected CPT peak LR guidance to 1.1e-5 to 2.2e-5 based on Apertus 8B pretraining peak of 1.1e-4 (from tech report Table 2). Q C1 mostly resolved.
  - §8: Removed Items A (Token Distillation adaptation) and C (LM-head calibration) per Distillation bracketing.
  - §11 Q C5: Tokenizer base identified as Mistral-Nemo tekken v3.
  - §12 verifications: Removed V11 (LM-head calibration prototype). Added V12 (cross-document attention masking), V13 (EoD loss masking), V14 (BoD/EoD token preservation), V15 (xIELU trainable scalars in optimizer), V16 (tokenizer byte-fallback sanity check).
  - §13: Explicit note about QK-Norm-aware techniques as general future consideration.

- **v0.5 (2026-05-20):** Incorporated external reviewer feedback: BPC/char-NLL replaces per-token PPL; LM-head calibration procedure; new-token integration diagnostic suite; stress-probe phase; language-ID drift as hard gate; bootstrap CIs; custom Greek evals; expanded polytonic exposure metrics; NeMo Curator + FineWeb2-HQ + GR-NLP-TOOLKIT + CLTK + Inspect AI tooling; NTP-for-bakeoff default flip; decontamination as V1 gating. Added §8 consolidated adaptation checklist.
- **v0.4 (2026-05-20):** Restructured §9 into Decisions / Lookups / Verifications with response fields.
- **v0.3 (2026-05-20):** Added §4.2.1 convergence-based language selection framework.
- **v0.2 (2026-05-20):** Restructured as design-space document with published-recipe comparisons.
- **v0.1 (2026-05-20):** Initial draft as a specific recipe.