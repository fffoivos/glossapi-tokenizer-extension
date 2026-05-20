# Greek Tokenizer Extension for Apertus — Working Document

*Status: Draft v0.12 — Open for iteration*

*Canonical sources for decisions referenced below:*
- *C3 arm / corpus → `docs/C3_CONVERGENCE.md`*
- *C3 source datasets, per-source rows + provenance → `docs/C3_TRAINING_DATASETS.md`*
- *Cutoff sweep / clean held-out slices → `docs/C3_CUTOFF_REPORT.md`*
- *Global decisions / 128-alignment / hard constraints → `docs/GLOBAL_DECISIONS.md`*
- *Existing Greek token IDs → `tokenizer_analysis/inspection/base/greek_tokens/`*
- *Phase A norm diagnostic → `runs/apertus_greek_diagnostic_20260511_v2/`*
- *Phase B v4 behavioral NLL (diversified, modern-Greek only) → `runs/apertus_greek_phase_b_v4_20260512/`*
- *Apertus Greek pretraining share → `APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`*

## 1. Goal

Extend the Apertus base model to handle Greek with native-quality fluency, depth, and cultural-conceptual specificity, by:

1. Extending the Apertus tokenizer (1,494 existing Greek tokens) via continued BPE training on authentic Greek corpora — primarily to improve compression efficiency.
2. Determining empirically (via the three-arm experiment in §5) whether and how new-token embedding initialization affects continued-pretraining quality.
3. Performing continued pretraining on academic Greek texts (and broader Greek corpora) to instantiate Greek-distinctive knowledge in the model.

The end state is an Apertus derivative with substantially improved Greek bytes-per-token efficiency, native Greek register coverage (including academic, philosophical, and polytonic Katharevousa material), and minimal regression in the base model's other capabilities.

**Project framing and hard constraint on multilingual preservation.**

This project does not currently have a specific deployment target. The goal is broadly "improve Greek" — to give Apertus stronger native-Greek capability across registers, without pre-committing to a particular downstream use case (academic, conversational, OCR post-processing, translation pipeline, etc.).

This under-determined deployment goal has one **hard constraint** attached:

> **Preserve Apertus's multilingual character.**

Apertus's defining property is that it is a fully-open, fully-compliant, multilingual model with native support for ~1,800 languages. The project will not ship a derivative that has bought Greek improvements at the cost of meaningfully degrading other languages. This propagates throughout the doc:

- §4 Constraint 1 (no translation-mediated learning) is partly motivated by the same concern — methods that anchor Greek to English bilingual signal can corrupt non-Greek capabilities.
- §8.5 forgetting prophylaxis: non-Greek replay ratio is not optional; the question is only how aggressive.
- §10 Q8a (gating criteria) is the operationalization of this constraint — perplexity gates on multiple well-pretrained languages disqualify arms that fail multilingual preservation regardless of Greek-side gains.
- §10 Q8d default-to-Vanilla on no-clear-winner: better to ship no extension than to ship something that won on broadly-Greek metrics at the cost of multilingual breadth.

Because the deployment is under-specified but the multilingual hard constraint is firm, the doc leans toward conservative defaults across the board — wide preservation gates, multi-register evaluation, default-to-simpler-when-tied. These are principled responses to the asymmetry between what's required (multilingual breadth) and what's optimized (Greek depth), not analytical conclusions independent of context.

**Note on empirical framing:** Phase A (§2.6) showed the existing Greek tokens are well-trained on average. Phase B v4 (§2.7) provides the behavioral cross-check: Apertus predicts modern Greek (both web and academic) at median NLL ≈ 0.95 — roughly 3× lower than its own English-on-English NLL (3.03). Ancient / polytonic / Katharevousa Greek is genuinely harder, but it's outside the C3 BPE training corpus and the deployment register. The case for extension is therefore *compression economy* (fewer tokens per Greek character → lower inference cost), not *lifting prediction quality*. Whether the extension actually pays off, and what initialization method works best, are still empirical questions addressed by the three parallel experiments in §5. We do not pre-commit to a method.

## 2. Project Context

### 2.1 Base model: Apertus-8B-2509 (verified config)

Model identifier: `swiss-ai/Apertus-8B-2509`. The swiss-ai org uses dated release names; there is no undated `Apertus-8B` repo.

| Property | Value | Notes |
|---|---|---|
| vocab_size | 131,072 (= 2¹⁷) | Cleanly divisible by 128, 256, 512 |
| hidden_size | 4,096 | Embedding dimension |
| intermediate_size | 21,504 | MLP dim |
| num_hidden_layers | 32 | |
| num_attention_heads | 32 | |
| num_key_value_heads | 8 | GQA 4:1 |
| **tie_word_embeddings** | **False** | **Untied — must initialize input embeddings AND LM head separately** |
| hidden_act | xIELU | Novel activation function |
| max_position_embeddings | 65,536 | RoPE-scaled (llama3-style) |
| Existing Greek tokens | **1,494** | Strict-Greek filter; per-id list at `tokenizer_analysis/inspection/base/greek_tokens/`. Phase A diagnostic uses a looser "contains any Greek codepoint" rule and reports 1,506. |

The 70B variant has the same `tie_word_embeddings: False` and follows similar conventions at scale.

### 2.2 Embedding architecture implications

**Untied embeddings mean we have two separate matrices to extend:**
- Input embedding matrix `E`: `[131072, 4096]` → resized to `[131072 + N_new, 4096]`
- LM head matrix `U`: `[131072, 4096]` → resized to `[131072 + N_new, 4096]`

Both must be initialized with the chosen method, applied independently (the two matrices have different empirical norm distributions and anisotropic structure, so norm matching must be done per-side).

### 2.3 Pretraining context (high-level)

Apertus was pretrained on a realised 13.5T-token budget (15T headline applies to the 70B; 8B skipped Stage 2). Multilingual training across 1,811 supported languages. Greek pretraining share has been measured directly — see §2.5.

**Training details:** Apertus uses the novel xIELU activation and the AdEMAMix optimizer, which may have minor implications for continued pretraining setup (we should match optimizer state if possible, or document the optimizer switch if not).

### 2.4 Corpora and pipeline

- **BPE training corpus:** GlossAPI + HPLT at 50/50 by training-token mass, on the wave-2-broad cleaner output (`greek_badness_score < 60`, `mojibake_badness_score ≤ 0.1`, `charset_greek_ratio ≥ 0.5`, openarchives.gr `needs_ocr=true` dropped). The trained arm is **C3 = `C3_wave2_broad_glossapi_plus_hplt_50_50`**; total vocab 156,672 = 131,072 base + 25,600 added. Scale: 14.4M training docs / ~100B chars (~50B per pool). The cutoff (how many of the 25,600 added units to keep) is the only open tokenizer-side decision. See `docs/C3_CONVERGENCE.md` and `docs/C3_TRAINING_DATASETS.md` for full provenance.

  **What's actually in the GlossAPI half** (per `docs/C3_TRAINING_DATASETS.md`, 19 source datasets, 546,920 docs post-filter): theses (`greek_phd` via didaktorika.gr / EKT, 37k docs), academic textbooks and theses (`Kallipos`, `Pergamos`, 20k combined), EU Parliament Greek-language data (`europaiko_koinovoulio`, 29k), Greek legislation (`eurlex-greek-legislation`, `AI-team-UoA/greek_legal_code`, public consultations from opengov.gr-diaboyleuseis, ~71k combined), the openarchives.gr aggregator (46k post-OCR-filter), `HuggingFaceFW/finepdfs-edu` Greek slice (209k), `HuggingFaceFW/finewiki` Greek (243k), `OPUS/OpenSubtitles-el-v2018` Greek (143k), plus smaller literature / school books / ecclesiastical / classical sources. **Net character: heavy on academic + legal + wiki, with substantial education-PDF and subtitle content; not "academic-only."** Note: the explicit polytonic filter Phase B v4 applied was on *evaluation*; the C3 BPE training corpus also implicitly skews modern because `greek_badness_score < 60` strips most ancient/Katharevousa content, but some polytonic material in classical / wikisource / openarchives sources may remain. This affects what merges exist in the C3 vocabulary, marginally.

  **What's in the HPLT half:** Greek slice of HPLT 2.0 (`ell_Grek`), filtered to bin ≥ 8 quality, no machine-translated docs, `greek_badness_score ≤ 60`. Only 28.6% of available HPLT clean60 (~13.9M of 48.6M docs) was sampled into C3; the remaining 71.4% (~34.7M docs) is the verified-virgin pool used for held-out evaluation slices.

- **BPE cutoff:** not yet decided. Empirical 25-point sweep (every multiple of 1024, from 1024 to 25600) on three verified-clean held-out slices is in `docs/C3_CUTOFF_REPORT.md`. Cumulative-fertility-savings checkpoints (avg over the 3 slices):
  - 25% by ~1,024
  - 50% by ~3,072
  - 75% by ~8,192
  - 90% by ~14,336
  - 95% by ~19,456
  - 99% by ~24,576
  
  Total possible savings base→25,600 is 1.122 absolute fertility. Updated candidate set:

  | candidate | added params (both matrices, ≈) | % of total fert savings | unused-added at this cutoff |
  |-----------|---------------------------------|-------------------------|-----------------------------|
  |     8,192 |                            67 M |                     75% |                  284 (3.5%) |
  |    16,384 |                           134 M |                     90% |                1,624 (9.9%) |
  |    20,480 |                           168 M |                     95% |               2,914 (14.2%) |
  |    24,576 |                           201 M |                     99% |               4,534 (18.4%) |

  All four are 128-aligned. **8,192 is a serious contender** if inference economics or staged-training cost matter at all.

- **Post-extension continued pretraining corpus:** Academic Greek texts and broader Greek material, mixed with original-distribution data as a forgetting prophylactic. Replay-ratio design informed by the measured 0.023% Greek pretraining share (§2.5).

- **Initialization methods under test:** Vanilla (no extension), ReTok, Distillation. See §5. Norm-matching defaults data-anchored from Phase A (§8.2).

### 2.5 Apertus pretraining Greek share — measured (2026-05-11)

Direct measurement: Apertus-8B consumed ≈ **3.11 B Greek tokens out of 13,545 B total = 0.023%**.

Per-source contribution:

| Source | Stage | Greek tokens consumed (B) |
|---|---|---:|
| FineWeb-2-HQ Greek (dominant route) | S1+S3+S4+S5 | 3.014 |
| EuroParl Greek (20 bitexts) | S5 | 0.078 |
| Clean-Wikipedia Greek | S5 | 0.019 |
| EuroBlocks-SFT Greek | S5 | 0.0001 |
| ParaDocs Greek | — | 0 (no Greek pairs in repo) |
| Institutional Books Greek (long-context phase only, gated) | — | not measured; headline-neutral |
| **Total** | | **3.111** |

Full method, per-stage math, citation chain in `APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`.

### 2.6 Existing Greek tokens — norm diagnostic (Phase A, 2026-05-11)

Per-token L2 norms on both `E` and `U` for every classified group, plus an empirical untrained-token floor. Per-group medians (full table in `runs/apertus_greek_diagnostic_20260511_v2/`):

| Group | E p50 | U p50 |
|---|---:|---:|
| Greek (n=1,506) | 5.047 | 3.797 |
| English-baseline (n=74,838) | 5.051 | 3.839 |
| CJK | 4.608 | 3.787 |
| Cyrillic | 5.195 | 3.865 |
| German | 4.597 | 3.652 |
| French | 4.103 | 3.518 |
| structural_non_linguistic | 4.911 | 3.583 |
| all-vocab | 4.925 | 3.799 |

Empirical untrained-floor median ‖U‖ = 0.4566. Greek p5 ‖U‖ / floor = 6.54.

**Conclusion:** Greek tokens are statistically indistinguishable from English-baseline on both matrices; no Greek-token tail near the empirical untrained floor. **Phase 1 grounding is not needed.**

**Reconciliation with §2.5.** 3.1 B Greek pretraining tokens / ~1,500 Greek vocab slots ≈ 2 M training occurrences per Greek token — well in the convergence regime, which is what Phase A picked up as healthy norms.

**Phase A caveats:**
- A handful of reserved `<SPECIAL_NNN>` slots leaked into English-baseline and bias its lower percentiles slightly downward. Greek still beats English's tail; conclusion robust.
- Norm diagnostic doesn't catch behaviorally-undertrained tokens. The behavioral cross-check via per-token NLL on real Greek text is **Phase B v4** (§2.7) — it confirms Phase A on modern Greek registers.
- Diagnostic is on the **base** model. Re-run for `Apertus-8B-Instruct-2509` if the extension target shifts.

### 2.7 Existing Greek tokens — behavioral NLL (Phase B v4, 2026-05-12)

Per-token NLL of Apertus on five diversified, register-matched held-out corpora. Sampling done with HuggingFace streaming-shuffle (`buffer_size=100_000`), GlossAPI filtered to drop `is_historical_or_polytonic` and `polytonic_ratio > 0.05`, GlossAPI further stratified across source shards to prevent any one source from dominating. Each slice has a recorded diversity audit (2,375 unique web domains on Greek HPLT vs 11 in the earlier sampling-buggy v3).

Native-group median NLL on its own slice, gated on min-occurrence ≥ 20:

| Slice | Native group | Median NLL | n token-ids |
|---|---|---:|---:|
| `hplt_el` (web Greek, diversified) | Greek | **0.958** | 1,493 |
| `glossapi_el_modern` (modern monotonic Greek) | Greek | **0.942** | 1,495 |
| `hplt_en` (web English) | English-baseline | 3.025 | 10,267 |
| `hplt_de` (web German) | German | 2.714 | 860 |
| `hplt_ru` (web Russian) | Cyrillic | 2.448 | 5,910 |

**Headline reading.** Apertus predicts modern Greek (both web and academic) at median NLL ≈ 0.95, about 3× lower than its own English-on-English NLL on diversified web English. Per-doc variance is small (e.g., `hplt_el` per-doc NLL std = 0.257 around median 1.42 across 5,262 docs / 2,375 domains).

**Magnitude caveat.** This is per-*token* NLL. Greek's higher fertility (more tokens per Greek word than English per English word, given the base tokenizer) means each Greek token carries less information than each English token, making per-token prediction structurally easier. The "3× better" headline does *not* translate to "Apertus generates Greek 3× better than English" — generating a Greek word may require predicting several easier tokens whose joint difficulty is comparable to a single harder English token. The valid reading is: **the existing Greek tokens are well-trained and well-predicted in context, and the extension is not needed to fix any per-token prediction-quality gap on the deployment register.**

**Register: ancient / polytonic / Katharevousa.** Not measured in v4 (explicitly filtered out). Earlier v3 sampling, which was dominated by Galen-era / Katharevousa material, hit median NLL 3.11 on what was effectively the polytonic register — comparable to English-on-English. This register is genuinely hard for Apertus but is **outside the project scope**: the C3 BPE training corpus (`wave-2-broad` cleaner + `greek_badness_score ≤ 60`) strips most polytonic content, and the deployment target does not include ancient Greek. If the project ever extends to polytonic, separate evaluation, possibly a different BPE training corpus, and the polytonic worst-token list become relevant.

**Behaviorally-hardest existing modern Greek tokens.** Top-50 list at `runs/apertus_greek_phase_b_v4_20260512/greek_worst_50_combined.json`. Top 5 by mean NLL: `Κων` (5.75), `Στις` (5.54), `λει` (5.45), `αποτέλε` (5.30), `δημι` (5.22). These are mid-length stems / partial stems whose continuation is locally underdetermined in heavily-inflected Greek (`Κων` prefixes Κωνσταντίνος, -ου, -α, -ούπολη…). The worst-token list isn't about register difficulty — it's about local-context ambiguity. Could be useful for a "worst-50 absorption check" against C3 cutoff candidates (§11).

**Reconciliation with Phase A.** Phase A showed Greek tokens are well-trained in norm. Phase B v4 confirms they're well-predicted behaviorally on the deployment register. Both findings are consistent and complementary; the v3 register-split reading is fully retracted as a sampling artifact (single-domain HPLT + polytonic-dominated GlossAPI).

**Implications:**
1. The case for the C3 extension is **purely compression economy** — fewer tokens per Greek character, lower inference cost — not lifting prediction quality. Modern Greek already predicts well at the per-token level.
2. The three-arm experiment (§5) is now more sharply about *can extensions deliver efficiency without losing the quality we already have*, rather than *can extensions lift quality at all*. The §10 Q8 asymmetric M_ext / M_van decision rule still applies and may be sharper than before — expect quality to be roughly tied; expect efficiency to be the deciding factor.
3. Soft constraint on cutoff from v4 §7.4: each new vocab unit should accumulate ≥ ~100k training occurrences during CPT to be at least as well-trained as the existing 1,506 Greek tokens (which sit at ~2M occurrences per token from pretraining). This favors smaller cutoffs (8K, 16K) or larger Greek CPT budgets — see §10 Q1.

Full method, per-doc CSVs, cross-group rows, and reconciliation across measurement generations in `runs/apertus_greek_phase_b_v4_20260512/`.

## 3. Decision Nodes

This section maps the decisions the project actually needs to make, in operational order, and what informs each. The rest of the doc — constraints (§4), experiments (§5), implementation (§8), open questions (§10) — is supporting material that each node draws from. Reading this section first gives a roadmap; reading the rest fills in why each decision is positioned the way it is.

**Architecture (not a decision — an underline).** Apertus-8B-2509 has `vocab=131,072`, `hidden=4,096`, **untied embeddings** (`tie_word_embeddings: False` — input embedding matrix E and LM head U must be initialized independently), xIELU activation, AdEMAMix optimizer, GQA 4:1. These shape implementation but aren't pending decisions. See §2.1–§2.3.

### Node 1 — BPE cutoff

- **Decision:** pick a cutoff for the C3-trained extension. Current candidates from the §2.4 sweep: 8K, 16K, 20K, 24K.
- **Informed by:** Pareto curve (§2.4: 75% / 90% / 95% / 99% of total fertility savings at the four candidates); Phase B v4 soft constraint (§2.7: each new vocab unit should accumulate ≥ ~100k training occurrences during CPT; at fixed budget, 8K gets ~3× the per-unit exposure of 24K); parameter cost (67M → 201M); worst-50 absorption check (§11 action item).
- **Determines:** parameter count, per-unit training exposure, init compute, total CPT-budget pressure — fixed across all three arms.
- **Weight:** very high.
- **Order:** must happen first. Everything downstream assumes a fixed cutoff.
- **Status:** leaning 8K (Pareto + 100k constraint both favor smaller cutoffs); pending final commitment.

### Node 2 — Training mix design

- **Decision:** pick CPT corpus mix ratios (Greek source weights) and non-Greek replay ratio.
- **Informed by:** C3 BPE corpus composition (§2.4: GlossAPI + HPLT 50/50 by char mass; CPT can match or differ); Apertus's measured 0.023% Greek pretraining share (§2.5); §1 hard multilingual-preservation constraint; Phase B v4 finding (§2.7: modern Greek is well-predicted, so CPT corpus design is about *delivering* compression economy, not *fixing* prediction-quality gap).
- **Determines:** forgetting prophylaxis effectiveness; the meaning of "Greek progress" in §10 Q8. Must be matched across all three arms.
- **Weight:** high.
- **Order:** before CPT setup.
- **Status:** starting point: 10–15% non-Greek replay (§8.5); specific Greek mix ratios TBD.

### Node 3 — Evaluation suite construction

- **Decision:** pick the slices that will actually be built and the native benchmarks that will actually be run.
- **Informed by:** existing held-out slices (`virgin_hplt`, `C3_val_clean`, `C3_test_clean` per §2.4); Phase B v4 slices already constructed and usable (`hplt_el`, `glossapi_el_modern` per §2.7); native Greek benchmark menu (§10 Q6: GreekMMLU, Belebele, Medical MCQA, OYXOY, greek-nlp/benchmark); §1 multilingual hard constraint → English / French / Russian / German regression slices required for §10 Q8 gates.
- **Determines:** what Node 4 thresholds reference; what the arm comparison is built from.
- **Weight:** high.
- **Order:** before Node 4 thresholds can be set; before experiments produce results.
- **Status:** §10 Q6 has the menu; pick the actually-running subset.

### Node 4 — Pre-commit decision-rule thresholds

- **Decision:** lock numerical values for X (preservation gate), M_progress (Greek improvement floor), M_ext (extension beats Vanilla), M_van (Vanilla beats extension), T (Distillation beats ReTok).
- **Informed by:** §1 hard multilingual constraint (spirit of X); §10 Q8 framework (structure); efficiency asymmetry (extension arms get ~30% inference cost reduction at 8K cutoff — motivates M_ext < M_van); Phase B v4 finding (§2.7: modern Greek already well-predicted, expect quality ties, efficiency decides).
- **Determines:** which arm wins.
- **Weight:** very high.
- **Order:** **hard temporal constraint** — must happen *before* any arm completes its CPT phase. Setting thresholds after seeing results converts the decision into post-hoc rationalization.
- **Status:** framework drafted in §10 Q8 (TENTATIVE); thresholds TBD; suggested starting points X=5%, M_progress=3–5%, M_ext=1–2%, M_van=3–5%, T=2–3% need sign-off.

### Node 5 — Three-arm experimental design

- **Decision:** confirm we run all three arms as an escalation ladder (Vanilla → ReTok → Distillation), at the chosen cutoff, with shared setup (same corpus, same total CPT token budget, same staged schedule with Stage 1 empty for Vanilla, same evaluation suite).
- **Informed by:** Phase A norm diagnostic (§2.6: substrate is well-trained, defensible for all three arms); Phase B v4 (§2.7: modern Greek well-predicted, so the question is efficiency vs quality trade-off, not quality lift); Token Distillation paper evidence (Dobler 2025, §6.3: published comparison shows TD beats simpler aggregation on representation quality); compute budget (each arm = 10B-token pilot per §8.7).
- **Determines:** project outcome.
- **Weight:** very high.
- **Order:** setup happens after Nodes 1–3 are resolved; results inform decision under Node 4.
- **Status:** framework in §5; pending all upstream nodes.

### Node 6 — Operational positioning vs Krikri (optional, late)

- **Decision:** pick one of the 5 formulations of "as good or better than Krikri" — or commit to no comparison claim. Options: match at equal compute / match on a specific register / match on native-sourced benchmarks only / win on open-data axis / no commitment.
- **Informed by:** Krikri facts (~56.7B Greek CPT compute on Llama-3.1-8B base — ~18× our pretraining Greek share); project compute budget; native benchmark availability (§10 Q6).
- **Determines:** writeup framing and external comparison claims.
- **Weight:** low–medium for the experiment; medium for project narrative.
- **Order:** can be late (writeup-time decision).
- **Status:** identified as open; not in active scope of the immediate decision sequence.

## 4. Constraints

### Constraint 1: Greek-authentic learning, no translation mediation

The model should learn Greek from authentic Greek data, not via translation equivalence with English. This rules out initialization methods that anchor on bilingual alignment (WECHSEL, Trans-tokenization, OFA), continued pretraining on machine-translated Greek corpora, and any auxiliary embeddings trained on parallel corpora.

Motivation: capturing what does *not* translate cleanly — Greek-specific concepts (φιλότιμο), discourse particles, register layers (Katharevousa, Demotic, polytonic), and academic/philosophical vocabulary with Greek-internal structure.

### Constraint 2: Smooth transition, preserve learned structures

New embeddings must respect the geometric/statistical properties the trained attention/MLP layers expect — norm distribution, anisotropic covariance structure, position relative to the embedding manifold. This rules out random init and norm-mismatched init; favors convex-hull methods + norm matching + staged training.

### Constraint 3: Compatible with continued BPE training

We are doing additive BPE extension, not tokenizer replacement. Every new token has a deterministic decomposition into existing tokens (either via the merge tree or via base-tokenizer retokenization). This favors methods that use this decomposition as their input signal.

## 5. The Three Experiments

We do not have evidence to choose an initialization method a priori. The three constraints above, plus Phase A's diagnostic findings, narrow the space; within that space, three approaches survive as serious candidates. **We run all three in parallel and let the results decide.**

The three experiments form a complexity ladder:

```
Vanilla  ────→  ReTok  ────→  Distillation
(no extension)  (extension +    (ReTok + gradient
                 static init)    descent on attention)
```

Each step adds work and (potentially) capability. The experiment determines whether the added work is justified.

### Experiment 1 — Vanilla

**Approach:** Continued pretraining on the Greek corpus with the *original* Apertus tokenizer. No vocabulary extension. No new embeddings to initialize.

**Rationale:** The "LLaMA Beyond English" paper (Yuan et al., 2024, arXiv:2401.01055) showed that CPT without vocabulary extension significantly outperformed Chinese LLaMA at much smaller data budgets. Phase A's finding that Greek tokens are well-trained on average means there's no representational gap for extension to close *on average*. Vanilla is therefore the load-bearing benchmark — extension methods must beat it on the agreed evaluation suite or they're adding parameters without payoff.

**Cost shape:** Pure CPT cost; no init cost. The simplest baseline to set up.

**What it doesn't fix:** Greek compression efficiency. Apertus's existing Greek vocabulary fragments Greek text into more BPE pieces than well-fitted languages would, with associated inference and sequence-length costs. Vanilla preserves that inefficiency.

### Experiment 2 — ReTok

**Approach:** Extend the tokenizer at the chosen cutoff. For each new token T with surface form s:

1. Retokenize s with the **base** Apertus tokenizer.
2. This yields a sequence of base-vocab pieces `p₁, p₂, …, pₖ`.
3. Initialize `E[T] = norm_match(mean(base_E[pᵢ] for i in 1..k))`.
4. Apply the same procedure to LM head rows `U[T]`.

Then run continued pretraining on the same Greek corpus.

**Rationale:** Static, closed-form init using only existing trained vectors. The new token starts at a sensible point in the convex hull of base embeddings; CPT does the work of teaching the model what the new tokens actually mean. Used (in some form) by EEVE-Korean, the "Accelerating Multilingual" paper, and the Chinese LLaMA family.

**Naming note.** "ReTok" is used here as the colloquial name for sub-token averaging in the literature. Strictly, the ReTok paper (Chen et al., 2024) describes a merge-order-chained variant where new tokens can depend on previously-initialized new tokens — call that ReTok-strict. The base-piece retokenization version above avoids the error-propagation problem and is what most production recipes (EEVE, etc.) actually do. We use this version.

**Cost shape:** Minutes of CPU to compute init (no GPU needed) + CPT cost.

**Fits all three constraints:** no cross-lingual signal, stays in convex hull by construction, BPE-compatible.

### Experiment 3 — Distillation (ReTok + gradient descent on attention)

**Approach:** Start with the ReTok initialization from Experiment 2. Then refine each new token's embedding via gradient descent so that downstream attention behavior at a target layer matches what the multi-subtoken sequence would have produced. Specifically:

1. Initialize E[T] ← ReTok (subtoken mean of base-piece decomposition).
2. Find contexts where the surface form s appears in real Greek text.
3. For each context, run two forward passes through the (frozen) base Apertus up to a chosen layer L:
   - Teacher: with the original tokenizer (s becomes multi-subtoken sequence).
   - Student: with the extended tokenizer (s is single token T, current E[T]).
4. Compute MSE between teacher and student hidden states at layer L, paired at downstream positions that attend to T (mapped to corresponding positions in the teacher).
5. Backprop through the frozen layers; update only E[T].
6. Repeat over ~25 contexts per token until convergence.

Then run continued pretraining on the same Greek corpus.

**Rationale:** ReTok uses only information stored in the embedding table. The semantics of a multi-subtoken composition like "πανεπιστήμιο" largely isn't stored in the embedding table at all — it's constructed in the Transformer layers via attention and MLP processing. Distillation extracts that learned representation by asking: *"what input embedding for the new token causes the model's downstream attention to behave the same as it would have with the original multi-subtoken sequence?"* The frozen transformer layers do the work; we just need the right input to trigger it.

In one line: **Distillation = ReTok + gradient descent on attention behavior**.

**Citation and code.** Dobler, Elliott, de Melo (2025), "Token Distillation: Attention-aware Input Embeddings for New Tokens", arXiv:2505.20133v2. Same author lineage as FOCUS. Official code: https://github.com/konstantinjdobler/token-distillation.

**Cost shape:** Minutes for ReTok init + ~1 hr GPU time for the distillation refinement (paper reports 2,500 tokens in ~10 min on one H100; linearly: 16K tokens ≈ 60 min) + CPT cost.

**Untied-embedding caveat.** Token Distillation as published learns *input embeddings only*. For Apertus's untied LM head `U`, we use one of three published patterns:
1. ReTok for U, Distillation for E (asymmetric init). Simplest.
2. Distillation for E + NTP-only objective on U (paper-recommended). One extra small objective.
3. Distillation + αNTP combined (paper's most robust variant). Auto-scaled multi-objective.

We default to pattern (2) for Experiment 3 unless implementation considerations push us elsewhere.

**Fits all three constraints:** no cross-lingual signal, lives inside the trained manifold by construction (hidden states are the model's own representations), BPE-compatible.

### Methods not in the experimental plan

The three experiments form the active comparison. For completeness:

- **A1 (literal ReTok with merge-order chaining)** — dominated by Experiment 2's base-piece variant (no error propagation). Not in plan.
- **B (CW2V, Yamaguchi et al., 2024)** — defensible alternative with strong theoretical/empirical paper, but adds Word2Vec training + constrained-optimization step without clearly outperforming Distillation. Not in plan; could be added as a fourth arm if results from the three above are inconclusive.
- **WECHSEL, FOCUS-bilingual, OFA, Trans-tokenization** — violate Constraint 1.
- **Random / mean-init** — violate Constraint 2.
- **Hypernetwork methods (ZeTT, Hyper-OFA)** — require hypernetwork training as a prerequisite; out of scope unless someone has trained one for Apertus.

## 6. Weaknesses and Open Considerations Per Experiment

This section is deliberately adversarial — laying out the strongest critiques and known caveats for each experimental arm. These don't change which experiments we run; they shape how we interpret results.

### 6.1 Weaknesses of Vanilla (Experiment 1)

**6.1.1 Doesn't fix Greek compression.** Whatever inefficiency Apertus has on Greek bytes-per-token remains. Inference cost is unchanged.

**6.1.2 May plateau at the existing tokenizer's expressiveness ceiling.** Greek long-words and inflectional patterns still get fragmented across multiple tokens, which is suboptimal for both inference cost and certain quality dimensions (token-boundary-sensitivity, multi-subtoken-word generation probability per Lesci et al. 2025).

### 6.2 Weaknesses of ReTok (Experiment 2)

**6.2.1 Linear-compositional prior is wrong for non-compositional units.** "πανεπιστήμιο" is not literally "παν" + "επιστήμιο" in a linear way. ReTok imposes this assumption anyway. Distillation directly answers this critique.

**6.2.2 Position-insensitivity.** `mean(a, b) = mean(b, a)`. Constituent-order information is lost.

**6.2.3 No distributional information.** ReTok uses only structural info — which pieces decompose the new token. It ignores how the new token is actually used in text.

**6.2.4 Norm matching is a uniform hack — now data-anchored.** Per-group targets from Phase A (§8.2) tighten this.

**6.2.5 No head-to-head winning paper for ReTok specifically.** Reviewers can ask why ReTok over Distillation, which has direct published superiority on representation-quality metrics. This is exactly why Experiment 3 exists alongside Experiment 2.

### 6.3 Weaknesses of Distillation (Experiment 3)

**6.3.1 Implementation complexity vs ReTok.** Requires forward passes, backward through frozen layers, per-token gradient descent. The added complexity is non-trivial even if compute is cheap.

**6.3.2 LM head story is awkward for our untied embeddings.** TD doesn't natively initialize the LM head. The three workable patterns are all defensible but add project surface area beyond pure Distillation.

**6.3.3 Validation breadth on language adaptation is limited.** TD has been evaluated on biomedical domain adaptation and on French. No published results on Greek specifically. Reasonable to expect transfer; not directly demonstrated.

**6.3.4 Known failure mode on tied embeddings.** Llama3.2-3B (the only tied-embedding test case in the paper) hit a degenerate-norm failure mode. Apertus is untied, so we're outside the failure regime, but the method has at least one known instability.

### 6.4 Weaknesses that apply to any vocabulary-extension experiment (2 and 3)

**6.4.1 Parameter overhead is non-trivial.** 67M–201M new parameters across both matrices, depending on cutoff (§2.4).

**6.4.2 Long-tail problem in our new vocab.** At 24,576 cutoff, 18.4% of new tokens are "unused-added." At 8,192, this drops to 3.5%.

**6.4.3 Corpus-distribution overfit.** The C3 50/50 GlossAPI+HPLT mix may produce a vocabulary optimized for registers the deployment never encounters.

**6.4.4 HPLT quality concerns — partially addressed.** Status (2026-05-11): HPLT clean60 release `fffoivos/hplt-greek-ge8-no-mt-clean60-wave4` is the active source. Modern-Demotic bias is real; polytonic/Katharevousa coverage remains corpus-distribution-limited.

### 6.5 Possible Mitigation Strategies (apply during experiment design)

- For ReTok/Distillation: hybrid init (compositional + small Greek-centroid pull); higher LR on new-token rows; staged training (§8.3).
- For Distillation specifically: use the TD+αNTP variant for combined E and U handling.
- For all extension arms: pilot the 8K cutoff variant alongside 16K to characterize the cutoff-dependence of the comparison.

## 7. Reference Implementations and Production Examples

### Korean LLM family (most relevant by structure)

- **EEVE-Korean-10.8B (Kim et al., 2024, arXiv:2402.14714).** Extended Llama-2 vocab to 40,960 tokens. ReTok-style subword-based init + 7-stage training with parameter freezing. Became leading Korean model on Open Ko-LLM Leaderboard.
- **RedWhale (Han et al., 2024, arXiv:2408.11294).** Extends and refines EEVE.
- **"Accelerating Multilingual Language Model" (Lim et al., 2024, arXiv:2401.10660).** Extends Llama-2-13B to Korean/Japanese. Explicitly ReTok-style mean-of-subword-embeddings init for LM head.

### Chinese LLM family

- **Chinese LLaMA / Alpaca (Cui et al., 2023, arXiv:2304.08177).** 32K → 49,953 vocab. Two-stage training: freeze transformer → train embeddings only → add LoRA.
- **ReTok (Chen et al., 2024, arXiv:2410.04335).** Llama3 + Chinese continued BPE. Paper's algorithm is merge-order-chained; production reuses commonly slide to the base-piece variant we use in Experiment 2.

### Theoretical and methodological

- **Dobler et al., 2025 (arXiv:2505.20133v2).** Token Distillation. Code: https://github.com/konstantinjdobler/token-distillation.
- **Yamaguchi et al., 2024 (arXiv:2407.05841).** Convex-hull theory + CW2V.
- **Hewitt, 2021.** Foundational mean-init with empirical covariance.

### Counterpoint

- **"LLaMA Beyond English" (Yuan et al., 2024, arXiv:2401.01055).** CPT without extension beats Chinese LLaMA at much smaller data budgets. This is Experiment 1's reason for being in the plan.

## 8. Implementation Considerations

### 8.1 Embedding architecture (verified for Apertus)

**Untied embeddings (`tie_word_embeddings: False`).** Initialize both `E` and `U` separately. Norm distributions differ; norm matching per-side.

### 8.2 Norm matching — data-anchored defaults

Per-group targets from Phase A (§2.6):

| New-token category | Norm-matching target (E median) | Norm-matching target (U median) |
|---|---:|---:|
| Greek-content tokens (most new tokens from C3) | **5.05** | **3.80** |
| Structural / non-linguistic tokens | **4.91** | **3.58** |
| (Fallback / global) | 4.93 | 3.80 |

Applies to Experiments 2 and 3.

### 8.3 Staged training (EEVE-style adaptation)

For Experiments 2 and 3 (extension arms):

1. **Stage 1 — Embedding-only training.** Freeze all base parameters. Train only new input-embedding rows + new LM head rows.
2. **Stage 2 — Embeddings + adapters.** Unfreeze input embeddings entirely + add LoRA adapters on attention/MLP.
3. **Stage 3 — Full continued pretraining.** Unfreeze all parameters. Reduced LR on base, higher LR on new-token rows.

For Experiment 1 (vanilla), Stage 1 is empty (no new embeddings) — either skip directly to Stage 2/3 or run a standard CPT schedule. The comparison stays apples-to-apples as long as total CPT tokens-seen budget is matched across arms.

### 8.4 Learning rate decoupling

For extension arms: higher LR on new-token rows than on the rest of the model. Typical ratio 5×–10×. Decay the asymmetry as new tokens converge.

### 8.5 Forgetting prophylaxis

Mix original-distribution data into the continued pretraining corpus for all three arms. Apertus's pretraining Greek share was 0.023% (§2.5); our CPT corpus is overwhelmingly Greek. A starting point is 10–15% non-Greek replay; the specific design is itself an experimental variable but should be matched across the three init arms.

### 8.6 Tooling

- HuggingFace `transformers`: `model.resize_token_embeddings()` handles both matrices (no-op for vanilla).
- ReTok: custom code (~50 lines) for base-tokenizer retokenization, mean computation, norm matching.
- Distillation: use the Token Distillation repo for `E`; pattern (2) for `U` requires adding an NTP objective on new LM head rows during the distillation training loop.
- BPE extension itself: handled in the C3 arm; see `docs/C3_CONVERGENCE.md`.

### 8.7 Training budget and schedule — TENTATIVE, forward-looking beyond core tokenizer-extension scope

The project's stated scope is tokenizer extension + initialization methodology. The continued-pretraining details that follow extend beyond that scope and are tentative — but worth having on paper because decisions made at the tokenizer-extension stage (cutoff, init method, eval suite design) can be shaped by what comes downstream. In particular, the Phase B v4 soft constraint (§2.7: each new vocab unit should accumulate ≥ ~100k training occurrences during CPT) couples directly to the CPT token budget, so the cutoff decision can't be fully decoupled from training-budget planning.

**Sequence length: keep at 4,096 throughout CPT** (not tentative — methodological hygiene). Apertus pretraining used 4k. Bundling context-length expansion with tokenizer adaptation would change two variables simultaneously and obscure the comparison. If long-context Greek capability becomes important downstream, it should be a separate continuation step after the tokenizer adaptation is locked.

**Phased token budget (tentative starting points):**

| Phase | Tokens | Trainable | LR range | Goal |
|---|---:|---|---|---|
| Stabilization | 0.5–1B | New embedding rows + new LM head rows only (everything else frozen) | 1e-4 to 3e-4 | Make new tokens usable before full-model training |
| Pilot CPT | 10B per arm | Full model, lower LR on base + higher LR on new-token rows (§8.4) | 1e-5 to 2e-5 | Compare Vanilla vs ReTok vs Distillation |
| Main CPT | 20–40B | Full model, WSD-style decay if achievable | 1e-5 to 2e-5 | Lock in Greek gains for the winning arm only |
| Optional targeted continuation | 5–10B | Full model or top/bottom layers only | 5e-6 to 1e-5 | If a specific register or downstream task remains weak |

**Other recipe choices (tentative):**

- **Batch size:** 2–4M tokens per global step (Apertus pretraining used 4.2M → 8.4M for the 8B). At 4M tokens/step, 10B pilot ≈ 2,500 steps; 20–40B main ≈ 5,000–10,000 steps.
- **Checkpointing:** every 250–500 steps or every 1B tokens, whichever first. Intrinsic eval every 100–250 steps.
- **Optimizer:** AdEMAMix + WSD if Apertus's optimizer state is portable and we trust the loading path; otherwise AdamW with short warmup and conservative peak LR. The portability question itself is unresolved — §10 Q4.

**Compute scale estimates (planning numbers, order-of-magnitude, not measured throughput):**

| Run | 8× H100-class | 8× A100-class |
|---|---|---|
| 1B stabilization | < 1 day | ~ 1 day |
| 10B pilot per arm | 1–3 days | 3–7 days |
| 20–40B main run | 2–6 days | 1–3 weeks |

Three pilot arms at 10B each plus stabilization is roughly 3–9 days on H100s for the comparison phase. **This is the load-bearing compute estimate for the whole project**; everything else (eval, deployment, optional continuation) is small by comparison.

**Status:** Numbers above are adapted from the Apertus technical report's training recipe and from CPT defaults in the multilingual-adaptation literature; they should be calibrated against actual throughput on the target cluster before commitment. The 10B pilot size in particular is a placeholder — could be smaller (5B) if compute is tight, or larger (20B) if early signal isn't separating the arms.

## 9. Other Possible Approaches (Open for Contribution)

(Unchanged from earlier versions — exploratory list of alternatives not currently in the experimental plan.)

### 9.1 Alternative tokenization paradigms
- BLT (Byte-Latent Transformer), H-Net, MorphBPE, GPE, character-level fallback.

### 9.2 Alternative initialization signals
- CW2V (Yamaguchi 2024) — could be added as a 4th experimental arm if the three give inconclusive results.
- Iterative refinement (Distillation → brief CPT → re-Distillation).
- Distillation init from a Greek-only model.
- Hypernetwork init (ZeTT, Hyper-OFA).
- Adversarial init.

### 9.3 Parameter-efficient training strategies
- LoRA-only, language-specific adapters, prefix tuning, embeddings-only fine-tuning.

### 9.4 Multi-channel or augmented embeddings
- Phonetic, etymological, morphological, script normalization.

### 9.5 Architectural modifications
- MoE with Greek expert, language-specific layer norms, Greek-specific blocks.

### 9.6 Curriculum and data strategies
- Curriculum learning, code-switched data, synthetic Greek (carefully filtered for Constraint 1).

### 9.7 Training objective modifications
- Span corruption for academic Greek, Greek NLU auxiliary losses, register-contrastive learning.

### 9.8 Post-training and alignment
- SFT, DPO, Constitutional AI variants.

### 9.9 Hybrid and external
- Greek model as distillation teacher, model merging, routing hybrid, retrieval augmentation.

### 9.10 Evaluation-driven and diagnostic
- Embedding manifold tracking, per-layer probing, catastrophic forgetting analysis, Greek-distinctive concept probes, active data selection.

### 9.11 Theoretical investigations
- Greek-specific subspaces, multilingual interference scaling, optimal vocab size as a function of corpus and target-language fraction.

Strong opinions welcome; pull requests encouraged.

## 10. Open Questions

1. **BPE cutoff: position on the Pareto frontier.** Given the smooth diminishing-returns curve (§2.4): 75% benefit at ~8k, 90% at ~16k, 95% at ~20k, 99% at ~24k. We need to pick one cutoff for the three-experiment comparison. Trades inference economics against representational capacity. Default lean: 8k or 16k depending on how strongly inference cost weighs.

   **Soft constraint from Phase B v4 (§2.7).** Existing Greek tokens received ~2M training occurrences each during Apertus pretraining (3.1B Greek tokens / 1,506 vocab slots) and predict modern Greek well at NLL ~0.95. For new C3 units to be at least as well-trained as the existing ones, each should accumulate ≥ ~100k training occurrences during Greek CPT. Larger cutoffs spread the CPT budget thinner across new units; at fixed CPT volume, 8K gets ~3× the per-unit exposure of 24K. This soft constraint favors smaller cutoffs unless the Greek CPT budget is substantial.

2. ~~**Whether to do a "Phase 1" grounding step.**~~ **Resolved (no).** Phase A.

3. **HPLT register coverage.** Modern-Demotic bias is real; polytonic/Katharevousa coverage is corpus-distribution-limited. If deployment target needs these registers, supplementing the GlossAPI corpus may be warranted.

4. **Continued pretraining schedule.** Total budget, mixing ratios, staged unfreezing, optimizer state handling. Must be matched across all three experiments for the comparison to be fair.

5. **Replay ratio for forgetting prophylaxis.** Apertus's 0.023% Greek pretraining share gives a concrete anchor for what to maintain on non-Greek. Specific value should be matched across all three arms.

6. **Greek evaluation methodology.** Existing clean held-out slices (`virgin_hplt`, `C3_val_clean`, `C3_test_clean`) need supplementing with:

   **Perplexity / language-modeling slices (for the three-arm internal comparison):**
   - Modern Greek web slice (`hplt_el`-style) — already exists from Phase B v4.
   - Modern Greek academic slice (`glossapi_el_modern`-style, polytonic-filtered) — already exists from Phase B v4. Note: Phase B v4 found web and academic modern Greek are essentially equivalent in difficulty for Apertus (median NLL 0.958 vs 0.942), so the register-stratification is less load-bearing than v0.8 assumed; both slices are useful for variance reduction but probably won't split the three arms.
   - English / French / Russian / German regression slices for the §10 Q8a preservation gates.
   - **Polytonic / Katharevousa probe — confirmed out of scope** per Phase B v4 §6.3. If the project ever pivots to polytonic, this becomes relevant; for the current scope, optional.

   **Native-sourced Greek benchmarks (for external positioning + downstream task validation, all preferred over MT-derived because they don't carry English translation artifacts):**
   - **GreekMMLU** (Zhang et al. 2026, arXiv:2602.05150) — 21,805 multiple-choice questions across 45 subjects, sourced from Greek academic / professional / governmental exams. Public release 16,857 + private leaderboard 4,948 for contamination-resistant evaluation. Methodologically the strongest available.
   - **Belebele Greek** (Bandarkar et al. 2024) — native-speaker-created reading comprehension; part of the ILSP Greek Evaluation Suite.
   - **Medical MCQA Greek** (Voukoutis et al. 2024) — native-sourced Greek medical knowledge; part of the ILSP Greek Evaluation Suite.
   - **OYXOY** (Sotiropoulos et al. 2024, arXiv:2309.07009) — NLI, word sense disambiguation, metaphor detection; native-sourced from the Dictionary of Standard Modern Greek.
   - **greek-nlp/benchmark** (Bakagianni et al., arXiv:2501.12826) — authorship attribution, text clustering for Greek legal texts.

   **MT-derived Greek benchmarks (secondary, for Krikri-comparability only):**
   - MMLU Greek, ARC-Challenge Greek, HellaSwag Greek, TruthfulQA Greek — included in ILSP Greek Evaluation Suite. Useful as comparison axes against published Krikri / Meltemi numbers, but known to carry translation artifacts; not weighted heavily in our decision rule.

   **Additional benchmarks worth tracking — TENTATIVE, forward-looking beyond core tokenizer-extension scope.** These extend evaluation beyond what's needed for the internal three-arm decision but become relevant if the project produces a deployable Greek model that we want positioned against Krikri / Meltemi on dimensions perplexity doesn't capture:

   - **Universal Dependencies Greek treebanks (UD Greek GDT, UD Greek GUD)** — POS / morphology / syntax. Fine-grained linguistic competence, complements perplexity-based signals. Particularly relevant given Greek's morphological productivity.
   - **elNER, Greek Legal NER** — named-entity tasks. Greek Legal NER aligns directly with the C3 corpus's heavy legal weighting (EUR-Lex + greek_legal_code + opengov diaboyleuseis).
   - **GreekSUM** — native Greek summarization. One of the cleaner options in a thin space; Greek summarization evaluation is genuinely underdeveloped.
   - **GreekBarBench** — legal long-form reasoning. Pairs naturally with the legal content in C3; if Apertus-Greek is positioned as having a legal-Greek advantage over Krikri, this is the benchmark for it.
   - **Safety / toxicity:** OGTD, DACHS hate-speech, AttaQ Greek. Not currently part of the §10 Q8 decision rule but should be checked before any deployment; useful to flag now so it doesn't surface as a blocker later.
   - **Mixed-script / Greeklish probe:** Greek-script vs Greeklish vs alternating-script alternation. Greeklish is a real Greek user-input channel that recent Greek transliteration work treats as a first-class register rather than incidental noise. Worth a dedicated small probe.

   The internal decision rule (§9) uses perplexity on the first group. Benchmark numbers from the second group inform external positioning and writeup but don't drive arm selection unless a benchmark gap is dramatic enough to override perplexity signal. The "tentative additional" group is for downstream / deployment evaluation and is not load-bearing for the tokenizer-extension comparison.

7. **Behavioral cross-check (Phase B) — DONE (v4, §2.7).** Diversified, register-matched held-out NLL confirms Phase A. Modern Greek (both web and academic) is well-predicted at median NLL ~0.95, ~3× lower than English-on-English. Polytonic / Katharevousa is harder but out of scope.

8. **How will we declare a winner? — TENTATIVE proposal, to be refined and signed off before any results come in.**

   With three experiments and a register-stratified evaluation suite, a clear ordering may or may not emerge. The decision is multi-axis and cannot be reduced to a single number. We pre-commit to **gates + an ordered set of progress criteria** so that the winner is determined by the rule, not by post-hoc reading of results.

   ### 8a. Gating criteria (preservation — disqualifies if violated)

   These gates are the operationalization of §1's hard constraint on multilingual preservation. An arm is disqualified, regardless of Greek progress, if it regresses any of the following by more than **X%** (suggested starting point: 5%, candidate to tighten):

   - English held-out perplexity vs base Apertus.
   - French held-out perplexity vs base Apertus.
   - Russian held-out perplexity vs base Apertus.
   - (Add: German vs base Apertus — Apertus is Swiss, German is heavily represented; meaningful regression here is a strong negative signal.)
   - (Optional: add other languages where deployment has known needs — e.g. Italian.)

   The language list above is the "well-trained reference set" Apertus shouldn't lose ground on. Phase A norm comparisons can guide the list — any language where Apertus has healthy norms is a fair preservation target.

   The threshold X is a value judgment about how much multilingual breadth we're willing to sacrifice for Greek depth. 5% is a conventional lenient bound; 2-3% would be tight; <1% would be a strict no-regression policy. Pick X explicitly before running.

   ### 8b. Progress measure (Greek improvement — the thing we want)

   Each arm must improve over base Apertus on Greek by at least **M_progress%** to qualify for shipping at all:

   - **Greek web held-out perplexity** must improve vs base Apertus by ≥ M_progress%.
   - **Greek academic held-out perplexity** must improve vs base Apertus by ≥ M_progress%.

   **Both must improve**, not just one. An arm that improves only web Greek but flatlines on academic Greek hasn't done the thing the project is for; an arm that improves only academic but regresses on web has shifted the failure mode rather than fixed it.

   Suggested starting point: M_progress = 3–5%. Below 3% is hard to distinguish from training-noise variance across runs.

   ### 8c. Decision rule (asymmetric Vanilla-vs-extension comparison)

   1. **Disqualify** any arm that fails a gate (8a) or fails to clear M_progress% on both Greek metrics (8b).
   2. Among qualifying arms, compare Vanilla vs the best extension arm:
      - **If an extension arm beats Vanilla by ≥ M_ext% on both Greek metrics → ship the best extension arm.** Suggested M_ext = 1–2% (small, because extension also delivers ~30% Greek inference cost reduction — see note below).
      - **If Vanilla beats all extension arms by ≥ M_van% on at least one Greek metric, without an extension closing the gap on the other → ship Vanilla.** Suggested M_van = 3–5% (larger than M_ext, because Vanilla must overcome the efficiency disadvantage to justify shipping).
      - **Otherwise (extensions and Vanilla within noise on both metrics) → ship the best qualifying extension arm.** Vanilla loses the within-noise case because it carries no efficiency benefit.
   3. **Between ReTok and Distillation when both qualify** → ship Distillation iff it beats ReTok by ≥ **T%** on at least one Greek progress metric *and* doesn't regress the other; otherwise ship ReTok (simpler wins ties). Suggested T = 2–3%.
   4. **If no arm qualifies** → none of the three are deployment-ready. Reconsider corpus, schedule, or whether the deployment requirements are feasible with current Apertus and compute.

   **Note on the M_ext vs M_van asymmetry — TENTATIVE.** The asymmetric threshold encodes the inference economics difference between Vanilla and the extension arms. Vanilla uses the base tokenizer with no change to Greek fertility; extension arms achieve ~30% reduction in Greek bytes-per-token at the 8K cutoff (or up to ~44% at 24K — see §2.4), which translates roughly to ~30%–44% inference cost reduction on Greek workloads. For Vanilla's quality lead to be worth losing this efficiency benefit, it must be substantial — hence M_van > M_ext. Conversely, extensions need only modest quality wins to be worth shipping because they bring efficiency along.

   The specific asymmetry magnitude is itself a value judgment about deployment economics. With no specific deployment target (§1), the suggested defaults encode a moderate prior that efficiency matters but doesn't override clear quality differences. A deployment with high Greek inference volume would justify wider asymmetry (smaller M_ext, larger M_van); a deployment that is research-only or quality-critical with low Greek volume would justify narrower or symmetric M. Sign off the specific values before running.

   ### 8d. The "no clear winner" outcome

   With the asymmetric rule, this case is now better-defined than in earlier drafts. If all three arms come out within noise on every Greek metric (Δ < ~1% across the board), and gates are passed, **ship the simplest qualifying extension arm**, because the efficiency benefit is the deciding factor. Only if no extension arm qualifies (all fail gates or M_progress) should we fall back to Vanilla.

   This is a change from v0.8's "default to Vanilla on no-clear-winner" — that default ignored the efficiency asymmetry. The new default-to-extension-on-ties reflects that adding parameters is worth it when the parameters buy efficiency the eval can't see directly.

   ### 8e. Soft tiebreakers (only when 8c is genuinely indeterminate)

   When the asymmetric rule in 8c still doesn't produce a unique winner (e.g., two extension arms within noise of each other on both Greek metrics, T threshold doesn't trigger):

   1. Implementation simplicity (ReTok > Distillation in the production maintenance sense).
   2. Risk profile (fewer moving parts = lower probability of weird production failures).
   3. Parameter count (smaller is better — fewer slots to undertrain in deployment), only at the same cutoff. Note: cutoff is the dominant parameter-count lever (8K = 67M, 24K = 201M), and that's fixed once chosen for the comparison, so this tiebreaker is small.

   ### 8f. Status

   This decision framework is **tentative**. Specific values that must be agreed and committed to writing **before** any of the three experiments completes its CPT phase:

   - **X** (preservation gate threshold for English / French / Russian / German held-out perplexity).
   - **M_progress** (minimum Greek improvement over base Apertus for any arm to qualify).
   - **M_ext** (extension-beats-Vanilla threshold; smaller, accounts for efficiency advantage).
   - **M_van** (Vanilla-beats-extension threshold; larger, must overcome efficiency loss).
   - **T** (Distillation-beats-ReTok threshold).
   - Exact list of gate languages.

   Doing this with results visible risks post-hoc rationalization; the rule is only as honest as the pre-commitment.

## 11. Immediate Next Actions

- [x] **Verify Apertus model card / config:** `vocab=131,072`, `hidden=4,096`, `tie_word_embeddings=False`, `hidden_act=xIELU`, `intermediate=21,504`, GQA 4:1.
- [x] **Pilot BPE compression analysis at multiple cutoffs — DONE for C3.** 25-point sweep at every 1024 from 1024 to 25,600 in `docs/C3_CUTOFF_REPORT.md`.
- [x] **Audit GlossAPI and HPLT-Greek for quality, register distribution, MT contamination — partial.** Current filters documented in §2.4 and §6.4.4.
- [x] **Phase A — measure L2 norm distribution of existing Greek tokens.** DONE 2026-05-11. Phase 1 grounding not needed.
- [x] **Identify Apertus's pretraining data composition + Greek share.** DONE. 0.023% Greek.
- [x] **Phase B (per-token NLL on diversified held-out Greek) — DONE (v4, 2026-05-12).** Modern Greek (web + academic) median NLL ~0.95, ~3× lower than English-on-English. v3 register-split retracted as sampling artifact. Artifacts at `runs/apertus_greek_phase_b_v4_20260512/`.
- [ ] **Worst-50 modern Greek tokens vs C3 absorption check.** For each cutoff candidate (8K, 16K, 20K, 24K), compute what fraction of Phase B v4's worst-50 modern Greek stems become subsumed by an extended-vocab unit at that cutoff. Cheap script, sharp signal about whether a cutoff actually targets the hard cases.
- [ ] **Decide on cutoff for the three-experiment comparison** (Q1). Default lean: 8K or 16K — informed by the v4 100k-occurrences soft constraint (§10 Q1) which favors smaller cutoffs.
- [ ] **Set up the three experiments to run in parallel:**
  - **Experiment 1 (Vanilla):** Apertus + original tokenizer + CPT on Greek corpus.
  - **Experiment 2 (ReTok):** Apertus + extended tokenizer at chosen cutoff + ReTok init for E and U + same CPT.
  - **Experiment 3 (Distillation):** Apertus + extended tokenizer + ReTok init then Distillation refinement on E + NTP-only on U + same CPT.
  
  All three share: same Greek corpus, same total CPT token budget, same replay ratio, same evaluation suite, same staged-training schedule (with Stage 1 empty for Vanilla).
- [ ] **Pre-register evaluation criteria** (Q8). Before results are in.
- [ ] **Greek evaluation suite — register-stratified.**
  - Existing: `virgin_hplt`, `C3_val_clean`, `C3_test_clean`.
  - Add: academic-Greek probe, polytonic/Katharevousa probe, English-regression benchmarks.
- [ ] **Read source papers in full:** Token Distillation (Dobler 2025), EEVE (Kim 2024), LLaMA-Beyond-English (Yuan 2024), Yamaguchi 2024.

---

*This is a living document. Updates expected as decisions are made and experiments complete.*

*Changelog:*
- *v0.12 (2026-05-12 later): Major structural change — inserted new §3 "Decision Nodes" as a roadmap of the actual decisions the project needs to make, in operational order, with each node's evidence base, weight, ordering constraint, and current status. Six decision nodes (BPE cutoff / training mix design / evaluation suite construction / pre-commit thresholds / three-arm experimental design / Krikri positioning) plus an "architecture underline" that's a constraint to communicate rather than a decision. The empirical findings stay in §2; the §3 nodes cross-reference them to make the evidence-to-decision flow explicit. All subsequent sections renumbered (Constraints §3→§4; Three Experiments §4→§5; Weaknesses §5→§6; References §6→§7; Implementation §7→§8; Other Approaches §8→§9; Open Questions §9→§10; Next Actions §10→§11). Internal cross-references updated throughout. External references to Phase B v4 doc (§6.3, §6.4) preserved unchanged. Earlier changelog entries retain their period-correct section references; they describe state at the time they were written.*
- *v0.11 (2026-05-12 later): Integrated three items from external review. (1) New §8.7 "Training budget and schedule" — tentative, forward-looking beyond tokenizer-extension scope — adds concrete phased token budgets (0.5–1B stabilization → 10B/arm pilot → 20–40B main → 5–10B optional), LR ranges, batch sizes, compute-time estimates. Includes the non-tentative methodological point that sequence length stays at 4,096 throughout CPT (don't change two variables at once). (2) §10 Q6 expanded with a "tentative, forward-looking" benchmark group: UD Greek treebanks (GDT, GUD), elNER + Greek Legal NER, GreekSUM, GreekBarBench, safety benchmarks (OGTD, DACHS, AttaQ Greek), and a mixed-script/Greeklish probe. These extend evaluation beyond the internal three-arm decision into deployment-positioning territory. License-posture concerns on Pergamos and didaktorika.gr (from the same external review) noted but not integrated; deferred until CPT corpus is being actively spec'd.*
- *v0.10.1 (2026-05-12 later): §2.4 enriched with accurate GlossAPI source breakdown from new `docs/C3_TRAINING_DATASETS.md` reference — replaces the rough "forums, web archives, Wikipedia, EELLAK, opengov" sketch with the actual 19-source composition (academic theses, EU parliament, legal corpora, finepdfs-edu, finewiki, OpenSubtitles, etc.) and adds training scale (14.4M docs / 100B chars). Net characterization: "heavy on academic + legal + wiki + education-PDF + subtitles, not academic-only." Earlier sketch corrected. Also flagged: 71.4% of HPLT clean60 unsampled, available as verified-virgin eval pool.*
- *v0.10 (2026-05-12): Phase B v4 results integrated. Diversified, register-matched, polytonic-filtered sampling produced clean behavioral cross-check: modern Greek (web + academic) median NLL ~0.95, ~3× lower than English-on-English (3.03). v3 register-split fully retracted as sampling artifact (single-domain HPLT + polytonic-dominated GlossAPI). New §2.7 with v4 headline + magnitude caveat (per-token NLL, not per-word). §10 Q1 gains a 100k-occurrences-per-new-unit soft constraint favoring smaller cutoffs. §10 Q6 marks polytonic/Katharevousa probe as confirmed out-of-scope (relevant only if project pivots). §10 Q7 status flipped to DONE. §11 adds the worst-50 modern Greek absorption check (cheap cutoff-decision verification). Decision rule (§10 Q8) unchanged — the asymmetric M_ext/M_van structure still applies and is sharper now: expect arms to be roughly tied on quality, expect efficiency to decide.*
- *v0.9 (2026-05-11 later): §10 Q6 expanded with native Greek benchmarks (GreekMMLU, Belebele Greek, Medical MCQA, OYXOY, greek-nlp/benchmark) plus secondary MT-derived ILSP suite for Krikri-comparability. §10 Q8c restructured to asymmetric Vanilla-vs-extension thresholds: M_ext (smaller) and M_van (larger) replace the single M, encoding the ~30% Greek inference-efficiency advantage of extension arms. §10 Q8d's "default to Vanilla on no-clear-winner" flipped to "default to extension" — within-noise ties go to extensions because they carry efficiency the eval doesn't see directly. §9e tiebreakers reduced. §9f status updated with new threshold list. Asymmetry explicitly flagged as TENTATIVE.*
- *v0.8 (2026-05-11 later): Added explicit project-framing paragraph in §1 stating (1) no specific deployment target, project goal is broadly "improve Greek," and (2) hard constraint to preserve Apertus's multilingual character. This was previously implicit; making it explicit anchors the conservative choices in §10 Q8 (gates), §8.5 (replay), and §4 Constraint 1 (no translation mediation) as principled responses to the asymmetry between required-multilingual-breadth and optimized-Greek-depth. §10 Q8a updated to reference the constraint explicitly as the basis for preservation gates.*
- *v0.7 (2026-05-11 later): Reframed from option-selection to three parallel experiments — we lack evidence to pre-commit. Three arms: (1) Vanilla, (2) ReTok, (3) Distillation. Distillation framed as "ReTok + gradient descent on attention behavior" — a complexity ladder where each step adds capability. CW2V and A1 moved to "alternatives not in plan." §10 added Q8 (pre-register evaluation criteria — expanded mid-session into a concrete tentative framework with gates on preservation languages [En/Fr/Ru/De], progress thresholds on Greek web + academic, and explicit decision rule with default-to-Vanilla on no-clear-winner).*
- *v0.6 (2026-05-11 later): Reverted Phase B v3 register-split framing — earlier results used non-representative HPLT slices. Phase B re-run in progress. Retained Phase A, Greek-share measurement, Token Distillation literature findings.*
- *v0.5 (superseded): Phase B v3 register-dependent findings — not relied on.*
- *v0.4: Phase A norm diagnostic and Greek pretraining share (0.023%). Config corrections.*
- *v0.3: Greek token count (1,494). C3 arm specifics. Cutoff sweep with 8K Pareto contender. Split ReTok into A1/A2. Promoted contextual init to Option C.*
- *v0.2: Verified Apertus config (partially), added criticism section, added open-ended techniques section.*
- *v0.1: Initial draft.*