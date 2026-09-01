# Greek CPT of Apertus-8B — Training Configuration (v1.0)

**Base model / checkpoint:** **`swiss-ai/Apertus-8B-2509`** — the released `main` checkpoint. We train from this; we do **not** have the GPU-hours to resume from an earlier (pre-decay) checkpoint. 32 layers, dim 4096, untied input/output embeddings, xIELU activation, QK-Norm, decoder-only.
**Optimizer:** AdEMAMix (Swiss-AI Megatron-LM fork).
**Status:** Parameter block finalized. Experiment design (token budgets, evaluation suite, success criteria) and the peak-LR sweep are specified separately.
**Provenance:** Values not listed as *changed* trace to the validated Task-1 regime (`04_cpt_training_regime_on_vanilla/goal/hyperparameters.json` + the iter-1192 as-run log). Changed values and their rationale are flagged below.

---

## 1. Optimizer — AdEMAMix

| Parameter | Value | Justification |
|---|---|---|
| β₁ (fast EMA) | **0.9** | Standard AdamW/AdEMAMix value; matches Apertus. AdEMAMix's design assumes the *slow* first-moment EMA β₃ is comfortably larger than β₁ (Pagliardini et al. 2025) — satisfied here (0.999 ≫ 0.9). (β₂ is the unchanged Adam *second*-moment EMA — a separate role.) |
| β₂ (second moment) | **0.995** *(starting value)* | Lower than Apertus's 0.999: a faster (slightly noisier) second-moment estimate adapts more quickly to the shifted (Greek) gradient statistics on a short run. β₂ is the ordinary Adam variance EMA and is free to tune — AdEMAMix's stability constraint is β₃ > β₁, which β₂ does not affect. Couples to warmup via 2/(1−β₂) (§2). To be swept in [0.99, 0.999]. |
| β₃ (slow EMA) | **0.999** | The AdEMAMix paper's value for **low-iteration** runs (App. C.1.5 / Fig. 17: β₃ = 0.999 beats 0.9999 at 32k–64k steps; 0.9999 wins only for longer runs) — and our CPT is short. Lower than Apertus's pretraining 0.9999. *(The paper flags fast distribution-shift only as a stated limitation with no β₃ value attached, so we lean on the low-iteration result, not a distribution-shift claim.)* |
| α (slow-EMA mixing) | **4** *(starting value)* | The paper's reference α = 8 was tuned with β₃ = 0.9999; no α is published for β₃ = 0.999. Both the lower β₃ and the large (~4.2M-token) batch argue for less slow-momentum weight, so α = 4 (≈ half of Apertus's 8) is the starting point, to be swept. |
| α / β₃ warmup | **Whole run, T_{α,β₃} = T** (paper schedulers) | A large β₃ diverges if applied from step 0, so both are warmed from β_start = β₁ (Pagliardini et al. 2025). Apertus used a **fixed 100k-step** warmup (Table C.4) — longer than our entire short CPT — so a run-length-relative schedule is used instead; the paper notes scheduling need not span all of training. *(Open — §6: confirm β₃ actually engages on a short run, and that the Megatron fork supports the schedule.)* |
| Weight decay | **0.1** | Apertus / bakeoff value; unchanged. Applied to all parameters including new embedding rows (no exclusion, since differential LR is not used). |
| Gradient clipping | **0.1** (global norm) | Apertus value; unchanged. Expected to be active most steps under AdEMAMix's long-momentum term. |
| ε | 1e-8 | Standard default. |

---

## 2. Learning-rate schedule — WSD (Warmup–Stable–Decay)

Shape follows Apertus's pretraining schedule (trapezoid, 1-sqrt cooldown), with the peak set to **half** of Apertus's. Both ends sit at 0.1× peak — Apertus's warmup starts at 0.1× peak, not 0. **The same warmup and decay apply in the experiments and the production run.**

| Phase | Span | LR | Shape |
|---|---|---|---|
| Warmup | **2/(1−β₂)** iters (β₂=0.995 → ~400, ≈12%) | 5.5e-6 → 5.5e-5 | linear |
| Stable | remainder (≈68% at β₂=0.995) | 5.5e-5 | constant |
| Decay | final **20%** | 5.5e-5 → 5.5e-6 | 1-sqrt |

- **Peak LR = 5.5e-5** = 0.5 × Apertus-8B's pretraining peak (1.1e-4). This sits on the *adaptation-favoring* side of the documented adaptation-vs-forgetting tradeoff (Ibrahim et al. 2024: a higher peak improves adaptation, a lower peak reduces forgetting). It is more aggressive than Apertus's own long-context continuation LR (0.1× peak = 1.1e-5) and than the project's prior 1.1e-5 runs; the higher peak is paired with the reduced α (4) to moderate effective update magnitude, and with replay + the decay tail to protect retention. **This peak is the primary subject of the planned peak-LR sweep.**
- **Warmup = 2/(1−β₂) iterations** (policy, decided 2026-06-09), linear, from 0.1× peak — the untuned-warmup horizon over which the second-moment estimate becomes reliable (Ma & Yarats 2021); a re-warm is needed because Adam-family optimizers overstep early even when restarted at a minimum. It **scales with β₂** across the sweep: 0.99→200, 0.995→400, 0.999→2000 iters. ⚠ **Coupling:** at the top of the β₂ sweep (0.999) this is ~2000 iters ≈ 62% of a 3,218-iter run — so β₂ and warmup must be chosen together (high β₂ is incompatible with this warmup rule on a short run).
- **Decay = final 20%, 1-sqrt**, to 0.1× peak — the cooldown shape Apertus used. 20% is the upper end of the 10–20% WSD sweet spot. Supplies the "re-decay" ingredient of the canonical CPT recipe (Ibrahim et al. 2024) that the constant-LR diagnostics omitted.

---

## 3. Loss — Goldfish

| Parameter | Value | Justification |
|---|---|---|
| Goldfish k = h | **50** | The exact Apertus config: a 2% token-masking rate (k = 50) with a 50-token hashing window (h = 50), calibrated by Xu (2025) (Apertus §3.1 / §F). The k = 50/h = 50 *values* are Apertus's, not from the Goldfish paper itself (Hans et al. 2024 validated small k≈3–4); we adopt Apertus's calibrated setting for parity. At k = 50 only 2% of tokens are dropped, so the downstream-performance cost is minimal. |

*(The Megatron fork already computes the Goldfish mask **in the dataloader** (`GPTDataset.__getitem__` → `apply_goldfish`), i.e. Apertus's "front-load during data loading" — no separate offline pass exists or is needed. The hash is uniform over the full 148,480 vocab (prime-modulus, no token-id branching), so the new arm-2 ids are masked identically to base ids — verified.)*

---

## 4. New-token initialization & vocabulary

| Item | Decision | Justification |
|---|---|---|
| Init method | **Token Distillation (TD), layer 11** | Empirically selected over ReTok and Centroid; layer 11 won on iter-0 intrinsic BPB (+0.040 full-scale) and new-token recall. TD distills the input rows E at the layer-11 target depth and learns the output rows U by cross-entropy (`learn_output_with_ce`). |
| Embedding stabilization | **None — subsumed by TD** | TD's relearning runs with the body frozen, so the new rows are already fitted to the converged body at init — the work a separate stabilization phase would do. |
| Differential LR (new rows) | **Not used** | Uniform LR. A higher LR for new rows has no publication support (the documented separate-embedding-LR practice sets it *lower* — e.g. Unsloth's 2–10× reduction for CPT); uniform LR was the regime that worked in the bakeoff and keeps the configuration simple. |
| Vocabulary size | 131,072 base + 17,408 modern Greek = **148,480** (= 256 × 580) | **Final.** The modern-only extended tokenizer at `03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_only_148480/` (the TD-11 bakeoff base). Cutoff 17,408 at the fertility knee (kept tokens fire >1,000×); the merge-preserving noise-token cleanup is **done** (69 noise tokens removed at build, 69 valid merges backfilled, append-only ids, 256-aligned). Polytonic (+5,120 → 153,600) is **deferred** to a separate downstream arm. Embedding resize: **17,408 new rows**. |

---

## 5. Training regime (inherited from Apertus / the bakeoff)

| Parameter | Value |
|---|---|
| Sequence length | 4096 |
| **RoPE base (θ / `rotary_base`)** | **500,000** |
| **`max_position_embeddings`** | **4096** |
| **RoPE scaling** | **`llama3`** — factor 8.0, high_freq 4.0, low_freq 1.0, original_max_pos 8192 |
| Global batch | 1024 sequences ≈ 4.19M tokens |
| Micro-batch | 2 |
| Precision | bf16 |
| Parallelism | TP = 2, PP = 1 |
| Attention masking | EoD + cross-document attention masking (`--reset-attention-mask --reset-position-ids`) |
| Data mixture | **new Greek (= 70% HPLT + 30% openarchives, unseen) is primary; replay = +35% OF the new-Greek budget** (24% multilingual + 4% code + 2% math + 5% Greek-replay). 10B new → 13.5B total; per-step shares ≈ 74% new / 18% ML / 3% code / 1.5% math / 3.7% Greek-replay. 3 held-out val sets (0.5B each: HPLT, openarchives, greek_phd) excluded from training; per-set loss at eval cadence. |

Held at the Apertus/bakeoff regime so the run stays faithful to the base model's training dynamics.

**Data-mixture note.** The **new Greek corpus (glossapi + HPLT**, post-dedup → decontamination → HPLT confident-only residue cleaning (`02_corpus_preparation/10_clean_hplt`) → PII masking) is the **primary** stream: its token budget is set first per experiment (a fixed amount — e.g. 10B — or all available), and replay is added **as a percentage OF that new-Greek count** (+35% total, so 10B new → 13.5B): **24% multilingual** (en/fr/de/ru… per the Vanilla-run B1 recipe), **4% code**, **2% math**, and a new **5% Greek replay** of *Apertus-original* Greek to anchor the model's existing Greek against drift (candidate source: the `apertus_overlap_drop` docs removed during dedup — i.e. Greek that was already in Apertus's pretraining). Shares are interleaver probabilities; effective token shares track them up to per-source exhaustion. The 24/4/2 sub-composition follows the Vanilla B1 recipe (`03_…/init_bakeoff/corpus_build/MIX_RECIPE.md`).

**Geometry note (verified).** The RoPE settings above (θ=500k, seq 4096, `llama3` ×8) are Apertus's **main-pretraining** geometry, not an override we invent — stated explicitly in the **Apertus paper §2.4 + Table C.4** ("RoPE base Θ = 500,000 during pretraining, which we extend in the long-context phase"; "context of 4,096 tokens during pretraining"; NTK/LLaMA-3 scaling, factor 8; the paper also lists "RoPE θ after 64k context expansion = 12,000,000"). It is verified identical in the pre-decay checkpoint config (`step2400000-tokens13112B`: `rope_theta=500000`, `max_position_embeddings=4096`, `rope_scaling.rope_type=llama3`), which is the load-bearing evidence. **Correction (2026-08-06): it is _not_ verified in the Vanilla-5B run's as-run config.** That run executed with `rope_scaling=null` (`04_cpt_training_regime_on_vanilla/RUN_LOG_20260528.md:1127-1129`), so it matches on θ and sequence length but *not* on the scaling block — which is precisely the component that turns out to be load-bearing. The earlier wording here claimed a verification that does not exist. The released `main` carries the *post*-long-context geometry (`rope_theta=12,000,000`, seq 65536); since we start from `main`, we **revert it to this seq-4096 pretraining geometry**. This reframes the archived Path-A geometry-probe decision (`04_cpt_training_regime_on_vanilla/_archive/.../cpt-plan.md` + `PATH_A_GEOMETRY_PROBE_PLAN.md`) — rope-500k/4096 is the model's actual pretraining geometry, not a "Path-B mistake." **Documented trade-off (restated 2026-08-06 — the previous "minor retention cost (xnli_ru −1.57 pp)" summary materially understated the probe).** What the May probe actually measured was the seq-4096 geometry **with the scaling block nulled**, and that configuration was bad: on `main` weights it drove Greek BPB to `1.2216` (against ~0.43 for the CPT arms) and GreekMMLU from `0.5280` to `0.4879`, and a 0.5B-token Path-A CPT beat it by **+5.51 pp** [3.79, 7.25] on the 3-task MCQ macro (`04_cpt_training_regime_on_vanilla/reports/path_a_probe_results_20260531.md`). The Vanilla-2B adversarial critique diagnosed why: forcing θ=500k onto 12M-trained weights phase-shifts Q·K, so "the model is *perturbed*, not *re-anchored*". **Keeping the `llama3` ×8 scaling block is the difference between that failure and the production config**, so it must not be dropped as an incidental detail.

**Direct validation on the production weights (2026-08-06).** A same-weights A/B now exists at iteration 0 of the full-8B run: identical safetensors, identical 16,159-item clean subset (manifest sha `61ed4ac9…`), only `config.json` differing. Under the released `12,000,000 / 65,536` geometry the anchor scores decontaminated GreekMMLU **0.35151** (choice NLL 1.4969); under this production `500,000 / 4,096 + llama3 ×8` geometry it scores **0.35782** (choice NLL 1.4586). The production geometry is **+0.63 pp better with lower NLL**, so the reverted geometry is validated rather than merely argued. Receipts: `07_full_8b_cpt_prelaunch/20260805T123000Z-d0-v1/initial_greekmmlu/` and `.../20260805T154100Z-d0-v3/initial_greekmmlu/`. **Known gap:** the production geometry (500k/4096 *with* scaling) has never been evaluated on the un-extended base weights — only the two flanking configurations were, so the three-way comparison is bracketed, not measured.

**The iteration-0 GreekMMLU drop is not a geometry artifact.** The anchor's `0.35782` against `Apertus-Base` `0.52796` is the reproduced cost of the tokenizer extension plus TD initialization: the May bakeoff's `TokenDistil-Init` scored `0.35396` on the same benchmark (`release/apertus-tokenizer-extension/benchmark-evals/native-greek-suite/native_mcq_per_task.csv`), i.e. the same dip to within half a point across two independently built models three months apart. It recovers with training (TD-5B `0.4693`; the 13.5B TD arm ended macro `0.5301` above base `0.4817`). *(Data mixture is specified in §5's data-mixture note above.)*

---

## 6. Open / to-confirm (not blocking the parameter block)

- ~~α/β₃ warmup on a short run~~ — **resolved (code-verified):** the fork's AdEMAMix warms over iterations (T_{α,β₃}=TRAIN_ITERS), β₃ warms from β₁ and reaches 0.999 by run end (long-memory engaged within the first ~10%), and `group["step"]` is restored on resume (no per-segment reset).
- ~~Goldfish mask offline~~ — **resolved (code-verified):** the fork computes the mask in the dataloader (`GPTDataset.__getitem__`); no offline pass needed; the hash is uniform over the extended 148,480 vocab.
- **5% Greek-replay source.** Wire the 5% *Apertus-original* Greek replay into the mix recipe — candidate source is the `apertus_overlap_drop` docs removed during dedup (Greek that was in Apertus's own pretraining); confirm and build it into `mix_builder.py`.
- **Citation hygiene (from `papers/CITATION_AUDIT.md`):** two supporting citations point the wrong way and should be re-framed or dropped — `optimal-embedding-lr` (2506.15025) actually argues for a *higher* embedding LR, not the uniform LR it's cited for; `jiang-tokenizer-aware-adaptation` solves the generation caveat with a *frozen*-body adapter, not by unfreezing. The uniform-LR decision still stands on the bakeoff empirics + ALLaM/Tokenization-Bottleneck practice; only the citations need fixing.
- **Experiment design (next):** evaluation suite, success criteria, and the peak-LR sweep that validates the 5.5e-5 choice. (Per-experiment token budget = the primary new-Greek amount, set per run.)

---

## References

Grouped by the part of the configuration they inform. **(value)** = directly fixes a value above; **(supporting)** = justifies a choice without pinning a number.

**Optimizer**
- AdEMAMix — Pagliardini, Ablin & Grangier 2025, *The AdEMAMix Optimizer: Better, Faster, Older* (ICLR 2025, arXiv 2409.03137). **(value)** β₁/β₂/β₃/α and their warmup schedulers; β₃ = 0.999 for low-iteration / non-stationary settings; useful (α, β₃) ranges are wide.

**Learning-rate schedule**
- Apertus — *Apertus v1 Technical Report* (arXiv 2509.14233). **(value)** 8B peak 1.1e-4, WSD with 1-sqrt cooldown, warmup from 0.1× peak over 16.8B tokens, final LR 0.1× peak.
- Hägele et al. 2024, *Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations* (NeurIPS 2024, arXiv 2405.18392). **(supporting)** The WSD scheduler and the "set max LR to half the cosine value" guideline Apertus followed — the origin of the "half" framing.
- Ibrahim et al. 2024, *Simple and Scalable Strategies to Continually Pre-train LLMs* (TMLR, arXiv 2403.08763). **(supporting)** Re-warm + re-decay + replay; higher peak → more adaptation, lower → less forgetting — the dial the 5.5e-5 peak sits on.
- Gupta et al. 2023, *Continual Pre-Training of LLMs: How to (re)warm your model?* (arXiv 2308.04014). **(supporting)** Re-warming for CPT; companion to Ibrahim.
- *A Practitioner's Guide to Continual Multimodal Pretraining* (arXiv 2408.14471). **(supporting)** Maps the adaptation-vs-retention curve over base LR; finds the pretraining-reference LR (~1e-5) optimal and larger LRs more forgetting-prone — the evidence the peak-LR sweep tests against.
- Stability gap — *Efficient Continual Pre-training by Mitigating the Stability Gap* (arXiv 2406.14833). **(supporting)** The temporary general-capability dip under CPT that replay and distribution-matching mitigate — the failure mode the decay tail + replay protect against.
- Ma & Yarats 2021, *On the Adequacy of Untuned Warmup for Adaptive Optimization* (arXiv 1910.04209). **(supporting)** Adam-family optimizers overstep early even when restarted at a minimum — why the re-warm is needed.

**Loss**
- Hans et al. 2024, *Be like a Goldfish, Don't Memorize! Mitigating Memorization in Generative LLMs* (NeurIPS 2024, arXiv 2406.10209). **(value)** The Goldfish loss; suppresses verbatim recall with little downstream impact.

**New-token initialization & vocabulary**
- Token Distillation — Dobler 2025 (arXiv 2505.20133; repo `konstantinjdobler/token-distillation`). **(value)** The init method: distill input rows E, CE-learn output rows U; its body-frozen relearning is what subsumes a separate stabilization phase.
- Artetxe, Ruder & Yogatama 2020, *On the Cross-lingual Transferability of Monolingual Representations* (arXiv 1910.11856). **(supporting)** Foundation of freeze-body / train-embeddings — why TD's relearning suffices and no separate stabilization is needed.
- EEVE — Kim et al. 2024 (arXiv 2402.14714). **(supporting)** Staged-freeze recipe for vocab expansion; old output rows must eventually unfreeze for logit-scale matching.
- Jiang et al. 2026, *Tokenizer-Aware Cross-Lingual Adaptation of Decoder-Only LLMs through Embedding Relearning and Swapping* (EACL 2026, 2026.eacl-long.357). **(supporting)** Embedding relearning on decoder-only LLMs at scale; mitigates forgetting; the generation caveat that argues for unfreezing the body rather than staying frozen.
- *Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size* (arXiv 2506.15025). **(supporting)** Embedding LR scales with vocabulary size under Adam — context for the uniform-LR (no differential-LR) choice.
- Vocabulary-extension CPT with uniform LR — ALLaM (arXiv 2407.15390); *The Tokenization Bottleneck* (arXiv 2511.14365). **(supporting)** Documented practice is uniform LR + good init — the basis for dropping differential LR.
- Differential-LR concept — ULMFiT, Howard & Ruder 2018 (arXiv 1801.06146); LLRD, Zhang et al. 2021, *Revisiting Few-sample BERT Fine-tuning* (arXiv 2006.05987). **(supporting)** Origin of per-group / discriminative LR; the documented direction puts embeddings *lower*, not higher — why the higher-new-row-LR idea was dropped.
- Tokenizer cutoff — Tao et al. 2024, vocabulary scaling (arXiv 2407.13623); Land & Bartolo 2024, "Fishing for Magikarp" / under-trained tokens (arXiv 2405.05417). **(supporting)** Vocabulary-size-optimal cutoff and the under-trained-token signature — the basis for the 17k cutoff and the separate noise-token cleanup.
