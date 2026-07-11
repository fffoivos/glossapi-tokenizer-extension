# Greek CPT of Apertus-8B — Training Configuration (v1.0)

**Base model / checkpoint:** **`swiss-ai/Apertus-8B-2509`** — the released `main` checkpoint. We train from this; we do **not** have the GPU-hours to resume from an earlier (pre-decay) checkpoint. 32 layers, dim 4096, untied input/output embeddings, xIELU activation, QK-Norm, decoder-only.
**Optimizer:** AdEMAMix (Swiss-AI Megatron-LM fork).
**Status:** **Frozen for the full-corpus probe.** Replay, peak LR, alpha, beta3 and beta2 sweeps are complete; see `05_token_distillation_cpt/PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`.
**Provenance:** Values not listed as *changed* trace to the validated Task-1 regime (`04_cpt_training_regime_on_vanilla/goal/hyperparameters.json` + the iter-1192 as-run log). Changed values and their rationale are flagged below.

---

## 1. Optimizer — AdEMAMix

| Parameter | Value | Justification |
|---|---|---|
| β₁ (fast EMA) | **0.9** | Standard AdamW/AdEMAMix value; matches Apertus. AdEMAMix's design assumes the *slow* first-moment EMA β₃ is comfortably larger than β₁ (Pagliardini et al. 2025) — satisfied here (0.999 ≫ 0.9). (β₂ is the unchanged Adam *second*-moment EMA — a separate role.) |
| β₂ (second moment) | **0.999** | Selected from the mechanically comparable `{0.99, 0.995, 0.999}` sweep at a fixed 400-iteration LR warmup. It had the best GreekMMLU and new-Greek held-out loss with only a negligible foreign-loss tradeoff. |
| β₃ (slow EMA) | **0.999** | The AdEMAMix paper's value for **low-iteration** runs (App. C.1.5 / Fig. 17: β₃ = 0.999 beats 0.9999 at 32k–64k steps; 0.9999 wins only for longer runs) — and our CPT is short. Lower than Apertus's pretraining 0.9999. *(The paper flags fast distribution-shift only as a stated limitation with no β₃ value attached, so we lean on the low-iteration result, not a distribution-shift claim.)* |
| α (slow-EMA mixing) | **4** | Sweep winner over `{0, 4, 8}`: strongest GreekMMLU with the best overall adaptation/retention compromise. |
| α / β₃ warmup | **Whole run, T_{α,β₃} = T** (paper schedulers) | A large β₃ diverges if applied from step 0, so both are warmed from β_start = β₁ (Pagliardini et al. 2025). Apertus used a fixed 100k-step warmup; our run-length-relative schedule was code-verified and used by every selected sweep arm. |
| Weight decay | **0.1** | Apertus / bakeoff value; unchanged. Applied to all parameters including new embedding rows (no exclusion, since differential LR is not used). |
| Gradient clipping | **0.1** (global norm) | Apertus value; unchanged. Expected to be active most steps under AdEMAMix's long-momentum term. |
| ε | 1e-8 | Standard default. |

---

## 2. Learning-rate schedule — WSD (Warmup–Stable–Decay)

Shape follows Apertus's pretraining schedule (trapezoid, 1-sqrt cooldown), with the peak set to **half** of Apertus's. Both ends sit at 0.1× peak — Apertus's warmup starts at 0.1× peak, not 0. **The same warmup and decay apply in the experiments and the production run.**

| Phase | Span | LR | Shape |
|---|---|---|---|
| Warmup | **400 iterations, fixed** (≈12% of the 13.5B sweep horizon) | 5.5e-6 → 5.5e-5 | linear |
| Stable | remainder (≈68% at the 13.5B horizon) | 5.5e-5 | constant |
| Decay | final **20%** | 5.5e-5 → 5.5e-6 | 1-sqrt |

- **Peak LR = 5.5e-5** = 0.5 × Apertus-8B's pretraining peak (1.1e-4). The completed sweep selected it as the loss-first adaptation/retention knee: nearly the high-LR GreekMMLU gain without their foreign held-out loss cost.
- **Warmup = 400 iterations, fixed**, linear from 0.1× peak. The beta2 sweep deliberately held this constant so beta2 was the only changed field. Do **not** reapply `2/(1-beta2)`: at the selected beta2=0.999 it would produce 2,000 iterations and a recipe that was never compared.
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
| Data mixture | **79% new Greek / 20% foreign replay / 1% old-Greek replay.** Foreign replay remains Apertus-family matched (FineWeb-Edu, FineWeb-2-HQ, StarCoderData, FineMath). Stable document-hash held-outs are excluded from training. |

Held at the Apertus/bakeoff regime so the run stays faithful to the base model's training dynamics.

**Data-mixture note.** New Greek is the primary stream. The replay sweep fixed the total-token shares at **79/20/1**; this supersedes the pilot's +35%-of-new-Greek composition. Exact realized source counts must be checked from the new full-corpus materialization before launch.

**Geometry note (verified).** The RoPE settings above (θ=500k, seq 4096, `llama3` ×8) are Apertus's **main-pretraining** geometry, not an override we invent — stated explicitly in the **Apertus paper §2.4 + Table C.4** ("RoPE base Θ = 500,000 during pretraining, which we extend in the long-context phase"; "context of 4,096 tokens during pretraining"; NTK/LLaMA-3 scaling, factor 8; the paper also lists "RoPE θ after 64k context expansion = 12,000,000"). It is verified identical in the pre-decay checkpoint config (`step2400000-tokens13112B`: `rope_theta=500000`, `max_position_embeddings=4096`, `rope_scaling.rope_type=llama3`) and in the validated Vanilla-5B run's as-run config. The released `main` carries the *post*-long-context geometry (`rope_theta=12,000,000`, seq 65536); since we start from `main`, we **revert it to this seq-4096 pretraining geometry**. This reframes the archived Path-A geometry-probe decision (`04_cpt_training_regime_on_vanilla/_archive/.../cpt-plan.md` + `PATH_A_GEOMETRY_PROBE_PLAN.md`) — rope-500k/4096 is the model's actual pretraining geometry, not a "Path-B mistake." Documented trade-off: the Path-A probe measured a minor retention cost for the seq-4096 geometry on `main` (xnli_ru −1.57 pp); revisit only if multilingual retention regresses. *(Data mixture is specified in §5's data-mixture note above.)*

---

## 6. Open / to-confirm (not blocking the parameter block)

- ~~α/β₃ warmup on a short run~~ — **resolved (code-verified):** the fork's AdEMAMix warms over iterations (T_{α,β₃}=TRAIN_ITERS), β₃ warms from β₁ and reaches 0.999 by run end (long-memory engaged within the first ~10%), and `group["step"]` is restored on resume (no per-segment reset).
- ~~Goldfish mask offline~~ — **resolved (code-verified):** the fork computes the mask in the dataloader (`GPTDataset.__getitem__`); no offline pass needed; the hash is uniform over the extended 148,480 vocab.
- ~~Replay fraction and old-Greek replay~~ — **resolved:** 79/20/1 of total tokens; see `05_token_distillation_cpt/PRODUCTION_MIX_DECISION_20260612.md`.
- **Citation hygiene (from `papers/CITATION_AUDIT.md`):** two supporting citations point the wrong way and should be re-framed or dropped — `optimal-embedding-lr` (2506.15025) actually argues for a *higher* embedding LR, not the uniform LR it's cited for; `jiang-tokenizer-aware-adaptation` solves the generation caveat with a *frozen*-body adapter, not by unfreezing. The uniform-LR decision still stands on the bakeoff empirics + ALLaM/Tokenization-Bottleneck practice; only the citations need fixing.
- ~~Peak LR / alpha / beta3 / beta2 sweeps~~ — **resolved:** see `05_token_distillation_cpt/PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md` and the machine-readable sweep audit.

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
