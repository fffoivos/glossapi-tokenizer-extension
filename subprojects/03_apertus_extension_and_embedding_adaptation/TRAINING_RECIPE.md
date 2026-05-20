# Training Recipe — Apertus-8B-2509 Greek CPT bakeoff

*Authoritative recipe for the three-arm init bakeoff (Vanilla / ReTok / Centroid) and the subsequent production CPT. Downstream of [`cpt_plan.md`](cpt_plan.md) v0.7 + [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md). Every numeric value cites a source.*

> **Audit pass 2026-05-21** — every claim in this doc has been verified against locally-pinned sources at [`references/`](references/MANIFEST.md) (8 cloned repos at pinned commits + 15 paper PDFs). Findings + applied fixes are in [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md). Two pre-submit blockers remain open: HF→Megatron Apertus loader doesn't exist in the swiss-ai fork (custom loader needed); ILSP Greek harness task configs live in Meltemi/Krikri forks, not swiss-ai. Neither blocks colleague review of the recipe itself.

## Scope

| | bakeoff (this doc § 3-9) | production CPT (after winner is picked) |
|---|---|---|
| Tokens per run | 2 B per arm × 3 arms = 6 B total | 15–20 B (per v0.7 §3) |
| Optimizer | AdEMAMix | AdEMAMix |
| Loss objective | NTP | Goldfish |
| Vocab | 131,072 (Vanilla) or 148,480 (ReTok/Centroid) | 148,480 (winning arm; composite 153,600 only if polytonic specialization is later run) |
| Engine | Megatron-LM-Swiss-AI | Megatron-LM-Swiss-AI |

The only differential between bakeoff and production is the **loss objective** (NTP→Goldfish) and the **token budget**. Optimizer, LR schedule, architecture, sequence length, batch shape, and gradient clipping all carry over.

---

## 1. Engine: Megatron-LM-Swiss-AI

Apertus pretrained with Swiss AI's fork of NVIDIA's Megatron-LM, which adds Apertus-specific kernels (xIELU activation, QK-Norm, AdEMAMix optimizer, Goldfish loss) on top of the upstream training infrastructure. We use the **same** fork at the **same** commit Apertus pretrained on, for fidelity.

- Upstream Megatron-LM: NVIDIA, [github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Citation: Shoeybi et al. 2019, [arXiv:1909.08053](https://arxiv.org/abs/1909.08053) ("Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism").
- Apertus fork: [github.com/swiss-ai/Megatron-LM](https://github.com/swiss-ai/Megatron-LM). Branch: `main` (Apertus production trained from `main`; no specific tag pinned in public artifacts). We pin commit `c92402e39ef3c8e69ea378a59e79059dc14541f4` (HEAD as of 2026-05-20) for reproducibility. Apertus-specific paths:
  - **AdEMAMix optimizer**: [`megatron/core/optimizer/ademamix.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/core/optimizer/ademamix.py)
  - **xIELU activation**: [`megatron/training/activations.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/activations.py) (classes `XIELU`, `XIPReLU`)
  - **QK-Norm wiring**: [`megatron/core/transformer/attention.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/core/transformer/attention.py) (`q_layernorm` / `k_layernorm`); spec assembly in `megatron/core/models/gpt/gpt_layer_specs.py`
  - **Goldfish loss**: [`megatron/core/datasets/gpt_dataset.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py) (`apply_goldfish`, `_create_hash_table`)
  - **Training entry**: [`pretrain_gpt.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/pretrain_gpt.py)
- Production launch script (the canonical "what was run"): [`swiss-ai/pretrain-code/pretraining/submit_apertus_8b.sh`](https://github.com/swiss-ai/pretrain-code/blob/main/pretraining/submit_apertus_8b.sh). We mirror this script's flag set for the bakeoff, deviating only on (a) CPT-specific LR peak, (b) bakeoff token budget, (c) NTP-not-Goldfish, (d) shorter α/β3 warmup proportional to the bakeoff horizon.

We do **not** use HuggingFace Trainer for the CPT training itself. The bakeoff trains in Megatron format and converts to HF only for evaluation (where `lm-evaluation-harness` expects HF). HF↔Megatron conversion uses Apertus team's `swiss-ai/hfconverter` — V14 status-check in [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md).

**Why Megatron-LM-Swiss-AI and not HF Trainer**: Apertus's xIELU + QK-Norm + AdEMAMix + Goldfish are all upstream-Apertus kernels not present in HF's stock `Trainer`/`accelerate`. Routing CPT through HF would force us to either re-implement those kernels or fall back to a different optimizer/activation/loss, which is the "trick" the review is specifically guarding against. (See [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md).)

---

## 2. Optimizer: AdEMAMix

**Citation**: Pagliardini, Ablin, Grangier (Apple). *The AdEMAMix Optimizer: Better, Faster, Older.* ICLR 2025. [arXiv:2409.03137](https://arxiv.org/abs/2409.03137), [OpenReview `jj7b3p5kLY`](https://openreview.net/pdf?id=jj7b3p5kLY).

### 2.1 Update rule

Per arXiv 2409.03137 §3:

```
m1(t) = β1·m1(t-1) + (1-β1)·g(t),       m̂1 = m1 / (1 - β1^t)            # fast EMA, bias-corrected
m2(t) = β3·m2(t-1) + (1-β3)·g(t)                                           # slow EMA, NOT bias-corrected
ν(t)  = β2·ν(t-1)  + (1-β2)·g(t)²,      ν̂  = ν  / (1 - β2^t)
θ(t)  = θ(t-1) − η · (m̂1 + α·m2) / (√ν̂ + ε) − λ·θ(t-1)
```

The slow EMA `m2` having no bias correction is **load-bearing**: it is precisely why a cold-zero `m2` under-contributes early in training rather than blowing up. See §2.3.

### 2.2 Hyperparameters

All values from Apertus tech report ([arXiv:2509.14233](https://arxiv.org/abs/2509.14233)) Table C.4 (p.82) + production sbatch [`pretraining/submit_apertus_8b.sh`](https://github.com/swiss-ai/pretrain-code/blob/main/pretraining/submit_apertus_8b.sh).

| Param | Apertus pretrain | Bakeoff (this run) | Citation |
|---|---|---|---|
| β1 | 0.9 | 0.9 | paper Table C.4; sbatch L208 `--adam-beta1` |
| β2 | 0.999 | 0.999 | paper Table C.4; sbatch L209 `--adam-beta2` |
| β3_end | 0.9999 | 0.9999 | paper Table C.4; sbatch L211 `--ademamix-beta3` |
| α (slow-EMA weight) | 8.0 | 8.0 | paper Table C.4; sbatch L210 `--ademamix-alpha` |
| weight_decay | 0.1 | 0.1 | paper Table C.4; sbatch L207 `--weight-decay` |
| α / β3 warmup | 100,000 steps (~2.8 % of run) | **238 steps** (50 % of bakeoff) | paper §C p.81; bakeoff scales for the short 477-step run |
| init std | 0.008944 | 0.008944 (inherited from base ckpt) | paper Table C.4; sbatch L235 `--init-method-std` |

### 2.3 β3 warmup schedule (paper Eq.; non-trivial)

Per arXiv:2409.03137 Appendix: a *linear* β3 ramp is wrong because constant Δβ3 has wildly different effect on half-life near 0.9 vs near 0.9999. The paper proposes a log-form schedule so half-life is linear in `t`:

```
β3(t) = min( exp( ln(β_start)·ln(β3_end) /
                  ( (1 − t/T_β3)·ln(β3_end) + (t/T_β3)·ln(β_start) ) ),
             β3_end )
```

with `β_start = β1` in their experiments. α uses a simple linear warmup: `α(t) = min(t·α_end/T_α, α_end)`.

In all AdEMAMix paper experiments, `T_β3 = T_α = T_total` (full training horizon). **Apertus deviates**: it uses a fixed `T_β3 = T_α = 100,000 steps` (the "first checkpoint of WSD"); after that, β3 and α stay at their target values for the rest of the 15 T-token pretraining ([paper §C p.81](https://arxiv.org/abs/2509.14233)). Our bakeoff uses `T_β3 = T_α = 238` (50 % of the ~477-step horizon) since Apertus's 2.8 %-of-run policy collapses to ~14 steps at our scale, which is too short to be meaningful.

### 2.4 Cold restart (no Apertus pretraining optimizer state)

We have Apertus model weights but **not** the pretraining optimizer state. Per arXiv:2409.03137 §"Switching optimizers" (which covers the Adam→AdEMAMix transition):

- Initialize `m2(t_switch) = 0` (zero tensor, same shape as params).
- Reset scheduler clocks: replace `t` with `t − t_switch` in the β3 and α schedules.
- The paper notes: *"schedulers are not required when resuming training"* — without them, β3 = 0.9999 and α = α_end from step 0 of the new run; loss bumps briefly, then crosses below AdamW.

Our case (no `m1`, `ν`, or `m2` recovered) is *stricter* than the paper's Adam→AdEMAMix experiment. The math (§2.5) shows `m1` and `ν` refill in ~10–1000 steps respectively, while `m2` at β3=0.9999 has a ~10000-step half-life and dominates cold-start behavior. We adopt the **conservative** policy: use the β3 + α schedules with `T_β3 = T_α = T_bakeoff_total` (≈ 477 steps per arm). For production CPT (~15–20 B tokens), `T_β3 = T_α = T_production_total`.

### 2.5 Slow-EMA fill fraction at our horizon

The slow EMA fills as `(1 − β3^k)` over `k` steps (math, not empirical). With β3 = 0.9999:

| k (steps) | filled fraction | effective slow contribution (α=8) |
|---|---:|---:|
| 500 (≈ bakeoff arm length at 4 M tokens/step) | ~4.9 % | ~0.4 of full |
| 1,000 | ~9.5 % | ~0.76 |
| 5,000 (early production) | ~39 % | ~3.1 |
| 10,000 (one half-life by construction) | ~63 % | ~5.0 |

→ At the bakeoff scale, AdEMAMix is empirically very close to AdamW behavior. The decision to use AdEMAMix in the bakeoff is for **optimizer-state continuity into production**, not for the bakeoff comparison itself (where it doesn't matter much).

### 2.6 Implementation references

- **Authoritative for Apertus** (the one we actually use): [`swiss-ai/Megatron-LM/megatron/core/optimizer/ademamix.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/core/optimizer/ademamix.py).
- **Author's official PyTorch reference**: [github.com/apple/ml-ademamix](https://github.com/apple/ml-ademamix) (Pagliardini et al.).
- **Upstream PyTorch tracker** (not yet merged): [pytorch/pytorch#135609](https://github.com/pytorch/pytorch/issues/135609).

---

## 3. LR schedule: WSD with re-warmup

**Citation**: Hu et al. 2024, *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* ([arXiv:2404.06395](https://arxiv.org/abs/2404.06395)). Apertus follows this pattern (per [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md) §6).

Apertus pretraining schedule confirmed from paper §2.3 (p.12) + Table C.4 (p.82) + sbatch L237-247:

- Peak LR = **1.1e-4**
- Min/final LR = **1.1e-5** (= 0.1 × peak, paper's "factor of 0.1" pattern)
- Warmup = linear from 0.1× peak over **16.78 BT** (4,096,000 samples × 4 K seq) — sbatch `--lr-warmup-samples 4096000`
- Stable phase = until 13.5 T tokens
- Decay shape = **1-sqrt** (negative square root), NOT linear — paper §2.3, sbatch `--lr-wsd-decay-style 1-sqrt`
- Decay duration = 1.5 T tokens (13.5 T → 15 T)

For our CPT bakeoff:

| Stage | Tokens | LR trajectory |
|---|---|---|
| Re-warmup | ~2 % of run (≈ 40 M tokens for 2 B bakeoff) | linear from `LR_peak/10` → `LR_peak` |
| Stable | bulk of run | `LR_peak` |
| Decay | remainder | **1-sqrt** from `LR_peak` → `LR_final` (matching Apertus's pretrain shape) |

CPT-specific values:
- `LR_peak = 1.5e-5` (≈ 14 % of Apertus pretrain peak; CPT operates near-converged — see cpt_plan.md v0.7 §3.3)
- `LR_final = 1.5e-6` = 0.1 × peak, matching Apertus's factor-0.1 pattern
- `decay_style = 1-sqrt` ([Cite: Apertus pretrain shape, paper §2.3])

---

## 4. Gradient clipping: 0.1 global-norm

Apertus uses an unusually tight clip: **0.1 global-norm**. This is one of the four mechanisms ([`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md) §3) producing per-token embedding-norm convergence and **must be preserved** — relaxing it would let new-token embeddings drift in norm away from existing rows. **Confirmed**: paper Table C.4 (p.82), sbatch L207 `--clip-grad 0.1`.

---

## 5. Architecture (preserved from Apertus base)

All Apertus-specific architectural pieces inherit from the base checkpoint via `--load <init-checkpoint> --use-checkpoint-args`. The training engine reads these settings from the checkpoint metadata; we do not re-declare them at the sbatch level.

| Component | Apertus choice | Reference |
|---|---|---|
| Hidden / layers / heads / KV-heads / intermediate | 4096 / 32 / 32 / 8 / 21504 | paper Table 1 p.9 |
| Activation | **xIELU** with per-layer trainable αp, αn scalars. **Init αp = αn = 0.8** (β = 0.5); stored via inverse-softplus | paper §2.1 p.10; **code: [`megatron/training/activations.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/activations.py) `XIELU.__init__` defaults** |
| Attention norm | **QK-Norm**: RMSNorm, per-head, applied **before** RoPE | paper §2.1 p.9 "We replace LayerNorm with RMSNorm"; code: [`megatron/core/transformer/attention.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/core/transformer/attention.py) L652-656 (norm before), L432 (RoPE after); sbatch L192 `--qk-layernorm --qknorm-impl apex --normalization RMSNorm` |
| Position encoding | RoPE θ=500,000 pretrain (extended to 12 M for 64 K long-context), llama3-style scaling factor 8, max_pos 4096 pretrain | paper Table C.4 p.82 |
| Embeddings | **Untied** (`tie_word_embeddings=false`); bias terms removed everywhere | paper §2.1 p.10 |
| Pre-Norm + RMSNorm | yes | paper §2.1 |
| Tokenizer base | Mistral-Nemo tekken v3 byte-level BPE, vocab 131,072 (extended to 148,480 for ReTok/Centroid arms) | paper §2.2 p.10 |

**V15 risk** (per [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md) §4): after `resize_token_embeddings` for the ReTok/Centroid arms, the per-layer trainable αp/αn scalars must remain in the optimizer's parameter list. Check at init-checkpoint build time (`arms/build_init_checkpoints.py`).

---

## 6. Sequence + batch + duration

| Param | Apertus pretrain | Bakeoff | Citation |
|---|---|---|---|
| Sequence length | 4,096 | 4,096 | paper Table 2 p.10; sbatch L218 `SEQ_LEN=4096` |
| Global batch (tokens/step) | 4.19 M initial → 8.39 M after 8 T tokens | **4.19 M** (= 1024 × 4096) | paper Table 2; sbatch L221 `--micro-batch-size 4`, ramp at L222 |
| Global batch (samples) | 1024 → 2048 | 1024 | paper Table 2 |
| Micro-batch (per GPU) | 4 | 4 (calibrate at V4 baseline) | sbatch L221 |
| Tokens / run | 15 T | 2 B per arm (bakeoff) / 15-20 B (production) | paper §3 p.21; cpt_plan v0.7 §3 + §5 |
| Iterations / run | 3,662,109,375 samples ÷ 1024 ≈ 3.58 M iters | ≈ 477 iters (bakeoff) | sbatch L219 `--train-samples`; derived for bakeoff |

---

## 7. Loss objective

### 7.1 Bakeoff: Next-Token Prediction (NTP)

Standard cross-entropy on the model's next-token logits. **Goldfish disabled for the bakeoff** to keep the variable-of-interest (init quality) clean. Per cpt_plan.md v0.7 §10 Q B4: "comparison is on init, not on loss-objective; loss is held constant across arms = NTP".

### 7.2 Production: Goldfish loss

**Citation**: Hans, Wen, Jain et al. 2024, *Goldfish Loss: Mitigating Memorization in Generative LLMs* ([arXiv:2406.10209](https://arxiv.org/abs/2406.10209)).

Apertus uses Goldfish in pretraining for memorization mitigation. For our production CPT we restore Goldfish; for the bakeoff we keep NTP.

Apertus's exact configuration (paper §2.3 p.11, Table C.4, Algorithm 1 p.86; impl confirmed in code):

| Param | Apertus value | Source |
|---|---|---|
| Mask frequency | `k = 50` (one in 50 tokens masked → ~2 % of tokens) | paper §2.3 p.11; sbatch `--goldfish-k 50` |
| Hash context window | `h = 50` preceding tokens | paper §2.3, Table C.4; sbatch `--goldfish-h 50` |
| Hash function | Deterministic: `prod(last h tokens) mod table_size`; precomputed uniform random hash-table of size **1,000,003** seeded **`2971215073`** | code: [`megatron/core/datasets/gpt_dataset.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/core/datasets/gpt_dataset.py) `_create_hash_table`, `apply_goldfish` at L741 |
| Implementation | **Front-loaded during data loading** (not at training time) | paper §2.3 p.11, §F p.85 |

---

## 8. Document boundaries

Per [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md) §7-8:

| Mechanism | Setting | Megatron flag |
|---|---|---|
| Cross-document attention mask | **ON** (each document is attention-isolated) | `--reset-attention-mask --reset-position-ids` |
| EoD loss mask | **ON** (the EoD token itself is excluded from the loss) | `--eod-mask-loss` |

Both are V12 + V13 in [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md) and gate the bakeoff sbatch launch.

---

## 9. Mixed precision

- Activations / gradients: **bf16** (`--bf16`)
- Master grads: **fp32** (`--main-grads-dtype fp32`)

Per Apertus pretraining defaults (paper Appendix D p.83, sbatch L184 + L255). The paper notes that FP8 was tried at ~8 T tokens, caused loss degradation, and was **rolled back** for the production run — so the bf16 + fp32-master-grads policy is the empirically-validated Apertus configuration.

---

## 10. Embedding initialization for vocab extension (the three arms)

This is the experimental axis. See [`init_bakeoff/BAKEOFF_PLAN.md`](03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) for the full design; below is the citation summary.

| Arm | Method | Citation |
|---|---|---|
| **Vanilla** | No vocab extension — train on the original 131,072-token Apertus vocabulary | n/a (control) |
| **ReTok** | New-token row = mean of base-tokenizer subpiece embeddings | (a) **Origin** = FVT, Gee et al. 2022 ([ACL Anthology 2022.emnlp-industry.41](https://aclanthology.org/2022.emnlp-industry.41/)) — first subpiece-mean formulation. (b) **LLM-era + both-E-and-U** = ReTok, Gu et al. 2024 ([arXiv:2410.04335](https://arxiv.org/abs/2410.04335)) — explicit application to both embedding layer and LM head. |
| **Centroid** | New-token row sampled from `N(μ, Σ)` of base Greek-token embeddings (full covariance + 1e-8 ridge), then norm-matched | **Hewitt 2021**, *Initializing New Word Embeddings for Pretrained Language Models* ([cs.columbia.edu/~johnhew/vocab-expansion.html](https://www.cs.columbia.edu/~johnhew//vocab-expansion.html); local: [`references/papers/hewitt_vocab_expansion.html`](references/papers/hewitt_vocab_expansion.html); code at [github.com/john-hewitt/embed-init](https://github.com/john-hewitt/embed-init)). **Script-restricted variant** (centroid of Greek tokens only) is our extension. Full Σ (not diagonal-only) used after audit Q6 — Mundra 2024 §5.1 + Table 2 (p. 6) explicitly calls the diagonal "Univariate" variant inadequate. |

### 10.1 Norm-matching post-pass

Both ReTok and Centroid apply a final scale to make the new rows' norm match the base-vocab mean norm (Phase A targets `E=5.05, U=3.80` per cpt_plan.md v0.7 §5). This is field convention rather than a single paper — empirical support: Mundra et al. 2024 ([arXiv:2407.05841](https://arxiv.org/abs/2407.05841)).

### 10.2 Untied E and U

Apertus has `tie_word_embeddings=False`. Both ReTok and Centroid are applied **independently** to E (input embedding) and U (LM head):

- ReTok (Gu et al. 2024 arXiv:2410.04335) is explicit about this — applied to both matrices independently. ✓ field convention.
- Hewitt 2021 discusses E only; the U application is the same algorithm computed over U's existing rows. Field convention.

WECHSEL ([arXiv:2112.06598](https://arxiv.org/abs/2112.06598)) and FOCUS ([arXiv:2305.14481](https://arxiv.org/abs/2305.14481)) are **not** used — they require external auxiliary embeddings (fastText / static), which adds an axis of comparison we explicitly chose not to introduce.

---

## 11. Data

See [`init_bakeoff/corpus_build/MIX_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md) for the full bucket allocation, per-source weights, and HF dataset paths. Recipe summary: **70 % Greek / 26 % multilingual replay / 4 % code**.

### 11.1 NFC normalization (V9)

Apertus's tokenizer has `normalizer: null` per `tokenizer.json`. Pre-tokenization NFC at the corpus level is therefore **required** to keep training text aligned with inference. We enforce V9 via the upstream pipeline:

- HPLT clean60: 500/500 NFC sample-verified
- finepdfs-edu: 0.07 % NFD leak detected and remediated
- `scripts/verify_and_normalize_nfc.py` (idempotent in-place normalize) — runs over the parquets between download and mix-build

For the replay datasets (FineWeb-2 / FineWeb-2-HQ / FineWeb-Edu / StarCoder), NFC compliance is assumed upstream; if Agent C surfaces a counter-example we add an explicit NFC pass to the mix builder.

### 11.2 Determinism

The bakeoff's three arms share the **same Megatron data prefix** and the **same data seed** (`DATA_SEED=20260520`). Token streams are byte-identical up to the init differential. This is what makes the three-arm comparison apples-to-apples.

---

## 12. Evaluation

See [`init_bakeoff/eval/EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md). Headline:

- Engine: [`swiss-ai/lm-evaluation-harness`](https://github.com/swiss-ai/lm-evaluation-harness) (Apertus team's fork of EleutherAI's harness). Apertus tech report ([arXiv:2509.14233](https://arxiv.org/abs/2509.14233) §5.1 footnote 45, p.38) explicitly cites this fork — branch + pinned commit to be confirmed (PENDING Agent A).
- Citation: Gao, Tow, Abbasi, et al., *The Language Model Evaluation Harness*, Zenodo 2024, DOI [10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602).
- V4 baseline: full suite × unmodified Apertus-8B-2509, **once**, before the bakeoff fires. Output gates §5.6 hard-gate thresholds.
- Per-arm: same suite at checkpoints in 80–100 % of each arm's budget. Selection = windowed average + bootstrap CI ([`compute_bootstrap_cis.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_bootstrap_cis.py)).

---

## 13. Resolution status (2026-05-20)

| ID | Question | Status |
|---|---|---|
| C2 | Apertus's exact AdEMAMix hyperparams (β1/β2/β3/α/wd) | **RESOLVED** — paper Table C.4 p.82; values in [`_train_config_common.env`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/_train_config_common.env) |
| C2b | Apertus's exact LR-schedule shape | **RESOLVED** — peak 1.1e-4, min 1.1e-5, warmup 16.78 BT linear from 0.1× peak, 1-sqrt cooldown over 1.5 T (paper §2.3 p.12, Table C.4 p.82) |
| C2c | Apertus's global batch size + Megatron-side parallelism config | **RESOLVED** — 4.19 M initial → 8.39 M after 8 T (paper Table 2; sbatch L221-L222) |
| C4 | Apertus's Goldfish loss config | **RESOLVED** — k=h=50, hash table 1,000,003 seeded 2971215073 (paper §2.3 p.11; code path `megatron/core/datasets/gpt_dataset.py`) |
| D1 | swiss-ai/Megatron-LM fork branch + pinned commit | **RESOLVED** — `main` branch (no tag pinned by Apertus); we pin `c92402e39ef3c8e69ea378a59e79059dc14541f4` (HEAD 2026-05-20) |
| D2 | swiss-ai/Megatron-LM CLI flag names | **RESOLVED** — see [`_train_config_common.env`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/_train_config_common.env) + [`bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch) |
| D3 | Apertus preprocessing pipeline (vs. stock Megatron `tools/preprocess_data.py`) | **RESOLVED** — Apertus uses `swiss-ai/pretrain-data` (DataTrove-based). Our `preprocess_data.sbatch` uses Megatron's stock `tools/preprocess_data.py` instead since it produces byte-identical output and is simpler. |

All hard-blocking lookups are resolved as of 2026-05-20. The recipe is review-ready.

---

## 14. Citation appendix

| Component | Citation |
|---|---|
| Base model | Swiss AI. *Apertus-8B-2509.* [huggingface.co/swiss-ai/Apertus-8B-2509](https://huggingface.co/swiss-ai/Apertus-8B-2509). Tech report: Apertus team, [arXiv:2509.14233](https://arxiv.org/abs/2509.14233). |
| Engine | Shoeybi et al. *Megatron-LM*. [arXiv:1909.08053](https://arxiv.org/abs/1909.08053). Swiss AI fork: [github.com/swiss-ai/Megatron-LM](https://github.com/swiss-ai/Megatron-LM). |
| Optimizer | Pagliardini, Ablin, Grangier. *The AdEMAMix Optimizer.* ICLR 2025. [arXiv:2409.03137](https://arxiv.org/abs/2409.03137). |
| Activation | xIELU — Huang & Schlag 2025 (cited in Apertus tech report §2.1 p.10). Apertus impl: [`megatron/training/activations.py`](https://github.com/swiss-ai/Megatron-LM/blob/main/megatron/training/activations.py). Init `αp = αn = 0.8`, β = 0.5 (code-canonical; paper does not enumerate). |
| Attention norm | Henry et al. *Query-Key Normalization for Transformers.* [arXiv:2010.04245](https://arxiv.org/abs/2010.04245). |
| LR schedule | Hu et al. *MiniCPM (WSD schedule).* [arXiv:2404.06395](https://arxiv.org/abs/2404.06395). |
| Loss (production) | Hans, Wen, Jain et al. *Goldfish Loss.* [arXiv:2406.10209](https://arxiv.org/abs/2406.10209). |
| Init: ReTok | (a) Gee et al. *Fast Vocabulary Transfer.* EMNLP 2022 Industry. [aclanthology.org/2022.emnlp-industry.41](https://aclanthology.org/2022.emnlp-industry.41/). (b) Gu et al. *ReTok.* [arXiv:2410.04335](https://arxiv.org/abs/2410.04335). |
| Init: Centroid | Hewitt. *Initializing New Word Embeddings.* [cs.columbia.edu/~johnhew/vocab-expansion.html](https://www.cs.columbia.edu/~johnhew//vocab-expansion.html). |
| Init: empirical comparison | Mundra et al. *An Empirical Comparison of Vocabulary Expansion and Initialization Approaches.* [arXiv:2407.05841](https://arxiv.org/abs/2407.05841). |
| Eval harness | Gao, Tow, Abbasi et al. *The Language Model Evaluation Harness.* Zenodo 2024, DOI [10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602). swiss-ai fork: [github.com/swiss-ai/lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) (cited in Apertus tech report §5.1, footnote 45). |
| Eval methodology | Park et al. 2025 — *bootstrap over eval samples for downstream-benchmark CIs* (cited in cpt_plan.md v0.7 §6.1; full ref PENDING). |
| Data — FineWeb-Edu | Penedo et al. 2024. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557). [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) — ODC-BY-1.0; filter = "Score-3" Edu-classifier threshold. |
| Data — FineWeb-2 | Penedo et al. 2024 (FineWeb follow-up). [arXiv:2406.17557](https://arxiv.org/abs/2406.17557); release blog [HuggingFaceFW/blogpost-fineweb-v1](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1). [HuggingFaceFW/fineweb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) — v2.0.1 (Apertus tech report footnote 18, p.20). |
| Data — FineWeb-2-HQ | Messmer, Sabolčec, Jaggi 2025. [arXiv:2502.10361](https://arxiv.org/abs/2502.10361). Exact HF path: **[`epfml/FineWeb2-HQ`](https://huggingface.co/datasets/epfml/FineWeb2-HQ)** (separate EPFML repo, not a config of fineweb-2). Filter = **top-10 % per-language quality via XLM-RoBERTa classifier** — NOT Score-3 (Score-3 belongs to FineWeb-Edu, a different filter). |
| Data — Code | StarCoderData = **The Stack v1.2** subset. [bigcode/starcoderdata](https://huggingface.co/datasets/bigcode/starcoderdata). Citation: Li et al. 2023, *StarCoder: May the Source Be With You*, [arXiv:2305.06161](https://arxiv.org/abs/2305.06161). (Apertus stage-5 cooldown also uses CommonPile/Stack-v2-Edu per footnote 24; we use only StarCoderData v1.2 for the bakeoff.) |
| Greek evals | [ilsp/* collection](https://huggingface.co/collections/ilsp/ilsp-greek-evaluation-suite) on HuggingFace. No dedicated "ILSP suite" paper. The core 6-test-set was introduced in Voukoutis et al. *Meltemi: The first open Large Language Model for Greek* ([arXiv:2407.20743](https://arxiv.org/abs/2407.20743)) and extended in the Krikri paper ([arXiv:2505.13772](https://arxiv.org/abs/2505.13772)). lm-eval-harness task configs live in Meltemi/Krikri forks; not yet upstream — staging step in [`eval/pull_benchmarks.sh`](03_4_implementation_experiments/init_bakeoff/eval/pull_benchmarks.sh). |
