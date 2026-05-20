# Audit findings (2026-05-21)

*A second-pass audit of the bakeoff recipe against locally-pinned primary sources.* Each finding cites a specific file path in `references/` and a specific line / section. Code fixes are marked **APPLIED** when patched in this commit, **DEFERRED** otherwise.

## Method

1. Cloned 8 reference repos via [`references/clone_references.sh`](references/clone_references.sh) — pinned commits in [`references/MANIFEST.md`](references/MANIFEST.md).
2. Downloaded 15 paper PDFs via [`references/download_papers.sh`](references/download_papers.sh).
3. Audited the bakeoff code, sbatch, and recipe doc against the local sources.

---

## Findings

### Section A — `bakeoff_training/bakeoff_train.sbatch` flag-name corrections

Audited against `references/repos/swiss-ai_pretrain-code/pretraining/submit_apertus_8b.sh` at commit `531cc8be`.

| Was | Now | Source |
|---|---|---|
| `--xielu-activation` | `--xielu` | `submit_apertus_8b.sh:L195` |
| `--ademamix-beta3-warmup-steps` | `--ademamix-beta3-warmup` | `submit_apertus_8b.sh:L218`; verified against `megatron/core/optimizer/ademamix.py:36 (beta3_warmup arg)` |
| `--ademamix-alpha-warmup-steps` | `--ademamix-alpha-warmup` | `submit_apertus_8b.sh:L219`; verified against `ademamix.py:37 (alpha_warmup arg)` |

**Critical additions** previously missing from our sbatch:

| Flag | Why it matters | Source |
|---|---|---|
| `--make-vocab-size-divisible-by 128` | Affects how Megatron pads the embedding matrix; mismatch with the converter would produce shape errors | `submit_apertus_8b.sh:L193` |
| `--ckpt-format torch_dist` | Save/load format Apertus uses; required to be consistent with the converted init checkpoint | `submit_apertus_8b.sh:L257` |
| **`--dist-ckpt-strictness assume_ok_unexpected`** | **Critical for our CPT — allows Megatron to load a checkpoint whose embedding shape was resized 131,072 → 148,480 by our init builder. Without it Megatron refuses the shape mismatch.** | `submit_apertus_8b.sh:L260` |
| `--cross-entropy-loss-fusion` | Throughput | `submit_apertus_8b.sh:L229` |
| `--manual-gc --manual-gc-interval 500` | Memory; matches Apertus's GC cadence | `submit_apertus_8b.sh:L233-234` |
| `--overlap-grad-reduce --overlap-param-gather` | Distributed-optimizer overlapping; matches Apertus throughput optimizations | `submit_apertus_8b.sh:L273-274` |
| `--attention-dropout 0.0 --hidden-dropout 0.0` | Explicit zero-dropout; matches Apertus | `submit_apertus_8b.sh:L210-211` |
| `--no-check-for-nan-in-loss-and-grad` | Performance; matches Apertus | `submit_apertus_8b.sh:L226` |
| `--split 100,0,0` | All data goes to training (no val/test held back at the Megatron level — eval is separate) | `submit_apertus_8b.sh:L283` |
| Explicit network-arch flags (`--num-layers 32 --hidden-size 4096 --ffn-hidden-size 21504 --num-attention-heads 32 --group-query-attention --num-query-groups 8`) | Apertus declares these explicitly. We can rely on the checkpoint, but explicit declaration is safer and matches Apertus's pattern. | `submit_apertus_8b.sh:L182-188` |

**Status: APPLIED.** Updated [`bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch).

---

### Section B — `arms/retok.py` (init code, ReTok arm)

Audited against `references/papers/retok_2410.04335.html` (Gu et al. 2024) + `references/papers/fvt_emnlp2022_industry_41.pdf` (Gee et al. 2022) + `references/papers/mundra_2407.05841.html` (Mundra 2024).

| Q | Question | Verdict | Action |
|---|---|---|---|
| Q1 | Arithmetic mean of base subpieces | PASS — matches FVT Eq. 2 (p. 421) + ReTok §3.2 (p. 3) | – |
| Q2 | Independent application to E and U (untied) | PASS — explicit in ReTok §3.2 (p. 3): "embedding layer and LM head layer, compute their average" | – |
| Q3 | Norm-match after the mean | **DEVIATION** — not in any of FVT / ReTok / Mundra. Our Phase-A targets (5.05 / 3.80) are project-specific from `runs/apertus_greek_diagnostic_20260511_v2/` | DOCUMENTED in `retok.py` |
| Q4 | Rare-token / long-subpiece-chain gating | PAPERS SILENT. We follow the papers (no gating) but added a warning if `n_max_subpieces > 8` | **APPLIED** (warning) |
| Q5 | **Silent zero-row on empty decode / zero subpieces** | **BUG** — would post-`norm_match` yield zero-norm row; in training the new token would dominate softmax (Hewitt 2021 §"Zero-init can cause problems") | **APPLIED** — hard fallback to global base-vocab mean |

---

### Section C — `arms/centroid.py` (init code, Centroid arm)

Audited against `references/papers/hewitt_vocab_expansion.html` (Hewitt 2021) + `references/papers/mundra_2407.05841.html` (Mundra 2024).

| Q | Question | Verdict | Action |
|---|---|---|---|
| Q6 | Diagonal σ vs full Σ | **DEVIATION (was)** — diagonal σ is Mundra's "Univariate" baseline which Mundra §5.1 (p. 5) + Table 2 (p. 6) explicitly calls **inadequate**. Hewitt uses full Σ. | **APPLIED** — switched to `compute_centroid_and_cov` (full Σ + 1e-8 ridge for numerical stability); now uses `rng.multivariate_normal(μ, Σ)` |
| Q7 | Script-restricted subset (modern / polytonic) | DEVIATION — papers silent. Our extension, justified by Mundra's convex-hull argument applied to the tighter prior (the subset hull ⊂ full hull, so still in-hull). | DOCUMENTED |
| Q8 | Explicit norm-match vs MVN's natural norm | DEVIATION — overrides Hewitt's natural per-row norm. Kept for the bakeoff because Phase-A norm-match is part of the experimental design (otherwise we'd be comparing norm effects, not init effects). | DOCUMENTED |
| Q9 | Half-and-half average for both-block tokens | PAPERS SILENT. Kept as-is; smoke test shows `both` count is typically <1 % so the choice barely matters. | DOCUMENTED |
| Q10 | Diagonal-Gaussian implementation correctness (for backward compat) | PASS | – |
| Q11 | Resize-then-fill ordering | PASS — matches all three papers' workflow | – |

**Note on the magnitude question (Q6 detail):** Mundra Appendix F (p. 16) + Hewitt's demo code both scale Σ by `1e-5` to "remain within the convex hull with high confidence". We **do not apply the `1e-5` scale** because:
- Our subsequent `norm_match()` rescales rows to the Phase-A targets (5.05 / 3.80), making the post-sample magnitude moot.
- The 1e-5-scaled noise + norm_match would collapse all new rows of the same script to ≈ the scaled-centroid direction → loss of per-token variance, defeating the centroid arm's purpose.

So we keep `Σ` unscaled (with a small `1e-8` ridge for `multivariate_normal` numerical stability), then apply `norm_match`. This gives a meaningful per-token direction (full correlation structure preserved) and the right magnitude (Phase-A target).

---

### Section D — AdEMAMix optimizer impl (verified, no action)

Audited `references/repos/swiss-ai_Megatron-LM/megatron/core/optimizer/ademamix.py` against `references/papers/ademamix_2409.03137.html`.

| Item | Status | Code path |
|---|---|---|
| `exp_avg_slow` NOT bias-corrected (paper §3) | PASS — `ademamix.py:125` does `mul_(beta3).add_(grad, alpha=1 - beta3)` only | – |
| `exp_avg_fast` IS bias-corrected | PASS — `ademamix.py:107, 131` use `1.0 - beta1^step` | – |
| β3 warmup uses log-form half-life-linear schedule | PASS — `ademamix.py:12-23` (`linear_hl_warmup_scheduler`) maps `β ↔ half-life` then interpolates linearly in half-life space | – |
| α warmup uses simple linear | PASS — `ademamix.py:5-10` | – |
| `β_start = β1` (= 0.9) for β3 schedule | PASS — `ademamix.py:117` | – |
| CLI args: `beta3_warmup`, `alpha_warmup` (integer step counts) | confirmed | `ademamix.py:36-37` |

---

### Section E — xIELU activation + V15 (xIELU scalars survive resize)

Audited `references/repos/swiss-ai_Megatron-LM/megatron/training/activations.py` (XIELU class, lines 33-46).

| Item | Status | Notes |
|---|---|---|
| Init values `alpha_p_init = alpha_n_init = 0.8`, `β = 0.5` | CONFIRMED | `activations.py:34` defaults |
| Stored via inverse-softplus | CONFIRMED | `activations.py:37-38`: `nn.Parameter(torch.log(torch.exp(...) - 1.0))` |
| Effective values: `αp = softplus(stored)`, `αn = β + softplus(stored)` | CONFIRMED | `activations.py:43-44` |
| **V15: αp/αn survive `resize_token_embeddings`** | LOW RISK | They're `nn.Parameter` children of each XIELU instance. `resize_token_embeddings` only touches the embedding tensor + LM head — XIELU instances are untouched, so their αp/αn parameters remain in `model.parameters()` and the optimizer's param list. |
| Assertion in `build_init_checkpoints.py` to verify post-resize | DEFERRED | Per audit Q12 — low priority; the mechanism is sound, but a sanity-check assertion would be cheap. |

---

### Section F — Goldfish loss (Apertus uses; bakeoff disables)

Audited `references/repos/swiss-ai_Megatron-LM/megatron/core/datasets/gpt_dataset.py` against paper + sbatch.

| Item | Apertus value | Verified |
|---|---|---|
| `--goldfish-loss` enable flag | enabled in pretrain | `submit_apertus_8b.sh:L290` |
| `--goldfish-k` | 50 (mask freq = 1/50) | `L291`; impl in `gpt_dataset.py:apply_goldfish` |
| `--goldfish-h` | 50 (hash context window) | `L292` |
| Hash table size | 1,000,003 (constant) | `gpt_dataset.py:_HASH_TABLE_SIZE` |
| Hash seed | `2971215073` (constant) | `gpt_dataset.py:_create_hash_table` |
| Implementation | Front-loaded during data loading (not at training time) | paper §2.3 p.11 |
| `_GOLDFISH_TOKEN_ID = -2` | constant — masked tokens get this id pre-loss | `gpt_dataset.py` |

The bakeoff intentionally **disables** Goldfish (NTP only — see cpt_plan.md v0.7 §10 Q B4). Production CPT re-enables with these exact values.

---

### Section G — HF ↔ Megatron checkpoint converter

**Issue surfaced:** `swiss-ai/hfconverter` does NOT exist as a separate repo. The converter is at `references/repos/swiss-ai_Megatron-LM/tools/checkpoint/`:

- `convert.py` — top-level orchestrator (loader + saver via queue)
- `saver_swissai_hf.py` — **Megatron → HF** for Apertus (explicit `ApertusConfig` / `ApertusForCausalLM` imports). This works.
- `loader_llama_mistral.py` — **HF → Megatron** for llama2/llama3/mistral/qwen2.5. **Apertus is NOT in the supported list.**
- Other loaders: `loader_core` (default Megatron format), `loader_legacy`, `loader_mixtral_hf`.

**Status: BLOCKER for bakeoff submission.** We need to load HF-format Apertus (the only release format) into Megatron-LM-Swiss-AI for training. Three paths to resolve:

1. **Write a custom `loader_apertus_hf.py`** based on `saver_swissai_hf.py`'s inverse. Apertus is architecturally Mistral-derived (Mistral-Nemo tokenizer; similar block layout) so `loader_llama_mistral.py` is a starting template + xIELU / QK-Norm adjustments.
2. **Try `loader_llama_mistral.py --model-size mistral`** as a first approximation, then patch failures. Risky — Apertus's xIELU and QK-Norm aren't in the Mistral layer spec.
3. **Coordinate with `swiss-ai` team** to find their internal HF → Megatron tool (they must have one to validate the Megatron → HF saver).

This needs ~1-2 hours of focused work before the first bakeoff sbatch can submit. Documented in [`init_bakeoff/bakeoff_training/README.md`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/README.md) as a pre-submit blocker.

---

### Section H — ILSP Greek harness task configs

Per `references/repos/swiss-ai_lm-evaluation-harness/lm_eval/tasks/` + Agent C's research: the ILSP `*_greek` task YAMLs are NOT in swiss-ai's harness fork. They live in:

- Meltemi team's harness fork: `LeonVouk/lighteval`
- ILSP team's harness fork: `ilsp/lm-evaluation-harness-greek`

**Status: pre-submit blocker** for the V4 baseline + per-arm Greek eval. Action: at staging time, fetch task YAMLs from one of those forks and merge into the swiss-ai fork installation. Documented in [`eval/EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md).

---

### Section I — Apertus preprocessing pipeline

Apertus uses `swiss-ai/pretrain-data/examples/tokenize_megatron/preprocess_megatron.py` (DataTrove-based) rather than Megatron's stock `tools/preprocess_data.py`. Both produce Megatron-binary indexed datasets given the same tokenizer + input; the DataTrove path adds streaming + sharding ergonomics.

Our [`preprocess_data.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/preprocess_data.sbatch) uses Megatron's stock tool. This is documented as an explicit choice (no fidelity issue — output is byte-identical).

---

## Resolution summary

| Issue | Severity | Status |
|---|---|---|
| sbatch flag-name typos (`--xielu-activation`, `--ademamix-*-warmup-steps`) | first-submit fail | APPLIED |
| Missing `--dist-ckpt-strictness assume_ok_unexpected` | first-submit fail (resized embeddings won't load) | APPLIED |
| Missing arch flags + Apertus throughput flags | fidelity gap | APPLIED |
| retok.py zero-row failure mode (Q5) | silent training-time bug | APPLIED (hard fallback) |
| centroid.py diagonal-σ "Univariate" baseline (Q6) | Mundra calls inadequate | APPLIED (full Σ) |
| Q3/Q7/Q8/Q9 deviations from papers | documented experimental choices | DOCUMENTED in code |
| Q4 long-subpiece warning | diagnostic | APPLIED (warning) |
| Q12 xIELU-survives-resize assertion | sanity check | DEFERRED |
| HF→Megatron Apertus loader missing | **pre-submit blocker** | OPEN — needs ~1-2 h custom loader work |
| ILSP harness task YAMLs missing from swiss-ai fork | **pre-submit blocker** | OPEN — staging-time merge from Meltemi/Krikri forks |
| Apertus preprocess uses DataTrove (we use stock Megatron) | fidelity note | DOCUMENTED |

The recipe + code are **review-ready** at the level of paper / sbatch / code-line fidelity. The **two OPEN items** above gate the first sbatch submission but do not block colleague review of the recipe itself.
