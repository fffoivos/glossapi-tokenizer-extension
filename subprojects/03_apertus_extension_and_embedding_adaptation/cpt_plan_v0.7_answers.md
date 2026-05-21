# cpt_plan v0.7 — Answers (2026-05-21)

*Compact response covering every question in [`cpt_plan.md`](cpt_plan.md) v0.7 §10 (decisions), §11 (lookups), §12 (verifications). For each item: current status + one-line answer or pointer. Detailed V audit in [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md); reviewer-facing summary in [`REVIEW_PRESENTATION.md`](REVIEW_PRESENTATION.md).*

Status legend: **RESOLVED** = answer known + cited · **LOCKED** = working default in code · **PENDING** = needs Fivos input · **DEFERRED** = explicitly out of scope for v0.7 · **NOT POSSIBLE** = answer doesn't exist in available sources.

---

## Q A — Decisions from Fivos (§10)

These are Fivos's calls, not technical lookups. v0.7's framing keeps placeholders.

| # | Question | Status | Notes |
|---|---|---|---|
| A1 | Capability targets | **DEFERRED** | v0.7 itself defers this; placeholder defaults flow downstream. |
| A2 | Total token budget for CPT post-init | **PENDING** | Working assumption: 15-20 B (cpt_plan §3). Needs explicit confirmation. |
| A3 | Compute timeline / deadline | **PENDING** | Bakeoff sizing assumes ~12 h-per-arm Clariden `normal` budget; production budget gated on A2. |
| A4 | Stakeholders / downstream consumers | **PENDING** | Determines decontamination scope (V1). Until set, we treat ILSP suite + Global-MMLU as comparison-grade. |
| A5 | Colleague sign-off on shuffled-bulk + annealing | **PENDING** | Both encoded in our recipes ([`bulk.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json), [`anneal.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json)); no objections received but no explicit sign-off either. |
| A6 | Specific downstream tasks | **PENDING** | Affects eval-suite emphasis (§5.6 weighted score); using v0.7 §5.6 midpoint weights until set. |
| A7 | Team structure | **PENDING** | Soft dependency. |

---

## Q B — Design decisions (§10)

All have working defaults locked into our recipes. None are explicitly confirmed by Fivos.

| # | Question | Status | Current value | Where |
|---|---|---|---|---|
| B1 | Outer Greek/non-Greek split | **LOCKED** | 70 % Greek / 24 % replay / 4 % code / 2 % math (post-Item-2 rebalance) | [`bulk.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json) |
| B2 | Code share | **LOCKED** | 4 % (StarCoderData) | [`bulk.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json) |
| B3 | Anneal composition priority | **LOCKED** | (d) balanced — 85/12/3 Greek/replay/code | [`anneal.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json) |
| B4 | Loss objective for bakeoff | **LOCKED** | NTP for bakeoff, Goldfish for production | [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) §7 |
| B5 | Init experiment budget per variant | **LOCKED** | 2 B tokens | [`BAKEOFF_PLAN.md`](03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) §5 |
| B6 | Adaptation work prioritization | **LOCKED** | per [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) + [`RISKS.md`](RISKS.md) | this commit |

---

## Q C — Apertus lookups (§11)

| # | Question | Status | Answer + citation |
|---|---|---|---|
| C1 | Apertus pretraining peak LR | **RESOLVED** | 1.1e-4; CPT divergence to 1.5e-5 (v0.7 §3.3). Source: Apertus paper Table 2; sbatch L245. |
| C2 | AdEMAMix optimizer hyperparams | **RESOLVED** | β1=0.9, β2=0.999, β3=0.9999, α=8, wd=0.1, α/β3 warmup 100k steps. Source: Apertus paper Table C.4 p.82; sbatch L208-219. Full table: [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) §2. |
| C3 | Per-language token shares | **PARTIAL — empirical path proposed** | Apertus tech report publishes **FW2 document counts** per language (Appendix G Table G.6 p.88), not training-mix token shares. Token shares can be derived by tokenizing the published datasets ourselves with Apertus's tokenizer, weighted by the per-stage iteration shares in Table 6 p.24. ~4-6 h on Clariden xfer; not yet run. |
| C4 | Goldfish loss configuration | **RESOLVED** | k = 50, h = 50, hash table size 1,000,003, seed `2971215073`, hash function `prod(last h tokens) mod table_size`. Source: Apertus paper §2.3 p.11 + code path `megatron/core/datasets/gpt_dataset.py:apply_goldfish`. |
| C5 | Apertus tokenizer config | **RESOLVED** | Mistral-Nemo tekken v3 byte-level BPE, vocab 131,072. Extension to 148,480 (modern-only) verified end-to-end (V14, V16). Source: Apertus paper §2.2 p.10. |

---

## Q D — Engineering lookups (§11)

| # | Question | Status | Answer + citation |
|---|---|---|---|
| D1 | Megatron-LM fork branch / commit | **RESOLVED** | `swiss-ai/Megatron-LM` `main` HEAD pinned at `c92402e39ef3c8e69ea378a59e79059dc14541f4`. Apertus production does not pin a tag (paper footnote 4). Source: [`references/MANIFEST.md`](references/MANIFEST.md). |
| D2 | FineWeb-2 Tier 3 language audit | **PENDING** | Token counts for `lat_Latn` / `hye_Armn` / `kat_Geor` / `sqi_Latn` / `mkd_Cyrl`. v0.7 says: under ~100 M → "preservation aspiration" not "active maintenance". Cheap to run via HF `datasets` `info()`; not done. |
| D3 | Apertus intermediate checkpoints | **PENDING** | Available on HF branches; useful for annealing-as-quality-meter. Not bakeoff-gating; not yet enumerated. |

---

## V1-V16 — Verifications (§12)

Detail per item in [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md). Summary below; status as of 2026-05-21 after Items 1-6 + audit pass:

| # | Status | One-line |
|---|---|---|
| V1 | NOT DONE | Decontamination scope — item-level dedup of clean-measurement benchmarks against training data. Scheduled Clariden `xfer`; gated on Q A4. |
| V2 | PARTIAL | Tokenizer side ✓ (`build_and_verify_ship_tokenizer.py`); model-resize forward pass scheduled for Clariden debug slot. |
| V3 | NOT DONE | Dataloader-state preservation — Megatron default; smoke verify on first Clariden debug submit. |
| V4 | NOT DONE | Run-to-run variance baseline on unmodified Apertus-8B-2509 — **gates §5.6 hard-gate thresholds**. ~3-4 h Clariden `normal`. |
| V5 | NOT DONE | Polytonic exposure audit — gates on CPT corpus build. |
| V6 | NOT DONE | Accent-normalized dedup re-verification — original ran with `preserve`; ~2-3 h Clariden `xfer`. |
| V7 | PARTIAL | Pull scripts ready ([`corpus_build/pull_*.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/)); Clariden login-node pull pending go-ahead. |
| V8 | READY | Goldfish hash uniformity. **Q C4 unblocked**; production-only (bakeoff is NTP). |
| V9 | DONE | NFC normalization — operationally enforced via [`normalize_nfc.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/normalize_nfc.sh) wrapper (Item 2). |
| V10 | DEFERRED | vLLM / SGLang compatibility — post-pilot. |
| V12 | DONE (config) | Cross-document attention mask — `--reset-attention-mask --reset-position-ids` in [`bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch). Runtime verification on first Clariden submit. |
| V13 | DONE (config) | EoD loss mask — `--eod-mask-loss` in sbatch. R8 (special-token-ID equality assertion) flagged in [`RISKS.md`](RISKS.md). |
| V14 | DONE | BoD/EoD special-token preservation in extended tokenizer ([`build_and_verify_ship_tokenizer.py`](03_3_cscs_experiments_kickoff/scripts/build_and_verify_ship_tokenizer.py)). |
| V15 | PARTIAL | xIELU αp/αn in optimizer param list — **mechanism audited LOW RISK** ([`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) §E); explicit assertion pending (R9). |
| V16 | DONE | Tokenizer byte-fallback — verified clean polytonic routing through new vocab. |

Summary distribution: **DONE 5** · **PARTIAL 3** · **READY 1** · **NOT DONE 5** · **DEFERRED 1**. (Was 2/1/0/8/3 in the 2026-05-20 snapshot.)

---

## What's still gating bakeoff submission

Five items, ordered by hardness:

1. **R1 (in [`RISKS.md`](RISKS.md))**: HF→Megatron loader roundtrip on unmodified Apertus-8B-2509. Procedure documented at [`megatron_patches/README.md`](03_4_implementation_experiments/init_bakeoff/megatron_patches/README.md); needs Clariden GPU + weights.
2. **V4 baseline**: ~3-4 h on Clariden `normal`. Sets §5.6 hard-gate thresholds.
3. **R3 (held-out eval slice)**: reconstruct val/test partition from the dedup audit on Clariden `xfer` (option B per `03_3 ANALYSIS.md`).
4. **V7 Clariden pull**: 30-60 min login-node `huggingface-cli download` execution.
5. **ILSP harness task YAMLs**: staging-time merge from Meltemi/Krikri forks before per-arm Greek eval can run.

Items 1-3 are real verifications; 4-5 are mechanical pulls.

---

## Pointers for the planner

- **Bakeoff implementation summary**: [`COMPLETENESS_CHECK.md`](COMPLETENESS_CHECK.md) (script-coverage axis: what we have vs. what v0.7 expects).
- **Training recipe**: [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) — every hyperparameter cited against Apertus paper / sbatch line / code path.
- **Audit findings**: [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) — locally-pinned-source audit; 9 sub-section findings, 4 patches applied.
- **Silent-failure risks**: [`RISKS.md`](RISKS.md) — 16 risks in 3 tiers + 7 cheap mitigations.
- **Reviewer entry point**: [`REVIEW_PRESENTATION.md`](REVIEW_PRESENTATION.md).
- **V detail**: [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md).
