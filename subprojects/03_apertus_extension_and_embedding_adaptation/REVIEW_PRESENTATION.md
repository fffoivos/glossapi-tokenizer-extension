# Review presentation — Apertus-8B-2509 Greek-CPT init bakeoff

*Single entry point for a reviewer. Every claim links to the artifact; every artifact links to the source. Trace any choice back to (a) the Apertus tech report (arXiv:2509.14233), (b) a peer-reviewed paper, or (c) explicit reasoning in `cpt_plan.md` v0.7.*

**State 2026-05-21 takeover.** Recipe + sbatch + eval + risk inventory have moved from paper review into live Clariden execution. R1 HF->Megatron->HF roundtrip passed on Apertus-8B-2509 (`2333864`) for standard tensors, with R17 xIELU/QK reset quantified. V4-HF job `2334245` produced valid artifacts for its listed tasks but omitted `global_mmlu`; corrected V4-HF and V4-post-conversion reruns are required before final §5.6 thresholds. Corpus build job `2334880` is running `prepare_greek_pool` with the DuckDB external-sort fix; the final `${SELECTED}` parquet is not complete yet. See [`SESSION_LOG_20260521.md`](SESSION_LOG_20260521.md), [`CSCS_OVERNIGHT_STATE.md`](CSCS_OVERNIGHT_STATE.md), and [`TAKEOVER_LOG_20260521.md`](TAKEOVER_LOG_20260521.md).

## TL;DR

We're running a closed-form three-arm init bakeoff (**Vanilla / ReTok / Centroid**, 2 B tokens per arm) to decide how to initialize 17,408 new modern-Greek token embeddings before a 15-20 B-token Apertus-8B CPT on the GlossAPI Greek corpus. Training engine: **Apertus's own** `swiss-ai/Megatron-LM` + `swiss-ai/pretrain-code`. Hyperparameters: **Apertus's exact values** from paper Table C.4 + production sbatch, with three CPT-specific deviations called out explicitly.

The recipe is **boringly faithful** to Apertus's pretraining wherever possible — no novel optimizers, no custom kernels. The extended-vocab arms isolate the embedding-init algorithm; the Vanilla control intentionally also uses the base tokenizer and base-tokenized data.

## What we built

| # | Item | Artifact |
|---|---|---|
| 1 | Dataset scripts (corpus pull + mix builder + NFC normalize) | [`init_bakeoff/corpus_build/`](03_4_implementation_experiments/init_bakeoff/corpus_build/) |
| 2 | Training recipe (hyperparameters + citations) | [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) |
| 3 | 2 B-after-init training infra (sbatch + HF→Megatron loader) | [`init_bakeoff/bakeoff_training/`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/) + [`init_bakeoff/megatron_patches/`](03_4_implementation_experiments/init_bakeoff/megatron_patches/) |
| 4 | Benchmarking (retention + Greek + tokenizer-fair metrics + new-token diagnostics) | [`init_bakeoff/eval/`](03_4_implementation_experiments/init_bakeoff/eval/) |
| 5 | Audit against locally-pinned primary sources | [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) + [`references/`](references/MANIFEST.md) |
| 6 | Silent-failure risk inventory | [`RISKS.md`](RISKS.md) |
| 7 | Question-by-question answers to v0.7 | [`cpt_plan_v0.7_answers.md`](cpt_plan_v0.7_answers.md) (planner-facing) + [`cpt_plan_v0.7_status.md`](cpt_plan_v0.7_status.md) (V detail) |

## How to review (~1 h)

1. **This document** (10 min) — the map.
2. [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) (20 min) — every hyperparameter cited to paper §/page or sbatch line.
3. [`init_bakeoff/BAKEOFF_PLAN.md`](03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) (10 min) — the three arms.
4. [`init_bakeoff/corpus_build/MIX_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md) (10 min) — the data mix.
5. [`init_bakeoff/eval/EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) (10 min) — eval suite + §5.6 hard-gate rubric.
6. [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) + [`RISKS.md`](RISKS.md) (10 min) — what's been verified vs. flagged.

## (1) Dataset scripts

The bakeoff trains on **one shuffled-mixture JSONL document stream**, built once on Clariden `xfer`, that all three arms read at the same shuffle seed. The **document order** is byte-identical across arms; **token IDs differ between Vanilla and ReTok/Centroid** because Vanilla is tokenized with the base 131,072 Apertus vocab (its embedding table only has those 131,072 rows) while ReTok/Centroid are tokenized with the extended 148,480 vocab. Two Megatron binary preprocessings are produced from the same JSONL (one per tokenizer); each arm loads the one that matches its embedding table. So the bakeoff is "same documents, different tokenizer where appropriate, plus different init" — not "byte-identical token IDs".

The Apertus-overlap drop + nanochat internal-dedup (`drop_intra_and_inter`) are applied **upstream** in `prepare_greek_pool.sh` per the runbook at [`03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md). The runbook order matters: Apertus-overlap removal first, then internal dedup, then `mix_builder` reads the resulting `${SELECTED}` parquet. All six Greek source-categories in `bulk.json` filter the same `${SELECTED}` pool by `source_dataset` value, so the upstream drop applies uniformly (was a B3 reviewer issue — only HPLT had it before).

| File | Purpose |
|---|---|
| [`mix_builder.py`](03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py) | Streaming `interleave_datasets` over local `${SELECTED}` parquet (Greek sources) + HF datasets (replay/code/math). Writes JSONL + manifest. |
| [`prepare_greek_pool.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/prepare_greek_pool.sh) | NEW: invokes `glossapi_corpus_cli mix-prepare-selected-input` per the runbook. Apertus-overlap-drop + `drop_intra_and_inter` → `${SELECTED}` parquet. |
| [`normalize_nfc.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/normalize_nfc.sh) | V9 enforcement — idempotent NFC pass between pull and mix-build. |
| [`recipes/bulk.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json) | 32 sources, **70 / 24 / 4 / 2** Greek / replay / code / math. Weights sum to 1.0. |
| [`recipes/anneal.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json) | 14 sources, 85 / 12 / 3 (production-anneal only; not used in bakeoff). |
| [`pull_greek_corpus.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/pull_greek_corpus.sh) | Login-node HF pull: GlossAPI Greek nanochat + Apertus-overlap drop overlay. |
| [`pull_replay_datasets.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/pull_replay_datasets.sh) | Login-node HF pull: FineWeb-Edu (Score-3 English), FineWeb2-HQ (T1+T2 high-resource), FineWeb-2 (T2+T3), StarCoderData, FineMath-3plus. |
| [`normalize_nfc.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/normalize_nfc.sh) | V9 enforcer — idempotent NFC pass between pull and mix-build. |

Data citations: FineWeb-Edu / FineWeb-2 [Penedo arXiv:2406.17557], FineWeb2-HQ [Messmer arXiv:2502.10361, top-10 % per-language XLM-R filter — **not** Score-3], StarCoderData [Li arXiv:2305.06161 = Stack v1.2], FineMath-3plus [Apertus stage-1 source per `submit_apertus_8b.sh:L29`].

## (2) Training recipe

Full table in [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md). Headline:

| Component | Value | Source |
|---|---|---|
| Engine | `swiss-ai/Megatron-LM` @ `c92402e3` | Apertus footnote 4 |
| Launcher | mirror of `swiss-ai/pretrain-code/pretraining/submit_apertus_8b.sh` | paper footnote 5 |
| Optimizer | **AdEMAMix** β1=0.9, β2=0.999, β3=0.9999, α=8.0, wd=0.1 | paper Table C.4; sbatch L208-219; Pagliardini et al. ICLR 2025 (arXiv:2409.03137) |
| Gradient clip | **0.1 global-norm** | paper Table C.4; sbatch L207 |
| LR schedule | **WSD with 1-sqrt cooldown** | paper §2.3 + Table C.4; sbatch L242 |
| Loss (bakeoff / production) | NTP / Goldfish k=h=50 | v0.7 §10 Q B4 / Apertus §2.3 p.11; Hans et al. arXiv:2406.10209 |
| Cross-doc attention + EoD mask | both ON | paper §2.1; sbatch L286-289 |
| Mixed precision | bf16 + fp32 master grads (FP8 rolled back) | paper Appendix D p.83 |
| Activation | xIELU (αp = αn = 0.8, β = 0.5) | paper §2.1; code `megatron/training/activations.py` |
| Attention norm | QK-Norm, per-head RMSNorm, **before** RoPE | code `megatron/core/transformer/attention.py` L652-656 |

**Three CPT-specific deviations** (each with explicit reason):
1. **LR peak 1.5e-5** (vs Apertus pretrain 1.1e-4): CPT is near-converged; standard practice.
2. **α + β3 warmup 238 steps** (vs Apertus 100k): Apertus's 2.8 % policy collapses to ~14 steps at our 477-step bakeoff scale — too short. Conservative 50 %-of-horizon for cold-restart.
3. **NTP loss for bakeoff only**: keeps loss constant across arms so the comparison isolates init.

**Choices explicitly rejected:** HF Trainer (would replace Apertus kernels silently); FP8 (Apertus rolled it back); custom optimizer; FOCUS / WECHSEL init (need external aux embeddings).

## (3) 2B-after-init training — `bakeoff_training/`

All three arms train under identical Megatron-LM-Swiss-AI conditions on Clariden (1 node × 4 × GH200 × 12 h, partition `normal`). They differ in three coupled things — picked by the per-arm switch in [`bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch):

- **`--load <init-checkpoint>`** — Vanilla → unmodified Apertus; ReTok / Centroid → respective resized + initialized checkpoints
- **`--tokenizer-model`** — Vanilla → `${BASE_TOKENIZER_DIR}` (131,072); ReTok / Centroid → `${EXT_TOKENIZER_DIR}` (148,480)
- **`--data-path`** — Vanilla → `${BASE_DATA_PREFIX}` (JSONL tokenized with base vocab); ReTok / Centroid → `${EXT_DATA_PREFIX}` (same JSONL tokenized with extended vocab)

Documents are byte-identical across the two preprocessings (same `mix_builder` output JSONL, same shuffle seed); only the token IDs differ. This is the corrected bakeoff shape after the round-2 B2 fix — previously all three arms were hardcoded to the 148,480 tokenizer + data path, which would have either crashed Vanilla on out-of-range IDs or silently corrupted the control arm.

| File | Purpose |
|---|---|
| [`_train_config_common.env`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/_train_config_common.env) | Shared config — every value cites paper §/page or sbatch line or code path. |
| [`preprocess_data.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/preprocess_data.sbatch) | One-time CPU job (`xfer`): Megatron `tools/preprocess_data.py` JSONL → `.bin` / `.idx`. |
| [`bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch) | Parameterized training job. Every flag annotated with the Apertus sbatch line it mirrors. |
| [`submit_all_arms.sh`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/submit_all_arms.sh) | Submits all three arms in parallel with shared seed. |

**Init checkpoints**: built by [`arms/build_init_checkpoints.py`](03_4_implementation_experiments/init_bakeoff/arms/build_init_checkpoints.py), then HF→Megatron-converted via our custom [`megatron_patches/loader_apertus_hf.py`](03_4_implementation_experiments/init_bakeoff/megatron_patches/loader_apertus_hf.py) (swiss-ai's stock loader covers llama/mistral/qwen but not Apertus's xIELU + QK-Norm + GQA + bias-free arch — see `megatron_patches/README.md`).

Three init methods:
- **Vanilla** ([`arms/vanilla.py`](03_4_implementation_experiments/init_bakeoff/arms/vanilla.py)) — control; trains on base 131,072 vocab.
- **ReTok** ([`arms/retok.py`](03_4_implementation_experiments/init_bakeoff/arms/retok.py)) — new-token row = mean of base subpieces. **Audited**: hard fallback added for empty-decode failure mode (Hewitt's "disaster case"). Origin: Gee et al. FVT 2022; LLM-era: Gu et al. arXiv:2410.04335.
- **Centroid** ([`arms/centroid.py`](03_4_implementation_experiments/init_bakeoff/arms/centroid.py)) — sample from `N(μ, Σ)` of base **Greek-script** embeddings + norm-match. **Audited**: switched to full Σ multivariate normal per Hewitt 2021 (was diagonal-σ "Univariate" — Mundra arXiv:2407.05841 calls inadequate).

Local smoke ([`arms/test_init_logic.py`](03_4_implementation_experiments/init_bakeoff/arms/test_init_logic.py)) green; both arms produce norm-matched [17408, 4096] rows.

## (4) Benchmarking — `eval/`

Two scopes, same task list:
- **V4 baseline** — Apertus-8B-2509 evaluated **twice** (post round-3 R17 decision):
  - **V4-HF**: full suite × unmodified HF Apertus. Absolute reference.
  - **V4-post-conversion**: full suite × Apertus after HF → Megatron → HF roundtrip via our loader (so xIELU + QK-Norm at `__init__` defaults — same path the bakeoff arms experience). **This is what sets the §5.6 hard-gate thresholds** because it's the apples-to-apples comparator for the arms.
- **Per-arm** — same suite at each arm's checkpoints in 80-100 % of its 2 B budget. Compared against V4-post-conversion for fidelity-loss-fair hard gates; against V4-HF for "what did vocab extension cost in absolute terms".

| File | Purpose |
|---|---|
| [`EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) | Task lists scoped to Apertus's **pretraining-eval Table 14** (not Table 22 post-training: GSM8K/HumanEval/IFEval excluded). §5.6 hard-gate placeholders marked PENDING(V4). |
| [`pull_benchmarks.sh`](03_4_implementation_experiments/init_bakeoff/eval/pull_benchmarks.sh) | Login-node HF pulls + clone `swiss-ai/lm-evaluation-harness`. |
| [`run_eval.sbatch`](03_4_implementation_experiments/init_bakeoff/eval/run_eval.sbatch) | Parameterized: `MODEL_PATH` + `OUTPUT_DIR` + `TASK_GROUP={full,retention_only,greek_only,safety_only}`. |
| [`compute_tokenizer_fair_metrics.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_tokenizer_fair_metrics.py) | §5.1 primary intrinsic metrics: BPC, NLL/char, NLL/word, tokens/word, STRR. Cross-tokenizer fair for Vanilla-vs-extended comparison. |
| [`LOSS_MEASUREMENT_POLICY.md`](03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md) | Explains why raw Megatron `lm loss` is health-only across different tokenizers and defines heldout BPC/BPB plus future dense `bpb`/`bpt`/base-new training-log fields. |
| [`compute_new_token_diagnostics.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_new_token_diagnostics.py) | §5.3 diagnostic suite — all 7 diagnostics over the 17,408 new IDs. `--embedding-only` mode for fast checks. |
| [`compute_bootstrap_cis.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_bootstrap_cis.py) | Bootstrap CIs (1000 resamples, 95 %) over `--log_samples` JSONL — per v0.7 §6.1 methodology. |
| [`summarize_bakeoff.py`](03_4_implementation_experiments/init_bakeoff/eval/summarize_bakeoff.py) | Aggregates per-arm JSONs into one markdown table for manual §5.6 review. **No automated weighted score** — selection stays manual against V4-derived thresholds. |

**Engine + citation:** Gao, Tow, Abbasi et al., Zenodo DOI [10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602). `swiss-ai/lm-evaluation-harness` fork cited in Apertus tech report §5.1 footnote 45.

**Reviewer flag:** ILSP Greek task YAMLs (`arc_greek`, `hellaswag_greek`, etc.) live in **Meltemi / Krikri** harness forks, **not** swiss-ai's. The current V4 path uses swissai-native Greek tasks; PF5 remains open if we want the ILSP-only tasks before final reporting.

## (5) Current state, scope gaps, and risks

**Clariden execution has started.** R1 is complete, V4-HF partial artifacts exist, and `prepare_greek_pool` is running. Local syntax checks remain green, but this packet should now be read with the live-state docs above.

**Scope coverage vs v0.7.** Full table in [`COMPLETENESS_CHECK.md`](COMPLETENESS_CHECK.md). Headline:

| v0.7 expectation | Status |
|---|---|
| Dataset scripts + mix recipe + NFC enforcement | ✓ (Items 1-2) |
| Training recipe + audited sbatch + HF→Megatron loader | ✓ (Items 5-6 + audit pass) |
| §5.1 tokenizer-fair metrics + §5.3 diagnostic suite | ✓ (Items 3-4) |
| §5.6 hard gates + selection score automation | ⚠ thresholds marked `PENDING(V4)`; selection deliberately manual |
| §6.2 custom Greek evals (polytonic continuation, accent accuracy, language-ID drift, register preservation) | ❌ deferred — v0.7 itself notes "1-2 weeks construction" |
| Held-out eval slice (post-Apertus-dedup) | ❌ reconstruction path documented; gated on Clariden xfer |

**Silent-failure risks.** Full inventory in [`RISKS.md`](RISKS.md) — 16 risks in 3 tiers. **Tier 1** (could invalidate the bakeoff entirely):

| | Risk | What catches it |
|---|---|---|
| R1 | HF→Megatron QKV interleaving in our `loader_apertus_hf.py` | CLOSED for standard tensors by job `2333864`; R17 remains open for xIELU/QK reset and requires V4-post-conversion comparison |
| R2 | Token-stream determinism across arms unverified (no MD5 of mix_builder output) | 15-min addition to `mix_builder.py`; not done |
| R3 | Held-out Greek eval slice cleanliness vs Apertus pretraining | reconstruct the dedup-audit val/test partition on Clariden xfer |
| R4 | ReTok / Centroid surface-form decode leading-space artifact | 30-min unit test on a sample of new tokens; not done |

**Tier 2** (5 risks) and **Tier 3** (7) in [`RISKS.md`](RISKS.md). 7 of these have **cheap mitigations** (~2 h total) — see `RISKS.md` §"Cheap mitigations available". Bundle not yet landed pending your decision.

## (6) Citation discipline

Every numeric or methodological choice maps to one of:
- **Apertus tech report (arXiv:2509.14233)** — paper §/page/Table
- **Apertus production sbatch** (`submit_apertus_8b.sh`) — line number
- **Apertus code** (`swiss-ai/Megatron-LM` @ `c92402e3`) — path + line
- **External peer-reviewed papers** — see [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) §14 + [`references/MANIFEST.md`](references/MANIFEST.md)

If a value cannot be cited against one of these, it's flagged `[Cite: PENDING]` or `internal` in context. **All hard-blocking lookups are resolved** (Q C1/C2/C4/C5/D1); Q C3 has an empirical-derivation path proposed in [`cpt_plan_v0.7_answers.md`](cpt_plan_v0.7_answers.md).

## (7) Hand to reviewer

Specific things to flag:

1. Any value that contradicts Apertus's pretraining settings without an explicit "deviation, because X" note in [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) §2-9.
2. Any embedding-init logic in `arms/{retok,centroid}.py` that doesn't match the cited papers' formulations (already audited at [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) §B-C; cross-check welcome).
3. Any task-ID / shot-count in `eval/EVAL_RECIPE.md` that doesn't match Apertus's Table 14 (p.38).
4. Any HF dataset path / version drift (FineWeb-2 v2.0.1, FineWeb2-HQ as `epfml/FineWeb2-HQ`, StarCoderData v1.2).
5. Any place where our "24 replay languages" framing or per-language token-share targets disagrees with an Apertus-pretrain replicator's view — Apertus tech report publishes only document counts (Appendix G), not training-mix token shares; we have an empirical-derivation proposal for Q C3 in [`cpt_plan_v0.7_answers.md`](cpt_plan_v0.7_answers.md).
6. **Tier 1 risks in [`RISKS.md`](RISKS.md)** — would you tolerate any of them being unverified at first-submit? (Our current plan: run roundtrip + reconstruct held-out slice as part of Clariden pre-submit; ReTok-decode unit test as a cheap mitigation we can land now.)

## (8) Day-1 execution plan (post-review)

```
[ela login]   bash corpus_build/pull_greek_corpus.sh      # ~30-60 min
              bash corpus_build/pull_replay_datasets.sh   # ~1-3 h
              bash eval/pull_benchmarks.sh                # ~30-60 min
              bash megatron_patches/install.sh $MEGATRON  # one-time

[Clariden xfer]
              bash corpus_build/normalize_nfc.sh          # V9 enforcement
              python3 corpus_build/mix_builder.py --target-tokens 100000000 …  # 100M dry-run
              python3 corpus_build/mix_builder.py --target-tokens 7000000000 … # ~6-10 h
              sbatch bakeoff_training/preprocess_data.sbatch                   # ~2-4 h

[Clariden debug]
              # Pre-submit gate: R1 roundtrip on unmodified Apertus-8B-2509
              # (procedure in megatron_patches/README.md)
              python3 arms/build_init_checkpoints.py --arms vanilla retok centroid
              # HF → Megatron conversion for each arm via tools/checkpoint/convert.py
              #   --loader apertus_hf --saver core
              # Each ~10-15 min

[Clariden normal]
              bash eval/run_apertus_baseline.sh           # V4 baseline; ~3-4 h
              # set §5.6 hard-gate thresholds from V4 numbers (manual)
              bash bakeoff_training/submit_all_arms.sh    # 3 × 12 h parallel
              # at every 500 M tokens:
              bash eval/run_bakeoff_arm_eval.sh <ckpt>
              sbatch eval/run_tokenizer_fair_metrics.sbatch
              sbatch eval/run_new_token_diagnostics.sbatch
              python3 eval/summarize_bakeoff.py <dirs> --out summary.md  # manual review
```

End-state Day 1: V4 baseline + first 500-1000 M tokens per arm. Day 2: full bakeoff complete, summary table + manual §5.6 selection.

## Authoritative artifact tree

```
03_apertus_extension_and_embedding_adaptation/
├── cpt_plan.md                       ← v0.7, USER-AUTHORED, canonical plan
├── REVIEW_PRESENTATION.md            ← THIS FILE
├── cpt_plan_v0.7_answers.md          ← planner-facing Q/V answers
├── cpt_plan_v0.7_status.md           ← V1-V16 verification detail
├── TRAINING_RECIPE.md                ← full hyperparam table with citations
├── apertus_fidelity_checklist.md     ← what we must preserve from Apertus
├── AUDIT_FINDINGS.md                 ← code-vs-paper audit + 4 patches applied
├── RISKS.md                          ← silent-failure inventory
├── COMPLETENESS_CHECK.md             ← script-coverage gap analysis
├── references/                       ← 8 pinned repos + 15 papers (regenerable)
├── 03_3_cscs_experiments_kickoff/
│   ├── ship/apertus_greek_modern_only_148480/  ← 148,480-vocab ship bundle (active)
│   ├── ship/apertus_greek_extended_153600/     ← 153,600 composite (parked)
│   └── scripts/verify_and_normalize_nfc.py     ← V9 enforcer
└── 03_4_implementation_experiments/
    └── init_bakeoff/
        ├── BAKEOFF_PLAN.md           ← three arms, slurm shape, fidelity constraints
        ├── arms/                     ← vanilla / retok / centroid + driver + smoke
        ├── corpus_build/             ← mix recipe + builder + pull + NFC scripts
        ├── bakeoff_training/         ← sbatch templates + shared env
        ├── megatron_patches/         ← HF→Megatron Apertus loader (custom)
        └── eval/                     ← V4 + per-arm + BPC + diagnostics + bootstrap
```
