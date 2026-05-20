# Review presentation — Apertus-8B-2509 Greek-CPT init bakeoff

*Single entry point for review. Tight by design — every claim links to the artifact where it lives, every artifact links to the source it derives from. The intent is that a careful reader can trace any choice back to either (a) the Apertus tech report, (b) a peer-reviewed paper, or (c) our own explicit reasoning in `cpt_plan.md` v0.7.*

**State as of 2026-05-21:** scripts + recipe + sbatch + eval are ready for review. No CSCS jobs submitted; only local validation has happened.

**Second-pass audit done.** A verification pass against locally-pinned primary sources (8 cloned repos at pinned commits + 15 paper PDFs at [`references/`](references/MANIFEST.md)) surfaced: (a) 3 sbatch flag-name typos that would have failed first submission (`--xielu`, `--ademamix-{beta3,alpha}-warmup`), (b) a missing flag (`--dist-ckpt-strictness assume_ok_unexpected`) critical for loading a resized-embedding checkpoint, (c) a latent zero-row bug in `retok.py` on empty decode, (d) diagonal-σ in `centroid.py` (Mundra "Univariate" baseline — paper calls inadequate). **All four fixed in this commit.** Two pre-submit blockers remain open (HF→Megatron Apertus loader; ILSP harness task YAMLs from Meltemi/Krikri forks). See [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) for the full audit + resolution table.

---

## TL;DR

We're running a closed-form three-arm init bakeoff (**Vanilla / ReTok / Centroid**, 2 B tokens per arm) to decide how to initialize 17,408 new modern-Greek token embeddings before a 15–20 B-token Apertus-8B CPT on the GlossAPI Greek corpus. The training infrastructure is **Apertus's own pretraining engine** (`swiss-ai/Megatron-LM` + `swiss-ai/pretrain-code`) and **Apertus's exact hyperparameters** (from arXiv:2509.14233 Table C.4 + production sbatch), with three principled CPT-specific deviations called out explicitly.

The recipe is **boringly faithful** to Apertus's pretraining choices wherever possible — no novel optimizers, no clever loss tricks, no custom kernels. The only experimental variable across the three arms is the embedding-init algorithm.

---

## What we built (in this PR)

The user's checklist mapped to artifacts:

| # | Item | Artifact | Status |
|---|---|---|---|
| 1 | Dataset scripts | [`init_bakeoff/corpus_build/`](03_4_implementation_experiments/init_bakeoff/corpus_build/) | ready |
| 2 | Training recipe | [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) | ready |
| 3 | 2 B-after-init training infra | [`init_bakeoff/bakeoff_training/`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/) | ready (untested on CSCS) |
| 4 | Benchmarking (Greek + regression) | [`init_bakeoff/eval/`](03_4_implementation_experiments/init_bakeoff/eval/) | ready (untested on CSCS) |
| 5 | "Not run yet, except as tests" | – | confirmed: no SLURM submissions; only `bash -n` syntax checks and `arms/test_init_logic.py` smoke (which is green) |
| 6 | Reasoned presentation + citations | **this document** + [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) §14 + [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md) + [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) + [`references/`](references/MANIFEST.md) | ready |
| 7 | Hand to review | this document — start at "What we built" then read in the order below | open |
| 8 | After review: run benchmarks + day-1 execution | [`init_bakeoff/README.md`](03_4_implementation_experiments/init_bakeoff/README.md) "End-to-end sequence" | gated on review |

---

## How to review (suggested reading order)

1. **This document** (10 min) — gives you the map.
2. [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) §1-9 (20 min) — every training hyperparameter cited against Apertus tech report + sbatch line number.
3. [`init_bakeoff/BAKEOFF_PLAN.md`](03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) (10 min) — the three arms, what they test.
4. [`init_bakeoff/corpus_build/MIX_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md) (10 min) — the data mix.
5. [`init_bakeoff/eval/EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) (10 min) — what we eval and why.
6. The scripts themselves (skim): `corpus_build/mix_builder.py`, `bakeoff_training/bakeoff_train.sbatch`, `arms/{vanilla,retok,centroid}.py`.

Total review budget: ~1 hour for a careful pass.

---

## (1) Dataset scripts — corpus_build/

The bakeoff trains on the **same shuffled-mixture JSONL** across all three arms. The stream is built once on Clariden `xfer`, tokenized once to Megatron binary, then read by all three GPU jobs with a shared seed (token streams are byte-identical across arms; the only differential is init).

| File | What it does |
|---|---|
| [`mix_builder.py`](03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py) | Streaming HF `interleave_datasets` from a JSON recipe with per-source weights + Apertus-overlap-drop filtering. Writes JSONL + a manifest sidecar that records the realized per-source token shares. |
| [`recipes/bulk.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json) | 31 sources: 70 % Greek (HPLT clean60 + GlossAPI literary/dialogue/academic/legal/dictionary) + 26 % multilingual replay + 4 % code. Weights sum to 1.0 (verified). |
| [`recipes/anneal.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json) | 14 sources: 85 / 12 / 3 Greek-curated-anneal / replay / code, for the production CPT's final 10–20 %. Not used in the bakeoff. |
| [`pull_greek_corpus.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/pull_greek_corpus.sh) | Login-node HF pull: our `fffoivos/glossapi-greek-nanochat-pretraining-dataset` + the Apertus-overlap-drop overlay parquet. |
| [`pull_replay_datasets.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/pull_replay_datasets.sh) | Login-node HF pull: FineWeb-Edu (English Score-3), FineWeb2-HQ (T1+T2 high-resource langs), FineWeb-2 standard (T2+T3), StarCoderData. |

**Citations:** FineWeb-Edu / FineWeb-2: Penedo et al. ([arXiv:2406.17557](https://arxiv.org/abs/2406.17557)). FineWeb2-HQ: Messmer, Sabolčec, Jaggi ([arXiv:2502.10361](https://arxiv.org/abs/2502.10361); top-10 % per-language XLM-R filter — **not** "Score-3", which belongs to FineWeb-Edu). StarCoderData: Li et al. ([arXiv:2305.06161](https://arxiv.org/abs/2305.06161); = Stack v1.2 subset, per Apertus footnote 23).

**Reviewer flags:**

- **Our "24 replay languages" framing (T1 + T2 + T3 = 8 + 11 + 5) is internal**, not from the Apertus tech report. Apertus enumerates 20 high-resource languages (Appendix G, p.88-89) that receive quality+toxicity filtering; we extended by 4 to cover region-specific small languages. Token-share targets per language are also our derivation — the tech report publishes only FineWeb-2 *document* shares, not training-mixture *token* shares.
- **NFC normalization (V9)** is enforced upstream of `mix_builder.py` via [`scripts/verify_and_normalize_nfc.py`](03_3_cscs_experiments_kickoff/scripts/verify_and_normalize_nfc.py) (idempotent). Apertus's tokenizer has `normalizer: null`, so pre-tokenization NFC is mandatory.

---

## (2) Training recipe

The full recipe is in [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md). Headline summary:

| Component | Choice | Why (citation) |
|---|---|---|
| Engine | `swiss-ai/Megatron-LM` @ commit `c92402e3...` (HEAD 2026-05-20) | The fork Apertus pretrained from. Confirmed Apertus-specific kernels at `megatron/core/optimizer/ademamix.py`, `megatron/training/activations.py` (xIELU), `megatron/core/transformer/attention.py` (QK-Norm), `megatron/core/datasets/gpt_dataset.py` (Goldfish). |
| Launcher | mirror of `swiss-ai/pretrain-code/pretraining/submit_apertus_8b.sh` | Apertus production launch script (paper footnote 5, p.9 — the canonical "what was run"). |
| Optimizer | **AdEMAMix** (β1=0.9, β2=0.999, β3=0.9999, α=8.0, wd=0.1) | Apertus tech report Table C.4 (p.82) + sbatch lines 207-213. Citation: Pagliardini, Ablin, Grangier, *AdEMAMix Optimizer: Better, Faster, Older*, ICLR 2025 ([arXiv:2409.03137](https://arxiv.org/abs/2409.03137)). |
| Gradient clip | **0.1 global-norm** | Apertus Table C.4; sbatch L207 `--clip-grad 0.1`. Load-bearing for per-token embedding-norm convergence — see [`apertus_fidelity_checklist.md`](apertus_fidelity_checklist.md). |
| LR schedule | **WSD with 1-sqrt cooldown** | Apertus §2.3 + Table C.4. Citation: Hu et al. *MiniCPM (WSD)*, [arXiv:2404.06395](https://arxiv.org/abs/2404.06395). |
| Loss (bakeoff) | NTP | v0.7 §10 Q B4: keep optimizer & loss as constants across arms — only init varies. |
| Loss (production) | Goldfish, `k = h = 50` | Apertus §2.3 p.11; hash table size 1,000,003, seed `2971215073`. Citation: Hans et al., [arXiv:2406.10209](https://arxiv.org/abs/2406.10209). |
| Cross-doc attention + EoD mask | both **ON** | Apertus §2.1 p.10. Flags `--reset-attention-mask --reset-position-ids --eod-mask-loss`. |
| Mixed precision | bf16 + fp32 master grads | Apertus Appendix D p.83. FP8 was tried at 8 T tokens, caused loss degradation, **rolled back**. |
| Activation | xIELU (αp = αn = 0.8, β = 0.5) | Apertus §2.1 p.10; init values are code-canonical (`megatron/training/activations.py` `XIELU.__init__` defaults — paper itself doesn't specify). |
| Attention norm | QK-Norm: per-head RMSNorm, **before** RoPE | Code-confirmed: `megatron/core/transformer/attention.py` L652-656 (norm) then L432 (RoPE). |

### Three CPT-specific deviations from Apertus pretraining

These are the *only* places we knowingly diverge, each with documented reason:

1. **LR peak = 1.5e-5** (Apertus pretrain: 1.1e-4). CPT operates near-converged, so a 7× lower peak is standard practice (e.g., Llama-3 CPT, Aya). See [`cpt_plan.md`](cpt_plan.md) v0.7 §3.3.
2. **α + β3 warmup = 238 steps** for the bakeoff (Apertus pretrain: 100,000 steps ≈ 2.8 % of training). Apertus's 2.8 %-of-run policy collapses to ~14 steps at our 2 B-token / 477-step bakeoff scale, which is too short. We use 50 % of the bakeoff horizon for the cold-restart warmup as the **conservative** choice (per the AdEMAMix paper §"Switching optimizers"). Production CPT scales back to Apertus's 2.8 % rule.
3. **Loss = NTP** for the bakeoff only (Apertus pretrain: Goldfish `k=h=50`). The bakeoff measures init quality, not loss-objective quality; loss is held constant across arms. Production CPT restores Goldfish with Apertus's exact config.

### What we explicitly chose **not** to do (and why)

- **No HuggingFace Trainer.** Apertus's xIELU / QK-Norm / AdEMAMix / Goldfish kernels are not in HF stock — routing CPT through HF Trainer would silently replace them with off-the-shelf substitutes. That's the "trick" the review is guarding against.
- **No FP8.** Apertus rolled it back after 8 T tokens for loss-degradation reasons. We don't second-guess.
- **No custom optimizer.** AdEMAMix everywhere (bakeoff + production) is locked. See the user's earlier "if AdEMAMix ≈ AdamW for the first ~500 steps anyway, why introduce a switch?" decision (locked 2026-05-21).
- **No FOCUS / WECHSEL init.** Those require auxiliary external embeddings; we restrict the bakeoff to subpiece-mean (ReTok) and centroid (Hewitt 2021), both of which use only the base model's existing embeddings.

---

## (3) 2 B-after-init training recipe — `bakeoff_training/`

The three arms train under **identical** Megatron-LM-Swiss-AI conditions on Clariden (one node × 4 × GH200 × 12 h, partition `normal`). They differ only in `--load <init-checkpoint>`.

| File | Purpose |
|---|---|
| [`_train_config_common.env`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/_train_config_common.env) | Sourced by all sbatches. Every value cites its source (paper section + table, or sbatch line, or code path). |
| [`preprocess_data.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/preprocess_data.sbatch) | One-time CPU job (`xfer` partition): Megatron's `tools/preprocess_data.py` to convert JSONL → `.bin` / `.idx`. Notes the DataTrove alternative Apertus actually used. |
| [`bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch) | Parameterized training job. Every Megatron CLI flag is annotated with the Apertus sbatch line it mirrors. |
| [`submit_all_arms.sh`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/submit_all_arms.sh) | Submits all three arms in parallel with a shared seed. Sanity-checks init checkpoints + data prefix exist before dispatching. |

**Init checkpoints** are built upstream by [`arms/build_init_checkpoints.py`](03_4_implementation_experiments/init_bakeoff/arms/build_init_checkpoints.py), then HF→Megatron converted via `swiss-ai/hfconverter`. Three init methods:

- **Vanilla** ([`arms/vanilla.py`](03_4_implementation_experiments/init_bakeoff/arms/vanilla.py)) — no-op; trains on the unmodified 131,072-token Apertus vocab. Control arm.
- **ReTok** ([`arms/retok.py`](03_4_implementation_experiments/init_bakeoff/arms/retok.py)) — new-token row = mean of base-tokenizer subpiece embeddings, applied independently to `E` and `U`. Origin: Gee et al. *FVT* ([EMNLP 2022](https://aclanthology.org/2022.emnlp-industry.41/)). LLM-era: Gu et al. *ReTok* ([arXiv:2410.04335](https://arxiv.org/abs/2410.04335)).
- **Centroid** ([`arms/centroid.py`](03_4_implementation_experiments/init_bakeoff/arms/centroid.py)) — new-token row sampled from `N(μ, Σ)` of base **Greek-script** token embeddings + norm-match. Origin: Hewitt 2021 ([vocab-expansion technical note](https://www.cs.columbia.edu/~johnhew//vocab-expansion.html)). **Script-restricted variant is our extension** (Hewitt averages all rows, not script-restricted).

The init smoke test ([`arms/test_init_logic.py`](03_4_implementation_experiments/init_bakeoff/arms/test_init_logic.py)) runs locally without needing the full HF model load — confirms ReTok-vs-Centroid produces sensibly-different vectors (cos similarity ~0.025 in the smoke run).

---

## (4) Benchmarking — `eval/`

Two scopes, **same** task list:

- **V4 baseline**: one run, full suite × unmodified `swiss-ai/Apertus-8B-2509`, before the bakeoff fires. Gates the §5.6 hard-gate thresholds.
- **Per-arm**: same suite × each arm's checkpoints in 80–100 % of its 2 B-token budget. Drives selection.

| File | Purpose |
|---|---|
| [`EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) | Task lists, cadence, statistical methodology. Scoped to Apertus's **pretraining-eval table** (paper Table 14, p.38) — Group 1 retention. GSM8K + HumanEval + IFEval are post-training-only (paper Table 22) and excluded. |
| [`pull_benchmarks.sh`](03_4_implementation_experiments/init_bakeoff/eval/pull_benchmarks.sh) | Login-node HF pulls + clones `swiss-ai/lm-evaluation-harness` (the Apertus team's fork, cited in tech report §5.1 footnote 45). |
| [`run_eval.sbatch`](03_4_implementation_experiments/init_bakeoff/eval/run_eval.sbatch) | Parameterized: takes `MODEL_PATH` + `OUTPUT_DIR` + `TASK_GROUP`. Captures harness commit in `run_metadata.json` for the audit trail. |
| [`run_apertus_baseline.sh`](03_4_implementation_experiments/init_bakeoff/eval/run_apertus_baseline.sh) | V4 baseline thin wrapper. |
| [`run_bakeoff_arm_eval.sh`](03_4_implementation_experiments/init_bakeoff/eval/run_bakeoff_arm_eval.sh) | Per-arm-checkpoint thin wrapper. |
| [`compute_bootstrap_cis.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_bootstrap_cis.py) | Post-processor: bootstrap CIs over `--log_samples` JSONL (1000 resamples, 95 % CI). |

**Engine + citation:** Gao, Tow, Abbasi et al., *The Language Model Evaluation Harness*, Zenodo 2024, DOI [10.5281/zenodo.12608602](https://doi.org/10.5281/zenodo.12608602). Apertus team's fork at [github.com/swiss-ai/lm-evaluation-harness](https://github.com/swiss-ai/lm-evaluation-harness) — cited in the tech report.

**Methodology citation:** bootstrap-over-samples (not over-runs) because pretraining benchmarks are deterministic; cited in [`cpt_plan.md`](cpt_plan.md) v0.7 §6.1 as Park et al. 2025 (full citation PENDING).

**Reviewer flag:** the ILSP Greek tasks (`arc_greek`, `hellaswag_greek`, etc.) have lm-eval-harness task configs in the **Meltemi / Krikri team's harness forks** (e.g. `LeonVouk/lighteval`, `ilsp/lm-evaluation-harness-greek`), **not** in swiss-ai's or EleutherAI's. `pull_benchmarks.sh` clones the swiss-ai fork as primary; the Greek task YAMLs need to be merged in at staging time before the V4 baseline runs. Confirm before submitting.

---

## (5) Nothing has been run

- No SLURM jobs submitted at CSCS Clariden during this session.
- Local `arms/test_init_logic.py` smoke ran green (no model load needed).
- `bash -n` syntax check planned before commit on `bakeoff_training/*.sbatch` and `bakeoff_training/submit_all_arms.sh`.
- `mix_builder.py` is not exercised against real data; a tiny `--target-tokens 100000` dry-run on Clariden `xfer` is the recommended first-step validation **after** review approval.

## (5b) Known scope gaps vs cpt_plan v0.7

Honest inventory: the recipe + sbatch are review-ready, but the bakeoff is **not yet end-to-end-runnable** because several pieces v0.7 specifies have not been implemented. Full table in [`COMPLETENESS_CHECK.md`](COMPLETENESS_CHECK.md). The reviewer should know these gaps exist before reading the rest of this document:

| Gap | Where in v0.7 | What it means |
|---|---|---|
| **BPC / NLL-per-Unicode-char / NLL-per-word / STRR / tokens-per-word not tooled** | §5.1 (primary intrinsic metrics) | v0.7 explicitly says per-token PPL is **not comparable** across arms (Vanilla 131,072-vocab vs ReTok/Centroid 148,480-vocab). The bakeoff's apples-to-apples comparison needs these tokenizer-fair metrics. We currently only have standard lm-eval-harness retention scores. |
| **New-token integration diagnostic suite not tooled** | §5.3 (read at every checkpoint) | 7 diagnostics over the 17,408 new IDs — rank-of-correct-new-token, embedding-norm distribution, cosine-similarity / effective-rank collapse, etc. Without these we lose visibility into failure modes (embedding collapse, dead rows) the bakeoff is specifically supposed to detect. |
| **Hard-gates + weighted selection score not automated** | §5.6 | Encoded in prose in [`EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md); not a script that produces a pass/fail + score per arm. |
| **Custom Greek evals deferred** | §6.2 | Polytonic continuation, accent/diacritic accuracy, morphology minimal pairs, language-ID drift, register preservation. v0.7 itself notes "~1-2 weeks construction"; deferred as a separate workstream. |
| **FineMath + OPUS Greek-English missing from pull script** | §4.4 | Apertus stage-1 uses `finemath-3plus-merge` (`submit_apertus_8b.sh:L29`); OPUS is v0.7-optional but uniquely valuable for classical philology. |
| **NFC normalize step not invoked** | §8 I1 + V9 | `verify_and_normalize_nfc.py` exists at `03_3_cscs_experiments_kickoff/scripts/` but no wrapper invokes it between corpus pull and mix-build. Greek nanochat is upstream-NFC, but FineWeb-2/HQ/Edu compliance is assumed not enforced. |
| **HF→Megatron Apertus loader missing** | §7.1 | No `loader_apertus_hf.py` in `swiss-ai/Megatron-LM/tools/checkpoint/`; only llama/mistral/qwen2.5. Custom loader needed (~1-2 h) before first sbatch submission. |
| **ILSP harness task YAMLs missing from swiss-ai fork** | §6.2 + V7 | Live in Meltemi/Krikri forks; staging-time merge needed before V4 baseline. |
| **`build_init_checkpoints.py` resize logic not audited** against `transformers.PreTrainedModel.resize_token_embeddings` for untied-E/U + V15 (xIELU scalars survive resize) | §5.2 + V2 + V15 | Mechanism is sound (audited code path shows xIELU αp/αn are `nn.Parameter` children, auto-registered) but no explicit assertion in the driver. |

**Plan to close these (agreed 2026-05-21):**

1. ✓ This section — surface the gaps so the reviewer sees them.
2. (in progress) Add FineMath + OPUS to pull script, rebalance bulk.json, add NFC normalize wrapper.
3. (next) Implement §5.1 BPC + tokenizer-fair metrics as a sidecar script.
4. (after) Implement §5.3 new-token diagnostic suite as a sidecar script.
5. (after that) Discuss §5.6 selection-score automation + the HF→Megatron Apertus loader with Fivos.

Items 1-4 close the gaps that block the bakeoff from producing its **primary intended metrics**. Items 5-6 are the next decisions.

---

## (6) Reasoning + citations summary

Every numeric or methodological choice in the recipe maps to one of these four sources:

| Source | Examples |
|---|---|
| **Apertus tech report (arXiv:2509.14233 v2)** | All optimizer hyperparams (β1/β2/β3/α/wd), grad clip 0.1, LR schedule (WSD 1-sqrt), batch shape, Goldfish config, cross-doc + EoD mask, bf16 + fp32 master grads, 20 high-resource langs. **Always cited with §/page/Table.** |
| **Apertus production sbatch (swiss-ai/pretrain-code submit_apertus_8b.sh)** | Exact Megatron CLI flag names. **Always cited with line number.** |
| **Apertus code (swiss-ai/Megatron-LM)** | xIELU init values, QK-Norm placement-before-RoPE, Goldfish hash table internals. **Always cited with code path + line number.** |
| **External peer-reviewed papers** | AdEMAMix (Pagliardini et al. ICLR 2025, arXiv:2409.03137), Goldfish (Hans et al., arXiv:2406.10209), QK-Norm (Henry et al., arXiv:2010.04245), WSD (Hu et al., arXiv:2404.06395), Megatron-LM (Shoeybi et al., arXiv:1909.08053), FineWeb (Penedo et al., arXiv:2406.17557), FineWeb2-HQ (Messmer et al., arXiv:2502.10361), StarCoder (Li et al., arXiv:2305.06161), FVT (Gee et al., EMNLP 2022), ReTok (Gu et al., arXiv:2410.04335), Hewitt embedding-init (technical note), Mundra et al. (arXiv:2407.05841), lm-eval-harness (Gao et al., Zenodo DOI 10.5281/zenodo.12608602). |

The **full citation table** is in [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) §14.

If a value cannot be cited against one of these four sources, it's flagged with "[Cite: PENDING]" or "internal" in the relevant doc. **As of 2026-05-21, all hard-blocking lookups are resolved.**

---

## (7) Hand to review

Reviewer is asked to flag:

1. Any place where we've chosen a value that contradicts Apertus's pretraining settings without an explicit "deviation, because X" note.
2. Any embedding-init logic in `arms/{retok,centroid}.py` that doesn't match the cited papers' formulations.
3. Any task-ID / shot-count in `eval/EVAL_RECIPE.md` that doesn't match what Apertus reports in Table 14 (p.38).
4. Any HF dataset path / version drift (FineWeb-2 v2.0.1, FineWeb2-HQ as `epfml/FineWeb2-HQ`, StarCoderData v1.2).
5. Any place where our "24 replay languages" or per-language token-share targets disagree with what an Apertus pretrain replicator would do — we don't have the ground-truth final-mix token shares from the tech report (only document counts).

---

## (8) Day-1 execution plan (post-review)

Once review approves, the order of operations on Clariden Day 1:

```
[xfer]   sbatch corpus_build/pull_greek_corpus.sh  + pull_replay_datasets.sh   # ~3-4 h total
[xfer]   verify_and_normalize_nfc.py normalize  /iopsstor/.../cpt_corpus/     # V9 enforcement
[xfer]   python3 mix_builder.py --target-tokens 100000000 --recipe …          # 100M-token dry-run; verify manifest
[xfer]   python3 mix_builder.py --target-tokens 7000000000 --recipe bulk.json # 7 B-token real run; ~6-10 h
[xfer]   sbatch bakeoff_training/preprocess_data.sbatch                       # ~2-4 h on 64 vCPU

[debug]  python3 arms/build_init_checkpoints.py --arms vanilla retok centroid  # ~30 min on 1 GPU
[debug]  hfconverter HF → Megatron for each of the 3 arms                     # ~10-15 min each

[normal] bash eval/run_apertus_baseline.sh                                    # V4 baseline; ~3-4 h
[normal] bash bakeoff_training/submit_all_arms.sh                             # 3 × 12 h in parallel

[normal] (every 500 M tokens) bash eval/run_bakeoff_arm_eval.sh <ckpt>         # per-arm eval
[normal] python3 eval/compute_bootstrap_cis.py <output-dirs>                  # CIs over samples
```

End-state after Day 1: V4 baseline numbers + first 500-1000 M tokens of each arm logged. Day 2: arms finish, full per-arm bootstrap CIs, selection per [`cpt_plan.md`](cpt_plan.md) v0.7 §5.6 hard gates + selection score.

---

## Authoritative artifact tree (post-this-PR)

```
03_apertus_extension_and_embedding_adaptation/
├── cpt_plan.md                       — v0.7, USER-AUTHORED, canonical plan
├── REVIEW_PRESENTATION.md            — THIS FILE
├── TRAINING_RECIPE.md                — full hyperparam table with citations
├── apertus_fidelity_checklist.md     — what we must preserve from Apertus, why
├── cpt_plan_v0.7_status.md           — V1-V16 verification status
├── 03_3_cscs_experiments_kickoff/
│   ├── ship/apertus_greek_modern_only_148480/    — 148,480-vocab tokenizer (active)
│   ├── ship/apertus_greek_extended_153600/       — 153,600 composite (parked, polytonic)
│   └── scripts/verify_and_normalize_nfc.py       — V9 enforcer
└── 03_4_implementation_experiments/
    └── init_bakeoff/
        ├── BAKEOFF_PLAN.md           — three arms, slurm shape, fidelity constraints
        ├── arms/                     — vanilla.py / retok.py / centroid.py + driver + smoke test
        ├── corpus_build/             — mix recipe + builder + pull scripts
        ├── bakeoff_training/         — sbatch templates (the engine-side of the recipe)
        └── eval/                     — V4 baseline + per-arm eval + bootstrap CI
```
