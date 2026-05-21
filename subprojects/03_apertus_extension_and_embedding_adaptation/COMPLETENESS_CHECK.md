# Completeness check vs cpt_plan.md v0.7 (2026-05-21)

*Honest inventory of what we have vs what v0.7 expects, structured as three tables: (a) script coverage, (b) legitimate-source audit, (c) citation + v0.7 connection. Followed by a prioritized list of next moves.*

This doc is the answer to: **"is everything downstream of v0.7, are the scripts complete, has everything been checked against its legitimate source, and is it cited?"**

The headline: **(b) and (c) are mostly complete. (a) has real gaps** — some addressable in <1 h, some that require new sidecar scripts before the bakeoff can produce its intended primary metrics.

---

## (a) Script coverage vs v0.7 expectations

| v0.7 section / artifact | Have it? | Status / location |
|---|---|---|
| **§2 mix shape** (70/30, 4 % code, anneal tail, Apertus-overlap drop) | ✓ | [`corpus_build/recipes/bulk.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/bulk.json) + [`anneal.json`](03_4_implementation_experiments/init_bakeoff/corpus_build/recipes/anneal.json) |
| **§4.4 replay sources** — FW-Edu Score-3, FW2-HQ, FW2, StarCoderData | ✓ | [`pull_replay_datasets.sh`](03_4_implementation_experiments/init_bakeoff/corpus_build/pull_replay_datasets.sh) |
| **§4.4 — FineMath** (`HuggingFaceTB/finemath`) | ✓ (Item 2) | Added 2026-05-21 at 2 % share of bulk mix; mix shape now 70/24/4/2 (Greek/replay/code/math) |
| **§4.4 — OPUS Greek-English / Greek-Latin parallel** | ⚠️ | Marked optional in v0.7; deferred (needs schema-adapter in `mix_builder.py` for `{translation: {el, en}}` rows). Not blocking the bakeoff. |
| **§8 I1 / V9 NFC normalization** of training corpus | ✓ (Item 2) | `corpus_build/normalize_nfc.sh` added 2026-05-21; idempotent wrapper around `verify_and_normalize_nfc.py`; invoked between corpus pull and `mix_builder.py` per the end-to-end sequence in [`init_bakeoff/README.md`](03_4_implementation_experiments/init_bakeoff/README.md). |
| **§5 three init arms** (Vanilla / ReTok / Centroid) | ✓ | [`arms/{vanilla,retok,centroid}.py`](03_4_implementation_experiments/init_bakeoff/arms/) — audited + patched 2026-05-21 ([`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md)) |
| **§5 `build_init_checkpoints.py` driver** | ✓ implemented | **Not audited** against `transformers.resize_token_embeddings` semantics for untied E/U (V2 + V15 still unconfirmed) |
| **HF→Megatron Apertus loader** | ❌ | **Open blocker** — no `loader_apertus_hf.py` exists in `swiss-ai/Megatron-LM/tools/checkpoint/`; only llama/mistral/qwen2.5 covered |
| **§5.1 BPC + NLL/char + NLL/word + STRR + tokens/word + compression ratio (PRIMARY intrinsic metrics)** | ✓ (Item 3) | [`eval/compute_tokenizer_fair_metrics.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_tokenizer_fair_metrics.py) + sbatch wrapper. Aggregates globally + per-source + per-register. Stats-only smoke green. Held-out eval slice still needs construction (the dedup-audit val/test partition is the natural source). |
| **§5.3 new-token integration diagnostic suite** (7 diagnostics: rank-of-correct-new-token, embedding-norm distribution, cosine-similarity / effective-rank collapse, etc.) | ✓ (Item 4) | [`eval/compute_new_token_diagnostics.py`](03_4_implementation_experiments/init_bakeoff/eval/compute_new_token_diagnostics.py) + sbatch wrapper. All 7 diagnostics in one run; `--embedding-only` skips D1-D5 for fast embedding-only health checks; `--skip-greedy` skips just D5. |
| **§5.6 hard gates + weighted selection score** | ⚠️ | Encoded in [`EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) prose; not implemented as a script that produces a pass/fail + score per arm. |
| **§6.2 retention + Greek benchmarks via lm-eval-harness** | ✓ | [`eval/EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) + [`pull_benchmarks.sh`](03_4_implementation_experiments/init_bakeoff/eval/pull_benchmarks.sh); ILSP task YAMLs from Meltemi/Krikri forks still need staging-time merge (open blocker) |
| **§6.2 custom Greek evals** (polytonic continuation, accent accuracy, morphology minimal pairs, language-ID drift, register preservation) | ❌ | Documented as "1-2 weeks construction" in v0.7 — explicitly deferred but not flagged as a scope gap in our review packet |
| **§6.4 stability diagnostics during training** | ✓ partial | Apertus's `--log-throughput --log-params-norm --log-memory-to-tensorboard` flags mirrored ([`bakeoff_train.sbatch`](03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch)); no per-bucket PPL split tooled |
| **Bakeoff training sbatch** | ✓ | Audited line-by-line against `submit_apertus_8b.sh` (4 flag-name typos fixed + 9 missing flags added in audit pass) |
| **Continuous testing during training** | ✓ partial | [`run_bakeoff_arm_eval.sh`](03_4_implementation_experiments/init_bakeoff/eval/run_bakeoff_arm_eval.sh) for periodic eval; no automated checkpoint-window selection-score computation |

---

## (b) Legitimate-source audit coverage

| Area | Audited against | Outstanding |
|---|---|---|
| AdEMAMix / xIELU / QK-Norm / Goldfish / WSD / cross-doc mask / EoD mask / mixed-precision policy | Apertus tech report (arXiv:2509.14233 v2) + `swiss-ai/Megatron-LM` code + `submit_apertus_8b.sh` | – |
| ReTok + Centroid init algorithms | FVT (Gee 2022) + ReTok (Gu 2024) + Hewitt 2021 + Mundra 2024 | – |
| `bakeoff_train.sbatch` flag set | line-by-line vs `submit_apertus_8b.sh` (commit `531cc8be`) | – |
| AdEMAMix optimizer impl | `references/repos/swiss-ai_Megatron-LM/megatron/core/optimizer/ademamix.py` vs paper §3 | – |
| xIELU activation impl | `references/repos/swiss-ai_Megatron-LM/megatron/training/activations.py` (XIELU class) | – |
| Goldfish hash table internals | `references/repos/swiss-ai_Megatron-LM/megatron/core/datasets/gpt_dataset.py` | – |
| `build_init_checkpoints.py` resize logic | – | **Not audited** against `transformers.PreTrainedModel.resize_token_embeddings` for untied-E/U semantics |
| Mix recipe weights → realized per-language token shares | mix_builder math (weights sum to 1.0); spot-checked | not verified against Apertus's actual per-language token shares (Q C3 still pending) |
| §5.1 BPC / NLL methodology | – | **Not implemented** |
| §5.3 diagnostic suite | – | **Not implemented** |

Local pinned sources at [`references/MANIFEST.md`](references/MANIFEST.md): 8 git-cloned repos (commits pinned) + 15 papers (HTML preferred, PDF fallback; 15 MB total).

---

## (c) Citation + v0.7 connection

Every artifact-doc cites v0.7 + the primary source for each numeric/algorithmic choice. Cross-check:

| Doc | Cites v0.7? | Cites primary source? |
|---|---|---|
| [`TRAINING_RECIPE.md`](TRAINING_RECIPE.md) | ✓ throughout | ✓ `references/papers/X.{html,pdf}` + sbatch:L + code path |
| [`BAKEOFF_PLAN.md`](03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) | ✓ §5 | ✓ apertus_fidelity_checklist.md |
| [`MIX_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md) | ✓ §2 + §4 | ✓ FineWeb / FineWeb2-HQ / StarCoderData citations |
| [`EVAL_RECIPE.md`](03_4_implementation_experiments/init_bakeoff/eval/EVAL_RECIPE.md) | ✓ §6.1 | ✓ lm-eval-harness Zenodo DOI |
| [`AUDIT_FINDINGS.md`](AUDIT_FINDINGS.md) | – (audit pass) | ✓ paper-section + line + commit |
| [`REVIEW_PRESENTATION.md`](REVIEW_PRESENTATION.md) | ✓ | ✓ |
| [`arms/{retok,centroid,_common}.py`](03_4_implementation_experiments/init_bakeoff/arms/) | – (code) | ✓ `references/papers/X.{html,pdf}` |

**Caveat:** none of the docs flag the (a) gaps. A reviewer reading [`REVIEW_PRESENTATION.md`](REVIEW_PRESENTATION.md) would not realize §5.1 BPC isn't computed, §5.3 diagnostics aren't computed, §5.6 selection score isn't automated, custom Greek evals are deferred, and FineMath / OPUS / NFC-invocation are missing. **Item 1 below closes this honesty gap.**

---

## Prioritized next moves

| # | Gap | Effort | Why now |
|---|---|---|---|
| 1 | Document the gaps in [`REVIEW_PRESENTATION.md`](REVIEW_PRESENTATION.md) explicitly so the reviewer can't miss them | 30 min | closes the (c) coverage gap immediately |
| 2 | Add FineMath + OPUS to `pull_replay_datasets.sh`, rebalance bulk.json to 70/24/4/2 (Greek/replay/code/math), add explicit NFC normalize step (`normalize_nfc.sh`) between pull and mix-build | 1 h | small gaps; Apertus uses FineMath in stage-1; V9 must be operationally satisfied |
| 3 | Implement §5.1 BPC + NLL/char + NLL/word + tokens/word + compression-ratio + STRR computation as a sidecar Python script | 4-6 h | THE primary metric for the bakeoff — without it we can't compare arms fairly across the Vanilla-vs-extended axis |
| 4 | Implement §5.3 new-token diagnostic suite as a sidecar Python script (7 diagnostics over the 17,408 new IDs) | 6-8 h | failure-mode detector — bakeoff intends to use it at every checkpoint to catch embedding collapse / dead rows / etc. |
| 5 | Implement §5.6 selection-score computation as post-processing over eval JSONs (hard gates + weighted score) | 2-3 h | turns hard-gates + weighted score from prose into a number |
| 6 | Write the HF→Megatron Apertus loader (open pre-submit blocker) | 1-2 h | blocks first sbatch submission |
| 7 | Custom Greek evals (polytonic continuation, accent accuracy, morphology, language-ID drift, register preservation) | 1-2 weeks | v0.7 acknowledges this is its own work block; can be deferred to a separate construction phase |

**Plan agreed with Fivos 2026-05-21:** do (1) + (2) immediately, then (3) + (4); discuss (5) + (6) after.

**Progress 2026-05-21:**

| # | Status | Commit |
|---|---|---|
| 1 | ✓ done | `f26c07a` — REVIEW_PRESENTATION.md "(5b) Known scope gaps" section |
| 2 | ✓ done | `16b886c` — FineMath + NFC normalize wrapper; mix now 70/24/4/2 |
| 3 | ✓ done | `d5bc6c0` — `compute_tokenizer_fair_metrics.py` + sbatch (BPC, NLL/char, NLL/word, STRR) |
| 4 | ✓ done | `28b71f2` — `compute_new_token_diagnostics.py` + sbatch (all 7 §5.3 diagnostics) |
| 5 | open | Selection-score automation (§5.6 hard gates + weighted score) — pending discussion |
| 6 | open | HF→Megatron Apertus loader — pending discussion |
| 7 | open | Custom Greek evals (§6.2) — pending decision (1-2 weeks construction) |

**Remaining sub-gap from item 3:** the held-out eval slice itself hasn't been constructed. The natural source is the dedup-audit val/test partition, which is part of the gcloud-access-loss situation. Either rebuild it on Clariden xfer (option B per `03_3 ANALYSIS.md`), or substitute a fresh deterministic holdout-by-doc-id-hash slice from the current corpus. Decision pending.
