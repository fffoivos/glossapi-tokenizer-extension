# papers — Reading library and citation audit for the CPT recipe

> **In one line:** 20 papers fetched, read in full and audited adversarially against the project's frozen hyperparameter doc — the audit found 18 of 40 cited claims not fully confirmed, including two that argue the *opposite* of what they were cited for.
> **Period:** landed 2026-06-11 (`a19c136f`, alongside [`../CURRENT_HYPERPARAMETERS.md`](../CURRENT_HYPERPARAMETERS.md), which it audits). **Status:** completed; the corrections were partly absorbed into the recipe doc and the tensions it flagged were settled empirically by the sweeps that followed.
> **Came from / led to:** the earlier pinned reference set in [`../03_apertus_extension_and_embedding_adaptation/references/`](../03_apertus_extension_and_embedding_adaptation/references/MANIFEST.md) (2026-05-21) → this → the hyperparameter freezes in [`../05_token_distillation_cpt`](../05_token_distillation_cpt/README.md) and the production run in [`../07_full_8b_cpt`](../07_full_8b_cpt/README.md).

## Why this existed

By mid-2026 the Greek CPT recipe was a dense table of numbers — peak LR 5.5e-5, AdEMAMix α=4/β₃=0.999, Goldfish k=h=50, a 17,408-token vocabulary extension — each carrying a paper citation as justification. Nobody had checked whether the papers actually said those things. This library exists to answer that: fetch every cited paper as full text, and for each citation record a verdict (`CONFIRMED` / `PARTIAL` / `OVERSTATED` / `UNSUPPORTED` / `CONTRADICTED`) with the passage that settles it and a concrete correction. It is an audit instrument, not a reading list.

## History

**Before it — the first reference set (2026-05-21).** [`../03_apertus_extension_and_embedding_adaptation/references/MANIFEST.md`](../03_apertus_extension_and_embedding_adaptation/references/MANIFEST.md) listed eight external repos (six Swiss-AI, plus Apple’s AdEMAMix reference implementation and the EleutherAI harness; two pinned to exact commits, the rest at HEAD) and 15 papers, with a citation convention down to file, line and commit. It overlaps this library on four sources (Apertus, AdEMAMix, Goldfish, and WSD — there sourced to MiniCPM, here to Hägele) and covers different ground: init methods (ReTok, FVT, Hewitt, Mundra), the engine (Megatron, QK-Norm) and the Greek eval suites (Meltemi, Krikri).

**The build.** [`fetch.sh`](fetch.sh) holds the whole manifest as 20 `slug|arXiv-id` pairs (one, `jiang-tokenizer-aware-adaptation`, resolves to an ACL Anthology URL), downloads each PDF and runs `pdftotext` into `txt/`, six at a time, skipping any paper whose text already exists and exceeds 4 KB. The result is `pdf/` (20 PDFs), `txt/` (20 extractions, 25 KB–392 KB each) and `notes/` (20 notes).

**The read.** Each note follows one shape: a TL;DR, a "Key load-bearing facts" list with line references into the extracted text, then a **Citation audit** block per claim (claim → verdict → evidence quoted from the paper → correction), and a "How we use it" paragraph. [`CITATION_AUDIT.md`](CITATION_AUDIT.md) consolidates all of them: 20 papers, 40 cited claims, **18 not fully confirmed** — 2 `CONTRADICTED`, 2 `UNSUPPORTED`, 6 `OVERSTATED`, 8 `PARTIAL`.

**The feedback into the recipe.** [`../CURRENT_HYPERPARAMETERS.md`](../CURRENT_HYPERPARAMETERS.md) as committed on the same day already carries an explicit "**Citation hygiene (from `papers/CITATION_AUDIT.md`)**" note in §6 and several of the audit's rewrites verbatim in its justification cells — the β₁ row now says AdEMAMix's condition is β₃ ≫ β₁ (not β₂ ≫ β₁), the β₃ row explicitly refuses the distribution-shift attribution, and the Goldfish row credits k=h=50 to Apertus rather than to the Goldfish paper.

## Outcome

- **Two citations were pointing backwards.** `optimal-embedding-lr` was cited to support a *uniform* embedding LR but argues for LR_emb = √d · LR_hidden, i.e. a **higher** one; `jiang-tokenizer-aware-adaptation` was cited as an argument for unfreezing the body but its remedy is a **frozen**-body LoRA adapter. Both are named in the recipe doc's own §6.
- **The five flagged tensions were settled by sweeping, not by citation.** The audit's headline advice — "treat 5.5e-5 as a sweep midpoint, not a derived value", "decide explicitly between uniform / higher-emb / lower-emb", "β₃=0.999 is supported only for low-iteration runs", "cite Apertus for k=h=50" — matches what the frozen recipe now records: peak LR 5.5e-5 selected as the sweep's adaptation/retention knee, α=4 chosen over {0,4,8}, β₂=0.999 chosen over {0.99, 0.995, 0.999}, uniform LR kept on bakeoff empirics, and warmup pinned at a fixed 400 iterations with an explicit "do **not** reapply `2/(1-beta2)`" — the exact rule the audit called speculative.
- **The audit was only half-absorbed.** The corrections landed in the table rows that carry the decisions, but `CURRENT_HYPERPARAMETERS.md`'s trailing `## References` list still repeats several flagged phrasings unrevised: "β₃ = 0.999 for low-iteration / **non-stationary** settings" (audit: `PARTIAL`), Hägele as the source of a "set max LR to half the cosine value" **guideline** (audit: `OVERSTATED`, it is one small-scale observation), Goldfish "with little downstream impact" (audit: `PARTIAL`, true only at k=3–4), Token Distillation's relearning as what "**subsumes** a separate stabilization phase" (audit: `UNSUPPORTED`), EEVE's "must eventually unfreeze" (audit: `PARTIAL`), and uniform LR + good init as "**documented practice**" (audit: `PARTIAL`, one chemistry paper).
- **The library is refreshable** — `bash fetch.sh` is idempotent — but nothing in it has been re-run or re-audited since 2026-06-11.

## Index

Grouped by the topic each note declares. "Feeds" names the section of [`../CURRENT_HYPERPARAMETERS.md`](../CURRENT_HYPERPARAMETERS.md) the paper was cited in and, where the concept appears in a subproject's own docs, that subproject.

### Base model — the anchor for everything

| Paper | Supplies | Feeds | Audit |
|---|---|---|---|
| [`apertus`](notes/apertus.md) (2509.14233) | Every Apertus recipe number: peak LR 1.1e-4, WSD + 1-sqrt cooldown, AdEMAMix pretraining betas, Goldfish k=h=50, rope geometry, batch doubling | §1–§5; the base for all of 03–10 | All CONFIRMED — "do not touch" |

### Optimizer → §1

| Paper | Supplies | Feeds | Audit |
|---|---|---|---|
| [`ademamix`](notes/ademamix.md) (2409.03137) | β₁/β₂/β₃/α semantics, the β₃/α warmup schedulers, the low-iteration β₃=0.999 result | §1; the optimizer sweeps in [`../05_token_distillation_cpt`](../05_token_distillation_cpt/README.md) and the Task-1 regime in [`../04_cpt_training_regime_on_vanilla`](../04_cpt_training_regime_on_vanilla/README.md) | 1 OVERSTATED (β₂-vs-β₁ stability), 1 PARTIAL (distribution shift), 3 CONFIRMED |

### LR schedule and warmup → §2

| Paper | Supplies | Feeds | Audit |
|---|---|---|---|
| [`hagele-wsd-scaling`](notes/hagele-wsd-scaling.md) (2405.18392) | The WSD/constant-plus-cooldown scaling analysis; origin of the "0.5× peak" framing | §2; WSD shape reused in [`../06_dataset_scheduling_experiments`](../06_dataset_scheduling_experiments/README.md), [`../07_full_8b_cpt`](../07_full_8b_cpt/README.md), [`../10_early_cooldown_causal_experiment`](../10_early_cooldown_causal_experiment/README.md) | OVERSTATED — an observation at 33M–360M, not a rule |
| [`ibrahim-cpt`](notes/ibrahim-cpt.md) (2403.08763) | Re-warm + re-decay + replay; the adaptation-vs-forgetting LR dial | §2; replay decisions in [`../05_token_distillation_cpt`](../05_token_distillation_cpt/README.md) | CONFIRMED |
| [`gupta-rewarm-cpt`](notes/gupta-rewarm-cpt.md) (2308.04014) | Re-warm-then-decay is needed to learn well downstream | §2 | PARTIAL — Pythia-410M, perplexity only |
| [`stability-gap-cpt`](notes/stability-gap-cpt.md) (2406.14833) | The V-shaped capability dip under CPT and distribution-matched replay as the mitigation | §2; replay sizing in [`../05_token_distillation_cpt`](../05_token_distillation_cpt/README.md) | CONFIRMED |
| [`practitioner-multimodal-cpt`](notes/practitioner-multimodal-cpt.md) (2408.14471) | Maps the adaptation/retention curve over base LR | §2 | PARTIAL — OpenCLIP; the 1e-5 optimum is a batch-rescaled pretraining peak |
| [`ma-yarats-warmup`](notes/ma-yarats-warmup.md) (1910.04209) | Adam oversteps early even when restarted at a minimum — why a re-warm is needed | §2 | OVERSTATED as a `2/(1-β₂)` horizon; the rule was later dropped for a fixed 400-iteration warmup |

### Loss → §3

| Paper | Supplies | Feeds | Audit |
|---|---|---|---|
| [`goldfish-loss`](notes/goldfish-loss.md) (2406.10209) | The 1/k token-masking loss | §3; goldfish config carried through [`../04_cpt_training_regime_on_vanilla`](../04_cpt_training_regime_on_vanilla/README.md), [`../06_dataset_scheduling_experiments`](../06_dataset_scheduling_experiments/README.md), [`../07_full_8b_cpt`](../07_full_8b_cpt/README.md), [`../08_targeted_8b_cpt_experiments`](../08_targeted_8b_cpt_experiments/README.md) | 1 UNSUPPORTED (never tests k=h=50; recommends k=3/4, h=13), 1 PARTIAL |

### New-token initialization → §4

| Paper | Supplies | Feeds | Audit |
|---|---|---|---|
| [`token-distillation`](notes/token-distillation.md) (2505.20133) | The selected init method — distil input rows E against the frozen body | §4; [`../03_apertus_extension_and_embedding_adaptation`](../03_apertus_extension_and_embedding_adaptation/README.md) (`TOKEN_DISTILLATION_PLAN.md`) and all of [`../05_token_distillation_cpt`](../05_token_distillation_cpt/README.md) | 1 OVERSTATED (it does not train U), 1 UNSUPPORTED (does not license skipping stabilization) |
| [`artetxe-crosslingual-transfer`](notes/artetxe-crosslingual-transfer.md) (1910.11856) | Freeze-body / train-embeddings transfer | §4 | CONFIRMED |
| [`eeve-vocab-expansion`](notes/eeve-vocab-expansion.md) (2402.14714) | The 7-stage parameter-freezing schedule for vocabulary expansion | §4 | PARTIAL — the unfreeze is a preliminary finding, not an ablation |
| [`jiang-tokenizer-aware-adaptation`](notes/jiang-tokenizer-aware-adaptation.md) (ACL 2026.eacl-long.357) | Embedding relearning on decoder-only LLMs at scale | §4 | **CONTRADICTED** for the claim it was cited for — its fix is a frozen-body LoRA |

### Embedding learning rate → §4

| Paper | Supplies | Feeds | Audit |
|---|---|---|---|
| [`allam-arabic-cpt`](notes/allam-arabic-cpt.md) (2407.15390) | The only clean uniform-LR + averaged-init data point for vocab-extension CPT | §4 (differential LR *not* used) | CONFIRMED |
| [`optimal-embedding-lr`](notes/optimal-embedding-lr.md) (2506.15025) | Theory of embedding LR under Adam | §4 | **CONTRADICTED** — argues for a *higher* embedding LR, scaling with width not vocabulary |
| [`ulmfit`](notes/ulmfit.md) (1801.06146) | Discriminative per-layer LRs | §4 | PARTIAL — STLR is a separate schedule, not the per-group idea |
| [`llrd-bert-finetuning`](notes/llrd-bert-finetuning.md) (2006.05987) | Layer-wise LR decay; lower layers get a lower LR | §4 | CONFIRMED as pointing *lower*, not higher |
| [`tokenization-bottleneck`](notes/tokenization-bottleneck.md) (2511.14365) | A worked vocab-extension CPT recipe (Llama3-8B, chemistry) | §4 | PARTIAL — one recipe, not "documented practice" |

### Vocabulary cutoff → §4

| Paper | Supplies | Feeds | Audit |
|---|---|---|---|
| [`tao-vocab-scaling`](notes/tao-vocab-scaling.md) (2407.13623) | Vocabulary has a compute-bounded optimum | §4 (the 17,408 cutoff); tokenizer sizing in [`../02_1_tokenizer_experiments`](../02_1_tokenizer_experiments/README.md) | OVERSTATED — it does not supply 17k; compute-optimal V for 8B is 60–90k |
| [`magikarp-undertrained-tokens`](notes/magikarp-undertrained-tokens.md) (2405.05417) | The under-trained-token taxonomy and detection signature | §4 (noise-token cleanup); [`../02_2_tokenizer_implementation`](../02_2_tokenizer_implementation/README.md) | 1 OVERSTATED (no ">1,000× firing" threshold exists), 1 PARTIAL |

## Where things are

| Artifact | Role |
|---|---|
| [`CITATION_AUDIT.md`](CITATION_AUDIT.md) | The consolidated audit: verdict table, the confirmed load-bearing list, and five named tensions to resolve before the LR sweep. Read this before any note. |
| [`notes/`](notes/) | 20 per-paper notes; each is self-contained (facts, per-claim verdicts with quoted evidence, and how the project uses the paper). |
| [`fetch.sh`](fetch.sh) | The manifest and the idempotent fetch/extract pipeline. |
| [`txt/`](txt/), [`pdf/`](pdf/) | Extracted full text (what the notes cite by line number) and the source PDFs. |
| [`../CURRENT_HYPERPARAMETERS.md`](../CURRENT_HYPERPARAMETERS.md) | The document under audit — its §6 carries the audit's own summary and its `## References` list is the part that was never re-swept. |

## Working documents

There are no plans, logs or status snapshots in this directory — it is a single-pass artifact. The one thing to be aware of when reading it: the notes and the audit were written against `CURRENT_HYPERPARAMETERS.md` as of 2026-06-11, and that document was re-frozen on 2026-07-11 (`e0ebe592`) after the optimizer and LR sweeps completed. Several claims the audit criticised no longer appear in the doc's decision cells; several still stand in its reference list.
