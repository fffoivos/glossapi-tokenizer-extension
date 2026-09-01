# Greek Apertus CPT launch resource specification

**Date:** 2026-08-01

**Status:** evidence map complete; **not yet authorized for a production launch**

**Reference implementation to scale:** [`06_25b_midtraining_probe`](06_25b_midtraining_probe/README.md)

**Obsolete production-size recipe / valid 25B diagnostic:** [`recipe_25b_midtraining.json`](06_25b_midtraining_probe/configs/recipe_25b_midtraining.json)

## 1. Purpose and authority

This document identifies the exact data, tokenizer, initialization, training
recipe, software and Clariden orchestration needed for the Greek
Apertus-8B continued-pretraining run. It distinguishes four kinds of evidence:

1. **Upstream facts:** the Apertus report, official SwissAI code/data pipeline,
   upstream dataset repositories, and the Token Distillation paper/code.
2. **Project decisions:** settings selected by our controlled CPT experiments.
3. **Published artifacts:** immutable Hugging Face revisions and their embedded
   manifests.
4. **As-run evidence:** Clariden receipts that hash the exact local payloads,
   binaries, code, model conversion and checkpoints used by a run.

The authority order for an actual launch is:

1. a newly approved, versioned machine recipe;
2. a clean immutable commit of this repository;
3. completed Clariden input, bridge, initialization, round-trip and smoke
   receipts bound to that commit and recipe;
4. the upstream and experimental evidence cited here.

Earlier prose does not override a machine receipt. In particular,
[`05_training_dataset_bridge`](05_training_dataset_bridge/README.md) remains a
shared implementation source for replay and binary-building utilities, but its
single-blend production launcher is superseded by Phase 06.

## 2. Frozen corpus-coverage requirement and open mixture decisions

The production run will use **one complete pass over the eligible training
portion of `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`**. The
published corpus contains 63,822,761,532 training tokens before the final
GreekMMLU/heldout exclusions. The exact production Greek-token numerator must
come from a new post-exclusion Clariden receipt and every eligible Greek
document identity must receive a consumed-or-explicitly-excluded disposition.

The checked-in Phase-06 recipe currently encodes a smaller diagnostic:

- 24,998,051,840 effective tokens over 5,960 iterations;
- phase 1 at 79% HPLT Greek, 20% foreign replay and 1% old-Greek replay;
- phase 2 at 79% total new Greek, split between HPLT and non-HPLT so that the
  whole run preserves the new corpus's natural HPLT share, plus the same 20%
  foreign and 1% old-Greek replay;
- randomized order with seed `20260609` and a phase boundary at iteration
  3,570.

The 25B horizon consumes only about 19.75B new-Greek tokens, approximately
30.9% of the published corpus, so it is **not the production run**. Phase 06 is
the reference implementation for receipt gates, two-phase resume and
evaluation, but its horizon, phase boundaries, capacity plan and confirmation
names must be regenerated for the full-corpus run.

The replay **mixture is still open**, as requested. Before launch we must decide
and record:

1. one blend or two-phase midtraining;
2. the macro shares for new Greek, foreign replay and exact old-Greek replay;
3. the internal foreign-replay shares by language, code and math;
4. whether the final phase should restore the natural HPLT/non-HPLT share;
5. the total token horizon derived from the post-exclusion full-Greek count and
   approved Greek share.

For scale, if the final mix remains 79% new Greek / 20% foreign replay / 1%
old-Greek replay and the pre-exclusion 63,822,761,532-token count is used, the
nominal run is:

| Component | Planning tokens |
|---|---:|
| Complete new-Greek corpus | 63,822,761,532 |
| Foreign replay | 16,157,661,147 |
| Old-Greek replay | 807,883,057 |
| **Total** | **80,788,305,737** |

At the current 4,194,304-token global batch, covering that nominal target
requires 19,262 iterations and yields 80,790,683,648 effective batched tokens,
about 2.38M above the fractional target. Final counts will shift slightly after
heldout/decontamination exclusions and exact integer sample allocation.

The current replay acquisition was sized for only about 5.0B foreign-replay
tokens. A provisional 79/20/1 full-corpus run needs about 16.16B foreign tokens,
so replay selection/capacity must be expanded by roughly 3.23x or a deliberate
repetition policy must be approved. Any change requires a new recipe ID,
rebuilt binaries, new complete-coverage and capacity proofs, a new
training-assets receipt and a new two-phase smoke. No 25B receipt may be reused.

## 3. Dataset A: the new Greek corpus

### 3.1 Immutable training input

Use the gated/public Hugging Face release
[`fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` at
`3f97cec48af502f4996cf8ff20b02660e2dd3d31`](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/tree/3f97cec48af502f4996cf8ff20b02660e2dd3d31),
not a moving branch. The Phase-06 recipe pins:

| Property | Frozen value | Evidence |
|---|---:|---|
| Documents | 51,839,746 | [dataset card](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/blob/3f97cec48af502f4996cf8ff20b02660e2dd3d31/README.md), [deduplicated manifest](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/raw/3f97cec48af502f4996cf8ff20b02660e2dd3d31/manifests/deduplicated_manifest.json) |
| Parquet files | 431 | same manifest |
| Parquet bytes | 140,506,550,248 | [`recipe_25b_midtraining.json`](06_25b_midtraining_probe/configs/recipe_25b_midtraining.json) and as-run materialization receipt |
| Training tokens | 63,822,761,532 | current-tokenizer Clariden count pinned in the recipe; a fresh receipt is required |
| HPLT training tokens | 44,054,228,362 | same token-count receipt |
| Identity | `(source_dataset, source_doc_id)` | recipe and published Parquet schema |

Every training row must retain `text`, `source_dataset` and `source_doc_id`.
The published schema also retains source metadata fields such as title, author
and `source_metadata_json`, so HPLT/non-HPLT, source-family, archive and
historical/polytonic slices can be selected without reconstructing provenance.
The Phase-06 materializer and freezer independently check required columns,
file count, row count, byte size and SHA-256 for all 431 Parquets:

- [`materialize_hf_v2.py`](06_25b_midtraining_probe/clariden/materialize_hf_v2.py)
- [`freeze_inputs.py`](06_25b_midtraining_probe/dataset/freeze_inputs.py)

### 3.2 What proves cleaning and deduplication

The release contains three separate evidence layers:

1. The [deduplication manifest](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/raw/3f97cec48af502f4996cf8ff20b02660e2dd3d31/manifests/deduplicated_manifest.json)
   records `status=passed`, 53,046,533 input rows, 51,839,746 survivors and
   1,206,787 removed rows.
2. The [dedup decision ledger](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/resolve/3f97cec48af502f4996cf8ff20b02660e2dd3d31/manifests/dedup_decision_ledger.parquet)
   has exactly 1,206,787 decisions and SHA-256
   `4d1eeacc5c6028abbd5dbbaa8cd6006734b481caec751a02dbf52c7621174a54`.
   The inventory has 431 rows and SHA-256
   `7e89b632ea8823dd392e5e0e66ab36a4230318ffdad491558223f05c8e84abb9`.
3. The [bibliography reconstruction receipt](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/raw/3f97cec48af502f4996cf8ff20b02660e2dd3d31/manifests/bibliography_reconstruction.json)
   binds the cleaned corpus back to the deduplicated release, preserves all
   51,839,746 rows, transforms 175,242 documents in nine shards, and leaves the
   other 422 shards unchanged. The transformation contract hash is
   `2ce5154065f31965983bde8427cc302d647ef8a89b7db43005ed030bbf38eead`;
   the apply-summary hash is
   `725c204aa363ceaaeb4c78cb13111f6401e3ae8bf948f2ab786b27cc287bf626`.

The underlying release-integrity design and source-specific provenance are
documented in [`release_integrity.md`](04_full_corpus_preparation/docs/release_integrity.md),
[`sources.json`](04_full_corpus_preparation/configs/sources.json) and the
[dataset card](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/blob/3f97cec48af502f4996cf8ff20b02660e2dd3d31/README.md).

### 3.3 Release-policy inconsistency that must be closed

The published [license override receipt](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2/raw/3f97cec48af502f4996cf8ff20b02660e2dd3d31/manifests/license_override_receipt.json)
records the 2026-07-28 owner directive to publish v2 with cleaned `libduth`
rows. It also explicitly says that this directive is **not rightsholder
permission**, does not supersede source terms, and that component-source terms
still apply.

That conflicts with the checked-in technical adjudication:

- [`source_license_adjudication.json`](04_full_corpus_preparation/configs/source_license_adjudication.json)
  marks `libduth` ineligible for both local training and redistribution because
  its pinned source card declares CC BY-NC-ND 4.0 and underlying author rights;
- [`source_license_adjudication.md`](04_full_corpus_preparation/docs/source_license_adjudication.md)
  lists `libduth` among the excluded sources;
- the published bibliography reconstruction receipt still contains
  `publication_ready=false`, despite the later override and actual publication.

Publication proves what was published; it does not by itself prove permission
to train or redistribute. Before a production launch, choose and receipt one of
the following:

1. written rightsholder permission or a legal review that permits the intended
   noncommercial training use;
2. an updated source adjudication, with its precise legal/evidence basis,
   consistent with the published override; or
3. a new dataset revision excluding `libduth`, followed by a fresh token count,
   mixture calculation and all Clariden receipts.

The bibliography-cleaning implementation evidence is retained in
[`BIB_CLEANING_HANDOVER_20260727.md`](02_corpus_preparation/15_clean_academic/BIB_CLEANING_HANDOVER_20260727.md).

## 4. Dataset B: replay from Apertus training sources

### 4.1 Meaning of “replay”

There are two evidence classes and they must not be conflated:

| Pool | Defensible claim | Evidence strength |
|---|---|---|
| Foreign replay | documents acquired from the same **public source families** used by Apertus | family-level provenance; not proof that each selected document was consumed by the released 8B model |
| Old-Greek replay | exact Nanochat identities that the audit flagged as strict, relaxed-exact or near overlaps with Apertus Greek pretraining | document-level overlap classification and Clariden build receipt; not a consumed-document manifest |

The Apertus technical report identifies FineWeb-Edu, FineWeb2/FineWeb2-HQ,
FineMath and StarCoder as pretraining components: [Apertus technical report,
Table 6 and Appendix C](https://arxiv.org/pdf/2509.14233). The official data
implementation is available at SwissAI's pinned
[`pretrain-data` commit `8af990b`](https://github.com/swiss-ai/pretrain-data/tree/8af990b9401101cf95acd02b066ed0c449789126),
including the [FineWeb-Edu Score-2](https://github.com/swiss-ai/pretrain-data/blob/8af990b9401101cf95acd02b066ed0c449789126/pipelines/fineweb-edu/main-score-2.py),
[FineWeb-2](https://github.com/swiss-ai/pretrain-data/blob/8af990b9401101cf95acd02b066ed0c449789126/pipelines/fineweb-2/main.py),
[FineMath](https://github.com/swiss-ai/pretrain-data/blob/8af990b9401101cf95acd02b066ed0c449789126/pipelines/finemath/main.py) and
[StarCoder code-pipeline](https://github.com/swiss-ai/pretrain-data/tree/8af990b9401101cf95acd02b066ed0c449789126/examples/code_pipeline)
implementations.

The public sources do not expose a complete manifest of the exact examples
consumed by `Apertus-8B-2509`. Therefore the foreign subset must be called
**Apertus source-family replay**, not “the exact original training examples.”
If exact document membership is required, that is a separate data-access
requirement and the current foreign replay is insufficient.

### 4.2 Immutable replay sources

The executable acquisition policy is
[`replay_acquisition.json`](05_training_dataset_bridge/configs/replay_acquisition.json).
It pins every repository and deterministic selection:

| Role | Repository and immutable revision | Current selection |
|---|---|---|
| English educational web | [`HuggingFaceFW/fineweb-edu@87f0914`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/tree/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9) | complete pinned 10BT sample; builder applies `score >= 3` |
| High-resource multilingual web | [`epfml/FineWeb2-HQ@c0c06e9`](https://huggingface.co/datasets/epfml/FineWeb2-HQ/tree/c0c06e94fd3a44ae9e802b2b0fc533817601eb5e) | two deterministic shards for each of 12 languages |
| Other multilingual web | [`HuggingFaceFW/fineweb-2@af9c133`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/tree/af9c13333eb981300149d5ca60a8e9d659b276b9) | one deterministic shard for each of 11 languages |
| Math | [`HuggingFaceTB/finemath@e92b25a`](https://huggingface.co/datasets/HuggingFaceTB/finemath/tree/e92b25a616738fe95dc186b64dfb19f9c8525594) | four `finemath-3plus` shards |
| Code | [`bigcode/starcoderdata@9fc30b5`](https://huggingface.co/datasets/bigcode/starcoderdata/tree/9fc30b578cedaec69e47302df72cf00feed7c8c4) | 28 files across Python, JavaScript, Java, Go, Rust, C++ and TypeScript |

The acquisition plan expects 355 selected files and 250,673,537,368 remote
bytes. [`acquire_replay_sources.py`](05_training_dataset_bridge/scripts/acquire_replay_sources.py)
resolves the exact commits, downloads only the selected paths, hashes every
payload and emits `replay_acquisition_receipt.json`. The Clariden launcher is
[`acquire_replay.sbatch`](06_25b_midtraining_probe/clariden/acquire_replay.sbatch).

The current internal foreign-pool weights come from
[`bulk_13b.json`](03_training_experiments/dataset_build/bulk_13b.json), then are
renormalized inside the foreign pool by the Phase-06 finalizer. They are a
project replay design, **not the published Apertus token distribution**:

- English: 6.3337% of the foreign pool;
- seven tier-1 languages: 4.3336% each;
- eleven tier-2 languages: 2.9935% each;
- five tier-3 languages: 2.0801% each;
- code: 13.3343%;
- math: 6.6672%.

These shares belong to the open mixture decision. The final approved recipe
must store exact rational/decimal weights, and the bridge receipt must report
planned and available **tokens**, not merely file or document proportions.

### 4.3 Exact old-Greek replay

Use these two pinned sources:

- [`fffoivos/glossapi-greek-nanochat-pretraining-dataset@e1d5413`](https://huggingface.co/datasets/fffoivos/glossapi-greek-nanochat-pretraining-dataset/tree/e1d54136a880ed1df2ed95a5445dabd230453207);
- [`fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z@54faa75`](https://huggingface.co/datasets/fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z/tree/54faa75b5e0b4fad01bf7bf5541210c741cb10b8),
  specifically the `cpt_final_overlay/apertus_overlap_drop_docs.parquet`
  artifact and its [summary](https://huggingface.co/datasets/fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z/raw/54faa75b5e0b4fad01bf7bf5541210c741cb10b8/artifacts/dedup_20260519T010924Z/cpt_final_overlay/summary.json).

The overlay reports 2,223,742 unique Nanochat documents overlapping Apertus
Greek pretraining: 1,129,821 `strict_exact`, 225,318 `relaxed_exact`, and
868,603 `near`. [`build_old_greek_replay.py`](05_training_dataset_bridge/scripts/build_old_greek_replay.py)
materializes the **exact** `(source_dataset, source_doc_id)` intersection with
that overlay. The Clariden build receipt must show 2,223,742 unique
input/overlay/output identities, zero unmatched identities and byte hashes for
the resulting Parquet. This proves that every pool row is one of the Nanochat
documents flagged by the overlap audit. It does not prove that every selected
document was consumed byte-for-byte by Apertus, especially for the `near`
class; only an original consumed-document manifest could prove that stronger
claim.

### 4.4 Decontamination, heldouts, phase separation and capacity

Before tokenized training binaries are accepted:

1. [`freeze_greekmmlu.sbatch`](06_25b_midtraining_probe/clariden/freeze_greekmmlu.sbatch)
   freezes all 16,632 GreekMMLU prompts and benchmark provenance.
2. [`freeze_decontamination_binding.py`](06_25b_midtraining_probe/dataset/freeze_decontamination_binding.py)
   binds the prompt file, benchmark manifest and implementation.
3. [`build_heldouts.py`](05_training_dataset_bridge/scripts/build_heldouts.py)
   selects deterministic heldouts; their document identities are excluded
   before phase assignment.
4. [`build_binary_shard.py`](05_training_dataset_bridge/scripts/build_binary_shard.py)
   applies heldout exclusion and GreekMMLU decontamination, tokenizes with no
   special-token insertion, appends EOD and emits retained/dropped ledgers.
5. [`finalize_phase_bridge.py`](06_25b_midtraining_probe/dataset/finalize_phase_bridge.py)
   hashes every `.bin`, `.idx` and ledger, proves no document identity appears
   twice across phases using a SQLite primary key, accounts for exact duplicate
   text by SHA-256, and requires at least 1.005x nonrepeating sample capacity
   plus one boundary sample for every phase, logical pool, source and physical
   prefix.

The finalizer **measures and discounts duplicate content for capacity**; it is
not a claim that it globally near-deduplicates the foreign replay against every
Greek document. If cross-pool near-deduplication is desired, it must be added as
a separate preprocessing decision and receipted before binary construction.

## 5. Tokenizer and model initialization

### 5.1 Tokenizer source of truth

Load the tokenizer from
[`fffoivos/apertus-tokenizer-extension@fcd33ec`, subfolder
`greek-modern-polytonic-tokenizer`](https://huggingface.co/fffoivos/apertus-tokenizer-extension/tree/fcd33ec09fb7d86bc072b3a4b3e890efa6473b66/greek-modern-polytonic-tokenizer):

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "fffoivos/apertus-tokenizer-extension",
    revision="fcd33ec09fb7d86bc072b3a4b3e890efa6473b66",
    subfolder="greek-modern-polytonic-tokenizer",
    trust_remote_code=True,
)
assert len(tokenizer) == 148_992
```

The [release manifest](https://huggingface.co/fffoivos/apertus-tokenizer-extension/blob/fcd33ec09fb7d86bc072b3a4b3e890efa6473b66/greek-modern-polytonic-tokenizer/manifest.json),
[release audit](https://huggingface.co/fffoivos/apertus-tokenizer-extension/blob/fcd33ec09fb7d86bc072b3a4b3e890efa6473b66/greek-modern-polytonic-tokenizer/release_audit.json),
[selection](https://huggingface.co/fffoivos/apertus-tokenizer-extension/blob/fcd33ec09fb7d86bc072b3a4b3e890efa6473b66/greek-modern-polytonic-tokenizer/selection.json)
and [suspicious-token review](https://huggingface.co/fffoivos/apertus-tokenizer-extension/blob/fcd33ec09fb7d86bc072b3a4b3e890efa6473b66/greek-modern-polytonic-tokenizer/suspicious_token_review.json)
prove:

- 131,072 Apertus base tokens + 17,408 modern-Greek merges + 512
  polytonic continuation merges = 148,992 tokens;
- contiguous IDs `0..148991`;
- exact base vocabulary and merge prefix;
- 512 dependency-safe sequential appended merges with no orphan/dummy entries;
- exact front end and Hugging Face sidecars;
- `148992 = 582 x 256 = 291 x 512`, so with TP=2 and
  `--make-vocab-size-divisible-by 256` the actual vocabulary needs **zero
  padding tokens**;
- `tokenizer.json` SHA-256
  `bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b`.

The local reconstruction/audit copies are under
[`SHIP_TOKENIZER_RECONSTRUCTION.md`](../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md)
and the adjacent `ship/apertus_greek_modern_polytonic_148992` directory.

### 5.2 Token Distillation evidence and exact production procedure

The method is from Dobler, Elliott and de Melo,
[“Token Distillation: Attention-Aware Input Embeddings for New
Tokens”](https://arxiv.org/abs/2505.20133). The vendored implementation is
[`konstantinjdobler/token-distillation` at commit
`35702b5809599ecd68b7845eca27a0d7b7cec0da`](../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/external/token-distillation/PINNED_UPSTREAM.md).

Our fixed-ID BPE extension cannot use upstream's high-level API, because that
API appends tokens itself. Our adapter
[`train_retok_td.py`](../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/train_retok_td.py)
loads the already-built tokenizer, maps the immutable new IDs and calls the
lower-level training loop.

The completed modern-Greek TD experiment is recorded by:

- [`retok_td_manifest.json`](../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/full_td_20260523T092602Z/layer11/retok_td_manifest.json):
  layer 11, one epoch, batch 8, LR `1e-4`, bf16, 17,377/17,392 requested
  trainable tokens completed (99.9138%);
- [`td_preservation_report.json`](../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/full_td_20260523T092602Z/layer11/td_preservation_report.json):
  zero non-embedding drift and exact preservation of all non-target rows;
- the published uncpt `experiment-checkpoints/TokenDistil-Init` subfolder at
  the tokenizer repository revision above.

The production 148,992-row initialization is rebuilt by
[`build_production_init.sbatch`](06_25b_midtraining_probe/initialization/build_production_init.sbatch):

1. Start from the **uncpt** 148,480-row layer-11 `TokenDistil-Init` checkpoint.
   The CPT-trained `TokenDistil-3.5B` checkpoint is explicitly forbidden.
2. Append IDs `148480..148991` in merge order. Initialize each input/output row
   from its merge parents, with norm calibration against the modern rows.
3. Run input-embedding Token Distillation for the appended rows using 25
   snippets/token, one epoch, batch 8, LR `1e-4`, target layer 11 and bf16. The
   configured minimum trained fraction is 90%; skipped rows remain their
   merge-chain initialization.
4. Because Apertus uses untied input/output embeddings, calibrate all 512 new
   output rows with next-token CE for 400 steps, sequence length 512, LR
   `2e-4`, max grad norm 1.0 and seed `20260729`.
5. [`verify_production_init.py`](06_25b_midtraining_probe/initialization/verify_production_init.py)
   must prove that old input rows, old output rows and all non-embedding
   parameters/buffers are bit-exact, and that every new row is finite and
   nonzero.
6. [`roundtrip_production_init.sbatch`](06_25b_midtraining_probe/clariden/roundtrip_production_init.sbatch)
   converts HF -> Megatron TP=2 -> HF and requires zero standard, R17, xIELU
   and QK-Norm drift before the checkpoint can be frozen as a training asset.

The expected as-run evidence paths are:

```text
$INIT_ROOT/production_init_verification.json
$ROUNDTRIP_ROOT/work/verification.json
```

Do not rely on a checkpoint name alone; both receipt files and their complete
model trees must be hash-bound by `training_assets_receipt.json`.

## 6. Training hyperparameters and their provenance

### 6.1 Public reference versus current settled recipe

The public [`eellak/greek-apertus` recipe at commit
`905f632`](https://github.com/eellak/greek-apertus/tree/905f63270059a340e71e9f844480630e5bab8a95)
is the reusable reference for the Apertus model bridge, guarded Megatron
trainer and inherited geometry. Its
[`HYPERPARAMETERS.md`](https://github.com/eellak/greek-apertus/blob/905f63270059a340e71e9f844480630e5bab8a95/docs/HYPERPARAMETERS.md)
correctly labels beta2, alpha and peak LR as open at that commit. It also still
describes the older 13.5B horizon and physical-order curriculum.

Therefore, do **not** copy its old defaults blindly. The later local sweep
decision [`PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md`](PRODUCTION_HYPERPARAMETERS_DECISION_20260711.md)
supersedes those scalar candidates, and the Phase-06 machine recipe supersedes
the old horizon/order. The public repo remains authoritative for its reusable
training/bridge implementation and the documented distinction between regime
and launch profile:

- [training runbook](https://github.com/eellak/greek-apertus/blob/905f63270059a340e71e9f844480630e5bab8a95/docs/TRAINING.md);
- [model bridge](https://github.com/eellak/greek-apertus/blob/905f63270059a340e71e9f844480630e5bab8a95/docs/MODEL_BRIDGE.md);
- [executable CPT environment](https://github.com/eellak/greek-apertus/blob/905f63270059a340e71e9f844480630e5bab8a95/configs/training/cpt.env);
- [guarded trainer](https://github.com/eellak/greek-apertus/blob/905f63270059a340e71e9f844480630e5bab8a95/scripts/train/train_apertus_cpt.sbatch).

The original model/training reference is the
[Apertus technical report](https://arxiv.org/pdf/2509.14233) and the official
[`submit_apertus_8b.sh` at `pretrain-code@531cc8b`](https://github.com/swiss-ai/pretrain-code/blob/531cc8be2f76064127cad99a61019f985a7c7ee2/pretraining/submit_apertus_8b.sh).

### 6.2 Production regime and diagnostic-only 25B values

| Group | Current value | Provenance class |
|---|---|---|
| Model | 32 layers, hidden 4096, FFN 21,504, 32 heads, 8 query groups, xIELU, QK-LayerNorm, RMSNorm, untied embeddings/output, bias-free linear | Apertus-8B architecture, inherited |
| Sequence | 4,096 | Apertus main-pretraining geometry, inherited |
| RoPE | max positions 4,096; theta/base 500,000; scaling enabled; factor 8.0 | corrected Apertus main-pretraining geometry, inherited |
| Vocab divisor | 256; actual vocab 148,992; zero padding | tokenizer audit + Megatron/TP=2 compatibility |
| Optimizer | AdEMAMix; beta1 0.9, beta2 0.999, beta3 0.999, alpha 4.0 | beta1 inherited; beta2/beta3/alpha selected by local controlled sweeps |
| AdEMAMix ramps | beta3 and alpha over the complete production run | policy selected locally; production step count must be regenerated |
| Weight decay / clip | 0.1 / 0.1 | Apertus inherited and held fixed in local sweeps |
| Initialization std | 0.008944 | Apertus inherited |
| LR | peak `5.5e-5`, final/warmup-init `5.5e-6` | local LR sweep; peak is half the Apertus pretraining peak |
| LR schedule | WSD, 400-iteration warmup, final 20% `1-sqrt` cooldown | local fixed/sweep regime; **not** the original Apertus warmup horizon |
| Batch | microbatch 2; global 1,024 sequences = 4,194,304 tokens | project geometry held fixed in sweeps; global batch matches Apertus stage-1 scale, microbatch is project-specific |
| Precision | bf16 parameters, fp32 main gradients | Apertus inherited |
| Loss | Goldfish `k=50`, `h=50` | Apertus inherited |
| Data semantics | reset attention mask, reset position IDs, EOD-mask loss | Apertus/project inherited |
| Order | randomized, seed `20260609` | Phase-06 project decision |
| Parallelism | TP=2, PP=1, distributed optimizer/communication overlap | validated project runtime |
| Horizon | all eligible Greek data once, plus approved replay; about 80.788B nominal / 19,262 iterations if 79/20/1 survives | full-corpus owner requirement; exact post-exclusion count pending |
| Saves/evals | approximately every 0.5B tokens and heldout loss approximately every 0.1B tokens, plus an exact terminal save | preserve token cadence while recomputing integer intervals for the full run |

The corrected RoPE geometry is important: the released
[`swiss-ai/Apertus-8B-2509@3162c99`](https://huggingface.co/swiss-ai/Apertus-8B-2509/tree/3162c99675aa588097cecd4a24b9aa1f712af477)
contains the post-long-context release configuration, while CPT must use the
main-pretraining geometry above. The actual trainer emits:

```text
--max-position-embeddings 4096
--position-embedding-type rope
--rotary-base 500000
--use-rope-scaling
--rope-scaling-factor 8.0
```

The executable definitions are
[`full_corpus_25b.env`](05_training_dataset_bridge/train/full_corpus_25b.env),
[`phase_config.env`](06_25b_midtraining_probe/train/phase_config.env) and
[`bakeoff_train.sbatch`](../03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training/bakeoff_train.sbatch).
The first two still hard-code the diagnostic 25B horizon and must be replaced
by a full-corpus recipe/config pair; only the trainer implementation and
settled non-horizon hyperparameters are reusable unchanged.

## 7. Clariden orchestration

### 7.1 Frozen software and paths

The current launch profile is:

| Resource | Value |
|---|---|
| Slurm account / partition | `a0140` / `normal` |
| Training allocation | 16 nodes x 4 GPUs = 64 GPUs |
| Training time limit | 8 hours per segment |
| Smoke allocation | 1 node x 4 GPUs |
| uenv | `pytorch/v2.9.1:v2`, view `default` |
| Megatron | [`swiss-ai/Megatron-LM@c92402e`](https://github.com/swiss-ai/Megatron-LM/tree/c92402e39ef3c8e69ea378a59e79059dc14541f4) |
| Repository | clean immutable checkout; commit recorded in assets receipt |

Default paths and every expected receipt are centralized in
[`paths.env`](06_25b_midtraining_probe/clariden/paths.env):

```text
STAGE_ROOT=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/$CPT_RUN_ID
RUN_ROOT=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/25b_midtraining/$CPT_RUN_ID
DATASET_ROOT=/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/greek-cleaned-v2-3f97cec4
TOKENIZER_DIR=/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_polytonic_148992
MEGATRON_DIR=/iopsstor/scratch/cscs/fffoivos/code/training/Megatron-LM-Swiss-AI
```

### 7.2 Receipt-producing preparation chain

On Clariden, from the clean checkout:

```bash
export REPO_ROOT=/iopsstor/scratch/cscs/fffoivos/repo/<clean-cpt-checkout>
export CPT_RUN_ID=<new-immutable-run-id>

# Inspect the dependency graph; no state change.
DRY_RUN=1 clariden/submit_data_pipeline.sh prereqs

# Materialize runtime/data/replay/benchmark/init and freeze their receipts.
DRY_RUN=0 CONFIRM_PREPARATION=1 clariden/submit_data_pipeline.sh prereqs

# After input_receipt.json and heldouts are ready, tokenize/build/finalize.
DRY_RUN=0 CONFIRM_PREPARATION=1 clariden/submit_data_pipeline.sh after-freeze

# After bridge + model round-trip pass, freeze all launch assets.
DRY_RUN=0 CONFIRM_PREPARATION=1 clariden/submit_data_pipeline.sh assets
```

The entry point is
[`submit_data_pipeline.sh`](06_25b_midtraining_probe/clariden/submit_data_pipeline.sh).
It never launches the 64-GPU run. It produces or requires:

```text
$RUNTIME_RECEIPT
$DATASET_ROOT/manifests/deduplicated_manifest.json
$BASE_INIT_HF/materialization_receipt.json
$STAGE_ROOT/receipts/replay_acquisition_receipt.json
$STAGE_ROOT/receipts/old_greek_build_receipt.json
$STAGE_ROOT/greekmmlu/decontamination_binding.json
$STAGE_ROOT/input_receipt.json
$STAGE_ROOT/heldouts/heldout_manifest.json
$INIT_ROOT/production_init_verification.json
$ROUNDTRIP_ROOT/work/verification.json
$STAGE_ROOT/bridge_manifest.json
$STAGE_ROOT/training_data.env
$STAGE_ROOT/training_assets_receipt.json
```

[`freeze_training_assets.py`](06_25b_midtraining_probe/initialization/freeze_training_assets.py)
requires a clean project checkout, exact Megatron commit, tokenizer hash,
zero-drift initialization/round-trip, completed bridge and hashed trainer/eval
dependencies.

### 7.3 Smoke and segment-launch pattern

After `training_assets_receipt.json` exists:

```bash
export TRAINING_ASSETS_RECEIPT=<absolute-path>

DRY_RUN=1 train/submit_smoke.sh
DRY_RUN=0 CONFIRM_GPU_LAUNCH=GREEK_CPT25B_SMOKE train/submit_smoke.sh
```

[`submit_smoke.sh`](06_25b_midtraining_probe/train/submit_smoke.sh) runs one
phase-1 iteration, saves, resumes one phase-2 iteration and requires finite
losses plus phase-relative index evidence `consumed_samples 8 -> 0`. Only its
passed `smoke_verification.json` can unlock a run. The current confirmation
string contains `25B`; the full-corpus successor should use a new run identity
so a diagnostic smoke cannot unlock production accidentally.

The diagnostic launcher demonstrates the required receipt-bound segment
pattern:

| Segment | Phase | Iterations | Resume requirement |
|---|---:|---:|---|
| 1 | 1 | 0 -> 1,785 | production init receipt |
| 2 | 1 | 1,785 -> 3,570 | checkpoint receipt at 1,785 |
| 3 | 2 | 3,570 -> 5,960 | checkpoint receipt at 3,570 |

Every live segment separately requires:

```bash
CONFIRM_GPU_LAUNCH=GREEK_CPT25B_64GPU
```

These exact boundaries and confirmation string are **not valid for the
full-corpus production run**. After the mixture decision, generate new segment
boundaries spanning the full derived horizon (provisionally 19,262 iterations
at 79/20/1), a new confirmation string, and an exact terminal-checkpoint rule.
The reusable launcher behavior is to validate frozen assets and smoke at job
start, start a checkpoint-evaluation watcher, and freeze each boundary
checkpoint for the next segment.

## 8. Evaluation, learning and forgetting

The run has two complementary evaluation paths:

1. **Heldout language-model loss every 25 iterations** for HPLT,
   non-HPLT, OpenArchives, Greek PhD, historical/polytonic Greek, English,
   German, Russian, Chinese, code, math and old Greek. These show adaptation
   and distribution-specific forgetting while training continues.
2. **GreekMMLU checkpoint evaluation at an approximately 0.5B-token cadence and
   at the exact terminal iteration.** The 25B diagnostic uses every 119
   iterations and terminal iteration 5,960; production must recompute the
   integer cadence and terminal iteration.
   [`watch_greekmmlu_checkpoints.sbatch`](06_25b_midtraining_probe/eval/watch_greekmmlu_checkpoints.sbatch)
   waits for complete checkpoint markers,
   [`submit_greekmmlu_checkpoint.sh`](06_25b_midtraining_probe/eval/submit_greekmmlu_checkpoint.sh)
   converts to HF and evaluates all 16,632 examples, and
   [`finalize_greekmmlu_checkpoint.py`](06_25b_midtraining_probe/eval/finalize_greekmmlu_checkpoint.py)
   freezes score and output hashes in `evaluation_receipt.json`.

Before launch, define a small immutable retention scoreboard from the same
foreign heldouts used during the sweeps (at minimum English, major languages,
code and math), plus base and initialized-model baselines. A loss curve without
the baseline cannot quantify forgetting. Every evaluation must bind the
checkpoint receipt, tokenizer revision, benchmark revision, harness commit and
exact result-file hashes.

## 9. Known gaps to close before “ready to launch”

| Gate | Current state | Required closure |
|---|---|---|
| Complete eligible Greek-corpus coverage | frozen owner requirement; current 25B recipe reaches only about 30.9% of the published pre-exclusion token count | after exclusions, freeze the eligible identity/token manifest; allocate every eligible document exactly once; prove each identity was consumed or explicitly excluded; derive the full horizon from that receipt |
| Final corpus mix and midtraining policy | intentionally open | approve exact macro/internal replay shares and phase policy; derive the horizon from complete eligible Greek coverage; version the recipe |
| Replay capacity for a full-corpus run | current acquisition is sized for about 5.0B foreign tokens | expand to the approved requirement (provisionally 16.16B at 79/20/1), or explicitly approve and receipt a repetition policy |
| `libduth` training/redistribution basis | owner override conflicts with technical source adjudication | permission/legal review, evidence-backed adjudication update, or exclude it in a new dataset revision |
| Bibliography publication flag | published receipt contains `publication_ready=false` | publish a corrected immutable metadata/receipt revision or document a signed adjudication explaining why the later override supersedes only this flag |
| Foreign replay claim | source-family provenance only | accept that wording, or obtain exact Apertus consumed-document manifests |
| Foreign replay proportions | current project weights, not proven original Apertus token shares | approve/recompute token shares and encode them in the new recipe |
| Cross-pool near duplicates | exact-content capacity accounting only | explicitly accept, or add a global cross-pool near-dedup stage |
| Frozen config dependency coverage | assets receipt hashes `phase_config.env` and trainer but not the sourced `full_corpus_25b.env` or `common_cpt.env` individually | add both files to `freeze_training_assets.py` dependencies and test job-start drift rejection |
| Job-start repository check | repository commit is recorded at freeze time, but preflight does not recheck HEAD/dirty state | revalidate commit and cleanliness at every smoke/production job start, or hash the whole relevant repo tree |
| Current Clariden state | prior receipts were observed, but live status could not be refreshed on 2026-08-01 because `clariden` rejected the local SSH certificate while `ela` remained reachable | restore Clariden SSH, run `submit_data_pipeline.sh status`, and freeze a fresh status snapshot |
| Production smoke | required and asset-specific | run only after all decisions/config fixes and new receipts |

The two dependency-freeze gaps are concrete launch-integrity bugs: a sourced
configuration could change after the assets receipt is created without being
detected by the current job-start preflight. They should be fixed before
trusting `training_assets_receipt.json` as a complete launch contract.

## 10. Final launch checklist

A production segment is authorized only when all boxes are true:

- [ ] post-exclusion manifest fixes the complete eligible Greek identity and
  token set, with every identity consumed exactly once or explicitly excluded;
- [ ] exact replay mixture, phase policy and full-corpus-derived horizon
  approved in a new immutable recipe;
- [ ] `libduth` and bibliography publication-policy inconsistencies closed;
- [ ] dataset/tokenizer/model/upstream revisions remain exactly those approved,
  or this document and recipe are revised;
- [ ] clean immutable training repository and clean Megatron commit frozen;
- [ ] source configuration dependencies added to the training-assets receipt;
- [ ] Greek release, replay acquisition and old-Greek intersection receipts
  completed and rehashed;
- [ ] GreekMMLU binding and all heldout exclusions completed before binaries;
- [ ] phase bridge proves complete eligible Greek coverage, identity
  disjointness, exact blend sums and >=1.005x unique replay capacity at every
  required level (or binds an approved repetition policy);
- [ ] production TD initialization preservation and HF/Megatron round-trip both
  pass with zero old-row/body drift;
- [ ] training assets receipt rebuilt after the last code/config change;
- [ ] one-node/four-GPU two-phase smoke passes and is bound to those exact
  assets;
- [ ] segment dry run reviewed, storage capacity checked and expected checkpoint
  plus evaluation paths recorded;
- [ ] a new full-corpus-specific confirmation (not
  `GREEK_CPT25B_64GPU`) supplied for exactly one segment.

Until those gates close, the project is ready for **resource assembly and
receipt-bound preparation**, but not for a production training submission.
