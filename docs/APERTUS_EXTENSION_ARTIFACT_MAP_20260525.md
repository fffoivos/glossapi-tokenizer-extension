# Apertus Tokenizer Extension Artifact Map

Date: 2026-05-25.

Purpose: make the current tokenizer-extension work reviewable as a product, not
as a dump of scripts and run logs. This map records what exists, where it lives,
and what role each artifact plays in the story.

## Source Boundary

This map is based on:

- the local repo state in `/home/foivos/Projects/glossapi-tokenizer-extension`;
- the live Hugging Face model repo listing for
  `fffoivos/apertus-tokenizer-extension`, checked at commit
  `e58e61a2c129b4dbdab2f05cd6c6152e086945f5`;
- the local Clariden inventory snapshot
  [`../subprojects/03_apertus_extension_and_embedding_adaptation/CLARIDEN_INVENTORY_20260524.md`](../subprojects/03_apertus_extension_and_embedding_adaptation/CLARIDEN_INVENTORY_20260524.md).

Clariden paths below are authoritative according to the inventory snapshot, but
they were not live-reprobed from this document-writing pass.

## One-Line Story

We trained a continuous BPE extension of `swiss-ai/Apertus-8B-2509` on a
Greek-focused corpus, selected a clean 17,408-token modern Greek extension,
built a deduplicated CPT mixture, ran initialization and Token Distillation
experiments, and found TD layer 11 to be the leading extension path after the
3.5B-token continuation.

## Public Names

The HF release now uses short names for the main actors:

| Human name | Meaning |
|---|---|
| `ModernGreek-148k` | selected modern Greek tokenizer extension |
| `ModernGreek-Polytonic-154k` | optional stacked polytonic/ancient Greek tokenizer |
| `CPT-7B-mix` | 70/24/4/2 CPT training-data recipe |
| `TokenDistil-Init` | Token Distillation initialization checkpoint location |
| `TokenDistil-2B` | Token Distillation checkpoint location after about 2B tokens |
| `TokenDistil-3.5B` | selected Token Distillation checkpoint location after about 3.5B tokens |
| `Vanilla-3.5B` | original-tokenizer comparison checkpoint location |
| `ReTok-3.5B` | retokenization baseline checkpoint location |
| `3.5B-comparison` | compact benchmark summary |

Technical labels such as `layer11`, `iter_0000834`, R17, TP=2, and run tags are
kept in manifests and location files instead of public path names.

## Artifact Graph

```text
swiss-ai/Apertus-8B-2509
  |
  | continuous BPE on GlossAPI + HPLT 50/50
  v
C3 full tokenizer candidate
  base 131,072 + 25,600 = 156,672
  |
  | cutoff sweep, curation, alignment, backfill
  v
modern Greek tokenizer extension
  base 131,072 + 17,408 = 148,480
  |
  | optional stacked polytonic/ancient extension
  v
extended Greek tokenizer
  base 131,072 + modern 17,408 + polytonic 5,120 = 153,600

fffoivos/glossapi-greek-nanochat-pretraining-dataset
  + nanochat internal dedup metadata
  + Apertus overlap drop overlay
  |
  v
selected_after_apertus_and_internal_dedup.parquet
  |
  | 70/24/4/2 Greek/replay/code/math bulk recipe
  v
bulk_mix.nfc.jsonl
  |
  | Megatron preprocessing
  v
bulk_mix_base_nfc_megatron/bulk_mix_text_document.{bin,idx}

modern tokenizer + CPT mix
  |
  | init arms: Vanilla, ReTok, Centroid, TD layer 11
  v
2B bakeoff and 3.5B continuation
  |
  v
TD layer 11 leading checkpoint and evaluation artifacts
```

## 1. Tokenizer Discovery

The discovery tokenizer is C3:

- method: continuous BPE from Apertus, not `add_tokens`;
- base vocab: `131,072`, preserved exactly;
- full candidate: `+25,600` added tokens, total `156,672`;
- corpus: GlossAPI plus HPLT at 50/50 by training-token mass;
- key docs:
  - [`C3_CONVERGENCE.md`](C3_CONVERGENCE.md)
  - [`C3_TRAINING_DATASETS.md`](C3_TRAINING_DATASETS.md)
  - [`../subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/REPORT.md`](../subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/REPORT.md)

Role in the public story: this is the source of the 25k candidate-token universe.

## 2. Chosen Modern Greek Extension

Canonical modern-only tokenizer:

- added units: `17,408`;
- total vocab: `148,480 = 131,072 + 17,408`;
- alignment: `148,480 = 128 x 1160 = 256 x 580`;
- first 131,072 Apertus ids preserved verbatim;
- first 1000 special/reserved ids preserved;
- 69 noisy in-cutoff C3 tokens structurally removed and backfilled;
- tokenizer SHA-256:
  `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`;
- local canonical tokenizer artifact:
  `subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/variants/c3_added_17408_curated_padded/tokenizer.json`;
- shipped HF-style bundle:
  `subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_only_148480/`.

Key decision doc:

- [`../subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md)

Role in the public story: this is the main tokenizer artifact.

## 3. Optional Polytonic Extension

Stacked tokenizer bundle:

- base vocab: `131,072`;
- modern Greek extension: `+17,408`;
- polytonic/ancient Greek extension: `+5,120`;
- total vocab: `153,600 = 256 x 600`;
- tokenizer SHA-256:
  `b1eeb739a564b3abd33c1b85a16162b8284d98f9ab5d67528d3cbe8a82e9cbad`;
- shipped HF-style bundle:
  `subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_extended_153600/`.

Role in the public story: secondary tokenizer artifact, useful for future
polytonic specialization, not the checkpoint line that was trained in the 3.5B
continuation.

## 4. CPT Dataset

The CPT dataset is not just "nanochat as is." It is:

1. `fffoivos/glossapi-greek-nanochat-pretraining-dataset`;
2. hard drop against Apertus pretraining overlap using
   `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`;
3. replay of nanochat internal dedup with `drop_intra_and_inter`;
4. source-weighted mixture with replay, code, and math.

Important source repo:

- dataset repo:
  `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`;
- overlay file inside it:
  `artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet`.

Important Clariden paths:

- selected Greek pool:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet`;
- final NFC JSONL mix:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.nfc.jsonl`;
- production base-tokenized Megatron prefix:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document`.

Measured sizes:

- mix-builder budget: about `7,000,141,612` extended-tokenizer tokens;
- rows: `5,754,172`;
- base-tokenized Megatron tokens: `9,831,704,774`;
- JSONL input bytes read during base preprocessing: `27,125,565,083`;
- final JSONL on Clariden inventory: about `41 GB`;
- production Megatron `.bin`: about `37 GB`.

Recipe:

- Greek: `70%`;
- non-Greek replay: `24%`;
- code: `4%`;
- math: `2%`.

Key docs:

- [`../subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../subprojects/03_apertus_extension_and_embedding_adaptation/03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md)
- [`../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md`](../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/MIX_RECIPE.md)
- [`../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/bulk_mix_text_document.manifest.json`](../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/production_base_nfc_preprocess_2367579/bulk_mix_text_document.manifest.json)

Role in the public story: this explains how the training data was made and how
to recreate or hydrate it.

## 5. Initialization Arms

Initial tested arms:

- Vanilla: new rows initialized by the baseline extension path;
- ReTok: retokenization-based initialization;
- Centroid: centroid-based initialization;
- Token Distillation, layer 11: ReTok plus hidden-state distillation for new
  input rows and CE path for untied output rows.

Important Clariden init paths:

- Vanilla TP=2 R17-patched:
  `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/vanilla/megatron_tp2_r17patched/release/`;
- ReTok TP=2 R17-patched:
  `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/retok/megatron_tp2_r17patched/release/`;
- Centroid TP=2 R17-patched:
  `/iopsstor/scratch/cscs/fffoivos/init_checkpoints/modern_only_148480/centroid/megatron_tp2_r17patched/release/`;
- TD layer 11 TP=2 R17-patched:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/release/`.

Important TD artifacts:

- TD coverage prepass:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/coverage_2b_modern_20260523T032424Z_nfc/`;
- TD layer-pilot HF outputs:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z/{last,layer11}/`;
- chosen TD HF checkpoint:
  `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z/layer11/`.

Key docs:

- [`../subprojects/03_apertus_extension_and_embedding_adaptation/TOKEN_DISTILLATION_PLAN.md`](../subprojects/03_apertus_extension_and_embedding_adaptation/TOKEN_DISTILLATION_PLAN.md)
- [`../subprojects/03_apertus_extension_and_embedding_adaptation/ARTIFACTS_AND_HYDRATION.md`](../subprojects/03_apertus_extension_and_embedding_adaptation/ARTIFACTS_AND_HYDRATION.md)

Role in the public story: this is the experiment family that tests how to
initialize the new tokenizer rows.

## 6. Training Runs And Checkpoints

Completed 2B-token runs:

- Vanilla:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_vanilla/`;
- ReTok:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_retok/`;
- Centroid:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/bakeoff_1node_chain_20260522_005620_centroid/`;
- TD layer 11:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/td_full25_layer11_2b_20260523T165038Z/`.

Important 2B checkpoint:

- TD final 2B:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/td_full25_layer11_2b_20260523T165038Z/checkpoints/iter_0000476`.

Completed 3.5B continuation:

- run family:
  `continuation_3p5b_20260524T143012Z`;
- TD final 3.5B checkpoint:
  `/capstor/scratch/cscs/fffoivos/runs/bakeoff/continuation_3p5b_20260524T143012Z_td_layer11/checkpoints/iter_0000834`;
- corresponding eval copies:
  `/capstor/scratch/cscs/fffoivos/runs/eval/continuation_3p5b_20260524T143012Z_td_layer11/iter_0000{585,715,834}_hf`.

Role in the public story: the final selected checkpoint, plus enough baseline
checkpoint pointers to understand the comparison.

## 7. Evaluation Results

Main compact result doc:

- [`../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/CONTINUATION_3P5B_RESULTS_20260525.md`](../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/trajectory_analysis_20260524/CONTINUATION_3P5B_RESULTS_20260525.md)

At iter 834:

| Arm | Greek aggregate | English retention | Multilingual | Heldout BPC, lower better |
|---|---:|---:|---:|---:|
| Vanilla | 0.4339 | 0.6782 | 0.4923 | 0.4724 |
| ReTok | 0.4246 | 0.6786 | 0.4864 | 0.5390 |
| TD layer 11 | 0.4344 | 0.6865 | 0.4967 | 0.5054 |

Reading:

- TD layer 11 is the best final benchmark arm overall.
- Vanilla still has the best heldout Greek BPC.
- ReTok wins some Greek knowledge tasks but trails overall.

Role in the public story: this justifies why TD layer 11 should be promoted as
the main checkpoint candidate, while Vanilla remains a useful compression-loss
reference.

## 8. Current Hugging Face Repo State

Live model repo:

- `fffoivos/apertus-tokenizer-extension`;
- checked commit:
  `e58e61a2c129b4dbdab2f05cd6c6152e086945f5`;
- file count: `1064`;
- top-level entries:
  - `.gitattributes`;
  - `README.md`;
  - `SHA256SUMS`;
  - `analysis/`;
  - `artifacts/`;
  - `continuous/`;
  - `experiments/`;
  - `fresh/`;
  - `manifest.json`;
  - `metadata/`;
  - `subprojects/`;
  - `tokenizers/`.

Current remote contents by main bucket:

| Bucket | File count | Meaning |
|---|---:|---|
| `tokenizers/` | 4 | canonical modern tokenizer bundle only |
| `experiments/` | 33 | compact intrinsic/firing-count evidence |
| `subprojects/` | 587 | large mirror of local implementation docs/scripts |
| `analysis/` | 94 | older tokenizer-analysis outputs |
| `artifacts/` | 282 | language-attribution artifacts |
| `continuous/` | 18 | older continuous-tokenizer bundles |
| `fresh/` | 8 | older fresh-tokenizer bundles |
| `metadata/` | 34 | older metadata bundles |

Large checkpoint/dataset marker files found in the live repo: `0`.

Role in the public story: the HF repo has useful compact evidence, but it does
not currently put the main actors first and it does not contain checkpoint
payloads.

## 9. What Should Be Center Stage

These are the artifacts that should be first-class in the public repo:

1. Modern Greek tokenizer, 148,480 vocab.
2. Optional polytonic tokenizer, 153,600 vocab.
3. CPT dataset source graph and hydration recipe.
4. TD layer 11 final checkpoint, preferably 3.5B iter 834, either as uploaded
   HF payload or as a first-class hydration pointer.
5. Baseline checkpoint pointers for Vanilla and ReTok at comparable 3.5B
   checkpoints.
6. Compact benchmark table and plots.
7. Provenance for cutoff selection, dedup, TD coverage, conversion, and eval.

## 10. What Is Supporting Evidence

Keep these reviewable, but not on the front page:

- C3 cutoff sweep reports;
- firing-count attribution;
- Apertus overlap dedup audit;
- TD coverage and layer-pilot reports;
- R17/xIELU/QK-Norm roundtrip verification;
- evaluation JSON summaries;
- training commands and job metadata.

## 11. What Should Be Archived

Move these out of the main path:

- stale four-arm tokenizer discovery framing;
- old raw C1/C2/fresh tokenizer bundles;
- dry-run plans;
- failed attempt logs;
- full Slurm stdout/stderr dumps;
- old review packets and temporary watcher logs;
- broad `subprojects/` mirrors that make the repo feel like a scratch dump.
