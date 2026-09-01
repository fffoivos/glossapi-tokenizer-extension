# 09.3 — Added-token adaptation audit

This sub-subproject answers one question the trajectory analysis left open:

> The 8B CPT run added 17,920 vocabulary entries (17,408 modern Greek + 512
> polytonic, taking the vocabulary from 131,072 to 148,992). GreekMMLU peaks at
> update 9,536 and is worse at the 18,284 terminal. **Did the added tokens
> actually adapt, and does their adaptation explain that peak?**

The answer is **yes to the first, no to the second**. See [`RESULTS.md`](RESULTS.md).

## Ownership boundary

- `07_full_8b_cpt` remains authoritative for the training recipe, the checkpoints
  and the raw evaluation receipts.
- `03_apertus_extension_and_embedding_adaptation` remains authoritative for the
  tokenizer extension, the Token-Distillation initialization and the original
  D1–D7 new-token diagnostic suite (`compute_new_token_diagnostics.py`).
- This sub-subproject owns only the post-hoc conclusion about how those added
  tokens behave in the released checkpoints, and the compact code that produced it.

## Why a new measurement was needed

The existing D1–D7 suite was run **only on the 2B-token init bakeoff arms** at
vocabulary 148,480. It was never run on any full-8B checkpoint, so no artifact in
this repository answered the question. D6/D7 are also weak by construction:
[`docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`](../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md)
already establishes that Apertus's 0.1 gradient clip, Pre-Norm/RMSNorm, QK-Norm
and logit saturation force per-token norm parity regardless of language share —
so norm parity proves saturation, not that a merge is well targeted.

## What is measured

Three per-token tests, all **paired on identical text** and run on **held-out**
documents, so the comparison is both contamination-free and confound-free:

| test | statistic | what it detects | source |
| --- | --- | --- | --- |
| T1 merged-vs-split likelihood | `logP(added token \| ctx) − Σ logP(base pieces \| ctx)` | a token that is alive but not worth its vocabulary slot | tokenisation invariance (Cao & Rimell 2021; Chirkova et al. 2023) |
| T2 hidden-state agreement | `cos(h_L[merged], h_L[last base piece])` at layers 11 and 30 | whether the token occupies the same representational slot as the phrase it replaced | Token Distillation ([arXiv:2505.20133](https://arxiv.org/abs/2505.20133)); inner lexicon ([arXiv:2410.05864](https://arxiv.org/abs/2410.05864)) |
| T3 echo probe | rank and log-probability of the token after a repetition prompt | behaviourally dead tokens | Land & Bartolo, EMNLP 2024 ([arXiv:2405.05417](https://arxiv.org/abs/2405.05417)) |

Layer 11 is not arbitrary: it is the layer the production Token-Distillation
initialization was fitted at (`recipe_8b_full_mixed.json`,
`initialization.target_layer = 11`). Layer 30 tests whether the merged and split
paths reconverge downstream.

## Input

The audit reads the **held-out sets** built by the CPT data bridge, not the
training parquets:

```text
/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/cpt25b_midtraining/
  20260731T124000Z-cpt25b-v1/heldouts/sets/
```

Six Greek-relevant sets are used (`val_historical_polytonic`,
`val_forget_old_greek`, `val_greek_phd`, `val_openarchives`, `val_non_hplt`,
`val_hplt`). They are excluded from training, GreekMMLU-decontaminated, and
PII-masked upstream.

Document selection is **greedy supply-first**: a document is kept only if it
supplies an added token still below a 16-occurrence floor. A uniform sample would
have drowned the rare tail in head tokens; this reaches 96.3% of added tokens at
the floor from 9,559 documents.

## Canonical artifacts

| artifact | role |
| --- | --- |
| [`presentations/ADDED_TOKEN_ADAPTATION.data.json`](presentations/ADDED_TOKEN_ADAPTATION.data.json) | compact result payload |
| [`evidence/`](evidence/) | coverage, throughput, smoke and readiness receipts |
| [`evaluation/`](evaluation/) | the audit and corpus-build code, as executed |
| [`analysis/`](analysis/) | payload reduction and cross-checkpoint comparison |

The three raw per-token payloads (~16 MB each) stay on CSCS; the compact payload
carries their paths, sizes and SHA-256 digests under `raw_payload_pointers`.

## Reproduce

```bash
# 1. build the coverage-balanced corpus (debug partition, ~25 s)
sbatch evaluation/stageA_build_corpus.sbatch
# 2. score three checkpoints concurrently (normal partition, ~39 min)
sbatch evaluation/stageB_audit.sbatch
# 3. reduce and compare
python3 analysis/build_payload.py <raw_dir> presentations/ADDED_TOKEN_ADAPTATION.data.json
python3 analysis/compare_checkpoints.py <raw_dir>
python3 test_added_token_contract.py
```

Throughput is measured, not assumed: `evaluation/probe_throughput.sbatch` reports
3.19 docs/s for the slowest of three concurrent single-GPU tasks at batch 8, which
is what sized the 02:00:00 walltime. Batch 24 is **not** usable — the full fp32
log-softmax over a 148,992-token vocabulary needs 31.5 GiB at 3,072-token
sequences and OOMs a 96 GB GH200.
