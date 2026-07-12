# Receipt-bound joint ToC+BIB sequence-model ladder

This is the comparison-only path for the recovered `STRUCT_2K` corpus:

- C0: frozen ToC and BIB LR+hysteresis heads;
- C1: engineered-feature joint BIOES CRF;
- C2: C1 plus stable character n-grams;
- N1: byte CNN, dilated line TCN, and masked joint BIOES CRF.

No label in this corpus is human gold. The surviving annotations were produced by GPT-5.5 at medium
effort and remain `LLM_silver`. The legacy `STRUCT_2K_gold.jsonl` filename is historical only. Metrics
measure retrospective agreement with that silver and cannot authorize production cleaning.

## Frozen handoff and correction policy

`struct2k_handoff_lock.json` pins the recovered handoff, including:

- the 4,273-entry `INVENTORY.sha256` and its own SHA-256;
- the exact raw joint JSONL, manifest, and sparse line-matrix hashes;
- 2,000 documents, 1,683,369 present lines, source counts, label counts, and historical split counts;
- the original source commit and annotation engine;
- eight explicit coordinate-typo corrections across seven documents.

The importer rehashes the complete inventory and replays every batch plus annotation into line labels.
It fails on any invalid span that is not exactly one of the eight pinned corrections. The corrections
repair malformed coordinates; they do not add or reinterpret semantic labels.

## Historical-test isolation

The historical split contains 1,392 `train` and 608 `test` documents. The old feature work already
used this corpus, so the 608 documents are not claimed as a new unbiased test. They are nevertheless
sealed from this experiment:

1. `struct2k_import` opens the handoff only to authenticate and replay it.
2. It writes rows only for the 1,392 historical-train documents.
3. It derives a deterministic source-stratified 80/20 train/validation split from those documents.
4. Its receipt records 608 historical-test documents excluded and zero historical-test rows emitted.
5. `bib_ladder prepare-selection` revalidates that receipt and creates the only views accepted by C0,
   C1, C2, N1, calibration, evaluation, and finalization.

The materialized source has no `test` row at all. The runner has no historical-test prediction path.
Exact-text and work identities remain grouped when the new split is derived.

C0 is a descriptive in-sample reference: both frozen C0 heads were fit on historical train, and the
new validation set is a subset of historical train. Calibration therefore maximizes joint action
recall at or above C0's descriptive silver action precision; no result is reported as a held-out gain.

## 1. Import on a Clariden CPU node

Transfer the complete handoff directory to Clariden without modifying its contents. Submit from an
exact, clean checkout:

```bash
sbatch --export=ALL,\
REPO_ROOT=/path/to/exact/classifier/worktree,\
PHASE04_CLARIDEN_DIR=/path/to/exact/classifier/worktree/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/clariden,\
PHASE04_EXPECTED_COMMIT=<40-character-commit>,\
STRUCT2K_HANDOFF_ROOT=<transferred-APERTUS_CLASSIFIER_HANDOFF_20260712>,\
STRUCT2K_IMPORT_RUN_ID=<operator-chosen-id>,\
CONFIRM_STRUCT2K_IMPORT=1 \
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/clariden/import_struct2k_joint.sbatch
```

Omit `CONFIRM_STRUCT2K_IMPORT=1` for the in-allocation plan-only check. The successful immutable output
contains:

- `struct2k.LLM_silver.jsonl`;
- `struct2k.LLM_silver.split.json`;
- `struct2k.handoff.audit.receipt.json`;
- `struct2k.LLM_silver.receipt.json`.

The script deep-verifies both its staging directory and its no-replace published directory.

## 2. Profile N1 on a Clariden CPU node

N1 requires a passed joint profile before the full ladder can run:

```bash
sbatch --export=ALL,\
REPO_ROOT=/path/to/exact/classifier/worktree,\
PHASE04_CLARIDEN_DIR=/path/to/exact/classifier/worktree/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/clariden,\
PHASE04_EXPECTED_COMMIT=<40-character-commit>,\
STRUCT2K_SOURCE_ROOT=<immutable-output-from-step-1>,\
N1_JOINT_PROFILE_RUN_ID=<operator-chosen-id>,\
CONFIRM_N1_JOINT_PROFILE=1 \
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/clariden/profile_joint_n1.sbatch
```

The profile performs a two-replica deterministic tiny-fit smoke and one full epoch. It passes only if
the projected eight-epoch fit plus 15% margin is at most nine hours, reserving three hours of the
12-hour allocation for the other arms and final verification. A failed time gate means the full joint
runner must not be submitted unchanged; N1 has no resume path.

## 3. Run the full joint comparison

Use the exact passed receipt from step 2:

```bash
sbatch --export=ALL,\
REPO_ROOT=/path/to/exact/classifier/worktree,\
PHASE04_CLARIDEN_DIR=/path/to/exact/classifier/worktree/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/clariden,\
PHASE04_EXPECTED_COMMIT=<40-character-commit>,\
STRUCT2K_SOURCE_ROOT=<immutable-output-from-step-1>,\
N1_JOINT_PROFILE_RECEIPT=<passed-receipt-from-step-2>,\
CLASSIFIER_JOINT_RUN_ID=<operator-chosen-id>,\
CONFIRM_JOINT_CLASSIFIER_COMPARISON=1 \
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/clariden/run_joint_ladder.sbatch
```

All three scripts request a normal Clariden CPU node, hide accelerators, require the pinned uenv and
exact clean commit, default to a dry-run, lock run IDs, stage into job-specific partial directories,
and publish with no-replace semantics. They are detached from the corpus-production DAG and were not
submitted as part of this implementation.

`run.receipt.json` binds the source receipt, exact selection, config, code commit, uenv/runtime,
profile, models, calibration decisions, validation predictions, metrics, and resource measurements.
The finalizer reproduces the predictions from the bound model artifacts and rejects extra files.

## 4. Record a deployable C0 choice and run fresh Rust parity

The ladder does not automatically choose a winner. If the operator selects C0,
run the bridge against an exact detector build:

```bash
sbatch --export=ALL,\
REPO_ROOT=/path/to/exact/classifier/worktree,\
PHASE04_CLARIDEN_DIR=/path/to/exact/classifier/worktree/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/clariden,\
PHASE04_EXPECTED_COMMIT=<40-character-commit>,\
STRUCT2K_SOURCE_ROOT=<immutable-output-from-step-1>,\
JOINT_LADDER_RUN_ROOT=<immutable-output-from-step-3>,\
REFERENCE_BIN=<exact-detector-build>/reference_detect,\
DETECTOR_BUILD_RECEIPT=<exact-detector-build>/build_receipt.json,\
STRUCTURAL_C0_BRIDGE_RUN_ID=<operator-chosen-id>,\
STRUCTURAL_SELECTION_RATIONALE=C0_is_the_only_arm_with_an_existing_Rust_implementation,\
CONFIRM_STRUCTURAL_C0_BRIDGE=1 \
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/clariden/build_joint_c0_bridge.sbatch
```

This emits a classifier-selection receipt and a fresh Python↔Rust parity
receipt over the imported validation partition. Runtime parity checks
implementation equivalence; that validation partition is derived from
historical train and is not independent quality evidence. Both receipts are
required by the official Stage52/54 path.

C1, C2 and N1 are Python research arms, not Rust deployment packages. The
bridge rejects them. Selecting one requires a separate reviewed Rust
port/export, artifact contract and exact probability/span parity package before
Stage52; a `.npz` or `.pt` checkpoint must never be presented as deployable
Rust evidence.

## Interpretation and production boundary

The recovered corpus contains 1,289 whole documents and 711 front/tail or other historical windows.
It is useful for comparing structured line models but does not independently measure running-prose
safety. Prose contamination, true main-text retention, and catastrophic prose-deletion metrics remain
unavailable. A later production candidate still requires the explicit post-ladder selection, a frozen
cleaning policy, Stage54, runtime parity and resource receipts, all configured deployment gates, and the separate receipt-bound
100-case high-risk deletion review (50 ToC and 50 BIB, zero catastrophic deletions). Until then the
production fallback remains no-op.
