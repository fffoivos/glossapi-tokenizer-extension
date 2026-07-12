# Receipt-bound BIB sequence-model ladder

This is the operational path for the comparison-only SPAN experiment:

- C0: frozen LR plus hysteresis parity baseline;
- C1: engineered-feature BIOES CRF;
- C2: C1 plus stable character n-grams;
- N1: byte CNN, dilated line TCN, and masked BIOES CRF.

**There is no human-gold dataset and no human annotation campaign is required or planned.** The
available labels are existing LLM-silver BIB judgments. Every metric below is retrospective replay
agreement with that silver, never human-gold accuracy or a production-safety estimate. If a later
independent estimate is useful, it can use newly sampled LLM-silver documents; nobody needs to label
2,000 lines by hand.

Because SPAN supplies no ToC judgments, the runner compares the frozen C0 BIB head itself and does
not score its unrelated ToC-head outputs as false positives. C1,
C2, and N1 likewise hard-disable ToC tags, and the report marks all ToC metrics unavailable.

Before this path was added, N1 was only a forward/export scaffold: it had no optimizer, checkpoint,
calibration, prediction CLI, or run receipt. C1/C2 had standalone fitting entry points, but no runner
bound them to a rehydration receipt.

## Sealed retrospective partition

The historical split named `test` is not an unbiased never-seen test: earlier model/feature work used
this corpus, and C0's STRUCT-2K training overlap with SPAN cannot be excluded. We therefore call it a
sealed retrospective comparison partition, not a final holdout. `bib_ladder prepare-selection` is the
sole gate allowed to open the complete rehydrated SPAN artifact. It recomputes the exact split from
current identities and the bound config, revalidates the source-unit receipt within the immutable
hydration root, and emits two immutable views:

- `selection.train-validation.jsonl`, used for fitting and validation calibration;
- `selection.validation.jsonl`, used for C0 and the four-arm validation report.

No historically named test row is copied. C0, C1, C2, N1, and the report process receive only those
views. The finalizer rejects extra files, malformed checkpoints, receipt drift, predictions that do
not reproduce from their model, and an accidentally produced historical-partition prediction.

C0 is explicitly reference-only and descriptive. All candidates emit the complete deletion-bias grid
and precision/recall frontier, then use the same rule: maximize BIB recall on the frontier at or above
C0's descriptive LLM-silver action precision. No reported difference is called a fair held-out gain.
The report does not choose a winner automatically and leaves production at no-op.

## N1 entry points

N1 fitting is deliberately gated to an aarch64 Clariden Slurm allocation with accelerators hidden:

First run `clariden/profile_n1.sbatch`. It performs a two-replica deterministic tiny-fit smoke and one
full training epoch, then refuses a full fit if the eight-epoch projection plus 15% margin exceeds 9
hours. This reserves three hours for C0/C1/C2 and final verification. The profile is mandatory because
Clariden `normal` is capped at 12 hours and N1 has no resume
path. No fitting smoke is run on the MacBook.

```bash
python3 -m sequence_models.char_tcn_crf train \
  --selection-silver selection.train-validation.jsonl \
  --selection-manifest selection.train-validation.split.json \
  --validation-silver selection.validation.jsonl \
  --selection-receipt selection.receipt.json \
  --config sequence_models/config.json \
  --reference-predictions c0.validation.predictions.jsonl \
  --profile-receipt n1.profile.receipt.json \
  --model-out n1.model.pt \
  --validation-predictions n1.validation.predictions.jsonl \
  --receipt-out n1.training.receipt.json \
  --uenv pytorch/v2.9.1:v2 \
  --code-commit <40-character-commit> \
  --confirm-clariden-cpu-only
```

The separate `predict` subcommand validates the current config, feature metadata, training receipt,
and all selection hashes. It accepts only emitted train or validation views.

## Clariden runner

Use `clariden/run_bib_ladder.sbatch` from an exact, clean classifier checkout. It is detached from the
full-corpus submission DAG and requests no GPU. Required exports are:

```bash
sbatch --export=ALL,\
REPO_ROOT=/path/to/exact/classifier/worktree,\
PHASE04_CLARIDEN_DIR=/path/to/exact/classifier/worktree/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/clariden,\
PHASE04_EXPECTED_COMMIT=<40-character-commit>,\
SPAN_REHYDRATION_ROOT=<immutable-output-from-07>,\
N1_PROFILE_RECEIPT=<passed-one-epoch-profile-receipt>,\
CLASSIFIER_COMPARISON_RUN_ID=<operator-chosen-id>,\
CONFIRM_CLASSIFIER_COMPARISON=1 \
subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval/sequence_models/clariden/run_bib_ladder.sbatch
```

Omit `CONFIRM_CLASSIFIER_COMPARISON=1` for the in-allocation plan-only check. A successful run publishes
the staged directory with no-replace semantics and verifies the published tree. `run.receipt.json`
binds the rehydration receipt, source-unit receipt, exact recomputed split/config, uenv and runtime
versions/thread settings, per-arm wall/RSS, all real checkpoints/predictions, and the retrospective
report. Resource thresholds in `config.json` are promotion-only; this replay measures resources but
does not apply those thresholds or make a promotion claim.

The recovered joint ToC+BIB corpus uses a separate import, profile, and comparison path documented in
`JOINT_LADDER_RUNBOOK.md`; this SPAN runner remains intentionally BIB-only.
