# GlossAPI Tokenizer Extension

Clean workspace for extending `swiss-ai/Apertus-8B-2509` for Greek.

## Active stage

**Tokenizer arm has converged to C3.** Open decision: the C3 cutoff
from the frozen grid `{10240, 15360, 20480, 25600}`. New agents should
start at:

- [docs/C3_CONVERGENCE.md](docs/C3_CONVERGENCE.md) — what's settled,
  what's open, where the artifacts live
- [docs/GLOBAL_DECISIONS.md](docs/GLOBAL_DECISIONS.md) — hard constraints
- [docs/ACTIVE_BACKLOG.md](docs/ACTIVE_BACKLOG.md) — the cutoff-decision
  work list

The four-arm exploration (`F1`, `F2`, `C1`, `C2`) is closed. Material
that still describes it is retained for traceability but should not
drive new execution.

## Method

- continuous BPE extension of Apertus on a `GlossAPI + HPLT` `50 / 50`
  mix (= the C3 arm), preserving Apertus front-end behavior exactly
- extend Apertus through `model.vocab` and `model.merges`, not
  `add_tokens(...)`
- consume the same CPT-ready dataset for tokenizer experiments and
  continued pretraining
- the old whole-word `add_tokens(...)` sweep is retained only as a
  legacy baseline and has been moved out of the active planning path

## Canonical Code Root

This repo is now the canonical source for the active tokenizer pipeline code.

That includes:
- `glossapi_corpus_cli/`
- stage orchestration scripts under `subprojects/`
- upload handoff scripts under `ops/upload/`
- smoke and efficiency harnesses under `ops/`
- the active test suite under `tests/`

The workspace copy under `/home/foivos/data/glossapi_work/` is no longer the development source of truth for pipeline code. It remains a data/workspace root and deployment target only.

## Test Matrix

The active repo-local verification matrix includes:
- script proof tests
- stage-to-stage contract tests
- resumability regressions
- tiny real-document end-to-end smoke runs
- efficiency smokes for streaming mix build and near-candidate execution

## Execution Shape

Two parallel tracks:
- tokenizer critical path: lock the eval manifests, build C3 merged
  variants at the four cutoffs, run the intrinsic + fertility bundle,
  pick the cutoff at the elbow, then implement the merge-rule extension
  in `subprojects/02_2_tokenizer_implementation` and hand off to
  `subprojects/03_apertus_extension_and_embedding_adaptation`
- dataset operational sidetrack: HF upload of the upstream dataset from
  a separate cheap uploader instance using the official large-folder
  upload path

The tokenizer critical path does not need to wait for the HF upload —
the C3 tokenizer already exists on the gcloud worker.

Earlier critical-path stages (dedup repair, HPLT filter rebuild, mix
build, four-arm training, four-arm comparison) are settled. The dedup
recovery and scale plans are retained for traceability:
- [PIPELINE_RECOVERY_AND_SCALE_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PIPELINE_RECOVERY_AND_SCALE_PLAN.md)
- [DEDUP_SCRIPT_REPAIR_PLAN.md](/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/_archive/01_1_corpus_dedup/DEDUP_SCRIPT_REPAIR_PLAN.md)

## Canonical Files

- **C3 convergence** (read first):
  - [C3_CONVERGENCE.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/C3_CONVERGENCE.md)
- project index:
  - [PROJECT_INDEX.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/PROJECT_INDEX.md)
- global decisions:
  - [GLOBAL_DECISIONS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/GLOBAL_DECISIONS.md)
- current status:
  - [CURRENT_STATUS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/CURRENT_STATUS.md)
- active backlog:
  - [ACTIVE_BACKLOG.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/ACTIVE_BACKLOG.md)
- machine-readable config:
  - [apertus_greek_extension.yaml](/home/foivos/Projects/glossapi-tokenizer-extension/config/apertus_greek_extension.yaml)

## Subprojects

Live:
- [02_apertus_tokenizer_spec](subprojects/02_apertus_tokenizer_spec/README.md)
- [02_1_tokenizer_experiments](subprojects/02_1_tokenizer_experiments/README.md) — **active** (C3 cutoff sweep)
- [02_2_tokenizer_implementation](subprojects/02_2_tokenizer_implementation/README.md) — gated on cutoff
- [03_apertus_extension_and_embedding_adaptation](subprojects/03_apertus_extension_and_embedding_adaptation/README.md) — gated on tokenizer freeze

Archived (DONE for the C3 shipping path) — see [subprojects/_archive/README.md](subprojects/_archive/README.md):
- `01_hplt_filtering`, `01_1_corpus_dedup`, `01_2_training_dataset_mix`, `01_0_cleaning_iteration_and_thresholds`

## Repo Policy

- `artifacts/` and local virtualenvs are ignored by git
- legacy baseline material is preserved under `legacy/`
- unresolved research decisions should be frozen only after explicit user confirmation
