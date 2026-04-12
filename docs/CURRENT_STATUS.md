# Current Status

## Active Phase

Parallel execution:
- tokenizer-spec freeze and GCP worker setup for the corrected HPLT rebuild
- tokenizer-data preparation from the corrected CPT-ready dataset once that slice exists

## What Is Settled

- the shipping method is merge-rule tokenizer extension, not whole-word `add_tokens(...)`
- Apertus compatibility is a hard constraint
- machine-translated HPLT content is excluded from the final training dataset
- `openarchives.gr` rows with `needs_ocr == true` must stay excluded from the CPT-ready dataset used for tokenizer work
- HPLT is being prepared for the upstream HF corpus dataset, not as a separate tokenizer-only corpus
- the downstream CPT/tokenizer builder is expected to stay lightweight after HF download
- the first discovery tokenizer runs are locked to `50k` vocab
- the mixed `GlossAPI + HPLT` tokenizer view is locked to `70/30` by training-token mass
- local tokenizer progress does not need to wait for the HF upload to finish once the filtered HPLT parquet slice exists locally
- the workspace has now been split into smaller subprojects

## What Exists Now

- the previously built `HPLT__ell_Grek_ge8_no_mt` slice is no longer sufficient for tokenizer/CPT use by itself
- the corrected HPLT slice must add a real `corpus.clean` pass and drop rows with `greek_badness_score > 60`
- the old score-only HF upload attempt has been stopped and should be treated as invalid
- the canonical Apertus constraints and tokenizer-extension direction are documented
- the old `add_tokens(...)` baseline is retained only as diagnostic background

## What Is Not Done Yet

- no frozen final HPLT filtering spec beyond the current working defaults
- no frozen HPLT upload-schema mapping writeup at the field-by-field level
- no rerun of the full prepared-source dataset with the corrected HPLT slice included as the finalized tokenizer/CPT input view
- no frozen downstream manifests for `GlossAPI-only` vs `GlossAPI + HPLT`
- no frozen held-out eval manifests derived from the refreshed upstream dataset
- no true Greek `BPE` discovery tokenizer
- no implemented merge-rule extension
- no model adaptation plan beyond high-level constraints

## Current Trust Boundary

Active planning and execution should use:
- [GLOBAL_DECISIONS.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/GLOBAL_DECISIONS.md)
- [ACTIVE_BACKLOG.md](/home/foivos/Projects/glossapi-tokenizer-extension/docs/ACTIVE_BACKLOG.md)
- the relevant subproject folders

Legacy baseline and exploratory material has been moved under:
- [legacy/](/home/foivos/Projects/glossapi-tokenizer-extension/legacy/README.md)

Execution note:
- `home` should not be used as a tokenizer worker
- tokenizer filtering/export/training workloads should run on GCP workers only
- GCP workers should be sized minimally for the step and stopped when done
