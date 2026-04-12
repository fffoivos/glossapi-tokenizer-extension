# Current Status

## Active Phase

Parallel execution:
- HPLT filtering, canonical-parquet build, and HF upload
- tokenizer-spec freeze and local tokenizer-data preparation

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

- the filtered `HPLT__ell_Grek_ge8_no_mt` slice has been built locally into the canonical source-parquet tree on `home`
- the HF upload of that slice is still running in the background
- the canonical Apertus constraints and tokenizer-extension direction are documented
- the old `add_tokens(...)` baseline is retained only as diagnostic background

## What Is Not Done Yet

- no frozen final HPLT filtering spec beyond the current working defaults
- no frozen HPLT upload-schema mapping writeup at the field-by-field level
- no rerun of the full local prepared-source dataset with HPLT included as the finalized tokenizer/CPT input view
- no frozen local downstream manifests for `GlossAPI-only` vs `GlossAPI + HPLT`
- no frozen held-out eval manifests derived from the refreshed local upstream dataset
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
