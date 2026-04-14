# Global Decisions

These are the active high-level decisions already made.

## Goal

Extend `swiss-ai/Apertus-8B-2509` for Greek in a way that generalizes well.

This means:
- discovering reusable Greek subword units through true `BPE` training
- not using whole-word `add_tokens(...)` as the shipping method
- extending Apertus through `model.vocab` and `model.merges`

## Hard Constraints

- match Apertus tokenization behavior as exactly as possible
- preserve the fixed first `1000` ids
- preserve special-token behavior
- preserve the regex split plus `ByteLevel` regime
- final vocab size must remain divisible by `128`
- `tie_word_embeddings = false`, so embeddings and `lm_head` both matter

## Dataset Integration Constraints

- HPLT must be integrated into `fffoivos/glossapi-greek-nanochat-pretraining-dataset`
- the integration target is the upstream broad corpus stage under `data/*.parquet`
- HPLT must be uploaded in the existing canonical `21`-column source-parquet schema
- the published source parquets remain undeduplicated; deduplication is applied downstream through refreshed `dedup_metadata`
- HPLT-specific provenance should live in `source_metadata_json`
- `title` and `author` should only be promoted to top-level if they are credibly available
- the downstream builder should stay lightweight after HF download
- local tokenizer progress does not need to wait for HF upload once the filtered local source-parquet slice exists

## Corpus Constraints

- the training corpus must be diverse
- the training corpus must be deduplicated before any proper tokenizer training
- that cleaning/dedup work is owned by the upstream dataset pipeline, not by a separate tokenizer-project builder stage
- HPLT should be sampled with metadata awareness, not raw prefix sampling
- HPLT should currently be treated with a provisional `>=8` quality-bin filter
- after that metadata filter, HPLT must also run through real `corpus.clean`-compatible quality scoring before it is accepted as the tokenizer/CPT slice
- rows with `greek_badness_score > 60` must be excluded from the HPLT tokenizer/CPT slice
- HPLT documents labeled `Machine translated or generated` must be excluded from the final training dataset
- same-source overlap between GlossAPI and HPLT should be reduced before final freeze
- `openarchives.gr` rows with `needs_ocr == true` must remain excluded from the CPT-ready dataset used for tokenizer work

## Dedup Repair Constraint

- the dedup implementation may be changed for efficiency, storage layout, resumability, and parallelism
- dedup functionality must remain the same
- exact and near dedup decisions must remain semantically equivalent after the repair
- the repaired path must pass golden equivalence, resume equivalence, and downstream contract tests before it becomes the live default

## Operational Constraint

Use the existing dataset-build scripts as the operational path. Do not invent a second independent release builder when the current work can be expressed through the existing release pipeline and overlays.

## Upload Constraint

- dataset publication must run on a separate cheap uploader instance, not on the tokenizer worker
- that uploader track must stay independent of the tokenizer critical path
- the uploader payload must include:
  - the complete filtered HPLT source parquet slice
  - the refreshed `dedup_metadata` bundle so downstream builder-time dedup works
- the upload path should use the official Hugging Face large-folder upload mechanism, not an ad hoc custom uploader
- `publish_hf_release.py` is the intended upload entrypoint and should use the official large-folder upload strategy
- the upload instance should be configured for Xet-backed uploads when available

## Experimental Structure

- compare `GlossAPI-only` vs `GlossAPI + HPLT`
- use a discovery tokenizer vocab fixed at `50k` for the first discovery runs
- the `50k` stage is discovery only, not the final number of new Apertus tokens
- discovery must use true `BPE` learning, not word-frequency additions or `add_tokens(...)`
- preserve the Apertus front-end behavior during discovery: same normalization, same regex split, same byte-level regime
- after discovery, diff learned units against Apertus and drop units that should not be merged back as new tokenizer entries
- run analytic cutoffs in the `10k` to `25k` region on Apertus-compatible merged variants
- the current working cutoff grid is:
  - `10240`
  - `15360`
  - `20480`
  - `25600`
- fertility tests must be run on those merged Apertus-compatible variants, not on a raw standalone discovery tokenizer
- only snap the shipped build to a `128`-aligned size after the elbow is identified
- the divisibility rule applies to the whole final tokenizer, not just the newly added units
- tokenizer experiments should read from the same CPT-ready dataset used for continued pretraining
- the mixed `GlossAPI + HPLT` view should use a `70/30` split by training-token mass

## Execution Structure

There are now two parallel tracks:

Execution boundary:
- `home` is coordination-only for this project
- do not run tokenizer filtering, export, or training workloads on `home`
- operational tokenizer work should run on GCP workers and be stopped when done

1. Tokenizer critical path
- salvage and repair the current dedup run without changing dedup semantics
- freeze downstream manifests from the CPT-ready dataset
- freeze eval manifests
- lock the literal Apertus tokenizer-replication checklist
- export BPE-training text on the chosen worker
- train discovery tokenizers on the chosen worker
- diff learned units against Apertus
- assemble Apertus-compatible merged tokenizer variants
- run fertility tests at multiple cutoffs
- implement the final merge-rule extension

2. Dataset operational sidetrack
- finish uploading the filtered HPLT slice into the upstream HF dataset
- rerun the full prepared-source dataset view locally with HPLT included, using the existing release scripts
- keep the upload path and the local tokenizer path decoupled
- refresh published `dedup_metadata` later as a separate step once the intended dataset state is settled

## Open Decisions

- exact HPLT-to-canonical-schema field mapping inside `source_metadata_json`
- exact literal tokenizer replication checklist beyond the already confirmed settings
- new-row initialization method
- multilingual replay ratio during full continued pretraining
