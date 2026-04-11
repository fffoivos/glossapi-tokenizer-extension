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
- HPLT must be uploaded in the existing canonical 21-column source-parquet schema
- HPLT-specific provenance should live in `source_metadata_json`
- `title` and `author` should only be promoted to top-level if they are credibly available
- the downstream builder should stay lightweight after HF download

## Corpus Constraints

- the training corpus must be diverse
- the training corpus must be deduplicated before any proper tokenizer training
- that cleaning/dedup work is owned by the upstream dataset pipeline, not by a separate tokenizer-project builder stage
- HPLT should be sampled with metadata awareness, not raw prefix sampling
- HPLT should currently be treated with a provisional `>=8` quality-bin filter
- HPLT documents labeled `Machine translated or generated` must be excluded from the final training dataset
- same-source overlap between GlossAPI and HPLT should be reduced before final freeze

## Experimental Structure

- compare `GlossAPI-only` vs `GlossAPI + HPLT`
- use a discovery tokenizer vocab around `40k-50k`
- run analytic cutoffs around `5k`, `10k`, `15k`, `20k`
- only snap the shipped build to a `128`-aligned size after the elbow is identified
- tokenizer experiments should read from the same CPT-ready dataset used for continued pretraining

## Open Decisions

- exact HPLT mixing ratio
- exact HPLT-to-canonical-schema field mapping inside `source_metadata_json`
- exact literal tokenizer replication checklist beyond the already confirmed settings
- new-row initialization method
- multilingual replay ratio during full continued pretraining
