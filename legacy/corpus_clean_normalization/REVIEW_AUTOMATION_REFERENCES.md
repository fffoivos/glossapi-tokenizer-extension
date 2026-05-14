# Review Automation References

This note preserves the main local and external references for the token-audit and review architecture. The goal is not to copy these projects directly. The goal is to borrow the right abstractions.

## Local GlossAPI References

### OCR/debug export contract

- [`ocr_render.py`](/home/foivos/glossAPI-development/src/glossapi/corpus/ocr_render.py)
  - `_build_match_index_rows(...)`
- [`phase_clean.py`](/home/foivos/glossAPI-development/src/glossapi/corpus/phase_clean.py)
  - `clean_ocr_debug(...)`
  - `clean_ocr_numeric_word_debug_docs(...)`
- [`tests/test_corpus_clean_enhancements.py`](/home/foivos/glossAPI-development/tests/test_corpus_clean_enhancements.py)

Relevant lessons:

- debug exports should have stable `match_id`s
- offsets should map to what the reviewer actually sees
- `manifest.jsonl`, `match_index.jsonl`, and `summary.json` are the right baseline contract

### Review materialization

- [`review_manifest_materialize.py`](/home/foivos/glossAPI-development/src/glossapi/scripts/review_manifest_materialize.py)

Relevant lesson:

- materialize review bundles by label/category rather than asking reviewers to search raw manifests manually

### Context review for block artifacts

- [`table_sentence_context_review.py`](/home/foivos/glossAPI-development/src/glossapi/scripts/table_sentence_context_review.py)

Relevant lessons:

- three-page or expanded local context is useful for block-level decisions
- review files should preserve both:
  - original context
  - hypothetical cleaned/replaced context

## External References

## 1. Snorkel / data programming / weak supervision

### Why relevant

We are not building a classifier-only system, but Snorkel is still useful conceptually for:

- heuristic labeling functions
- noisy signal aggregation
- selective human review on uncertain cases

### References

- Snorkel documentation homepage:
  - <https://docs.snorkel.ai/>
- Snorkel weak supervision guide:
  - <https://docs.snorkel.ai/docs/25.user-guide/weak-supervision/overview>
- Ratner et al., *Data Programming: Creating Large Training Sets, Quickly*:
  - <https://proceedings.neurips.cc/paper_files/paper/2016/file/6709e8d64a5f47269ed5cea9f625f7ab-Paper.pdf>

### What to borrow

- treat heuristics as first-class signals
- aggregate multiple weak signals instead of trusting one detector
- use reviewed failures to refine heuristics iteratively

## 2. Argilla / Distilabel

### Why relevant

Argilla and Distilabel are good references for:

- suggestion-first review
- LLM-generated annotations that humans later confirm
- durable datasets/manifests with searchable metadata

### References

- Argilla docs:
  - <https://docs.argilla.io/latest/>
- Distilabel docs:
  - <https://distilabel.argilla.io/latest/>
- Distilabel components gallery:
  - <https://distilabel.argilla.io/dev/components-gallery/>

### What to borrow

- model suggestions should be stored, not hidden
- reviewer-facing records need rich metadata and filters
- the review UI layer should be optional, not entangled with extraction

## 3. Prodigy-style review

### Why relevant

Prodigy is a useful reference for adjudication patterns:

- multiple annotation versions
- stable example hashes / IDs
- explicit review/merge step

### References

- Prodigy docs homepage:
  - <https://prodi.gy/docs/>
- Prodigy review docs:
  - <https://prodi.gy/docs/review>

### What to borrow

- keep heuristic, model, and human outputs as separate versions
- add an adjudicated final decision layer
- keep stable IDs across reruns

## 4. Gemini structured outputs and batch execution

### Why relevant

For the production review backend, the important needs are:

- strict JSON outputs
- batch execution over large manifests
- durable request/result artifacts

### References

- Gemini structured output docs:
  - <https://ai.google.dev/gemini-api/docs/structured-output>
- Gemini OpenAI compatibility docs, including Batch API:
  - <https://ai.google.dev/gemini-api/docs/openai>
- Gemini optimization docs:
  - <https://ai.google.dev/gemini-api/docs/optimization>

### What to borrow

- strict schema validation for each review task
- JSONL-based batch manifests
- batch execution for offline corpus review at scale

### Important implementation note

The official hosted path is clear today for Gemini API and Gemini Batch API.

The plan should not assume that a Google-hosted "Gemma 4" review backend is equally available or equally documented. If we want a robust first implementation, Gemini is the safer documented target.

## 5. Human-in-the-loop methodology

### Why relevant

This work is fundamentally HITL:

- machine finds candidates
- model proposes judgments
- humans adjudicate high-risk cases
- rules improve iteratively

### References

- Wu et al., *A Survey of Human-in-the-loop for Machine Learning*:
  - <https://arxiv.org/abs/2108.00941>

### What to borrow

- do not collapse discovery, judgment, and intervention into one step
- keep the intervention loop explicit
- optimize for low-cost human review on the ambiguous cases

## What Not To Copy Blindly

- Web-quality heuristics such as terminal-punctuation or short-line ratios are not automatically appropriate for PDF-extracted academic markdown.
- Generic LLM annotation demos often assume short independent examples, while our cases require line/block/document context.
- Not every suspicious token should become a rule. Review must decide:
  - extraction failure
  - formatting artifact
  - valid but rare content

## First-Pass Recommended Stack

Recommended stack for this project:

1. Rust matcher plus local manifests
2. structured review prompts with strict JSON schema
3. Gemini Batch for large review sets
4. local materialized review bundles for high-risk adjudication
5. optional Argilla only if we need a heavier annotation UI

This gives us:

- speed in detection,
- scale in structured review,
- safety through human escalation,
- and a durable audit trail for each rule promotion decision.
