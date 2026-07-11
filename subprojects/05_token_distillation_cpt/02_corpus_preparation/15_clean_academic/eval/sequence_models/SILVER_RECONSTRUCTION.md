# Deterministic silver reconstruction

No human-gold corpus exists for this track. All tracked annotation decisions are model/LLM-produced
and every converted row is stamped `annotation.status=LLM_silver`. Nothing here can authorize a Rust
production change.

## What is usable

The committed audit (`silver_inventory.json`) covers all 17 annotation JSON files, all 16 top-level
unit/manifest files, nine feature/model artifacts, and the ten scripts that produced or consumed the
structural supervision. Its immutable inventory hash is
`e453b13d4d2ff4c1f1912d15ec61bcf3ce4932f6ccc74fc0aa54b610b81c05b7`.

- SPAN: 3,339 joined Opus decisions over 1,738 documents and 3,186 positive bibliography spans.
  Joined windows/documents are Greek PhD 1,596/639, OpenArchives 971/558, and Kallipos 772/541. These are BIB-only window
  annotations; they contain no ToC supervision and no independent running-prose judgment.
- Section-scale: 1,985 joined decisions (Kallipos 524, Pergamos 1,461), but only row identities and
  evidence quotes are tracked. They cannot be joined to work/document identity or reconstructed as
  complete ordered line sequences.
- Boundary and goal-review annotations remain diagnostics. They do not constitute line targets.
- The frozen LR/smoothing JSON files contain fitted coefficients, not recoverable raw supervision.

Today the fit-ready count is therefore **0 line rows**. The precise missing dependency for the SPAN
conversion is all 240 files named by the tracked `units/SPAN_batchpaths.json`: `batch_0000.json` through
`batch_0239.json`. Their ordered-name digest is
`913416375b3743ab86eeb4f23ad49487edb783bbde8545788627398b4ad6d481`.
The one absent annotation is `S00772`; it is omitted, never inferred.

The missing `units/STRUCT_2K_gold.jsonl` (or its original unit plus annotation files) is the exact
dependency for reconstructing joint BIB/ToC silver. It is absent from this checkout and Git history.

## Recovery and conversion

Copying existing artifacts from the user's home/archive is the only permitted recovery action. Do not
regenerate annotations. On a Clariden CPU node, first rerun the audit, then hydrate the BIB-only data:

```bash
cd subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval
python3 -m sequence_models.silver_reconstruct audit --output "$SCRATCH/silver-audit.json"
python3 -m sequence_models.silver_reconstruct hydrate-span \
  --unit-dir "$HOME/recovered/SPAN" \
  --tokenizer-json "$HOME/tokenizers/ModernGreek-148k/tokenizer.json" \
  --output "$SCRATCH/span.LLM_silver.jsonl" \
  --split-manifest "$SCRATCH/span.LLM_silver.split.json" \
  --receipt "$SCRATCH/span.LLM_silver.receipt.json"
```

Hydration refuses partial/extra batch inventories, conflicting overlap text, tokenizer hash drift,
identity collisions, exact-text or work leakage, output overwrite, and split-manifest drift. It merges
overlapping windows, preserves absolute physical-line coordinates, pins ModernGreek-148k token counts,
and emits atomic outputs. Source-balanced allocation is deterministic and label-blind.

If an existing `STRUCT_2K_gold.jsonl` is recovered, the minimal import hook is:

```bash
python3 -m sequence_models.silver_reconstruct import-legacy \
  --input "$HOME/recovered/STRUCT_2K_gold.jsonl" \
  --tokenizer-json "$HOME/tokenizers/ModernGreek-148k/tokenizer.json" \
  --output "$SCRATCH/struct2k.LLM_silver.jsonl" \
  --split-manifest "$SCRATCH/struct2k.LLM_silver.split.json" \
  --receipt "$SCRATCH/struct2k.LLM_silver.receipt.json"
```

This uses a metadata-only first pass, then streams one document at a time through 512-line tokenization
batches, contract validation, and an atomic temporary JSONL. Memory is bounded by the identity ledger
plus the largest document. It rebuilds the split instead of trusting the legacy locked test and records
the source file hash. It does not infer an annotator or adjudication history.

## Research ladder and safety

C0 replays the frozen LR plus hysteresis baseline. C1/C2 may fit BIB-only SPAN silver with
`--evidence-tier LLM_silver --target bib`; their CRF masks make ToC emission/transition paths impossible.
N1 accepts `target_classes=("BIB",)` for the same reason. Joint BIB/ToC comparison requires recovered
STRUCT silver. No fitting is permitted locally or on GPU, and this work does not submit a job.

Silver reports retain line/token/span/document agreement and work-clustered bootstrap intervals, but
null safety metrics needing independent prose labels. Promotion status is always `no_op`. After a
candidate is frozen, an optional small high-risk packet can be produced with
`silver_reconstruct false-deletion-packet`; a person must review it manually and the tool never
auto-adjudicates. The packet reuses the existing `failure_analysis.py`/`span_signals.py` review
taxonomy rather than forking it. Even a clean packet does not substitute for the configured
conservative gates.
