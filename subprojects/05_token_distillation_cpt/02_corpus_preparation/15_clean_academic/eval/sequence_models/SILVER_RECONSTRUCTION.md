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

The already-fitted two-head LR baseline and its evaluation outputs remain usable. The count of raw
line rows that can be loaded for a *new fit directly from this checkout* is **0**. The precise missing
dependency for converting the committed SPAN decisions into new fit rows is all 240 files named by the
tracked `units/SPAN_batchpaths.json`: `batch_0000.json` through
`batch_0239.json`. Their ordered-name digest is
`913416375b3743ab86eeb4f23ad49487edb783bbde8545788627398b4ad6d481`.
The one absent annotation is `S00772`; no decision is inferred. Its sampled text is retained as
negative-only (`O`) exactly as the historical `span_seq_data.load()` path did, and that assumption is
called out in the reconstruction receipt.

The 2,000-document joint BIB/ToC LLM annotation run was completed and produced the surviving fitted
models. Its ignored intermediate export, historically misnamed `units/STRUCT_2K_gold.jsonl`, and the
original unit/annotation directory were not committed. Those raw intermediates are absent from this
checkout, Git history, Clariden search and authenticated Hugging Face search; no replacement annotation
run is proposed. The 26 Phase-04 source directories are now staged on Clariden,
but the classifier job must wait for the fresh passed `ACQUISITION_EXISTING_ONLY=1` receipt; failed job
`2735391` is not acceptable evidence.

## Receipt-bound text recovery and conversion

Do not regenerate annotations. The missing SPAN batch files contain text prompts, while their annotation
decisions are committed, so the SPAN text can be rehydrated deterministically from exact source snapshots
without making a new annotation decision. This does not recreate the separate raw 2,000-document joint
annotation export. The recovery tool accepts only explicitly listed, SHA-256-checked artifacts at
immutable repository revisions:

- Greek PhD and OpenArchives: document Parquet or `.jsonl[.zst]`, selected by the route-bound ID;
- Greek PhD receipt-bound document Parquet, with explicit document-ID and text-precedence columns;
- Kallipos: section Parquet, grouped by `filename`, ordered by `id`, and joined with the original
  two-newline separator.

The source-artifact receipt has this shape (every placeholder must be replaced):

```json
{
  "schema_version": "span-source-artifacts-v1",
  "sources": {
    "greek_phd": {
      "repo_type": "dataset",
      "repo_id": "OWNER/REPOSITORY",
      "revision": "40_HEX_COMMIT",
      "format": "jsonl_documents",
      "fields": {"document_id": "doc_id", "text_precedence": ["text", "document", "content"]},
      "artifacts": [{"path": "/scratch/source.jsonl.zst", "repository_path": "data/source.jsonl.zst", "sha256": "64_HEX_SHA256"}]
    },
    "openarchives": {
      "repo_type": "dataset",
      "repo_id": "OWNER/REPOSITORY",
      "revision": "40_HEX_COMMIT",
      "format": "jsonl_documents",
      "fields": {"document_id": "doc_id", "text_precedence": ["text", "document", "content"]},
      "artifacts": [{"path": "/scratch/source.jsonl.zst", "repository_path": "data/source.jsonl.zst", "sha256": "64_HEX_SHA256"}]
    },
    "kallipos": {
      "repo_type": "dataset",
      "repo_id": "OWNER/REPOSITORY",
      "revision": "40_HEX_COMMIT",
      "format": "parquet_sections",
      "fields": {"filename": "filename", "order": "id", "section": "section"},
      "artifacts": [{"path": "/scratch/Dataset_Kallipos.parquet", "repository_path": "Dataset_Kallipos.parquet", "sha256": "64_HEX_SHA256"}]
    }
  }
}
```

### Derive the receipt from Phase-04 acquisition

Do not hand-copy Clariden paths or hashes when a passed Phase-04 acquisition receipt exists. The
builder cross-checks that receipt against its exact source lock and tracked `sources.json`, selects only
the requested repository paths, verifies stat identity and LFS SHA-256 bindings, and inspects the
required Parquet/JSONL fields:

```bash
python3 -m sequence_models.build_span_source_receipt \
  --acquisition-receipt "$ACQUISITION_RECEIPT" \
  --source-lock "$SOURCE_LOCK" \
  --sources-config "$SOURCE_CONFIG" \
  --greek-phd-route nanochat_base \
  --openarchives-route nanochat_base \
  --kallipos-route kallipos_sections \
  --output "$SCRATCH/span-source-artifacts.json"
```

All three source route choices are mandatory; the builder never prefers a newer artifact implicitly.

- `--greek-phd-route nanochat_base` selects exactly
  `data/greek_phd.part-00000.parquet` and `part-00001.parquet`, with
  `source_doc_id` → `text` and `source_dataset=greek_phd`. Their hash-shaped ID domain is compatible
  with the SPAN manifest, but the text is a processed Nanochat representation rather than the raw
  Mozilla JSONL used during annotation.
- `--greek-phd-route greek_phd_v2` selects the standalone v2 Parquet. It is not a silent upgrade:
  the tracked registry exposes URL/DOI identifiers rather than the historical hash `doc_id`. The route
  therefore requires an explicit `--greek-phd-document-id-column` and
  `--allow-unverified-greek-phd-id-domain`; without a proven mapping, rehydration is expected to fail
  the exact manifest-document join.
- `--greek-phd-route mdc_raw_forensic` is the only supported route for the quarantined current MDC v1
  archive. It requires the quarantine receipt, the v2 coordinate/projection audit, the exact observed
  archive SHA-256, and `--allow-quarantined-mdc-comparison-only`. The builder recomputes the archive,
  safe-extraction manifest and selected-shard hashes; cross-binds the quarantine-v2, immutable safe
  extraction and audit receipts to the tracked manifest and annotations; and requires source-coordinate
  integrity to pass. Every forensic audit freshly extracts the archive with a Python extractor that
  rejects traversal, duplicates, links, devices and FIFOs, then compares that fresh tree to the retained
  extraction. The publisher checksum mismatch is
  preserved in every downstream receipt. This route can produce comparison evidence only and must not
  be represented by a hand-authored generic source receipt.

The forensic evidence that motivated this route found exact source coordinates for all 640 requested
Greek-PhD documents and all 1,597 sampled windows. Under the historical document-union projection,
1,279 declared positive spans partition into 1,271 exact nonempty, 6 adjusted nonempty, and 2
zero-effective silver spans; 4 declarations escape their own unit window. These are LLM-silver
diagnostics, not source drift and not human adjudication. The v2 audit recomputes and receipts those
counts rather than hard-coding an exception list into hydration. It also receipts the exact selected
document-text digest and `doc_id`/`text → document → content` field-use counts; rehydration recomputes
and enforces all three before writing units.
- `--kallipos-route kallipos_sections` selects `Dataset_Kallipos.parquet` and reproduces the historical
  filename/id section grouping. `--kallipos-route nanochat_base` instead selects the processed
  `data/Apothetirio_Kallipos.parquet` document representation and remains an explicit comparison route.
- `--openarchives-route nanochat_base` selects the five pinned `data/openarchives.gr*.parquet`
  artifacts with `source_doc_id` → `text` and `source_dataset=openarchives.gr`. This retained base
  predates the registry's current replacement/resegmentation candidate and is the preferred recovery
  attempt, but it is still stamped snapshot-unverified.
- `--openarchives-route openarchives_current` explicitly selects the current raw
  `data/openarchives/**/*.jsonl.zst` replacement/resegmentation family with `doc_id` and
  `text → document → content` precedence. It is not presumed text-equivalent: the first full recovery
  attempt failed closed when `S01278` requested line 1,831 but the current document ended at line
  1,797. The failed immutable run is evidence for choosing the retained base, not grounds to clip the
  historical window.

The raw forensic route adds these mandatory builder arguments:

```bash
--greek-phd-route mdc_raw_forensic \
--mdc-quarantine-receipt "$MDC_QUARANTINE_RECEIPT" \
--mdc-span-audit-receipt "$MDC_SPAN_AUDIT_RECEIPT" \
--mdc-expected-observed-sha256 "$MDC_EXPECTED_OBSERVED_SHA256" \
--allow-quarantined-mdc-comparison-only
```

The Clariden wrapper additionally requires
`CONFIRM_QUARANTINED_MDC_COMPARISON_ONLY=1` and rejects
`EXPECTED_SPAN_ARTIFACT_SHA256` on this route, preventing a rehydrated current object from being
self-pinned as an independently verified historical snapshot.

The inspected Phase-04 lock contains all of those path families. The already staged Nanochat Greek and
OpenArchives files expose `source_doc_id`/`text` with hash-shaped IDs, while the registry declares the
required standalone OpenArchives and Kallipos fields; the passed acquisition schema audit confirms
them after download. The raw MDC Greek + Nanochat-OpenArchives + section-Kallipos route is therefore
the defensible next attempt at the 3,340-unit text join without `home`. A lock or a partially downloaded
tree is not authorization:
the builder requires a completed `full_cpt_acquisition_receipt_v1` with `status=passed`. Even after a
successful exact document/coordinate join, historical label-text snapshot equivalence remains
`rehydrated_unverified_snapshot` unless an independently recorded expected SPAN artifact SHA-256
matches.

The manifest-bound `span_rehydration_layout.json` preserves the historical batching: 2,483 base
units in 178 batches, followed by 857 extension units in 62 new batches. This is why the exact result
is 240 batches rather than the 239 produced by globally rechunking 3,340 units.

On a Clariden CPU node, first record the tracked missing-artifact audit, then rehydrate the text batches
and hydrate the BIB-only silver. The external recovery receipt—not the unchanged tracked-directory
audit—is the authority for the recovered files:

```bash
cd subprojects/05_token_distillation_cpt/02_corpus_preparation/15_clean_academic/eval
python3 -m pip install -r sequence_models/requirements-rehydration.txt
python3 -m sequence_models.silver_reconstruct audit --output "$SCRATCH/silver-audit.json"
python3 -m sequence_models.silver_reconstruct rehydrate-span-units \
  --source-artifacts "$SCRATCH/span-source-artifacts.json" \
  --output-dir "$SCRATCH/SPAN-rehydrated" \
  --receipt "$SCRATCH/SPAN-rehydrated.receipt.json"
python3 -m sequence_models.silver_reconstruct hydrate-span \
  --unit-dir "$SCRATCH/SPAN-rehydrated" \
  --unit-rehydration-receipt "$SCRATCH/SPAN-rehydrated.receipt.json" \
  --tokenizer-json "$HOME/tokenizers/ModernGreek-148k/tokenizer.json" \
  --output "$SCRATCH/span.LLM_silver.jsonl" \
  --split-manifest "$SCRATCH/span.LLM_silver.split.json" \
  --receipt "$SCRATCH/span.LLM_silver.receipt.json"
```

Text recovery uses the tracked `win_lo`/`win_hi` coordinates and the original physical-line numbering,
omitting blank lines exactly as `build_span_units.py` did. It refuses missing or duplicated requested
documents, partial/extra/reordered units or batches, window overflow, artifact hash drift, ambiguous
Kallipos section order, and output overwrite. Every input artifact and every output batch is hashed.
Full source shards naturally contain non-target documents; those rows are counted and ignored, while
the extracted document set must equal the manifest request exactly and a repeated requested `doc_id`
fails closed.
Hydration additionally refuses conflicting overlap text, tokenizer hash drift, identity collisions,
exact-text or work leakage, split-manifest drift, and malformed annotation coordinates. It then
reproduces the historical `span_seq_data.py` semantics exactly: merge present nonblank lines from all
sampled windows by document and label only the intersection with each inclusive declared span. It does
not invent or rewrite a boundary. Missing endpoints, spans escaping their own unit window, adjusted
nonempty projections and zero-effective spans are recorded as coordinate-only diagnostics. A
zero-effective positive declaration contributes no positive line; sampled lines remain `O`, matching
the historical comparison dataset. Any such output remains explicit LLM-silver comparison evidence,
never verified gold or a production authorization.

Without `--expected-artifact-sha256`, the receipt is explicitly
`rehydrated_unverified_snapshot`. A matching reference directory is useful diagnostic evidence but does
not silently upgrade that status. If an independently recorded original snapshot digest is recovered,
validate it first with `span-snapshot-digest`, then rerun recovery with the explicit digest:

```bash
python3 -m sequence_models.silver_reconstruct span-snapshot-digest \
  --unit-dir "$ARCHIVE/original-SPAN"
python3 -m sequence_models.silver_reconstruct rehydrate-span-units \
  --source-artifacts "$SCRATCH/span-source-artifacts.json" \
  --output-dir "$SCRATCH/SPAN-rehydrated-verified" \
  --receipt "$SCRATCH/SPAN-rehydrated-verified.receipt.json" \
  --expected-artifact-sha256 "$EXPECTED_64_HEX_SNAPSHOT_SHA"
```

An explicit mismatch fails atomically. An unverified rehydration may still be used as clearly labelled
LLM-silver comparison evidence, but it never establishes historical snapshot equivalence and this
receipt can never authorize a production promotion.

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
null safety metrics needing independent prose labels. Silver model selection is comparison-only; no
2,000-line human annotation effort is required or implied. Any future structural-application run must
freeze an approved policy before Stage10. After a candidate is frozen, its separate production-safety
path also requires Stage54 and a receipt-bound manual review of exactly 100 high-risk predicted removals
(50 ToC + 50 BIB), with zero catastrophic deletions and every configured
retention/contamination gate. The packet
can be produced with `silver_reconstruct false-deletion-packet`; the tool never auto-adjudicates it.
It reuses the existing `failure_analysis.py`/`span_signals.py` taxonomy rather than forking it. A clean
packet alone does not substitute for the configured conservative gates, and the rehydration receipt
itself always remains non-promotion-authorizing.

## Clariden CPU execution (explicit, not in the corpus DAG)

`04_full_corpus_preparation/clariden/07_rehydrate_span_silver.sbatch` performs the receipt build, text
rehydration, and LLM-silver hydration on one CPU allocation. It is deliberately absent from
`submit.sh`, refuses non-Slurm/local execution, requires an exact clean Git commit, and defaults to a
no-write dry run unless `CONFIRM_CLASSIFIER_RESEARCH=1` is exported. Example from Clariden:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/train-apertus-with-glossapi
export PHASE04_CLARIDEN_DIR="$PWD/subprojects/05_token_distillation_cpt/04_full_corpus_preparation/clariden"
export PHASE04_EXPECTED_COMMIT="$(git rev-parse HEAD)"
export ACQUISITION_RECEIPT=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/source_locks/COMPLETED.receipt.json
export SOURCE_LOCK=/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/source_locks/EXACT.lock.json
export GREEK_PHD_ROUTE=nanochat_base
export KALLIPOS_ROUTE=kallipos_sections
export CLASSIFIER_RUN_ID=span-silver-20260712

# Dry run: resolves and prints the immutable plan, then exits without writes.
sbatch --export=ALL "$PHASE04_CLARIDEN_DIR/07_rehydrate_span_silver.sbatch"

# Execution requires a new, unused CLASSIFIER_RUN_ID/output root.
export CONFIRM_CLASSIFIER_RESEARCH=1
sbatch --export=ALL "$PHASE04_CLARIDEN_DIR/07_rehydrate_span_silver.sbatch"
```

This job creates no labels. It reconnects the 3,339 existing Opus bibliography-window decisions to
receipt-bound text; the missing `S00772` decision remains missing while its sampled lines retain the
historical negative-only treatment, and no ToC target is inferred. This
BIB-only output cannot satisfy Stage54's joint-head evidence contract. The current CPT run does not need
it for application because the tracked policy is `audit_only` and Stage58 is an explicit no-op.
