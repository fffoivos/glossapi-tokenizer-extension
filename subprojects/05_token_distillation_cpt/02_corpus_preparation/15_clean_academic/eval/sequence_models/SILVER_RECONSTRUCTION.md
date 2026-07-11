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
The one absent annotation is `S00772`; it is omitted, never inferred.

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

- Greek PhD and OpenArchives: `.jsonl` or `.jsonl.zst`, selected by `doc_id`;
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
  --kallipos-route kallipos_sections \
  --output "$SCRATCH/span-source-artifacts.json"
```

Both route choices are mandatory; the builder never prefers a newer artifact implicitly.

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
- `--kallipos-route kallipos_sections` selects `Dataset_Kallipos.parquet` and reproduces the historical
  filename/id section grouping. `--kallipos-route nanochat_base` instead selects the processed
  `data/Apothetirio_Kallipos.parquet` document representation and remains an explicit comparison route.
- OpenArchives always selects the pinned `openarchives_current` raw `data/openarchives/**/*.jsonl.zst`
  files with `doc_id` and the original `text`, `document`, `content` precedence.

The inspected Phase-04 lock contains all of those path families. The already staged Nanochat Greek
files expose `source_doc_id`/`text` with hash-shaped IDs, while the registry declares the required
OpenArchives and Kallipos fields; the passed acquisition schema audit must confirm the latter after
download. The Nanochat-Greek + current-OpenArchives + section-Kallipos route is therefore sufficient
**in principle** to attempt the 3,340-unit text join without `home`. A lock or a partially downloaded
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
exact-text or work leakage, split-manifest drift, annotation spans outside `win_lo`/`win_hi`, and any
positive span whose start/end coordinate is absent from the rehydrated nonempty lines. The resulting
silver receipt records this coordinate-alignment check. Source-balanced allocation remains
deterministic and label-blind until the existing annotation join.

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
receipt-bound text; the missing `S00772` decision remains missing, and no ToC target is inferred. This
BIB-only output cannot satisfy Stage54's joint-head evidence contract. The current CPT run does not need
it for application because the tracked policy is `audit_only` and Stage58 is an explicit no-op.
