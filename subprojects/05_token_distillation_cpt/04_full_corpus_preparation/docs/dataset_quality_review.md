# Dataset quality diagnostics and review site

This lane answers two different questions without conflating them:

1. **What do the selected new datasets look like?** Stage
   `35-dataset-quality-sample` profiles the exact documents selected by the
   Stage 30 source-review policy. It is the default, prompt feedback path.
2. **What are the selected-population cleanliness rates?** Optional Stage
   `15-dataset-quality-full` profiles every selected canonical candidate row and
   can resume across Slurm time limits.

Both modes use the same pinned GlossAPI Rust modules and per-document schema.
Neither writes cleaned corpus text. A sample statistic is always labelled
`review_sample`, displays its document denominator, and must not be described as
corpus-wide. A full scan names its selected and excluded source IDs and is not
promoted to a corpus-wide claim merely because every row in that selected
population was read.

## Repository universes

The presentation deliberately has 29 entries: the 25 post-cutoff repositories
plus four older repositories with material post-cutoff changes. This is an
inventory universe, not a claim that all 29 currently have normalized text.

The current normalization registry is different:

- `nanochat_base` contains 18 exact historical `source_dataset` groups and is
  excluded from quality diagnostics by default;
- current candidate routes cover most text-bearing inventory entries;
- Kallipos and Pergamos replacement routes are useful corpus checks but are not
  members of the 29-entry post-cutoff inventory, so their metrics are recorded
  as supplemental and do not create extra site pages;
- legacy OpenGov and current EUR-Lex are inventory replacement audits rather
  than blind additive routes.

The site therefore performs an explicit repository-ID join. Missing metrics are
shown as missing, never borrowed from a similarly named Nanochat source.

## What cannot yet be evaluated

`glossAPI/istorima`, `glossAPI/modern-greek-dictionary`, and
`glossAPI/ert-press` are externally registered Mozilla Data Collective Parquet
archives rather than payloads acquired into the receipt-bound canonical run.
Their provenance and provisional value can be reviewed,
but no cleanliness, variability, Rust, or document-sample statistic exists
until the exact MDC payload is acquired, hashed, normalized, and added to a new
immutable run. User confirmation of license terms does not substitute for a
payload receipt.

Likewise, `pandemos` is metadata-only, while `elstat` and
`hellenic-parliament-legislative-work` are empty scaffolds. Their pages explain
that state and show no fabricated document metrics.

## CPU-only pinned Rust runtime

Prepare a clean detached GlossAPI checkout at commit
`6f29a2825559c540ab342fc77ae4457cf3556f2a` on Clariden at
`$GLOSSAPI_QUALITY_ROOT`. Then submit:

```bash
CONFIRM_LAUNCH=1 ./clariden/submit.sh build-quality-runtime
```

`06_build_glossapi_quality_runtime.sbatch` is a CPU-only compute-node job. It:

- builds `glossapi_rs_noise` and `glossapi_rs_cleaner` with `maturin` and
  `cargo --locked` in job-local target directories;
- installs only the two wheels into a staged, relocatable module directory;
- binds module hashes, both Cargo locks, Python, `rustc`, `cargo`, the pinned
  `maturin` version, and the clean source commit in
  `glossapi_rust_quality_build_receipt_v1`;
- atomically renames the complete module-and-receipt directory into its
  versioned runtime path;
- imports and rehashes the published modules after the rename.

Compilation is never performed on the Mac or a login node. Profiling refuses a
dirty/different source checkout, a different imported module path, or any module
hash drift.

## Representative and full profiling

After Stage 30 has produced `requests.jsonl`, the default command is:

```bash
CONFIRM_LAUNCH=1 PIPELINE_RUN_ID=<run-id> \
  ./clariden/submit.sh dataset-quality-sample
```

Stage 35 first exports only the exact primary review sample from the canonical
shards. High-precision identifier patterns are masked; this is not a claim that
generic names, addresses, or identifying context have been anonymized. Raw
`source_doc_id` values are not persisted. Complete `http://`, `https://`, and
`www.` URL spans in exported sample text are masked, including their query and
fragment material. This targeted masking still does not make the samples
anonymous or safe to publish. The relocatable packet receipt binds the exact
Stage30 request hash, canonical shards, profile-text hashes, redaction
implementation hashes, and persistent per-shard export checkpoints. The packet
receipt also points to a compact, text-free site attestation. Before emitting
that attestation on Clariden, the exporter rehashes every checkpoint fragment,
revalidates the exact normalization/acquisition identity closure, and records
the current masking implementation hashes. Its contract, checkpoint directories,
receipts, and fragments must remain beneath the packet checkpoint root and are
opened component-by-component without following symlinks. Checkpoint receipt
objects are validated against the same exact-key, non-boolean integer, SHA, and
relative-path contract as their JSON schema before reuse. The packet, receipt,
and attestation are inputs to the Rust profiler, so the usual 100/200 per
exact-source sampling strata remain intact.

For the optional population scan:

```bash
CONFIRM_LAUNCH=1 QUALITY_MODE=full_scan PIPELINE_RUN_ID=<run-id> \
  ./clariden/submit.sh dataset-quality-full
```

Full scan checkpoints are immutable 4,096-document batch directories bound to
the normalized shard SHA. Resume with `submit.sh resume dataset-quality-full`.
Set `INCLUDE_NANOCHAT_BASE=1` only for an intentional 54-million-document base
audit; the default scans candidates, not Nanochat.

Checkpoint receipts and outputs are opened by walking every path component with
`O_NOFOLLOW`; absolute, non-canonical, traversal, duplicate, and symlinked paths
are rejected. Selected physical shard paths are globally unique regardless of
source labels or hashes. Every batch has its own directory containing exactly
`receipt.json` and `documents.parquet`; receipt parents, derived output paths,
and device/inode output identities cannot alias. Each checkpoint inventory
entry binds the source ID, canonical shard path and SHA, batch index, and
half-open row interval. A `full_scan` summary is valid only when those intervals
start at zero, are contiguous and non-overlapping, end at every selected
shard's declared row count, and sum to the consolidated document and repository
denominators. A `review_sample` summary instead binds its single synthetic shard
to the exact masked packet filename, SHA, and document count recorded in the
contract. The compact site handoff repeats these validations and reopens each
receipt/output beneath the quality output root before attesting it.

The zero-badness/zero-Greek case is an explicit guard state because the current
Rust noise scorer can return zero when no Greek remains after table filtering.
It is never labelled “clean” on score alone.

The summary exposes document length, Greek-letter share, repeated/one-token
line rates, HTML, mojibake and replacement markers, Markdown tables, simple
ToC/bibliography header heuristics, Diavgeia footer/ADA/personnel and metadata
signals, structural edge-template concentration, Rust badness, and diagnostic
cleaner loss. The ToC/BIB header heuristics are not classifier accuracy or
authorization to delete text; receipt-bound trained-classifier results remain
pending in the separate structural-cleaning lane.

## Dataset-review presentation site

Agent 3 accepts one compact immutable directory, headed by
`dataset_review_site_handoff.json`. It contains only receipts, summaries,
review documents, and any bounded public-preview packet—never Parquet shards,
corpus exports, checkpoints, or model files. Every member is role-, byte-,
SHA-256-, schema-, run-ID-, and producer-commit-bound before the static site
opens it. Public material is labelled **public source excerpt**. A transformed
copy is labelled **identifier-masked pipeline review sample** and states the
transformation; neither label makes a confidentiality claim about public source
data.

Build a staging site from the handoff while Agent 1 is running (a fixture
handoff produces `UI/fixture ready` only):

```bash
RUN_ID=full-corpus-v3-YYYYMMDD
STAGING="$HOME/presentations/train-apertus-with-glossapi/full-corpus-v3-dataset-review.staging-$RUN_ID"
python scripts/build_dataset_review_presentation.py build \
  --handoff-dir /path/to/agent1-handoff \
  --output-dir "$STAGING"

python scripts/build_dataset_review_site.py serve \
  --site-dir "$STAGING" --port 8767
```

Publication performs a fresh full handoff rehash, refuses fixture/incomplete
handoffs and existing targets, verifies every emitted file, and atomically
renames the staging directory:

```bash
FINAL="$HOME/presentations/train-apertus-with-glossapi/full-corpus-v3-dataset-review"
python scripts/build_dataset_review_presentation.py publish \
  --handoff-dir /path/to/agent1-handoff \
  --staging-dir "$STAGING" \
  --output-dir "$FINAL"

python scripts/build_dataset_review_site.py serve \
  --site-dir "$FINAL" --port 8766
```

The server binds only `127.0.0.1`. The generated site has no CDN, analytics,
web fonts, or automatic external requests; uses a strict CSP; keeps opaque
site-local sample IDs; fetches text JSON only on demand; inserts it with DOM
`textContent`; and refuses a tampered, extra, or symlinked site file before
serving. The `site_manifest.json` binds all inputs and output files, while
`site_acceptance_report.md` provides exact receipts, missing-evidence state,
commands, and the browser/screenshot checklist.
