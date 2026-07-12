# Dataset quality diagnostics and private review site

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
opened component-by-component without following symlinks. The packet, receipt,
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
are rejected. Each checkpoint inventory entry binds the source ID, canonical
shard path and SHA, batch index, and half-open row interval. A `full_scan`
summary is valid only when those intervals start at zero, are contiguous and
non-overlapping, end at every selected shard's declared row count, and sum to
the consolidated document and repository denominators. The compact site
handoff repeats this validation and reopens each receipt/output beneath the
quality output root before attesting it.

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

## Private local site

Transfer only these compact/private Stage 35 artifacts to the Mac: the quality
summary, `dataset_quality_site_handoff.json`, Stage30 requests, the masked
complete-sample packet, its packet receipt, and
`complete_review_samples_site_attestation.json`. Do not transfer quality
checkpoint Parquets, sample-export checkpoint fragments, normalized shards, or
corpus directories. The two compact attestations are generated on Clariden
after those large dependencies have been rehashed there. Site generation itself
is lightweight coordination work:

```bash
python scripts/build_dataset_review_site.py build \
  --quality-summary /path/to/dataset_quality_summary_v1.json \
  --quality-handoff-receipt /path/to/dataset_quality_site_handoff.json \
  --review-requests /path/to/requests.jsonl \
  --complete-samples /path/to/complete_review_samples.jsonl \
  --complete-samples-receipt /path/to/complete_review_samples_receipt.json \
  --complete-samples-attestation /path/to/complete_review_samples_site_attestation.json

python scripts/build_dataset_review_site.py serve
```

The default target is
`~/presentations/train-apertus-with-glossapi/full-corpus-v2-dataset-review`.
The server binds only `127.0.0.1`. The site has 29 dataset pages, no CDN or
external resources, a restrictive CSP, percentage-formatted rates, and an
unmissable sample/full-scan scope label. Complete documents live in separate
mode-0600 JSON files, are fetched only on request, and are inserted with DOM
`textContent`; corpus HTML never executes. Canonical sample/document IDs never
enter the site: each build uses an ephemeral random HMAC key to derive opaque
site-local IDs and filenames. The builder rejects quality fields outside the
strict aggregate contract, requires exact acquired/normalized identity evidence
before upgrading an external repository's availability state, and reruns the
current masker idempotently on every complete document immediately before site
emission. It snapshots and hashes every local input before parsing, then rehashes
the same inputs immediately before atomic publication. The manifest binds every
input and every emitted file, and the local server refuses tampered, extra, or
symlinked content before binding to `127.0.0.1`. The material remains sensitive
and must not be shared.
