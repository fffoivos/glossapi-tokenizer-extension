# Full-corpus decontamination, deduplication and release

These stages operate on the CPU side of Clariden `normal` nodes after Stage50
source/PII cleaning and terminal source admission. They stream Parquet row
groups, use DuckDB spill for global joins and never load the roughly 170 GB
corpus into memory. Stage58 always runs before them: it either applies
pre-authorized, Stage54-promoted bibliography/ToC spans last or records an
explicit no-op and copies Stage50 text. The current `audit_only` policy takes
the no-op path. Ordinary cleaning chains stop after Stage50 or the optional
post-clean admission review. A separate, confirmed `chain-finalize-noop` or
`chain-finalize-promoted` command freezes the finalization choice and only then
submits the downstream order:

1. Stage50 source/PII cleaning, optional post-clean source review, then the
   separately confirmed, required structural-last Stage58;
2. GreekMMLU decontamination;
3. existing GlossAPI exact/near deduplication;
4. release materialization, exact token waterfall and validation;
5. gated Hugging Face publication of redistribution-eligible rows only.

## Freeze GreekMMLU inputs

Build the canonical query JSONL with the existing
`02_corpus_preparation/30_decontaminate/scripts/build_decontamination_queries.py`
tool. Every GreekMMLU row must carry its query schema, exact repository,
configuration, immutable revision, example ID and split. Then bind both the
query bytes and builder summary to the tracked registry and immutable
Hugging Face dataset commit:

```bash
python scripts/freeze_greekmmlu_queries.py \
  --queries-jsonl "$RUN/queries/greekmmlu.jsonl" \
  --output "$RUN/queries/greekmmlu.jsonl.manifest.json" \
  --dataset-revision "$GREEKMMLU_COMMIT" \
  --required-split test \
  --registry "$REPO/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json" \
  --builder-summary "$RUN/queries/build-summary.json"
```

`decontaminate_full_corpus.py` fails closed if that sidecar is missing, the
query checksum drifts, the dataset revision is not a full commit, the registry
or builder summary differs, or the observed split is not exactly the tracked
GreekMMLU split.

## Decontaminate

```bash
python scripts/decontaminate_full_corpus.py \
  --input "$RUN/cleaned" \
  --output "$RUN/decontaminated" \
  --dropped "$RUN/decontamination_dropped" \
  --ledger "$RUN/decontamination_ledger" \
  --manifest "$RUN/manifests/decontamination.json" \
  --queries-jsonl "$RUN/queries/greekmmlu.jsonl" \
  --workers "${SLURM_CPUS_PER_TASK:-64}"
```

The filter does not remove documents for an answer surface alone. Removal
requires an exact long eval prompt, exact question plus nearby correct answer,
or an aligned question match that simultaneously passes 85% unique 8-gram
coverage, 85% deterministic 64-permutation MinHash similarity and a nearby
correct answer. Question-only candidates remain in training but are preserved
in the decision ledger for audit. Text is copied byte-for-byte; every decision
is bound to the post-cleaning text hash.

## Stage and run the established deduplicator

```bash
python scripts/run_full_corpus_dedup.py \
  --input "$RUN/decontaminated" \
  --staged-input "$RUN/dedup_input" \
  --state-root "$RUN/dedup_state" \
  --run-root "$RUN/dedup_run" \
  --manifest "$RUN/manifests/dedup.json" \
  --temporary-directory "$RUN/duckdb_tmp" \
  --workers "${SLURM_CPUS_PER_TASK:-64}"
```

The wrapper does not implement deduplication. It records the hash and Git
commit of `glossapi_corpus_cli/text_dedup.py` and invokes its public
`dedup-text run` CLI with the established 128-permutation, 32x4 LSH, token
5-shingle and 0.85 threshold defaults. The adapter sets dedup `source_doc_id`
to canonical `stable_uid`, retains the upstream ID in
`upstream_source_doc_id`, and rejects duplicate identities before execution.
Use `--stage-only` only to inspect the immutable staged contract.

## Materialize and validate

```bash
python scripts/materialize_release.py \
  --input "$RUN/decontaminated" \
  --cleaning-manifest "$RUN/manifests/cleaning.json" \
  --decontamination-manifest "$RUN/manifests/decontamination.json" \
  --dedup-manifest "$RUN/manifests/dedup.json" \
  --dedup-decisions "$RUN/dedup_run/final/dedup_decisions_content_bound.parquet" \
  --output "$RUN/release" \
  --manifest "$RUN/manifests/release.json" \
  --token-waterfall "$RUN/manifests/token_waterfall.json" \
  --cleaning-ledger "$RUN/cleaning_ledger" \
  --decontam-ledger "$RUN/decontamination_ledger" \
  --temporary-directory "$RUN/duckdb_tmp" \
  --threads 32

python scripts/validate_release.py \
  --release "$RUN/release" \
  --manifest "$RUN/manifests/release.json" \
  --cleaning-manifest "$RUN/manifests/cleaning.json" \
  --decontamination-manifest "$RUN/manifests/decontamination.json" \
  --dedup-manifest "$RUN/manifests/dedup.json" \
  --dedup-decisions "$RUN/dedup_run/final/dedup_decisions_content_bound.parquet" \
  --output "$RUN/manifests/release_validation.json" \
  --temporary-directory "$RUN/duckdb_tmp" \
  --threads 32
```

The waterfall reports exact tokenizer counts already recorded by the cleaner,
by source, stage and reason: source cleaning, high-confidence PII masking,
combined ToC/bibliography removal, policy drops, GreekMMLU, strict exact,
relaxed exact and near dedup. Its final count must equal the token mass of kept
dedup decisions.

The release contains two disjoint views. `training/data` includes every
training-eligible dedup survivor. `redistribution/data` is an explicit public
allowlist for survivors also approved for redistribution. Titles, authors, raw
metadata and raw upstream IDs never pass through; selected upstream IDs are
replaced by domain-separated hashes. Validation rehashes every file and proves
both directions of public/private completeness plus exact content, safe
provenance, eligibility and hashed-metadata parity. It also independently
rechecks every content-bound dedup decision against the decontaminated input.

## Gated publication

Publication is a dry run unless `--execute` is present. Manual gating is the
default, and the uploader's folder is hard-bound to `redistribution/data`:

```bash
python scripts/publish_release.py \
  --release "$RUN/release" \
  --release-manifest "$RUN/manifests/release.json" \
  --validation-receipt "$RUN/manifests/release_validation.json" \
  --repo-id fffoivos/glossapi-greek-cpt-redistributable-delta-v2 \
  --output "$RUN/manifests/publication-dry-run.json"

HF_TOKEN="$HF_TOKEN" python scripts/publish_release.py \
  --release "$RUN/release" \
  --release-manifest "$RUN/manifests/release.json" \
  --validation-receipt "$RUN/manifests/release_validation.json" \
  --repo-id fffoivos/glossapi-greek-cpt-redistributable-delta-v2 \
  --gate-mode manual \
  --remote-mode new-empty \
  --output "$RUN/manifests/publication.json" \
  --execute
```

The publisher rejects symlinks, unvalidated files, local checksum drift, and a
token waterfall whose current hash differs from the release manifest. Any
remote payload, including a partial upload from a failed attempt, fails with an
instruction to inspect and delete/recreate the repository manually or choose a
new empty repository. The temporary uploader cache is discarded and the
publisher performs no automatic resume or remote cleanup. The exact remote inventory
is the generated, checksum-bound `README.md`, validated `data/**/*.parquet`, and
the explicit provenance receipts. It verifies every remote path/size/SHA-256 at
the returned Hugging Face commit and records that commit in the immutable
publication receipt. This repository is the redistributable delta only, not the
full private training corpus. See `docs/release_integrity.md` for the full
contract and schema links.
