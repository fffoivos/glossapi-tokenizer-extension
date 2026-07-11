# Full-corpus decontamination, deduplication and release

These stages operate on Clariden CPU nodes after `apply_cleaning_policy.py`.
They stream Parquet row groups, use DuckDB spill for global joins and never
load the roughly 170 GB corpus into memory. Bibliography/ToC spans have already
been applied by the cleaner at this point, so the order is:

1. cleaning, PII masking and optional approved ToC/bibliography spans;
2. GreekMMLU decontamination;
3. existing GlossAPI exact/near deduplication;
4. release materialization, exact token waterfall and validation;
5. gated Hugging Face publication of redistribution-eligible rows only.

## Freeze GreekMMLU inputs

Build the canonical query JSONL with the existing
`02_corpus_preparation/30_decontaminate/scripts/build_decontamination_queries.py`
tool. Every row must expose `question`, choices, the correct answer and a split,
or the whole file must have a frozen `--default-split`. Then bind it to the
immutable Hugging Face dataset commit:

```bash
python scripts/freeze_greekmmlu_queries.py \
  --queries-jsonl "$RUN/queries/greekmmlu.jsonl" \
  --output "$RUN/queries/greekmmlu.jsonl.manifest.json" \
  --dataset-revision "$GREEKMMLU_COMMIT" \
  --required-split test \
  --default-split test \
  --registry "$REPO/subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/native_greek_benchmark_registry.json"
```

`decontaminate_full_corpus.py` fails closed if that sidecar is missing, the
query checksum drifts, the dataset revision is mutable or a required split is
absent. A manifest may include both train and test in `required_splits`; all
listed splits are normalized and indexed.

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
  --dedup-decisions "$RUN/dedup_run/final/dedup_decisions.parquet" \
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
  --dedup-decisions "$RUN/dedup_run/final/dedup_decisions.parquet" \
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
training-eligible dedup survivor. `redistribution/data` is the subset also
approved for redistribution and excludes raw source metadata and author
columns. Validation rehashes every file and gates duplicate stable IDs, exact
cleaned-text hashes, text/hash drift, missing provenance, eligibility leaks,
dedup join coverage and the redistribution subset relation.

## Gated publication

Publication is a dry run unless `--execute` is present. Manual gating is the
default, and the uploader's folder is hard-bound to `redistribution/data`:

```bash
python scripts/publish_release.py \
  --release "$RUN/release" \
  --release-manifest "$RUN/manifests/release.json" \
  --validation-receipt "$RUN/manifests/release_validation.json" \
  --repo-id fffoivos/glossapi-greek-cpt-full-corpus-v2

HF_TOKEN="$HF_TOKEN" python scripts/publish_release.py \
  --release "$RUN/release" \
  --release-manifest "$RUN/manifests/release.json" \
  --validation-receipt "$RUN/manifests/release_validation.json" \
  --repo-id fffoivos/glossapi-greek-cpt-full-corpus-v2 \
  --gate-mode manual --execute
```
