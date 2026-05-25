# CPT Dataset Build Runbook

This is the repeatable path for building an Apertus CPT dataset from the
published nanochat corpus.

## Inputs

- Source corpus:
  `fffoivos/glossapi-greek-nanochat-pretraining-dataset`
- Nanochat internal dedup bundle:
  `dedup_metadata/wave2_20260426_builder_metadata_v2_latest_cleaner_20260507/builder_metadata`
- Apertus hard-drop overlay:
  `fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`
  at `artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet`

The correct order is:

1. Download the published nanochat source rows.
2. Hard-exclude `apertus_overlap_drop_docs.parquet`.
3. Replay nanochat internal dedup with `drop_intra_and_inter`.
4. Build the final CPT mix/export from the remaining pool.

The order matters: if Apertus-overlapping docs are removed first, an internal
duplicate family can still keep a fresh alternate representative.

## GCP Scratch VM

Use a scratch VM with a large disk. A practical starting point is a
`c4-highmem-32` or larger VM with a 1-2 TB SSD/PD disk. Build artifacts should
be uploaded to Hugging Face or GCS before teardown.

Example setup on the VM:

```bash
sudo apt-get update
sudo apt-get install -y git python3-pip
python3 -m pip install --user duckdb pandas pyarrow huggingface_hub typer blake3 polars

git clone https://github.com/<your-org-or-user>/glossapi-tokenizer-extension.git
cd glossapi-tokenizer-extension
export PYTHONPATH="$PWD"
```

## Download Inputs

Do not build from the old local `/home/foivos/data/glossapi_work/hf_release_publish`
copy. Hydrate from the live HF repo so the `source_dataset` names and derived
`doc_key` values match the Apertus overlay and the `wave2` dedup bundle.

```bash
export WORK=/mnt/disks/cpt_build
mkdir -p "$WORK"

huggingface-cli download fffoivos/glossapi-greek-nanochat-pretraining-dataset \
  --repo-type dataset \
  --local-dir "$WORK/nanochat" \
  --include 'data/*.parquet' \
  --include 'dedup_metadata/latest.json' \
  --include 'dedup_metadata/wave2_20260426_builder_metadata_v2_latest_cleaner_20260507/builder_metadata/*'

huggingface-cli download fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z \
  --repo-type dataset \
  --local-dir "$WORK/apertus_audit" \
  --include 'artifacts/dedup_20260519T010924Z/cpt_final_overlay/*'
```

## Build Fresh Deduped Pool

```bash
export DEDUP_ROOT="$WORK/nanochat/dedup_metadata/wave2_20260426_builder_metadata_v2_latest_cleaner_20260507/builder_metadata"
export APERTUS_DROP="$WORK/apertus_audit/artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet"
export SELECTED="$WORK/cpt/selected_after_apertus_and_internal_dedup.parquet"

python3 -m glossapi_corpus_cli.cli mix-prepare-selected-input \
  --output-root "$WORK/nanochat" \
  --selected-input-path "$SELECTED" \
  --exclude-doc-keys-path "$APERTUS_DROP" \
  --dedup-metadata-root "$DEDUP_ROOT" \
  --dedup-action drop_intra_and_inter \
  --dedup-exact-stage strict_and_relaxed \
  --dedup-similarity-threshold 0.85 \
  --dedup-inter-dataset-policy share_aware
```

`$SELECTED` is the deduplicated, Apertus-fresh source pool. If the CPT recipe
uses the whole remaining pool, this is the main dataset artifact.

## Optional Source Mix

If the CPT run needs a sampled or weighted mix, build it from `$SELECTED`:

```bash
python3 -m glossapi_corpus_cli.cli mix-build-from-selected-input \
  --selected-input-path "$SELECTED" \
  --mix-output-path "$WORK/cpt/final_cpt_mix.parquet" \
  --source-mix-config-path "$WORK/cpt/source_mix.json"
```

## Publish And Teardown

Upload the final selected pool or mix before deleting the VM:

```bash
huggingface-cli upload <target-hf-dataset-repo> "$WORK/cpt" . --repo-type dataset
```

Then verify the HF repo contains the final parquet(s), summaries, and the
`cpt_final_overlay/summary.json` provenance before tearing down the VM.
