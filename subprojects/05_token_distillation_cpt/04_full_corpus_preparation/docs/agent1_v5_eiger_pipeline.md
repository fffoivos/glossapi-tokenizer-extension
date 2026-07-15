# Agent 1 v5: Eiger CPU production pipeline

This is the executable handoff for the 18 new sources plus the pinned Nanochat
base. It implements the user-approved order:

1. clean complex repetition and generated-image artifacts, then convert HTML to
   valid GFM;
2. apply GlossAPI Markdown normalization, Rust cleaner evaluation, Rust noise
   evaluation, and character statistics;
3. build the Nanochat envelope (`source_dataset`, `source_doc_id`, `text`,
   `title`, `author`, `source_metadata_json`) and preserve the Nanochat quality
   columns;
4. schema-cast (without changing text) and combine the pinned Nanochat base,
   then publish a private versioned Hugging Face snapshot;
5. run exact plus DataTrove MinHash deduplication, verify LSH candidates with
   actual 5-token-shingle Jaccard, materialize a decision ledger, and publish a
   second private versioned snapshot.

## Frozen contracts

- Config: `configs/agent1_v5_eiger_pipeline.json`
- Python runtime pins: `configs/agent1_v5_requirements.txt`
- GlossAPI: `a2aace04fbae61ed58931be1a1237a52d1b8ddb3`
- DataTrove: v0.9.0 / `87f7bad5c4a56ec648265fbf0b91d7d226bad428`
- Nanochat: `e1d54136a880ed1df2ed95a5445dabd230453207`
- Dedup: Greek tokenizer, accents preserved, 5-token shingles, 128 64-bit
  permutations as 32 bands × 4 hashes, seed 1, actual Jaccard ≥ 0.85, maximum
  5,000 documents per LSH bucket group.

The original Nanochat base always wins a mixed duplicate component. Among new
documents, the representative ordering is quality score, cleaning loss, text
length, then stable source/document identity.

## Storage bridge

The acquisition receipt currently points to MLP Iopsstor. Eiger is on HPCP and
uses Capstor, so the selected 18 sources plus Nanochat must first be staged by a
Clariden `xfer` job. The staging program hashes every copied source against its
pinned LFS/blob SHA-256 and emits an Eiger-visible narrowed acquisition receipt.

```bash
sbatch --account=a0140 --partition=xfer --time=23:30:00 \
  --export=ALL,PIPELINE_ROOT="$PIPELINE_ROOT",CONFIG="$CONFIG",SOURCE_ACQUISITION_RECEIPT="$SOURCE_RECEIPT",STAGED_INPUT_ROOT="$CAPSTOR_INPUT",STAGED_ACQUISITION_RECEIPT="$CAPSTOR_RECEIPT" \
  "$PIPELINE_ROOT/slurm/agent1_v5_eiger/stage_acquisition_xfer.sh"
```

## Submit and babysit

The Eiger-side Hugging Face token file must be mode 600. Its contents are read
only inside `xfer` jobs and are never placed in an sbatch argument or receipt.

```bash
python3 scripts/submit_agent1_v5_eiger.py \
  --pipeline-root "$PIPELINE_ROOT" \
  --config "$PIPELINE_ROOT/configs/agent1_v5_eiger_pipeline.json" \
  --acquisition-receipt "$CAPSTOR_RECEIPT" \
  --run-root "$CAPSTOR_RUN_ROOT" \
  --run-id "$RUN_ID" \
  --hf-token-file "$HF_TOKEN_FILE" \
  --account a0140 \
  --monitor
```

Setup and bootstrap are waited on before the dynamic arrays are sized. Each
major expensive implementation has a `debug` canary. Production work runs in
bundles on Eiger `normal`, publication on `xfer`, and receipts/merge gates use
`prepost` where they fit the 30-minute limit. Any missing text, empty post-clean
text, receipt mismatch, schema drift, incomplete task coverage, row-waterfall
failure, public HF repository, or remote checksum mismatch stops the DAG.

The submission state is written outside the not-yet-created run root at
`.<run-id>.coord/submission_state.json`. It can be reattached with:

```bash
python3 scripts/submit_agent1_v5_eiger.py \
  --monitor-state "/capstor/path/.${RUN_ID}.coord/submission_state.json"
```
