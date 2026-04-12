# Active Backlog

## Tokenizer Critical Path

1. Freeze the local downstream builder inputs for:
- `GlossAPI-only`
- `GlossAPI + HPLT`

2. Verify that `openarchives.gr` rows with `needs_ocr == true` are still excluded from the CPT-ready local dataset used for tokenizer work.

3. Freeze the held-out eval manifests from the same local upstream source-parquet tree.

4. Lock the literal Apertus tokenizer-replication checklist, including the exact tokenizer files and a toy extension proof.

5. Export local BPE-training text for the two training views from the CPT-ready local dataset.

6. Start true Greek `BPE` discovery experiments locally from the frozen manifests.

7. Diff learned Greek units against Apertus `model.vocab` and `model.merges`.

8. Run the analytic cutoff sweep around `5k`, `10k`, `15k`, `20k` new units.

9. Only after the elbow is known, choose the shipped `128`-aligned extension size.

10. Implement and test the merge-rule extension.

## Dataset Operational Sidetrack

1. Keep uploading the filtered HPLT slice into `fffoivos/glossapi-greek-nanochat-pretraining-dataset`.

2. Rerun the full prepared-source dataset locally with HPLT included, using the existing dataset scripts rather than inventing a new release path.

3. Keep tokenizer work moving off the local prepared dataset without waiting for the HF upload to finish.

4. Refresh published `dedup_metadata` only after the intended dataset state is settled; do not block tokenizer work on that refresh.

## Immediate Risks

- the exploratory HPLT review sample is not the same thing as the final upload-ready HPLT slice
- the exact tokenizer-replication spec is still missing some literal details and a proof-of-mechanism test
- the current HF uploader strategy is poor for observability and recovery on very large patches
- the old baseline workflow still exists for reference, so active work must stay within the new canonical files
