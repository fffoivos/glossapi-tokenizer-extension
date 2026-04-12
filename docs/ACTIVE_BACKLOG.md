# Active Backlog

## Tokenizer Critical Path

1. Rebuild the HPLT slice on a GCP worker with the real final filter path:
- quality bins `>=8`
- exclude `Machine translated or generated`
- run real `Corpus.clean(..., write_cleaned_files=False, drop_bad=False)` scoring
- drop rows with `greek_badness_score > 60`

2. Freeze the worker-side downstream builder inputs for:
- `GlossAPI-only`
- `GlossAPI + HPLT` at `70/30` by training-token mass

3. Verify that `openarchives.gr` rows with `needs_ocr == true` are still excluded from the CPT-ready dataset used for tokenizer work.

4. Freeze the held-out eval manifests from the same prepared source-parquet tree.

5. Lock the literal Apertus tokenizer-replication checklist, including the exact tokenizer files and a toy extension proof.

6. Export BPE-training text for the two training views from the CPT-ready dataset on the chosen GCP worker.

7. Start true Greek `BPE` discovery experiments from the frozen worker-side manifests.

8. Diff learned Greek units against Apertus `model.vocab` and `model.merges`.

9. Run the analytic cutoff sweep on merged variants at:
- `10240`
- `15360`
- `20480`
- `25600`

10. Only after the elbow is known, choose the shipped `128`-aligned extension size.

11. Implement and test the merge-rule extension.

## Dataset Operational Sidetrack

1. Replace the stopped score-only HPLT upload attempt with a rebuilt GCP-side slice that includes the real `Corpus.clean` gate.

2. Rerun the full prepared-source dataset with HPLT included, using the existing dataset scripts rather than inventing a new release path.

3. Keep tokenizer work moving off the prepared dataset without waiting for the HF upload to finish.

4. Refresh published `dedup_metadata` only after the intended dataset state is settled; do not block tokenizer work on that refresh.

## Immediate Risks

- the exploratory HPLT review sample is not the same thing as the final upload-ready HPLT slice
- the exact tokenizer-replication spec is still missing some literal details and a proof-of-mechanism test
- the current HF uploader strategy is poor for observability and recovery on very large patches
- the old baseline workflow still exists for reference, so active work must stay within the new canonical files
