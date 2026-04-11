# Active Backlog

## Immediate Order

1. Finalize HPLT filtering rules, including the exclusion of `Machine translated or generated` content.
2. Freeze the HPLT-to-canonical-schema mapping for upload into `data/*.parquet`.
3. Produce upload-ready HPLT parquet file(s) in the existing canonical 21-column schema.
4. Integrate those parquet file(s) into `fffoivos/glossapi-greek-nanochat-pretraining-dataset`.
5. Freeze the lightweight downstream builder config for `GlossAPI-only` vs `GlossAPI + HPLT`.
6. Freeze the held-out eval manifests.
7. Only then start true Greek `BPE` discovery experiments.

## Immediate Risks

- the exploratory HPLT review sample is not the same thing as the final upload-ready HPLT slice
- the exact tokenizer-replication spec is still missing some literal details
- the old baseline workflow still exists for reference, so active work must stay within the new canonical files
