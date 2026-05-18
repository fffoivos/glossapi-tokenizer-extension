# Firing-count run — 2026-05-18

This is the completed cloud run for the canonical
`c3_added_17408_curated_padded` tokenizer selected in
[`CHOSEN_CUTOFF.md`](CHOSEN_CUTOFF.md).

## Artifact Locations

Local output, intentionally ignored by Git:

```text
variants/c3_added_17408_curated_padded.firing_counts/
```

Hugging Face artifact location:

```text
fffoivos/apertus-tokenizer-extension
experiments/02_1_7_intrinsic_eval_sweep_20260518/firing_counts_c3_added_17408_curated_padded/
```

Tracked provenance:

```text
manifests/firing_count_20260518_run_summary_augmented.json
```

## Inputs

The run used the exact C3 BPE training split, not the upstream
`mix.parquet` fallback:

| input | GCS object | size | generation |
|---|---|---:|---:|
| train text | `gs://testbucketglossapi/c3_train_mix/train.parquet` | 44,174,219,259 bytes | 1779079559870759 |
| row-aligned manifest | `gs://testbucketglossapi/c3_train_mix/train_manifest.parquet` | 1,031,147,228 bytes | 1779079571531114 |

The augmented run summary records the GCS MD5 and CRC32C hashes.

## Execution

| field | value |
|---|---|
| run id | `20260518t044858` |
| GCS prefix | `gs://testbucketglossapi/firing_counts_20260518t044858` |
| K shards | 8 |
| worker machine | `c4-highcpu-32`, `europe-west4-a` |
| sharder machine | `c4-highcpu-32`, `europe-west4-a` |
| sharder wall | 2,127 s |
| fleet wall, inferred | 980 s |
| local aggregation wall | 39 s |
| end-to-end wall, inferred | 3,193 s |

The worker self-delete permission was unavailable for the default compute
service account. The fleet VMs were cleaned up manually after the run;
this is recorded in the augmented summary.

## Outputs

The final bundle contains:

```text
glossapi_nanochat_only.parquet
hplt_only.parquet
glossapi_nanochat_plus_hplt.parquet
source_dataset_token_counts.parquet
source_dataset_summary.parquet
run_summary.json
run_summary_augmented.json
provenance/
```

The cloud shard and per-shard partial objects are not canonical
artifacts. They are reproducible intermediates and should not be copied
to Git or Hugging Face.

## Corpus Totals

| component | rows | chars | tokenized tokens | token share |
|---|---:|---:|---:|---:|
| `glossapi_nanochat_only` | 517,791 | 46,998,438,596 | 12,394,053,979 | 49.79% |
| `hplt_only` | 13,883,763 | 52,258,739,417 | 12,497,462,491 | 50.21% |
| **combined** | **14,401,554** | **99,257,178,013** | **24,891,516,470** | **100.00%** |

The 50/50 design is approximately balanced by tokenized tokens, not by
row count: HPLT contributes 96.4% of rows but 50.2% of token mass because
GlossAPI rows are much larger.

## Added-token Firings

| component | added-token firings | added-token firing share | zero added tokens |
|---|---:|---:|---:|
| `glossapi_nanochat_only` | 3,356,312,291 | 27.08% | 0 |
| `hplt_only` | 4,740,755,772 | 37.93% | 27 |
| `glossapi_nanochat_plus_hplt` | 8,097,068,063 | 32.53% | 0 |

All 17,408 added tokens fire in the GlossAPI-nanochat component. In the
combined C3 corpus, only one added token has fewer than 100 firings.

## Source Pattern

The GlossAPI-nanochat component is concentrated:

| source | tokenized tokens | share of all tokens | added-token firing rate |
|---|---:|---:|---:|
| `openarchives.gr` | 7,069,580,546 | 28.40% | 27.22% |
| `greek_phd` | 3,685,891,778 | 14.81% | 25.14% |
| `Apothetirio_Pergamos` | 523,970,573 | 2.10% | 27.37% |
| `HuggingFaceFW/finewiki` | 239,034,764 | 0.96% | 27.03% |
| `Apothetirio_Kallipos` | 186,153,872 | 0.75% | 28.84% |

`openarchives.gr` and `greek_phd` together provide most of the
GlossAPI-nanochat token mass.

## Validation

The post-run audit checked:

- all component parquets have exactly 148,480 rows and ids `0..148479`;
- rate columns equal `fire_count / denominator`;
- `glossapi_nanochat_only + hplt_only == glossapi_nanochat_plus_hplt`
  exactly;
- per-source long counts sum back to the component parquets exactly;
- `source_dataset_summary.parquet`, `source_dataset_token_counts.parquet`,
  `run_summary.json`, and the augmented summary agree on the 18 sources;
- no nulls in final artifact columns.

## Interpretation Caveat

The firing-count data measures usage, not linguistic quality. Some
highly GlossAPI-specific added tokens are meaningful Greek/polytonic
units, while others are document-layout or PDF/math residue. Downstream
token attribution should therefore combine firing rates with the
curation categories from `02_1_5_added_token_curation`.
