# CPT-7B-mix

Human name: `CPT-7B-mix`.

Payload status: recipe and hydration pointer only; the large JSONL and Megatron
indexed dataset are not uploaded in this model repo.

Clariden selected Greek pool:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet
```

Clariden final NFC JSONL:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.nfc.jsonl
```

Clariden Megatron prefix:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document
```

Mix recipe:

- Greek: `70%`;
- non-Greek replay: `24%`;
- code: `4%`;
- math: `2%`.

Measured size:

- rows: `5,754,172`;
- mix-builder target: `7,000,141,612` extended-tokenizer-budget tokens;
- base-tokenized Megatron tokens: `9,831,704,774`.

Dataset bundle:

```text
../datasets/CPT-7B-mix/
```
