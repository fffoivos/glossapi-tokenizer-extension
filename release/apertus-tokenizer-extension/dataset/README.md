# CPT 7B Mix

Status: recipe, provenance, and hydration pointer.

This is the text mix used for the bakeoff and continuation line. The large JSONL
and Megatron `.bin/.idx` payloads are not included in this model repo.

Composition:

- Greek: `70%`;
- non-Greek replay: `24%`;
- code: `4%`;
- math: `2%`.

Key Clariden paths:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.nfc.jsonl
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document
```

The final base-tokenized Megatron artifact has `9,831,704,774` tokens.

Source graph:

```text
source-graph.json
```
