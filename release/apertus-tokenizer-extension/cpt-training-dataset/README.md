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

## Token Counts With ModernGreek-148k

The full staged HPLT clean60 Wave4 source slice was also counted with the
selected `ModernGreek-148k` tokenizer in this repo
(`greek-extension-tokenizer/`, SHA-256
`358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`):

| Source slice | Files | Rows | Tokens, no EOD | Tokens, +1 EOD/doc |
|---|---:|---:|---:|---:|
| `HPLT/ell_Grek_ge8_no_mt_clean60` | 250 | 48,728,774 | 44,195,950,025 | 44,244,678,799 |

This count uses `add_special_tokens=false`; the EOD column is for Megatron-style
planning where one document separator is added per row. Full machine-readable
metadata is in `token-counts.json`.

Source graph:

```text
source-graph.json
```
