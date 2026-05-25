# Legacy Hugging Face Layout Index

The previous Hugging Face repo layout included useful but noisy top-level
folders:

- `analysis/`;
- `artifacts/`;
- `continuous/`;
- `experiments/`;
- `fresh/`;
- `metadata/`;
- `subprojects/`;
- `tokenizers/`.

Those paths are retained for traceability. The release entrypoints are now the
curated folders at the repo root:

- `tokenizer/`;
- `checkpoints/`;
- `training-data/`;
- `results/`;
- `provenance/`;
- `code-links/`.

Do not treat this file as a deletion plan. It is an index explaining the old
layout while the new additive structure is reviewed.

