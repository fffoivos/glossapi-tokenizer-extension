# 02.2 Tokenizer Implementation

## Scope

Implement the actual compatible Greek `BPE` discovery and merge-rule extension workflow.

## Already Decided

- do not ship `add_tokens(...)`
- patch Apertus through `model.vocab` and `model.merges`
- preserve all old token ids
- append only new ids
- emit a manifest of every newly added unit
- enforce final vocab divisibility by `128`

## Required Checks

- exact preservation of the first `1000` ids
- exact preservation of special-token behavior
- exact preservation of regex split and byte-level behavior
- non-Greek smoke test after extension

## Sub-subprojects

### `vocab_lang_attribution/`

Per-token language attribution for the 131,072 Apertus base-vocab entries.
Tokenizes ~1 B Apertus-tokens per canonical language (1,933 canonical keys
covering FineWeb-2 1,811 langs + Wikipedia / EuroParl / ParaDocs / FineWeb-Edu /
FineWeb-HQ / DCLM-Edu) and emits raw per-language histograms plus token
metadata. Output joinable with the E / U embedding arrays by `token_id` to
support per-language embedding-structure experiments (norm distributions,
script-family overlap, "out of place" detection, candidate-init lookup
for the C3 extension).

- Read first: [`vocab_lang_attribution/RUN_REPORT.md`](vocab_lang_attribution/RUN_REPORT.md)
- Scripts: [`vocab_lang_attribution/scripts/`](vocab_lang_attribution/scripts/)
- Staging (per-worker raw `.npy` pulls): `vocab_lang_attribution/staging/<worker_idx>/`
- Final outputs (post-aggregation): `vocab_lang_attribution/outputs/`

