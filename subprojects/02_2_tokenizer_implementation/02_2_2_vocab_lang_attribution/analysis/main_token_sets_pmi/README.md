# PMI Main Token Sets

This directory contains the current language-token attribution pass used
by the Apertus tokenizer-extension experiments.

The method combines two independent signals:

1. empirical per-token firing counts from
   `02_2_2_vocab_lang_attribution/outputs/histogram_matrix.npz`;
2. strict char-admissibility masks from
   `02_2_1_char_language_membership/artifacts/token_language_bitmask.parquet`.

For each cap-hit canonical key (87 keys with at least 1B observed
Apertus-token firings), `build.py` computes PMI against the count-pooled
cap-hit marginal and emits three token sets:

- `tables/<key>__masked.txt` — Variant A, the main set: PMI/count test
  plus char admissibility for the target language.
- `tables/<key>__unmasked.txt` — Variant B, the PMI/count test only.
- `tables/<key>__delta.txt` — `B \ A`, tokens rate-promoted but rejected
  by the char mask.

Current knobs are recorded in `manifest.json`: `alpha = 0.5`,
`delta = 1.0`, `min_count = 100`, and `marginal_floor = 1_000_000_000`.

## Current Result

Latest rebuild: `2026-05-15T15:31:54Z`, char-tool schema v5.

- Apertus vocab size: 131,072.
- Covered by at least one masked set: 113,184 tokens (86.35%).
- Uncovered: 17,888 tokens (13.65%).
- Char scope consumed by this pass: 88 language bits, 47 family bits,
  29 script bits.
- Unmapped cap-hit keys: `gmh_Latn`, `und_Mong`, `und_Kana`,
  `und_Grek`, `und_Cyrl`.

## Files

- `build.py` — deterministic builder for the token sets and audit
  tables.
- `coverage_audit.py` — explains why uncovered tokens did not promote.
- `overlap_analysis.py` — inspects pairwise overlap between masked sets;
  family clusters are derived from the char-tool manifest.
- `summary.tsv` — one row per target key with masked/unmasked counts,
  mass coverage, and PMI range.
- `overlap_matrix.tsv` — pairwise intersection counts between masked
  token sets.
- `uncovered_tokens.tsv` — uncovered-token audit categories.
- `weights_used.json` — weights used for the diagnostic `pmi_training`
  column.
- `manifest.json` — provenance, input hashes, and current knobs.
- `tables/` — the 87 x 3 text token-set outputs.

`per_token_pmi.parquet` is intentionally not tracked in Git because it
is a binary audit artifact. The latest copy is published with the
Hugging Face artifact bundle for this run.

## Rebuild

Run from the repository root after regenerating the char-membership
artifacts and vocabulary histogram:

```bash
python3 subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/build.py
python3 subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/coverage_audit.py
python3 subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/analysis/main_token_sets_pmi/overlap_analysis.py
```

