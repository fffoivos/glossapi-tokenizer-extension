# Loss Measurement Policy

This is the repo-wide rule for reading loss in Apertus tokenizer-extension and
model-adaptation experiments.

## Short Rule

Do not compare Vanilla, ReTok, TokenDistil, or Centroid with raw Megatron
`lm loss` when the compared arms use different tokenizers. Megatron `lm loss`
is per-target-token cross entropy in nats, so it changes with softmax size and
with how many bytes each tokenizer packs into one token.

Cross-tokenizer conclusions use heldout tokenizer-fair BPB plus downstream
benchmarks. Raw `lm loss` is health telemetry and within-arm trajectory only.

## Metric Hierarchy

1. **Selection loss:** heldout `BPB` from
   `compute_tokenizer_fair_metrics.py`, measured on the same scored text.
   Lower is better.
2. **Dense training telemetry:** when patched Megatron logs include `bpb`,
   `bpt`, `base_loss`, `new_loss`, and `n_new`, use those for in-flight
   diagnostics. They must be computed over the same loss-mask positions as
   optimizer `lm loss`; positions masked out by EOD, padding, or Goldfish
   must not enter these measurement denominators.
3. **Raw training loss:** `lm loss` stays useful for instability, spikes,
   NaNs/skips, and within-arm trends. It is not a Vanilla-vs-extended
   scoreboard.
4. **Legacy naming:** old reports and JSON keys may say `BPC` or
   `bpc_bits_per_byte`. In this project that is a historical alias for bits
   per UTF-8 byte, not bits per Unicode character. New scripts should prefer
   `BPB` and `bpb_bits_per_byte` while reading the legacy alias.

## Script Contract

- Metric JSONs should write `bpb_bits_per_byte` and may also write the legacy
  `bpc_bits_per_byte` compatibility alias.
- Training-log parsers should read `bpb` / `bpb_batch` /
  `bits_per_byte_batch`, `bpt` / `bytes_per_token_batch`, and the base/new
  target split fields.
- When `bpb` and `bpt` are both logged, parsers should be able to verify:
  `bpb ~= lm_loss / ln(2) / bpt`.
- Vanilla should have `n_new=0` and `base_loss ~= lm_loss`; extended arms
  should have nonzero `n_new` on the Greek-heavy CPT mix.
- Plots and reports must label raw `lm loss` as diagnostic-only whenever
  multiple tokenizer vocabularies are present.

The implementation-level canonical file for the current Apertus bakeoff is
[`../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md`](../subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/LOSS_MEASUREMENT_POLICY.md).
