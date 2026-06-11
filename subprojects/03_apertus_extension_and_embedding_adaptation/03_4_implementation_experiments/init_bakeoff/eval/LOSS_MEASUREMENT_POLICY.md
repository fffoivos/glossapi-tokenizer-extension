# Loss Measurement Policy

This project trains and evaluates arms with different tokenizers:

- Vanilla: Apertus base vocab, 131,072 tokens.
- Extended arms: Greek extension vocab, 148,480 tokens.

Because the tokenizer changes both the output softmax size and the number of
tokens needed to represent the same text, raw Megatron `lm loss` is not a fair
head-to-head score across arms. It is per-target-token cross entropy in nats.
Use it for health checks, within-arm trends, optimizer stability, and detecting
spikes. Do not use it to decide whether Vanilla beats TD/ReTok/Centroid.

## Metric Hierarchy

1. Heldout checkpoint `BPB` plus downstream evals are the selection evidence.
2. Dense training-log `bpb` is useful in-flight telemetry when the Megatron
   logging patch is present, but it is still batch-local and does not replace
   heldout checkpoint BPB.
3. Raw `lm loss` is health and within-tokenizer trajectory telemetry only.
4. Legacy `BPC` / `bpc_bits_per_byte` names are read as BPB aliases.

## Selection Metrics

The cross-tokenizer selection metrics are per-byte / per-character metrics on
the same heldout text:

- `BPB`: bits per UTF-8 byte, lower is better. This is the canonical loss
  metric for Vanilla-vs-extended comparisons.
- `BPC`: legacy label used by earlier artifacts for the same bits-per-byte
  quantity (`bpc_bits_per_byte`). Do not read it as bits per Unicode character.
- `NLL per Unicode character`: character-normalized companion metric.
- `NLL per word`: human-facing language scale.
- `tokens/word`, `chars/token`, compression ratio, `STRR`: tokenizer-efficiency
  and whole-word preservation signals.

For current bakeoff and continuation runs, the authoritative tokenizer-fair
loss evidence comes from `compute_tokenizer_fair_metrics.py` at checkpoint
eval time. Sparse heldout BPB beats dense raw `lm loss` when they disagree.
The metrics JSON now writes both `bpb_bits_per_byte` and the historical
`bpc_bits_per_byte` alias for compatibility.

## Dense Training Logs

For Megatron runs with the logging patch, adopt in-flight tokenizer-fair logging
alongside `lm loss`. These are measurement-only fields and must not change the
optimizer loss:

```text
bpb: <bits_per_byte_batch> |   # aliases: bpb_batch, bits_per_byte_batch
bpt: <bytes_per_token_batch> | # alias: bytes_per_token_batch
base_loss: <mean_ce_on_target_id_lt_base_vocab> | # alias: base_target_loss
new_loss: <mean_ce_on_target_id_ge_base_vocab> |  # alias: new_target_loss
n_new: <count_new_target_positions> |
```

Implementation policy:

- Compute all metrics over the exact same loss-mask positions that contribute
  to `lm loss`. Positions masked out by EOD, padding, or Goldfish are excluded
  from the denominators and from the base/new split.
- Reduce numerators and denominators across context-parallel and data-parallel
  ranks exactly like `lm loss`, so printed values are global for the training
  step, not local to one shard.
- `bpt` is the mean UTF-8 bytes per target token over active loss positions.
  The cheap implementation is a precomputed `bytes_per_id` tensor gathered by
  `labels`.
- `bpb = (lm_loss_nats / ln(2)) / bpt`.
- `base_loss` and `new_loss` split the same per-position CE by target ID. Use
  `base_vocab_size=131072` for Apertus 8B. Vanilla has no new target positions;
  emit `n_new=0` and leave `new_loss` empty/NaN/0 consistently.

Verification before production:

- Run a short Vanilla and TD smoke after the patch.
- Check `abs((lm_loss / ln(2) / bpt) - bpb) < 0.001` on logged rows.
- Check Vanilla has `n_new=0` and `base_loss ~= lm_loss`.
- Check TD/ReTok have nonzero `n_new` on the Greek-heavy CPT mix.
- Keep the optimizer path unchanged: these fields are emitted after the loss
  values exist and must not feed back into gradient computation.
- Label any raw `lm loss` plot as diagnostic-only if it contains both Vanilla
  and extended-tokenizer arms.

## Reading Existing Runs

Runs before this logging patch only have dense raw `lm loss`. For those runs:

- Treat `lm loss` as diagnostic only.
- Compare arms using heldout checkpoint BPB and downstream evals.
- If a plot shows raw `lm loss`, label it explicitly as unfair across tokenizers.
- Dense BPB proxies derived from average bytes/token are exploratory only; they
  are useful for intuition, not for final selection.
