# Loss Measurement Policy

This project trains and evaluates arms with different tokenizers:

- Vanilla: Apertus base vocab, 131,072 tokens.
- Extended arms: Greek extension vocab, 148,480 tokens.

Because the tokenizer changes both the output softmax size and the number of
tokens needed to represent the same text, raw Megatron `lm loss` is not a fair
head-to-head score across arms. It is per-target-token cross entropy in nats.
Use it for health checks, within-arm trends, optimizer stability, and detecting
spikes. Do not use it to decide whether Vanilla beats TD/ReTok/Centroid.

## Selection Metrics

The cross-tokenizer selection metrics are per-byte / per-character metrics on
the same heldout text:

- `BPC` / `BPB`: bits per UTF-8 byte, lower is better.
- `NLL per Unicode character`: character-normalized companion metric.
- `NLL per word`: human-facing language scale.
- `tokens/word`, `chars/token`, compression ratio, `STRR`: tokenizer-efficiency
  and whole-word preservation signals.

For current bakeoff and continuation runs, the authoritative tokenizer-fair
loss evidence comes from `compute_tokenizer_fair_metrics.py` at checkpoint
eval time. Sparse heldout BPC/BPB beats dense raw `lm loss` when they disagree.

## Dense Training Logs

For future Megatron runs, adopt in-flight tokenizer-fair logging alongside
`lm loss`. These are measurement-only fields and must not change the optimizer
loss:

```text
bpb: <bits_per_byte_batch> |
bpt: <bytes_per_token_batch> |
base_loss: <mean_ce_on_target_id_lt_base_vocab> |
new_loss: <mean_ce_on_target_id_ge_base_vocab> |
n_new: <count_new_target_positions> |
```

Implementation policy:

- Compute all metrics over the exact same loss-mask positions that contribute
  to `lm loss`. This includes EOD masking, padding masking, and Goldfish masking.
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

## Reading Existing Runs

Runs before this logging patch only have dense raw `lm loss`. For those runs:

- Treat `lm loss` as diagnostic only.
- Compare arms using heldout checkpoint BPC/BPB and downstream evals.
- If a plot shows raw `lm loss`, label it explicitly as unfair across tokenizers.
- Dense BPB proxies derived from average bytes/token are exploratory only; they
  are useful for intuition, not for final selection.
