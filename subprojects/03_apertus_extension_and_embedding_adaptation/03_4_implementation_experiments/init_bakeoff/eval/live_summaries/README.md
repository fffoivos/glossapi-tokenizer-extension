# Live Summaries Metric Labels

This directory contains historical copied eval artifacts. The Markdown tables
have been relabeled to the canonical project name, `BPB`, meaning bits per
UTF-8 byte. Some underlying JSON files still use the legacy compatibility key
`bpc_bits_per_byte`.

Read these as:

- `BPB` in historical Markdown tables = bits per UTF-8 byte, lower is better.
- `bpc_bits_per_byte` in historical JSON = legacy alias for
  `bpb_bits_per_byte`.
- `NLL/char` remains the separate Unicode-character-normalized companion
  metric.
- Raw Megatron `lm loss` is per-token CE and is diagnostic only across
  different tokenizer vocabularies.

The source scripts now emit and prefer BPB labels while preserving readers for
legacy fields.
