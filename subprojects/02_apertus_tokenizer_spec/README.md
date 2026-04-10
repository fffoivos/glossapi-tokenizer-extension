# 02 Apertus Tokenizer Spec

## Scope

Pin down the exact Apertus tokenizer behavior that must be reproduced.

## Already Confirmed

- tokenizer family: `BPE`
- base vocab size: `131072`
- fixed front block: first `1000` ids
- `add_bos_token = true`
- `add_eos_token = false`
- `add_prefix_space = false`
- no normalizer
- regex split plus `ByteLevel` pretokenization
- `model.ignore_merges = true`
- `tie_word_embeddings = false`
- Hugging Face fast-tokenizer artifacts are the consumer-facing format

## Why This Matters

If tokenizer behavior diverges from Apertus, any extension and embedding retraining path becomes harder to trust.

