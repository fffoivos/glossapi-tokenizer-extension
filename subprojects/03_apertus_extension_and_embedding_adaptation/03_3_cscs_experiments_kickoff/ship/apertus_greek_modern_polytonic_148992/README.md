# ModernGreek + Polytonic Greek 148,992

This is the production tokenizer selected on 2026-07-29 for tokenizing the
bibliography-cleaned Greek CPT corpus.

## Vocabulary contract

- Apertus base vocabulary: `131,072`
- Modern Greek continuation: `17,408`
- Polytonic/ancient Greek continuation: `512`
- Actual BPE vocabulary: `148,992 = 256 x 582`
- External padding entries: `0`
- Token IDs: contiguous `0..148,991`
- Appended merge rules: `512`
- Orphan appended vocabulary entries: `0`

Every new vocabulary entry is the ID-sequential output of one dependency-safe
appended BPE merge. The previous 148,480 vocabulary IDs and complete merge
prefix are byte-for-byte unchanged. The normalizer, pre-tokenizer,
post-processor, decoder, BPE options, and first 1,000 special/reserved tokens
are unchanged from the verified ModernGreek-148k bundle.

## Load from Hugging Face

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "fffoivos/apertus-tokenizer-extension",
    subfolder="greek-modern-polytonic-tokenizer",
)
assert tokenizer.vocab_size == 148_992
assert len(tokenizer) == 148_992
```

For reproducible corpus accounting, pin the exact Hugging Face commit reported
in the token-count plan rather than following `main`.

## Selection and audit

The precommitted gate selected the `+512` continuation over `+1,024`.
On the fixed evaluation streams, `+512` reduced ancient token count by `7.62%`
and kept modern BPB regression to `0.1375%`, inside the `0.5%` guard. The
`+1,024` arm did not earn the required ancient-BPB margin.

All suspicious ByteLevel fragments in the chosen cutoff were reviewed against
their merge descendants and real corpus firings. None remained unresolved or
represented standalone mojibake.

Evidence:

- `release_audit.json` — vocabulary, alignment, merge, front-end, sidecar, and
  Transformers runtime checks
- `selection.json` — calibrated model-probe decision
- `suspicious_token_review.json` — suspicious-token decisions and corpus
  firings

Tokenizer SHA-256:

```text
bbb08e71929b519c5c2362338b0fc6a0e99955cb8fdbf0729ae1311117e6561b
```
