# TD initialization release tokenizer binding

The private 8B and 1.5B Token-Distillation initialization releases bind the
raw SHA-256 of the emitted `hf_roundtrip/tokenizer.json`:

```text
37c110e765160f64a22bf913f714a40744c84208d0ec22d8f22b8232b1923c34
```

This is the byte-level value the canonical checkpoint release verifier checks.
The 8B training-geometry tokenizer has the same bytes. The older 1.5B
training-geometry artifact records a compact JSON serialization with SHA-256
`358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`.
The two JSON values were independently parsed and found exactly equal: no
vocabulary, merge, normalizer, or special-token value differs. The 1.5B
round-trip receipt remains the evidence for the initialized tensors and logits;
the release contract intentionally binds the emitted artifact's exact bytes.

Both releases retain the pinned tokenizer repository and revision:

```text
fffoivos/apertus-tokenizer-extension@fcd33ec09fb7d86bc072b3a4b3e890efa6473b66
```
