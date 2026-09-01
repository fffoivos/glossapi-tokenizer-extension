---
license: other
library_name: transformers
tags:
  - apertus
  - greek
  - token-distillation
  - initialization
---

# Apertus 1.5B Token-Distillation initialization — 148,480 vocabulary

Private, reusable initialization checkpoint for the matched Apertus 1.5B
HPLT-to-OpenArchives continuation experiment. It is **not** a completed CPT
model and must not be presented as one.

It derives from `swiss-ai/Apertus-v1.1-1.5B` at revision
`dbe8919b2f0389888bada6b3a19e81e0ef4286c1`, with the vocabulary replaced by
the 148,480-token Modern-Greek extension from
`fffoivos/apertus-tokenizer-extension` at revision
`fcd33ec09fb7d86bc072b3a4b3e890efa6473b66`.

The checkpoint has untied input and output embeddings. Its training geometry
is fixed at a 4,096-token context, RoPE base 500,000, and Llama-3 RoPE scaling
factor 8.0. The release transaction verifies the full checkpoint inventory,
tokenizer SHA-256, tensor shapes, embedding geometry, and exact
HF-to-Megatron-to-HF tensor and logits round trip before publication.
The emitted tokenizer serialization is byte-bound in the release receipt; the
serialization-binding note records the semantic-equivalence check against the
1.5B training-geometry serializer.

Use this artifact only as the pinned initialization for the associated CPT
recipe. It is private because it is an experimental intermediate artifact.
