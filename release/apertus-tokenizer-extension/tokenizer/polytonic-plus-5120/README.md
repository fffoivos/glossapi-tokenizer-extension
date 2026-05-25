# Polytonic-Stacked Tokenizer, 153600 Vocab

This tokenizer stacks an additional polytonic/ancient Greek extension on top of
the modern Greek 148480-vocab tokenizer.

Properties:

- base model: `swiss-ai/Apertus-8B-2509`;
- base vocab preserved: `131072`;
- added modern Greek tokens: `17408`;
- added polytonic/ancient Greek tokens: `5120`;
- total vocab: `153600`;
- alignment: divisible by `128` and `256`;
- first 1000 special/reserved ids preserved;
- `tokenizer.json` SHA-256:
  `b1eeb739a564b3abd33c1b85a16162b8284d98f9ab5d67528d3cbe8a82e9cbad`.

This tokenizer is not the one used for the 3.5B TD layer 11 continuation. That
run used the modern-only tokenizer.

