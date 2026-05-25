# Modern Greek Tokenizer, 148480 Vocab

This is the canonical modern Greek tokenizer extension for Apertus.

Properties:

- base model: `swiss-ai/Apertus-8B-2509`;
- method: merge-rule BPE extension, not `add_tokens`;
- base vocab preserved: `131072`;
- added modern Greek tokens: `17408`;
- total vocab: `148480`;
- alignment: divisible by `128` and `256`;
- first 1000 special/reserved ids preserved;
- `tokenizer.json` SHA-256:
  `358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`.

Selection evidence is in:

```text
../../provenance/tokenizer-selection/CHOSEN_CUTOFF.md
```

