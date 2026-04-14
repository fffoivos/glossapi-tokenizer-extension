# TODO

- pin the literal split regex in a canonical file
- pin the exact post-processor template
- pin the full special-token inventory and ids
- add a toy compatibility test that proves a patched tokenizer still round-trips correctly under the HF stack
- add a tiny real-document tokenizer smoke run so the Apertus-compatible path is checked on real Greek text, not only toy strings
- add a toy merge-rule extension proof that exercises `model.vocab` and `model.merges` without using `add_tokens(...)`
