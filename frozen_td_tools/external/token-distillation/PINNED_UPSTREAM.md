# Pinned Upstream Token Distillation

Vendored source: `konstantinjdobler/token-distillation`

- Upstream URL: `https://github.com/konstantinjdobler/token-distillation`
- Pinned commit: `35702b5809599ecd68b7845eca27a0d7b7cec0da`
- License: MIT, copied in `LICENSE`
- Vendored subset:
  - `token_distillation/`
  - `pyproject.toml`
  - package `README.md`
  - `LICENSE`

Excluded from the vendored copy: upstream `.git`, CI metadata, paper assets,
large lockfiles, examples, and paper-only scripts. The runtime code we need for
the Apertus adapter is the package implementation, especially
`token_distillation.train_loop.train_embeddings` and
`token_distillation.ahocorasick`.

For our production path, do not call the high-level
`TokenDistillation.run(...)` entry point directly. It calls
`target_tokenizer.add_tokens(...)`, while the Apertus ReTok arm uses an exact
merge-extended tokenizer with fixed IDs. The adapter should load the shipped
student tokenizer and checkpoint directly, build the phrase-to-new-ID mapping
from `[131072, 148480)`, and then call the lower-level training loop.
