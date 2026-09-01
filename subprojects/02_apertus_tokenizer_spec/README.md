# 02 — Apertus Tokenizer Spec

> **In one line:** a two-file spec that pinned the exact `swiss-ai/Apertus-8B-2509` tokenizer behaviour every later extension had to reproduce; written once in April 2026, never revised, and it held all the way to the production CPT tokenizer.
> **Period:** 2026-04-10 → 2026-04-14 (commits `f21eed85` … `a062d0aa`). **Status:** completed (spec frozen); the six verification tasks in [`TODO.md`](TODO.md) were never closed *in this directory* — they were satisfied downstream instead.
> **Came from / led to:** repo restructure (`f21eed85`) → this → [`../02_1_tokenizer_experiments/`](../02_1_tokenizer_experiments/README.md) and [`../02_2_tokenizer_implementation/`](../02_2_tokenizer_implementation/README.md)

## Why this existed

The whole program rests on *extending* Apertus's tokenizer rather than replacing it. If the extended tokenizer's front end diverged from Apertus in any way — a different normalizer, a different pre-tokenizer regex, a renumbered special-token block — then every downstream claim about embedding adaptation and continued pretraining would be measuring the divergence rather than the Greek. This subproject wrote down what "identical to Apertus" means, as a checkable list, before any tokenizer was trained.

## History

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| 2026-04-10 | Directory created in the subproject restructure; `README.md` and `TODO.md` written | Contract pinned: BPE family, base vocab `131072`, fixed front block of the first `1000` ids, `add_bos_token=true`, `add_eos_token=false`, `add_prefix_space=false`, no normalizer, regex split + `ByteLevel` pre-tokenization, `model.ignore_merges=true`, `tie_word_embeddings=false`, HF fast-tokenizer artifacts as the consumer format | `f21eed85` |
| 2026-04-12 | One-line edit to `TODO.md` during the parallel tokenizer/HPLT re-plan | No change to the contract | `0a8a50be` (1 insertion, 1 deletion) |
| 2026-04-14 | Two verification tasks added to `TODO.md`: a real-document Greek smoke run, and a toy merge-rule extension proof exercising `model.vocab`/`model.merges` without `add_tokens(...)` | Verification scope widened; `README.md` untouched | `a062d0aa` |

`README.md` has never been edited since 2026-04-10. That is the whole history of this directory — the spec was right the first time.

## Outcome

- The contract became the acceptance test used everywhere downstream: the continuous-BPE trainer emits `front_end_contract_check.json` and `replication_check.json` per arm ([`../02_1_tokenizer_experiments/02_1_1_tokenizer_training/README.md`](../02_1_tokenizer_experiments/02_1_1_tokenizer_training/README.md)), every cutoff variant inherits it by copying the front-end JSON byte-for-byte ([`../02_1_tokenizer_experiments/02_1_2_cutoff_variant_builder/README.md`](../02_1_tokenizer_experiments/02_1_2_cutoff_variant_builder/README.md)), and the ship bundles were re-checked constraint by constraint in [`../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md`](../03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/SHIP_TOKENIZER_RECONSTRUCTION.md) §4 (first-1000 ids: 0 mismatches; normalizer/pre-tokenizer/decoder byte-identical to Apertus).
- Independently corroborated in 2026-05-17 by a side-by-side diff against `mistralai/Mistral-Nemo-Base-2407`, which confirmed each field value and found that Apertus inherited Mistral's BPE table verbatim, changing only the front block (514 → 1000 reserved ids) and truncating 486 trailing BPE entries — [`../02_1_tokenizer_experiments/02_1_6_representation_policy_analysis/11_tokenizer_provenance.md`](../02_1_tokenizer_experiments/02_1_6_representation_policy_analysis/11_tokenizer_provenance.md).
- The two artifacts the TODO asked for and this directory never produced — a canonical file holding the literal split regex, and a standalone round-trip compatibility test — do not exist here. The regex is transcribed in `11_tokenizer_provenance.md`; the round-trip check was run ad hoc as part of the ship-bundle verification, not as a committed test.

## Working documents

- [`TODO.md`](TODO.md) — the six verification tasks (regex file, post-processor template, special-token inventory, round-trip test, real-document smoke run, merge-rule extension proof). Historical; none were closed in this directory.
