# 02.2 Tokenizer Implementation

## Scope

Implement the actual compatible Greek `BPE` discovery and merge-rule extension workflow.

## Already Decided

- do not ship `add_tokens(...)`
- patch Apertus through `model.vocab` and `model.merges`
- preserve all old token ids
- append only new ids
- emit a manifest of every newly added unit
- enforce final vocab divisibility by `128`

## Required Checks

- exact preservation of the first `1000` ids
- exact preservation of special-token behavior
- exact preservation of regex split and byte-level behavior
- non-Greek smoke test after extension

