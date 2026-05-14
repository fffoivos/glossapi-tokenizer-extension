# char_language_membership

Sub-subproject of `02_2_tokenizer_implementation`.

## Goal

For every Apertus vocab token, emit a 55-bit mask identifying which
**(language, script, encoding)** triples the token's chars are
*compatible with*. The bitmask answers: **which triples can we
rule out for this token?**

Built for rejection, not classification. We never assign a token to
a single language; we tell the consumer "every triple not in this
bitmask is excluded with confidence".

See `PLAN.md` for the full design — scope, proxy assumption for
Mistral-Nemo's (private) tokenizer-training data, strict rule, the
four closures (script-filter, case, NFD, script-range fallback),
substrate-all-bits, validation requirement.

## Inputs

- **Scope** — 55 (language, script, encoding) triples defined in
  `languages.yaml`. Derived from Apertus's documented pretrain mix
  (`docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`) used as a
  proxy for Mistral-Nemo's unpublished tokenizer-training language
  list. See `PLAN.md § How we approach the missing-tokenizer-training-
  data problem`.
- **Per-locale character data** — CLDR `characters.json` subsets
  (`exemplarCharacters`, `index`, `numbers`, `punctuation`) from
  cldr-json (release pinned in `languages.yaml`). Fetched once, cached
  under `data/cldr/<release>/`.

## Outputs

- `artifacts/char_language_bitmask.parquet` — codepoint → 55-bit mask.
- `artifacts/token_language_bitmask.parquet` — token → AND/OR masks +
  status. Built against the Apertus snapshot under
  `~/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/`.
- `artifacts/manifest.json` — char-build provenance: bit assignments,
  CLDR release, closures applied, proxy-assumption note.
- `artifacts/token_manifest.json` — apply-step provenance: Apertus
  snapshot path + revision SHA, char-table build timestamp, status
  counts, AND-popcount distribution.

## Files

- `PLAN.md` — full design and rationale.
- `TODO.md` — open work.
- `languages.yaml` — 54-triple source of truth (bit assignments).
- `scripts/build_char_language_bitmask.py` — codepoint-level build.
- `scripts/apply_to_apertus_vocab.py` — token-level apply.
- `scripts/validate.py` — strict-rule sanity checks.
- `data/cldr/<release>/` — cached CLDR JSON (gitignored).
- `artifacts/` — built tables.

## Run

```bash
PROJECT=/home/foivos/Projects/glossapi-tokenizer-extension
SNAP=/home/foivos/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/snapshots/3162c99675aa588097cecd4a24b9aa1f712af477

$PROJECT/.venv-hplt-review/bin/python scripts/build_char_language_bitmask.py
$PROJECT/.venv-hplt-review/bin/python scripts/apply_to_apertus_vocab.py --apertus-snapshot $SNAP
$PROJECT/.venv-hplt-review/bin/python scripts/validate.py
```

## Dependencies

- `pyyaml`, `pyarrow` — standard Python tooling.
- No system deps: per-locale exemplar character sets are fetched
  from the cldr-json repo over HTTP and parsed in pure Python. The
  pinned release in `languages.yaml` makes the build reproducible.
