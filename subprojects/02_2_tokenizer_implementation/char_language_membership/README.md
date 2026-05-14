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
- `languages.yaml` — 55-triple source of truth (bit assignments).
- `scripts/build_char_language_bitmask.py` — codepoint-level build.
- `scripts/apply_to_apertus_vocab.py` — token-level apply.
- `scripts/validate.py` — strict-rule sanity checks (phase 1 char,
  phase 2 token-level audit gate).
- `scripts/query_codepoint.py` — **read this first** before consuming
  the char-level parquet directly. The parquet is a sparse table; a
  direct lookup that returns 0 for a missing codepoint will
  false-reject. `query_codepoint.codepoint_bits()` reproduces the
  build-time and apply-time substrate-aware fallback, so any consumer
  gets the same semantics the apply script does.
- `data/cldr/<release>/` — cached CLDR JSON (gitignored).
- `artifacts/` — built tables.

## How the char parquet is meant to be read

`char_language_bitmask.parquet` is **sparse with fallback required**.
It stores:

- Every codepoint that received explicit language evidence (CLDR
  exemplar + case/NFD/script-range-fallback closures), and
- The substrate codepoints we explicitly seeded (ASCII printable, a
  small supplementary list, `EXTRA_SUBSTRATE_CODEPOINTS` —
  fullwidth Latin / digits, ordinal markers, MICRO SIGN, NBSP, …).

It does **not** store the vast majority of Unicode substrate
(arbitrary emoji, exotic supplementary-plane symbols, less common
punctuation). For those codepoints, a direct lookup returns nothing
and the consumer must apply the same fallback rule the build and
apply scripts use:

```python
# Pseudocode — see scripts/query_codepoint.py for the real thing
if cp in table:                                return table[cp]
if cp in EXTRA_SUBSTRATE_CODEPOINTS:            return ALL_BITS
cat = unicodedata.category(chr(cp))
if cat == "Lm" or cat[0] in "NPSZ" or cat in ("Cc", "Cf"):
                                                return ALL_BITS
return 0       # Letter/mark in a script we don't model — reject all
```

`scripts/query_codepoint.py` exposes `load()`, `codepoint_bits()`,
`token_bits_and()`, `token_bits_or()` so consumers don't have to
re-implement this. Use it.

The token parquet (`token_language_bitmask.parquet`) has no such
caveat — `apply_to_apertus_vocab.py` already applied the fallback
when computing each row's `bitmask_and` and `bitmask_or`.

## Bitmask storage

Both `bitmask` (char table) and `bitmask_and` / `bitmask_or` (token
table) are Parquet `binary(16)` columns — fixed-width 16-byte
little-endian buffers (128-bit budget, vs. today's 55 used bits).
Convert to a Python int with `int.from_bytes(buf, "little")` and back
with `mask.to_bytes(16, "little")`. We picked binary over uint64
because audit gaps already foreshadow extensions past 64 bits
(Kazakh / Belarusian / Mongolian Cyrillic, Pashto / Sindhi / Uyghur
Arabic, Khmer / Lao / Tibetan / Sinhala / Ethiopic scripts, etc.).

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
