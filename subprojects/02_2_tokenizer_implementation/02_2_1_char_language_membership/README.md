# 02_2_1_char_language_membership

Sub-subproject of `02_2_tokenizer_implementation`.

## Goal

For every Apertus vocab token, emit **three parallel bitmasks** that
identify which **(script, language-family, language)** triples the
token's chars are compatible with. Each layer answers the same
question — *which categories can we rule out for this token?* — at a
different resolution:

| layer | bits used | example for bare-ASCII ` the` | example for `ß`-containing token |
|---|---|---|---|
| **language** | 55 | all 28 Latin locales | only `de` |
| **family** | 31 | all 8 Latin families | only `Germanic-Latn` |
| **script** | 22 | only `Latn` | only `Latn` |

A token's chars that are language-discriminating (`ñ`, `ß`, `ł`, `中`,
Greek polytonic) narrow at every layer simultaneously; chars that
aren't (bare ASCII, `α` shared between Greek encodings, Han chars
shared across CJK) narrow only at the coarser layers.

Built for **rejection**, not classification. We never assign a token
to a single language; we tell the consumer "every triple not in this
bitmask is excluded with confidence". Three layers means the
consumer gets the strongest available rejection at any resolution
they care about. See `PLAN_v3_HIERARCHICAL.md` for full rationale.

## Scope semantics — what the bits mean (and don't)

Every bit at every layer means **"some in-scope locale admits this
char"** — positive CLDR evidence (plus the documented closures: case,
NFD, script-range fallback, post-fallback NFD, substrate override).

Crucially, the **script layer is not a Unicode-script detector**.
A Latin codepoint outside every in-scope locale's CLDR — `ō`
(U+014D, used in Japanese romaji, Latvian, Hawaiian, Māori; none in
our scope) — gets **zero bits at every level**. Token AND collapses
to 0, which rejects every in-scope (script, family, language). That
is the desired strict behaviour. Consumers wanting "what Unicode
script is this codepoint" should call `unicodedata.name(chr(cp))`
and look at the prefix — that's a different question with its own
fast answer; we don't bake it in here.

Rationale (from `PLAN_v3_HIERARCHICAL.md § Why projection-only`):

1. The whole artifact is "what we model and can reject." Adding a
   free Unicode-script-detection axis on top would mix semantics —
   `script_and` would mean a different kind of thing than
   `family_and` and `bitmask_and`.
2. Consumers wanting broad Unicode-script detection can compute it
   themselves in a few lines. The bitmask's job is to express which
   in-scope categories admit each char.
3. It is the conservative extension of v2.2. v2.2 had no script
   layer; every bit was positive language evidence. v3 adds script
   and family as projections of that same evidence — no new
   inclusion rules.

## Inputs

- **Scope** — 55 (language, script, encoding) triples in
  `languages.yaml`, 31 families in `families.yaml`, 22 scripts in
  `scripts.yaml`. Derived from Apertus's documented pretrain mix
  (`docs/APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`) used as a
  proxy for Mistral-Nemo's unpublished tokenizer-training language
  list. See `PLAN_v3_HIERARCHICAL.md § Resolved decisions`.
- **Per-locale character data** — CLDR `characters.json` subsets
  (`exemplarCharacters`, `index`, `numbers`, `punctuation`) from
  cldr-json (release pinned in `languages.yaml`). Fetched once,
  cached under `data/cldr/<release>/`.

## Outputs (schema v4)

`artifacts/char_language_bitmask.parquet` — one row per codepoint:

| column | type | meaning |
|---|---|---|
| `codepoint` | `uint32` | Unicode scalar value |
| `script_bits` | `binary(16)` | 22-bit script mask, little-endian |
| `family_bits` | `binary(16)` | 31-bit family mask, little-endian |
| `bitmask` | `binary(16)` | 55-bit language mask, little-endian |
| `char` | `string` | the character itself, for inspection |
| `num_langs` | `uint8` | popcount of `bitmask` |
| `category` | `string` | Unicode general category |

`artifacts/token_language_bitmask.parquet` — one row per Apertus
vocab token (131,072 rows):

| column | type | meaning |
|---|---|---|
| `token_id` | `uint32` | Apertus token id |
| `token_bytes` | `binary` | decoded raw bytes (post ByteLevel inversion) |
| `decoded_text` | `string` | UTF-8 decode, or `null` if invalid |
| `script_and` / `script_or` | `binary(16)` | script-layer AND / OR across token chars |
| `family_and` / `family_or` | `binary(16)` | family-layer AND / OR |
| `bitmask_and` / `bitmask_or` | `binary(16)` | language-layer AND / OR |
| `num_chars` | `uint16` | decoded codepoint count |
| `status` | `string` | `text` / `text_with_unmodeled_letters` / `no_in_scope_chars` / `partial_utf8` / `byte_unmapped` / `special` |

Manifests:

- `artifacts/manifest.json` — char-build provenance: bit assignments
  at all three layers, CLDR release, closures applied, proxy-
  assumption note. `schema_version: 4`.
- `artifacts/token_manifest.json` — apply-step provenance: Apertus
  snapshot path + revision SHA, char-table build timestamp, status
  counts, per-layer AND-popcount distributions.

All mask columns are uniform `binary(16)` — see
`PLAN_v3_HIERARCHICAL.md § Storage / schema` for the rationale (one
decode rule for every column; negligible storage cost vs uneven
widths; wire-format headroom for future audit-driven additions).

## How the parquets are meant to be read

`char_language_bitmask.parquet` is **sparse with fallback required**.
It stores every codepoint with positive language evidence plus
explicitly seeded substrate (ASCII, the small supplementary list,
`EXTRA_SUBSTRATE_CODEPOINTS`). It does **not** store the vast
majority of Unicode substrate. For codepoints not in the table the
consumer must apply the same substrate-aware fallback rule the build
and apply scripts use:

```python
# Pseudocode — see scripts/query_codepoint.py for the real thing
if cp in table:                                       return table[cp]
if cp in EXTRA_SUBSTRATE_CODEPOINTS:                  return ALL_BITS
cat = unicodedata.category(chr(cp))
if cat == "Lm" or cat[0] in "NPSZ" or cat in ("Cc", "Cf"):
                                                       return ALL_BITS
return 0   # Letter/mark in a script we don't model — reject all
```

`scripts/query_codepoint.py` exposes the consumer-facing helpers:

- `load(parquet, languages_yaml)` → `(lang_table, all_lang_bits)`
  for backwards-compat single-layer usage.
- `load_all(parquet, languages_yaml, families_yaml, scripts_yaml)` →
  dict with all three tables and their ALL_BITS values.
- `codepoint_bits(cp, table, all_bits)` → mask int, applying the
  fallback above. Use this — don't re-implement.
- `token_bits_and(text, table, all_bits)` /
  `token_bits_or(text, table, all_bits)` — single-layer aggregation.
- `token_bits_and_three(text, loaded)` /
  `token_bits_or_three(text, loaded)` — three-layer aggregation,
  returns `(script, family, language)` tuples.

`token_language_bitmask.parquet` has no sparse-table caveat —
`apply_to_apertus_vocab.py` already applied the substrate fallback
when computing every row's six mask columns. Read them directly.

## Bitmask decode

Every mask column is little-endian fixed-width binary:

```python
mask_int = int.from_bytes(row["bitmask"], "little")
encoded = mask_int.to_bytes(16, "little")
```

Bit positions are stable wire identifiers — never reused. See
`scripts.yaml` / `families.yaml` / `languages.yaml` for the
authoritative bit assignments at each level.

## Files

- `PLAN_v3_HIERARCHICAL.md` — **active design** (v4 schema). Read
  first.
- `PLAN.md` — preserved v2.2 design (language-layer-only); useful
  background for the closures and substrate rule that v3 extends.
- `TODO.md` — open work.
- `scripts.yaml` — 22-script source of truth.
- `families.yaml` — 31-family source of truth.
- `languages.yaml` — 55-locale source of truth.
- `scripts/_common.py` — shared constants and helpers (bitmask
  encoding, substrate rule, derivation functions).
- `scripts/build_char_language_bitmask.py` — codepoint-level build.
- `scripts/apply_to_apertus_vocab.py` — token-level apply.
- `scripts/validate.py` — phase 1 char checks + derivation
  consistency, phase 2 token wire-format + recompute gate +
  out-of-scope audit gate.
- `scripts/query_codepoint.py` — **consumer-facing entrypoint**.
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
