# Greek — per-script research notes

> One of two encodings (`el` modern monotonic, `el-polyton` polytonic)
> in our scope. Central to the Apertus extension project. Status:
> coverage verified against authoritative sources; no `languages.yaml`
> changes required.

## Sources consulted

- CLDR cldr-misc-full 48.2.0 — `el/characters.json`, `el-polyton/characters.json`.
- Apertus tech-report §2.2 (tokenizer is Mistral-Nemo tekken v3,
  `normalizer: null`).
- CLAUDE.md "Empirical Apertus tokenizer facts" — verified against
  the cached vocab.
- 1982 Greek government decree on monotonic orthography (Π.Δ.
  207/9-3-1982) — defines the modern monotonic system: a single
  acute accent (tonos) and dialytika; eliminates all polytonic
  breathings and the grave / circumflex.
- Wikipedia: "Greek alphabet", "Polytonic orthography", "Greek
  numerals".
- TLG (Thesaurus Linguae Graecae) — UC Irvine; the standard
  polytonic-encoding reference for classical Greek.
- Unicode 16.0 Greek and Coptic block (U+0370–03FF) and Greek
  Extended block (U+1F00–1FFF).

## Empirical Apertus baseline

From CLAUDE.md, verified against the snapshot at
`~/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/`:

- Single-token vocab hits for Greek: bare `α` (U+03B1), monotonic
  `ά` (U+03AC), final sigma `ς` (U+03C2). All modern monotonic.
- **No single-token merges** for any polytonic codepoint —
  `ἄ` (U+1F04 psili+oxia), `ᾴ` (U+1FB4 tonos+ypogegrammeni),
  oxia `ά` (U+1F71), combining acute U+0301, or NFD-decomposed
  α + ◌́.

My audit of the v3 token table confirms this empirically:

- 1,507 Apertus vocab tokens contain at least one Greek codepoint
  (block U+0370–03FF or U+1F00–1FFF).
- All 1,507 are `status: text`. None are polytonic-containing.
- 0 tokens have `bitmask_and` set to `el-polyton` only (the encoding-
  discriminating case).
- 1,501 of the 1,507 have **both** `el` and `el-polyton` language bits
  set in their `bitmask_and` — i.e. every Greek token in Apertus is
  admissible in both modern and polytonic encoding because no token's
  chars carry the polytonic encoding's discriminating features.

**Consequence**: the `el` vs `el-polyton` split is char-level
informative (the codepoint `ἀ` U+1F00 has only the polytonic bit;
modern `α` has both) but **token-level vacuous on the inherited
Mistral-Nemo vocab**. Adding polytonic content via the C3 extension
work in this repo will populate the polytonic-only column with
actual signal — at that point the split becomes empirically useful
for tokens too.

## Coverage state

Per-codepoint audit of every Unicode codepoint in U+0370–03FF and
U+1F00–1FFF against our char table:

| block | block size | in table | not in table |
|---|---|---|---|
| Greek and Coptic (U+0370–03FF) | 144 | 127 | 17 |
| Greek Extended (U+1F00–1FFF) | 256 | 218 | 38 |

The 55 "not in table" codepoints fall into three groups:

### Group A — unassigned Unicode (skip)

`U+0378 U+0379 U+0380–0383 U+038B U+038D U+03A2 U+1F16 U+1F17 U+1F1E
U+1F1F U+1F46 U+1F47 U+1F4E U+1F4F U+1F58 U+1F5A U+1F5C U+1F5E U+1F7E
U+1F7F U+1FB5 U+1FC5 U+1FD4 U+1FD5 U+1FDC U+1FF0 U+1FF1 U+1FF5 U+1FFF`

These are reserved-or-unassigned Unicode codepoints. Correctly
absent.

### Group B — Greek-specific punctuation & marks, all substrate-category

These all carry the correct semantics via the apply-time substrate
fallback (`_common.codepoint_bits`) — Unicode category is `Lm`, `Sk`,
or `Po`, which the rule treats as ALL_BITS. The char table itself
doesn't store them; consumers using `query_codepoint.py` get the
right answer; consumers reading the parquet directly without the
fallback would get 0 (false rejection — documented in README as the
"sparse with fallback required" contract).

Modern-Greek-text-relevant:

| codepoint | char | name | category | apply-time bits |
|---|---|---|---|---|
| U+0374 | `ʹ` | GREEK NUMERAL SIGN | Lm | ALL_BITS |
| U+0375 | `͵` | GREEK LOWER NUMERAL SIGN | Sk | ALL_BITS |
| U+037A | `ͺ` | GREEK YPOGEGRAMMENI (combining iota subscript) | Lm | ALL_BITS |
| U+037E | `;` | GREEK QUESTION MARK | Po | ALL_BITS |
| U+0384 | `΄` | GREEK TONOS | Sk | ALL_BITS |
| U+0385 | `΅` | GREEK DIALYTIKA TONOS | Sk | ALL_BITS |
| U+0387 | `·` | GREEK ANO TELEIA (Greek high-dot, semicolon-equivalent) | Po | ALL_BITS |

Polytonic standalone breathing/accent marks (all category `Sk`,
all → ALL_BITS via substrate):

`U+1FBD KORONIS, U+1FBF PSILI, U+1FC0 PERISPOMENI, U+1FC1 DIALYTIKA AND
PERISPOMENI, U+1FCD PSILI AND VARIA, U+1FCE PSILI AND OXIA, U+1FCF
PSILI AND PERISPOMENI, U+1FDD DASIA AND VARIA, U+1FDE DASIA AND OXIA,
U+1FDF DASIA AND PERISPOMENI, U+1FED DIALYTIKA AND VARIA, U+1FEE
DIALYTIKA AND OXIA, U+1FEF VARIA, U+1FFD OXIA, U+1FFE DASIA`.

All correct under strict-rejection-with-substrate semantics — they
contribute zero exclusion power because every script's text could
contain them (in classical quotations, linguistic discussion,
etc.).

### Group C — Greek-block symbol that doesn't apply to our scope

`U+03F6 ϶ GREEK REVERSED LUNATE EPSILON SYMBOL` — used in
mathematical notation. Category `Sm`. Substrate via the rule.

## Coverage of in-text Greek codepoints (Apertus vocab)

60 unique Greek codepoints actually appear in Apertus vocab tokens.
**All 60 are in our table with both `el` and `el-polyton` bits set
correctly.** No coverage gap at the codepoint level. The CLAUDE.md
empirical-facts list (bare α, monotonic ά, final sigma, etc.) all
verified.

Capital letters with tonos — `Ά Έ Ή Ί Ό Ύ Ώ` — are present via case
closure (the v2.1 fix). CLDR `el` exemplar has lowercase tonos forms
only; case closure adds the precomposed uppercase precursors.

Greek numeral / archaic letters used as numerals — `Ϛ ϛ` (stigma =
6), `Ϝ ϝ` (digamma = 6 archaic), `Ϟ ϟ` (koppa = 90), `Ϡ ϡ` (sampi =
900) — are all in the table with both Greek bits. Picked up by the
script-range fallback over the basic Greek block.

## Decisions

1. **No changes to `languages.yaml` required.** CLDR + closures
   already cover everything that appears in the Apertus vocab and
   everything that needs language-evidence treatment.
2. **No changes to `EXTRA_SUBSTRATE_CODEPOINTS` recommended.** The
   ~22 missing Greek punctuation/mark codepoints (U+0384, U+0387,
   U+1FBD…1FFE, etc.) are all correctly handled by the existing
   substrate rule (Unicode category Lm/Sk/Po → ALL_BITS at apply
   time). Adding them to the explicit seed list would set precedent
   for enumerating every script's script-specific substrate, which
   would balloon the seed. Consumers reading the parquet directly
   need to apply the fallback rule (or use `query_codepoint.py`);
   that contract is documented in README § "How the parquets are
   meant to be read".
3. **The el / el-polyton split stays.** It's vacuous for the
   inherited Mistral-Nemo Apertus vocab but the C3-extension work
   shipping in this repo will introduce polytonic-containing tokens
   the split is designed to discriminate.

## Sanity-check assertions (already in validate.py)

- `α` (U+03B1) → `el` + `el-polyton` language bits, both Greek
  script bits, both Greek family bits.
- `ἀ` (U+1F00) → `el-polyton` only at every level. Modern-Greek
  rejection works on polytonic-only tokens.
- `Ά` (U+0386, via case closure) → `el` + `el-polyton`.
- `Έ` (U+0388), `Ή` (U+0389), `Ώ` (U+038F) → same.
- Combining ypogegrammeni `U+0345` → `el-polyton` via post-fallback
  NFD closure (only reachable from precomposed polytonic chars).

All pass in the current `validate.py` phase 1.

## Followups / open items

- **C3 extension impact.** Once the cutoff-decision work ships
  polytonic tokens into the extended Apertus vocab, re-run the
  audit. Expect the polytonic-only `bitmask_and` count to rise from
  0 to non-zero, and the language-level split to start contributing
  real exclusion power at the token level.
- **Greek numeral usage in modern text.** `α´` `β´` `γ´` are common
  for ordinals (especially in formal/legal text). The numeral sign
  U+0374 is handled by substrate fallback; alpha/beta/gamma are in
  scope. So a token like `α´` (if it exists in vocab) would AND-
  pass for both Greek bits. No change needed; flagged for awareness.
- **Greek question mark substitution.** Modern Greek text typically
  writes the question mark as the Latin semicolon U+003B, not the
  dedicated U+037E. ASCII `;` is in our table (substrate, ALL_BITS).
  U+037E is also ALL_BITS via fallback. Both end up not
  contributing to exclusion — correct.
