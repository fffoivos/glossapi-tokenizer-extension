# language_isolation/ — three filter modes per (canonical_key, lang_code)

Generalises `german_review/tiered_attribution.py` to any
(canonical_key, lang_code) pair and emits, per run, three filter sets
that correspond to the **with-assumption / without-assumption** spectrum
the user asked for.

## The three modes

| mode | tiers included | semantics | when to use |
| --- | --- | --- | --- |
| `strict`       | T0 only            | **No assumption.** Token's chars certify L exclusively among the 55 in-scope locales. Char evidence alone. | "List only tokens that *must* be L". For English this is **empty** by structure (en's CLDR exemplar is ASCII, a subset of every other Latin locale). For German it's the 103 ß-bearing tokens. |
| `premise`      | T0 ∪ T2            | **With assumption.** Token's chars are all L-admissible (`bitmask_and` has L's bit set) AND it fires in L's dataset. Premise: in L's dataset, an L-admissible token is treated as L. Defeasible. | "Pool of L-like content excluding universal substrate". Best for embedding-norm and centroid analyses where infrastructure tokens would skew the result. |
| `premise_sub`  | T0 ∪ T2 ∪ T3       | Same as `premise` plus substrate (T3 = popcount-55 universal punctuation/digit/whitespace). | "Full corpus footprint of L". Best for analyses that need the complete language profile including infra. |

T4 (char-excluded by some non-L letter, e.g. Polish `ć` in the de
dataset) and T5 (unknown standalone) are NEVER in any mode.

## Run

```bash
cd subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution
python3 analysis/language_isolation/build_isolation_sets.py \
  --canonical-key deu_Latn  --lang-code de
python3 analysis/language_isolation/build_isolation_sets.py \
  --canonical-key eng_Latn_fineweb_hq  --lang-code en
python3 analysis/language_isolation/build_isolation_sets.py \
  --canonical-key eng_Latn  --lang-code en
```

Outputs per (canonical_key, lang_code):

- `tables/<L>/<canonical_key>__strict.jsonl` — T0 ids (no premise)
- `tables/<L>/<canonical_key>__premise.jsonl` — T0 ∪ T2 ids
- `tables/<L>/<canonical_key>__premise_sub.jsonl` — T0 ∪ T2 ∪ T3 ids
- `tables/<L>/<canonical_key>__summary.tsv` — token count + mass per mode
- `tables/<L>/<canonical_key>__manifest.json` — provenance pins
  (char-membership schema version, language bit, total sample, tier breakdown).

## Current results

| canonical_key | lang | strict tokens | strict mass% | premise tokens | premise mass% | premise+sub tokens | premise+sub mass% |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `deu_Latn`            | de | 103 | 0.49 | 77,748 | 84.55 | 80,591 | 99.82 |
| `eng_Latn_fineweb_hq` | en | **0** | 0.00 | 76,593 | 81.66 | 79,175 | 99.73 |
| `eng_Latn` (wiki)     | en | **0** | 0.00 | 74,308 | 75.07 | 77,014 | 99.51 |

**Headline asymmetry:** German strict-mode set is small but non-empty
(103 ß-bearing tokens, 0.49 % mass) — char evidence alone certifies
these as German. English strict-mode is structurally empty — there is
no character in the 55-locale exemplar union that is exclusively
English, so no token is char-certified as English-only.

**Domain-shift finding:** the two English samples differ at the
premise level by 2,285 tokens / +6.6 pp mass (FineWeb-HQ shows more
letter-bearing tokens), confirming the wiki↔web domain shift the
reviewer flagged.

## Downstream consumer pattern

For each embedding-analysis use case in
`03_1_greek_embedding_diagnostic/`, pick the mode whose meaning matches
the question:

| question | mode |
| --- | --- |
| "How does L use the embedding space distinctively?" | `strict` if non-empty, else `premise` |
| "Centroid of all L-like content" | `premise` |
| "Full corpus footprint of L (incl substrate)" | `premise_sub` |
| "Cross-language confusion: which tokens land in both L and L'?" | `premise` for each, intersect |

The diagnostic pipeline loads `<canonical_key>__<mode>.jsonl` exactly
as it currently loads `base_greek_tokens.jsonl` — same one-row-per-id
schema with a `count` field added.

## Generalising to other languages

The script takes any `(canonical_key, lang_code)` pair where
`lang_code` is in `02_2_1_char_language_membership/artifacts/manifest.json`'s
`languages` list. So `--canonical-key ell_Grek --lang-code el` or
`--canonical-key rus_Cyrl --lang-code ru` works without code changes.
The `strict` mode size varies dramatically by language:

- Single-locale scripts (Greek, Hebrew, Hindi, Thai, Korean, …):
  strict is **large** — essentially the whole script is char-certified.
- Multi-locale scripts with distinctive chars (some Cyrillic locales,
  Polish, Czech, German, Vietnamese, …): strict is non-empty but
  small relative to premise.
- Latin locales whose CLDR exemplar is a subset of others (en, id, …):
  strict is **empty by structure**. Premise is the only available
  filter.
