> **Historical reference.** Pre-convergence cleaning-iteration work. The converged tokenizer arm is **C3** (see [../../docs/C3_CONVERGENCE.md](../../docs/C3_CONVERGENCE.md)). Kept for traceability; do not treat as live planning.

# Wave 3 Tokenizer Review - 2026-04-28

This review covers the tokenizer artifacts from the strict wave-3 production
run:

`/home/foivos/runs/wave3_20260428/production_strict_v2`

Local artifact archive:

`/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_1_tokenizer_experiments/runs/production_strict_v2`

The local archive excludes large parquet exports and continuous-BPE work
shards, but includes tokenizer JSON files, split summaries/manifests, logs,
reclean validation, and provenance files.

## Completion Status

Production completed at `2026-04-28T17:47:28+0000`.

Tokenizers:

- `F1_glossapi_only`: fresh tokenizer, done.
- `F2_hplt_only`: fresh tokenizer, done.
- `C1_glossapi_plus_hplt_70_30`: continuous tokenizer, done.

C1 summary:

- base tokenizer: `swiss-ai/Apertus-8B-2509`
- base vocab: 131,072
- target vocab: 156,672
- added tokens: 25,600
- added merges: 25,600
- total runtime: about 8,404 seconds
- final tokenizer SHA256:
  `74a092cd55c258b725cbd55c28b21ac25da4afb728ce73b5c103c7463d190319`

Validation files present:

- `training_summary.json`
- `front_end_contract_check.json`
- `replication_check.json`
- `all_done.json`

The front-end contract check reports the expected C1 vocab size of `156672`.

## Split Sanity

All three production split summaries use strict badness filtering:

- `allow_missing_badness_scores=false`
- `has_greek_badness=true`
- `has_mojibake_badness=true`
- `greek_badness_score < 60`
- `mojibake_badness_score <= 0.1`

The corrected GlossAPI + HPLT train split resolves the 70/30 source mix after
strict filters:

- OpenArchives: 17,329,922,848 chars, 37.32%
- Greek PhD: 15,173,047,987 chars, 32.68%
- HPLT: 13,930,816,971 chars, 30.00%
- total: 46,433,787,806 chars

HPLT was scoped to the `HPLT/ell_Grek_ge8_no_mt_clean60` source and reclean
validation showed no missing HPLT Greek badness scores after the fill step.

## Tokenizer Vocab Review

Important decoding note: raw tokenizer JSON uses byte-level BPE display forms.
Raw-looking strings such as `Î·` are often normal byte-level encodings of Greek
text, not corpus mojibake. I decoded tokens through the byte-level alphabet
before judging suspicious residues.

### F1 GlossAPI Fresh

- vocab size: 50,000
- Greek-containing decoded tokens: 34,442
- glyph/PDF-residue token hits: 12
- PostScript literal hits: 1
- true decoded mojibake marker hits, excluding byte-fallback replacement
  tokens: 8
- Cyrillic token hits: 24

The remaining glyph-family hits are small, but F1 still exposes GlossAPI-side
residuals more strongly than HPLT, as expected.

### F2 HPLT Fresh

- vocab size: 50,000
- Greek-containing decoded tokens: 44,691
- glyph/PDF-residue token hits: 0
- PostScript literal hits: 0
- true decoded mojibake marker hits, excluding byte-fallback replacement
  tokens: 1
- Cyrillic token hits: 5

HPLT looks clean by the artifact classes we were targeting.

### C1 GlossAPI + HPLT 70/30 Continuous

For the C1-added slice only, token IDs `>= 131072`:

- added vocab items reviewed: 25,600
- Greek-containing decoded tokens: 25,163
- glyph/PDF-residue token hits: 9
- PostScript literal hits: 1
- true decoded mojibake marker hits, excluding byte-fallback replacement
  tokens: 1
- Cyrillic token hits: 0

The full C1 vocab contains many inherited base-tokenizer Cyrillic and byte
fallback tokens, so the added-token slice is the relevant signal for whether
this cleaning wave introduced or reinforced bad residue. On that measure,
Cyrillic/homoglyph residue does not look like a current blocker.

## Corpus Spot Checks

I sampled contexts from the corrected C1 train parquet for the remaining
glyph/PostScript-looking patterns.

The `GLYPH`-family hits are mostly explained by protected or legitimate
contexts:

- `glyph[followsequal]` inside an algorithm/code-like block.
- `Glyphosate` in agricultural/toxicology text, which is legitimate English
  content and should not be removed by a bare `glyph` substring rule.

The one actionable residue is `/hyphenminus` inside numeric slash ranges:

- `4.600/hyphenminus5.600 KDa`
- `0.3/hyphenminus 100%`
- `50.000/hyphenminus250.000 κατοίκους`

The cleaner strips standalone `foo /hyphenminus bar` and preserves real URL
paths, but numeric dotted slash ranges are currently classified as URL-like and
therefore protected. This is small enough not to rerun the production wave, but
it is a real follow-up.

## Follow-Up Issues Opened

- `https://github.com/eellak/glossAPI/issues/99`
  Deferred mojibake/Cyrillic/homoglyph calibration.
- `https://github.com/eellak/glossAPI/issues/100`
  Narrow URL-like protection so numeric `/hyphenminus` ranges get cleaned.
- `https://github.com/fffoivos/glossapi-tokenizer-extension/issues/1`
  Avoid repeated full scans during budgeted split export and source-mix
  summaries.

## Conclusion

This is a good state for the current wave. The strict HPLT filtering is in
place, the 70/30 mix was corrected after filters, the continuous tokenizer
finished successfully, and the added C1 vocabulary is dominated by Greek tokens
with very little targeted residue.

I would not rerun the production wave for the remaining `/hyphenminus` and
protected `glyph[...]` cases. They are documented as follow-ups, and the current
artifacts are suitable for inspection and next-step use.
