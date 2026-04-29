# Glyph/PostScript Matcher Analysis - 2026-04-29

This is the context-blind scan requested after the fresh tokenizer review found
residual `GLYPH...`, `/hyphenminus`, and tiny mojibake tokens in the GlossAPI
fresh tokenizer.

## Tool

Rust scanner:

`subprojects/01_0_cleaning_iteration_and_thresholds/rust/glyphscan`

Built binary:

`subprojects/01_0_cleaning_iteration_and_thresholds/rust/glyphscan/target/release/glyphscan`

The matcher is intentionally context-blind. It matches:

- structured glyph markers:
  - `GLYPH<...>`
  - `GLYPH&lt;...&gt;`
  - `GLYPH(...)`
  - `glyph[...]`
  - `glyph<c=...,font=/...>`
  - `<c=...,font=/...>glyph`
- repeated/bare `GLYPH` stems without word-boundary protection:
  - `GLYPH`
  - `GLYPHGLYPH`
  - `GLYPHGLYPH...`
  - also embedded `glyph`, for discovery only
- PostScript/PDF residue:
  - `/uniXXXX`
  - `/gNN`, `/gidNN`
  - `CID+`
  - six-letter font subset names like `/ABCDEF+FontName`
  - the existing Rule-A PostScript literal list, including `/hyphenminus`,
    `/space`, `/period`, `/elipsis`, `/glyph`, etc.

Outputs:

- `summary.json`
- `documents.csv`
- `lines.jsonl`
- `samples_by_bin.jsonl`
- `family_summary.csv`
- `match_key_summary.csv`
- `count_bin_summary.csv`
- `coverage_bin_summary.csv`
- `matrix_summary.csv`
- `context_summary.csv`

The tool records:

- document-level counts
- line-level counts
- match families and exact normalized match keys
- count bins: `1-3`, `4-8`, `9-16`, `17-32`, `33+`
- non-whitespace match coverage bins: `00-10%`, ..., `90-100%`
- whether a match is glued to alphanumeric characters on either side
- deterministic random samples per count/coverage bin
- optional line-aligned compare output via `--compare-root`

## Production-Like Release Scan

Input root:

`/home/foivos/data/glossapi_work/unified_corpus_release/data`

Run directory:

`subprojects/01_0_cleaning_iteration_and_thresholds/runs/glyphscan_glossapi_release_detail_20260429T0038`

Command:

```bash
glyphscan \
  --input-root /home/foivos/data/glossapi_work/unified_corpus_release/data \
  --output-dir subprojects/01_0_cleaning_iteration_and_thresholds/runs/glyphscan_glossapi_release_detail_20260429T0038 \
  --workers 8 \
  --batch-size 64 \
  --sample-per-bin 40 \
  --max-line-records 100000
```

Runtime:

- wall clock: 8m16s
- max RSS: about 5.5GB
- rows scanned: 592,003
- lines scanned: 499,371,375
- docs with matches: 44,359
- matched lines: 5,125,201
- total matches: 80,575,919

The first unoptimized run took 28m52s. Removing per-match regex
classification brought the detail run down to 8m16s on the same root.

## Family Counts

| family | matches |
|---|---:|
| `glyph_structured` | 41,992,485 |
| `postscript_gid` | 29,716,230 |
| `postscript_literal_embedded` | 3,821,320 |
| `postscript_uni` | 3,274,590 |
| `postscript_literal_token` | 1,303,477 |
| `glyph_stem_embedded` | 360,212 |
| `glyph_stem_token` | 99,640 |
| `font_subset` | 7,618 |
| `cid_prefix` | 347 |

Important interpretation: this is deliberately context-blind, so some families
include false positives from URLs, base64 image payloads, and ordinary words.
The counts are discovery counts, not direct cleaner deletion counts.

## Top Exact Keys

| key | matches | interpretation |
|---|---:|---|
| `glyph_structured` | 41,992,485 | true PDF extractor residue, often `GLYPH<...>` dumps |
| `/hyphenminus` | 4,901,046 | true PostScript replacement for hyphen/minus; often glued into ranges or words |
| `/g3` | 2,378,775 | mostly true glyph-index residue, but some URL/base64 false positives |
| `/g28` | 1,725,522 | mostly glyph-index residue |
| `/g27` | 906,528 | mostly glyph-index residue |
| `/g302` | 869,538 | mostly glyph-index residue |
| `/g306` | 776,051 | mostly glyph-index residue |
| `/g1` | 703,254 | glyph-index residue plus some false positives |
| `/glyph` | 432,399 | mostly slash-prefixed glyph literal or structured marker fragments |
| `/uni1f77` | 268,010 | Unicode-name residue inside Greek words |
| `/uni1f79` | 236,307 | Unicode-name residue inside Greek words |
| `/uni03f1` | 168,271 | Unicode-name residue inside Greek words |

Selected literal counts:

| literal | matches |
|---|---:|
| `/hyphenminus` | 4,901,046 |
| `/glyph` | 432,399 |
| `/elipsis` | 64,512 |
| `/at` | 51,390 |
| `/bullet` | 29,963 |
| `/equal` | 9,378 |
| `/less` | 6,017 |
| `/period` | 4,462 |
| `/space` | 2,402 |
| `/ellipsis` | 1 |

Repeated bare `GLYPH` keys:

| key | matches |
|---|---:|
| `glyph_repeat_1` | 15,970 |
| `glyph_repeat_2` | 4,762 |
| `glyph_repeat_3` | 2,910 |
| `glyph_repeat_4` | 1,606 |
| `glyph_repeat_5` | 1,045 |
| `glyph_repeat_6` | 573 |

Total repeated bare `GLYPH` stem matches:

`27,453`

## Count And Coverage Distribution

Line count bins:

| count bin | matched lines |
|---|---:|
| `1-3` | 3,504,484 |
| `4-8` | 621,351 |
| `9-16` | 318,154 |
| `17-32` | 264,206 |
| `33+` | 417,006 |

Coverage bins:

| non-whitespace coverage | matched lines |
|---|---:|
| `00-10%` | 1,358,932 |
| `10-20%` | 791,259 |
| `20-30%` | 465,226 |
| `30-40%` | 392,375 |
| `40-50%` | 486,825 |
| `50-60%` | 449,499 |
| `60-70%` | 197,371 |
| `70-80%` | 88,970 |
| `80-90%` | 157,076 |
| `90-100%` | 737,668 |

The high-coverage bins are the strongest candidates for line dropping. The
low-coverage bins mix true small residues with URL/base64 false positives and
should not be dropped blindly.

## Context Distribution

| context | matched lines |
|---|---:|
| `prose_or_other` | 3,352,334 |
| `markdown_table` | 810,103 |
| `markdown_list` | 546,538 |
| `markdown_heading` | 228,578 |
| `latex_math` | 134,287 |
| `url_or_link` | 34,259 |
| `code_fence` | 19,102 |

The classifier is deliberately lightweight. `prose_or_other` includes true
prose, OCR/PDF dumps, and base64 markdown-image lines that do not begin as
normal URLs.

## Sample-Based Findings

### `/hyphenminus`

This is real and common. It appears in:

- numeric ranges:
  - `150/hyphenminus159`
  - `75/hyphenminus77`
- word splits:
  - `δε/hyphenminus ξιοο`
  - `ηπατο/hyphenminus παθών`
- explicit spaced hyphen:
  - `repeat /hyphenminus back jammer`

Root cause:

PDF/PostScript extraction emitted glyph names instead of actual punctuation.
The current cleaner strips standalone `/hyphenminus`, but the earlier tokenizer
review showed that numeric dotted slash ranges can be URL-protected and survive.
For many cases, replacement with `-` is probably better than deletion.

### Structured `GLYPH<...>`

Examples include:

- `GLYPH&lt;31&gt; η τομή...`
- `GLYPH&lt;133&gt; nite ranked alphabet`
- dense OCR/PDF dumps with `GLYPH&lt;129&gt;` repeated dozens of times
- markdown tables containing `N$GLYPH<137>GLYPH<137>`

Root cause:

PDF extraction placeholders and font-map failures. High-count/high-coverage
lines are clearly non-content and should be dropped. Low-count lines sometimes
contain recoverable prose where stripping the span is preferable.

### Bare/repeated `GLYPH`

Examples include:

- `GLYPHGLYPHGLYPHGLYPH...` mixed with `Æ`, `Œ`, and `<!-- text-missing -->`
- heading/list lines dominated by repeated `GLYPH`
- embedded English words such as `glyphic`

Root cause:

There are two groups:

- true bare placeholder residue: repeated all-caps `GLYPH...`
- discovery false positives from ordinary words: `glyphic`, `Glyphosate`, etc.

The cleaner should continue targeting uppercase repeated bare `GLYPH` stems
without using a lowercase substring rule.

### `/gNN` glyph indexes

Examples include:

- single-line pure residues:
  - `/g3`
  - `/g1`
- table/code-ish glyph-index runs:
  - `/g28/g46/g32`
  - `/g570/g655/g577/g32 /g635...`
- formula residue:
  - `V IR /g32 /g32 /g117 /g32 4 15 60V`

Root cause:

PDF glyph-index extraction. High-coverage lines are usually pure residue. Low
count lines may occur in formulas or real text and need context/coverage gates.

### `/uniXXXX`

Examples include:

- `/uni03BCε` for Greek `με`
- `σύ/uni03BCφωνα`
- `/uni1F79που`
- `Μακάριος, Συναξαριστ/uni1F75ς`

Root cause:

Unicode code-point names leaked into words. This is more promising as a
normalization/folding follow-up than as deletion. Decoding `/uni03BC` to `μ`
would often recover Greek text.

### `/elipsis`

Examples are mostly table-of-contents leader lines:

- `/elipsis/elipsis/elipsis...`

Root cause:

PostScript glyph name for ellipsis, often repeated as visual leaders. It can
probably be normalized to `...` or bucketed with dot/ellipsis cleanup.

### Base64/Image False Positives

Large base64 markdown image lines create context-blind false positives:

- `/gNN`
- `/at`
- font subset-like `/ABCDEF+...`

Root cause:

The matcher is intentionally context-blind. These should not be used as direct
deletion evidence for individual tokens. They do reinforce that data-image
payload stripping is important before glyph/PostScript matching.

## Local Before/After Proxy Compare

Compare run:

`subprojects/01_0_cleaning_iteration_and_thresholds/runs/glyphscan_unified_vs_release_compare_20260428T2355`

Input:

`/home/foivos/data/glossapi_work/unified_corpus/data`

Compare root:

`/home/foivos/data/glossapi_work/unified_corpus_release/data`

This is only a local proxy compare. It is not the exact wave-3 pre/post worker
root.

Result:

- rows scanned: 168,078
- matched lines: 3,266,206
- matches: 31,845,635
- retained line records all had the same match count in the compare line

Interpretation:

The local `unified_corpus` to `unified_corpus_release` line-aligned comparison
does not show useful cleaning deltas for retained samples. It appears to be a
mostly schema/release placement comparison, not the wave-3 recleaning delta.
For exact pre/post impact, run `glyphscan` with `--compare-root` against the
actual wave-3 original canonical root and wave-3 cleaned canonical root on the
worker or a copied local mirror.

## Cleaner Implications

The context-blind matcher is good for discovery, but not safe as a direct
cleaner rule. The useful implementation implications are:

1. Keep repeated uppercase `GLYPH...` in the existing glyph/PDF-artifact family.
   The bare repeated forms are real, but lowercase/embedded `glyph` creates
   false positives.
2. Add a later targeted fix for `/hyphenminus`:
   - numeric/word contexts should likely become `-`, not empty string
   - real URLs and paths still need protection
3. Consider a future `/uniXXXX` recovery pass:
   - many samples are Greek words with one leaked Unicode-name codepoint
   - this should be calibrated separately, not rushed into the current wave
4. Treat high-count or high-coverage `/gNN` and structured `GLYPH<...>` lines
   as strong line-drop candidates.
5. Keep base64/data-image stripping ahead of context-blind glyph matching.

## Artifacts

Primary detail run:

`subprojects/01_0_cleaning_iteration_and_thresholds/runs/glyphscan_glossapi_release_detail_20260429T0038`

Previous broad run:

`subprojects/01_0_cleaning_iteration_and_thresholds/runs/glyphscan_glossapi_release_20260428T2320`

Local proxy compare:

`subprojects/01_0_cleaning_iteration_and_thresholds/runs/glyphscan_unified_vs_release_compare_20260428T2355`

Code smoke:

`subprojects/01_0_cleaning_iteration_and_thresholds/runs/glyphscan_code_smoke_20260429T0035`
