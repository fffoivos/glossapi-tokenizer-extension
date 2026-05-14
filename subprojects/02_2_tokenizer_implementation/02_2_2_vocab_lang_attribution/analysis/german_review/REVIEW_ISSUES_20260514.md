# Review issues — German↔English analysis (2026-05-14)

External-reviewer findings logged here for triage. Each item:

1. The finding (verbatim).
2. Verification against the actual artifacts.
3. Proposed fix.
4. Priority / dependency notes.

No code changes applied yet — awaiting direction on which to fix first.

## Issue 1 (MAJOR) — dataset-source confound

> "German is from epfml/FineWeb2-HQ while English is from HuggingFaceFW/clean-wikipedia.
> So claims like 'German has more Latin and less substrate' because of compound nouns
> are plausible, but currently over-attributed to language rather than corpus/domain mix."

**Verification.** Confirmed exactly:

```
deu_Latn:  1,005,786,784 tokens · 100 % from epfml/FineWeb2-HQ          (web crawl, HQ)
eng_Latn:  1,005,327,142 tokens · 100 % from HuggingFaceFW/clean-wikipedia  (Wikipedia only)
ell_Grek:  1,002,997,071 tokens · 100 % from epfml/FineWeb2-HQ          (web crawl, HQ)
```

The `name` field even flags `eng_Latn` as `<wiki-only: en>`. The English
corpus is Wikipedia-only because the run's source-map prioritised
`clean_wikipedia` for `en` after FineWeb-2 / FineWeb-HQ caps were
exhausted. So **the English histogram is structurally Wikipedia-shaped**
while German and Greek are FineWeb2-HQ-shaped (web text).

Consequences for the claims in `firing_rate_mapping_report.md`:

- "German has more Latin (84.6 %) and less substrate (15.3 %) than English
  (75.4 %, 24.4 %) because of compound nouns" — **partly or wholly
  attributable to Wikipedia vs web text**, not necessarily language.
  Wikipedia has high substrate fraction (citations, tables, infoboxes,
  numbers, brackets). Web text has more prose substrate ratio.
- "English has higher BPE fertility" — possibly contaminated by domain.
- The 0.43 % vs 2.38 % hard-rejection comparison is also domain-confounded
  to some degree (Wikipedia has different loanword density than web text).

**Proposed fix.** Two options, both worth doing:

1. **Relabel everywhere** — every comparison "English vs German" in
   reports + viz captions becomes "English Clean-Wikipedia vs German
   FineWeb2-HQ" with explicit dataset-source disclosure. The numbers
   stay, the framing changes from language-attribution to corpus-pair-
   attribution. **Lowest effort, highest honesty.**
2. **Add a matched comparison** — re-run the tokenisation on a matched
   slice: German Clean-Wikipedia vs English Clean-Wikipedia (and/or
   German FineWeb-2-HQ vs English FineWeb-2-HQ). Both source datasets
   contain both languages, so a matched re-run is feasible. This isolates
   language from domain. **Higher effort, gives us the real language
   answer.**

Option 1 is mandatory regardless; option 2 is the methodologically
correct addition.

Priority: **highest** — affects every conclusion drawn from the
German↔English comparison.

## Issue 2 (MAJOR) — script-bucket misnamed as "family"

> "Some scripts still use the old 'family' concept to mean script buckets
> reconstructed from language bits, e.g. family:Latn. Since char membership
> now has real script_and and family_and, this naming is misleading and
> can diverge from the v4 hierarchy. I'd rename this layer to script_bucket
> or move these analyses onto script_and."

**Verification.** Confirmed in:

- `firing_rate_mapping.py` (lines ~52, build of `family()` helper from
  bitmask_and language bits, returning labels like `family:Latn`).
- `distribution_shift.py` (line ~42, same).

These predate the v4 char schema. The v4 `script_and` column now gives
the **real** script bucket directly per token (22 bits). My helper
reconstructs script buckets from the 55-bit language mask by checking
which language family has any bit set — which yields the same result
today but is fragile (a change to the language bit assignments would
silently shift the reconstructed bucket).

**Proposed fix.** Rewrite the family() helper to:

1. Load `script_and` directly from `token_language_bitmask.parquet`.
2. Map `script_and` bit positions ↔ ISO 15924 codes from
   `02_2_1_char_language_membership/artifacts/manifest.json` (the
   `scripts` list).
3. Bucket label = the script with its bit set, or `script_mixed` if
   multiple, or `script_none` if zero. Distinguish from substrate (popcount
   of bitmask_and == 55) and unknown (status not evaluable).
4. Rename the internal label from `family:Latn` to `script:Latn` and the
   variable from `fam` to `script_bucket`.

This change is mechanical but touches several analysis scripts and the
generated TSVs / JSONs. Will require regenerating downstream artifacts.

Priority: **high** — naming conflicts with the v4 schema and will cause
drift if not fixed before more analyses pile on.

## Issue 3 (MEDIUM) — viz histogram silently drops out-of-range tokens

> "The viz log-ratio histogram silently drops out-of-range tokens. It bins
> only -4 < log10(de/en) <= 4 and has no catch-all bins. Very German-heavy
> tokens can exceed that, so the visualization can understate the sharp
> German-only tail."

**Verification.** Confirmed in `viz/build_data.py`:

```python
edges = [-4, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
# ...
for b in bins:
    if b["lo"] < rat <= b["hi"]:
        b["tokens"] += 1
        ...
        break
# tokens with rat <= -4 or rat > 4 fall through the for-loop with no break
# and are silently discarded
```

The Laplace-smoothed log-ratio for a token with `count_de = 1,000,000`
and `count_en = 0` is `log10(1,000,000.5 / 0.5) ≈ 6.30` — outside the
bin range. There are ~3,500 such German-only tokens (no English firings)
in T2, plus the T0 set whose chars rule English out. Their masses are
dropped from the histogram, **understating the German-only tail**.

**Proposed fix.** Replace closed bins with open-ended sentinel bins:

```python
edges = [-inf, -4, -3, -2.5, -2, -1.5, -1, -0.5, 0,
                  0.5,  1,  1.5,  2,  2.5,  3,  4, inf]
```

Label the outer bins `≤ -4` and `> 4`. Histogram becomes complete.

Priority: **medium** — fixable in one edit; would not change reported
numbers in the markdown reports but will visibly change the histogram
rendering.

## Issue 4 (MEDIUM) — hard-coded 55 language bits

> "The analysis hard-codes 55 language bits in several places. This should
> come from the manifest, especially now that the membership schema is
> actively changing."

**Verification.** Confirmed:

- `tiered_attribution.py:52`: `N_LANG_BITS = 55`
- `viz/build_data.py:51`: `N_LANG_BITS  = 55`
- `firing_rate_mapping.py:64`: `N_BITS = 55`

(plus a few `popcount == 55` checks for substrate detection)

**Proposed fix.** Read from the manifest:

```python
manifest = json.load(open(CLM / "artifacts/manifest.json"))
N_LANG_BITS = manifest["levels"]["language"]["bits_used"]
```

Apply across all three scripts. Use a small shared helper in
`vocab_lang_attribution/scripts/_common.py` to avoid duplication.

Priority: **medium** — silent breakage risk if the schema grows.

## Issue 5 (MEDIUM) — viz skips T1 instead of sharing tier logic

> "The viz intentionally skips T1 rather than sharing the tiering logic
> from tiered_attribution.py. Today T1 is empty, so the output is fine,
> but this will silently misclassify if T1 becomes non-empty after
> membership changes."

**Verification.** Confirmed in `viz/build_data.py:57`:

```python
# Note in source: "skipping T1 since family tier is structurally empty
# for both en and de — see report."
```

The viz `tier()` helper hardcodes the en/de cases where T1 is empty by
construction. If schema changes introduce new languages or auxiliary
chars, T1 could become non-empty and the viz would silently misclassify
those tokens as T2.

**Proposed fix.** Extract the tier-assignment logic from
`tiered_attribution.py` into a shared helper module (under
`vocab_lang_attribution/scripts/_common.py` or a new
`02_2_3_token_classification/lib.py`). Both `tiered_attribution.py` and
`viz/build_data.py` import from there. Single source of truth.

Lighter alternative: have the viz consume the `tier_summary.json` /
per-token tier output written by `tiered_attribution.py` directly.

Priority: **medium** — bug becomes real as soon as the char schema
grows. Worth fixing before the language_category_promotion build.

## Issue 6 (MINOR) — stale doc numbers

> "German sample is reported as 1,005,777,069, but metadata and reruns
> show 1,005,786,784. The tier report says the sum is 99.99%, but the
> JSON/rerun accounts for the full sample."

**Verification.** Confirmed:

- `firing_rate_mapping_report.md:5` says German sample `1,005,777,069`.
- `lang_metadata.json` says `1,005,786,784`.
- The 9,715-token discrepancy is small (probably my earlier rounding
  before final aggregation), but yes — it's stale.

For the tier report:

- `tiered_attribution_report.md` headline says "99.99 %".
- Empirically the per-tier mass percentages sum to 99.99 % because of
  floating-point rounding in the per-row pct (4-decimal display); the
  actual integer mass sums to the exact sample total.

**Proposed fix.** Rerun `tiered_attribution.py` against the current
artifacts, copy the updated `total_de` straight from the JSON output
into the report, and either:

- accept "100.00 %" as the rounded display value if the raw integer
  masses really do sum to total_de, OR
- compute the sum and write it exactly.

Priority: **low** — pure documentation tightening. Bundle with the
fixes above.

## Suggested order of attack

1. **Issue 1 option 1** (relabel framing to "wiki-only English vs
   FineWeb2-HQ German") — touches every German↔English document and is a
   prerequisite for any further claims to be honest. **First.**
2. **Issue 2** (rename "family" → "script_bucket" / switch to
   `script_and`) — schema-level mismatch, do before adding more analyses.
3. **Issues 3, 4, 5** — bundle into one analysis-code-quality pass:
   open-ended histogram bins, manifest-driven `N_LANG_BITS`, shared tier
   helper. All in the same file group.
4. **Issue 6** — drop into the pass for issues 1–5.
5. **Issue 1 option 2** (matched-corpus rerun: de+en from the same
   source) — separate workstream, ~2 h of tokenisation + aggregation.
   Independent of 1–4 but the comparison is only meaningful after option
   1's relabelling lands.

Awaiting direction on which to take up first (or in what bundle).
