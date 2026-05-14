# `font_name_literal` matcher category — spec

**Status**: draft. Needs Gemini-review confirmation (Feat-A wave) before
promoting bare-form literals; the `+FontName` prefixed forms and
`/<SUBSET>+<FONT>` regex are direct-signal and can ship without review.

**Counter surfaced by this category**:
`page_font_marker_count` (part of the three-counter
`drop_low_salvage_pages` design).

## Literal set — empirically filtered

Probe on `openarchives.gr` first 5,000 rows (2026-04-21) showed bare
font names are false-positive magnets (`Georgia` = 193 hits, `Symbol`
= 117 — legitimate Greek prose). Signal is concentrated in the
PDF-subset-naming-convention forms.

### Direct-strip literal set (no review needed)

PDF font-subset prefixed forms — `+<FontName>` where the `+` comes
from the PDF structure, not from prose:

```
+TimesNewRoman
+Times-Roman
+Palatino
+Linotype
+Helvetica
+Arial
+Courier
+CourierNew
+CenturyGothic
+CenturySchoolbook
+Symbol
+ZapfDingbats
+Garamond
+Minion
+Baskerville
+Bookman
+Calibri
+Verdana
+Tahoma
+Georgia
+CMU
+CMR
+CMBX
+CMSY
+CMMI
+LMRoman
+LMSans
+STIX
+ImprintMT
```

Rationale: the `+` prefix never appears in Greek prose before a
font-name-looking token. Zero false positives expected.

### Regex for PDF font-subset names (direct, no review)

Matches the Adobe font-subset naming convention
`/<6-uppercase-chars>+<FontName>`:

```
/[A-Z]{6}\+[A-Z][A-Za-z0-9-]+
```

Examples: `/XQDMQS+CenturyGothic`, `/NUMPTY+ImprintMTnum`. These are
entirely PDF-extraction artefacts and should never appear in Greek
prose.

### Candidate bare-name literal set (Gemini-review-gated — Feat-A wave)

Only promote if the Feat-A review wave shows them as reliable noise
signals **without** the `+` prefix:

```
Palatino           Linotype      Helvetica      TimesNewRoman
Times-Roman        Courier       CourierNew     ZapfDingbats
Garamond           Minion        Baskerville    Bookman
CenturyGothic      CenturySchoolbook
CMU                CMR           CMBX           CMSY           CMMI
LMRoman            LMSans        STIX           ImprintMT
```

**Reject** (too many legitimate hits in Greek prose):

```
Georgia (country name)
Symbol (common Greek word "σύμβολο" / "symbol")
Arial (very common as a name)
Verdana, Tahoma (low hit rate, inconsistent)
Calibri (borderline; keep in +Calibri form only)
```

## Wiring

- Lives as a new Aho-Corasick literal set in the matcher
  (`glossapi_rs_noise`) — NOT in the cleaner's BAD_LINE_AC, because
  this is a *counter* surface, not an immediate line-rejection surface.
- Per-page count surfaces via
  `PAGE_CATEGORY_MATCH_COUNTS["font_name_literal"]` in the existing
  debug-export machinery.
- The regex part goes into an existing Rust regex set pattern for the
  category (or a new `regex_set` slot in the category spec).

## Testing

Positive cases (must match):
- `… /XQDMQS+CenturyGothic> ǕǐGLYPH …` → matches `+CenturyGothic` AND
  the regex.
- `… font: +Palatino, 10pt …` → matches `+Palatino`.
- `… /NUMPTY+ImprintMTnum …` → matches via regex.

Negative cases (must NOT match in direct-strip mode):
- `Η Γεωργία εξελίσσεται σε τουριστικό προορισμό.` (Georgia the country) →
  `Georgia` is NOT in the direct-strip set, so no match.
- `σύμβολο του κράτους` → `Symbol` is NOT in direct-strip set (only
  `+Symbol`), so no match.
- `Ο Άρειος (Arial) ήταν ο ιδρυτής του Αρειανισμού.` → `Arial` not in
  direct-strip, no match.

## Promotion path

1. Ship direct-strip literal set + PDF-subset regex into matcher as
   part of PR 3 (review-infrastructure) OR as a dedicated small PR
   right after PR 2 normalize.rs.
2. Run full-corpus matcher pass to get `page_font_marker_count` values
   distribution.
3. Stratified-sample 50–70 cases across the density range for Gemini
   review (Task 1 — cleaning anchor review) to decide which bare-form
   literals to add.
4. Use the same full-corpus pass to supply stratified samples for the
   page-level threshold calibration wave (Task 2).
