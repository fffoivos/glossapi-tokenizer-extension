# Stub — Commercial market footprint as selection criterion

Status: **OPEN**. Not yet investigated.

## Hypothesis

Both Mistral and Apertus are commercial offerings, not pure research.
The language priority lists may reflect the markets each
provider intends to sell into:

- **Mistral**: Western European + East Asian commercial markets. The
  Mistral-11 list includes the major Western Europe + East Asia
  business languages + Hindi (India strategic market) + Arabic
  (Gulf strategic market).
- **Apertus**: Swiss government + Swisscom enterprise + GDPR-compliant
  EU customers. Apertus's named priorities (apertus.ai apps:
  "Multilingual competence (German, French, Italian, English)" +
  "Your data stays in Europe") track this market exactly.

## Sources to check

### Mistral commercial markets

- Mistral's customer / partner announcements 2023-2026.
- Mistral pricing pages — do they list supported languages?
- Mistral case studies / industry papers — which industries/regions?
- Press coverage of Mistral funding rounds — investor geography.

### Apertus / Swisscom commercial markets

- Swisscom Apertus product announcement (we saw 404s on direct URLs).
- Swiss-AI Initiative business engagement statements.
- apertus.ai apps page language priorities (already verified —
  "German, French, Italian, English").

### EU market commitments

- EU AI Act language requirements (Apertus claims EU AI Act
  compliance — does the Act prescribe language coverage?).
- GDPR-related multilingual obligations.

## What to test

### Q1 — Does Mistral-11 track Mistral's commercial market geography?

Mistral's primary markets (per public commercial announcements):
- Western Europe (France, Germany, UK, Spain, Italy)
- North America (English)
- East Asia (Japan, Korea) — Mistral has partnerships in Tokyo
- Gulf / Middle East — Mistral has UAE partnership
- India — Mistral announced India strategy

Mistral-11 = en, fr, de, es, it, pt, zh, ja, ko, ar, hi. Map this
onto the commercial markets:

| Market | Language | In Mistral-11? |
|---|---|---|
| Western Europe core | en, fr, de, es, it, pt | ✓ × 6 |
| East Asia | zh, ja, ko | ✓ × 3 |
| Gulf | ar | ✓ |
| India | hi | ✓ |
| **Russia / CIS** | ru | **✗** |
| Africa | sw, ha, am, yo | **✗ × all** |
| Southeast Asia (deep) | id, vi, th | **✗ × all** |

Pattern: Mistral-11 covers exactly the markets Mistral has commercial
presence in. **Russia/CIS is the visible omission** — consistent with
geopolitical concerns about commercial AI in Russia post-2022.

### Q2 — Does Apertus's commercial priority list track Swiss / EU markets?

Apertus's apps page list: German, French, Italian, English. Map:

| Market | Language | In Apertus apps list? |
|---|---|---|
| Switzerland | de, fr, it, rm | de/fr/it ✓; rm in compliance scope but not headline |
| EU big-5 | en, de, fr, es, it | en/de/fr/it ✓; es ✗ |
| Greek market | el | **✗** |

Apertus's commercial framing is **Swiss-cantonal first, EU GDPR
second**. Greek is not a primary commercial market for Apertus.

### Q3 — Does the absence/presence track GDP rather than population?

Languages with high-GDP speaker bases (English, German, French,
Japanese, Korean) tend to be in commercial priority lists; languages
with high-population-but-lower-GDP bases (Bengali, Hindi, Indonesian,
Vietnamese) are inconsistently included.

Check: does Mistral-11 fit "top-by-GDP-of-speaker-base"? This would
explain the Korean inclusion (S Korea high GDP) vs Bengali exclusion
(Bangladesh lower GDP) and Vietnamese exclusion from Mistral-11.

## Why this matters for Greek

If Mistral-11 and HQ-20 are commercial-market-driven, Greek's
inclusion in HQ-20 (Greek market = ~$200B GDP, EU member) but
exclusion from Mistral-11 (smaller market relative to Mistral's
target Western+East Asian commercial focus) is internally consistent.

A "fair share by commercial value" frame would put Greek at small EU
member tier — roughly Polish-tier (Poland ~$700B GDP, ~$200B for Greece
× similar HQ-20 vocab). This is broadly consistent with the empirical
HQ-20 cluster, but doesn't generate a sharply different number from
the existing analysis.

## Output format

A markdown doc, ~800-1500 words:

1. Mistral commercial market geography.
2. Apertus / Swisscom commercial market geography.
3. Pattern fit assessment for both lists.
4. Verdict.

## Priority

**LOWER-MEDIUM.** Commercial markets correlate heavily with the
"top-by-web" pattern already established. This investigation might
not generate a fundamentally new perspective — but it would explain
the Russia-omission and the Korean/Hindi/Arabic-inclusion in
Mistral-11 that the speaker-count and pure-web hypotheses don't.

## Estimated effort

~30-45 min — Mistral's commercial press is public; Apertus's apps
page is local.
