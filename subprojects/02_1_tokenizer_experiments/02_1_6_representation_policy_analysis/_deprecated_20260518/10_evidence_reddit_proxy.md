# Stub — Reddit per-language footprint (pre-2024 API closure)

Status: **OPEN, LOW-PRIORITY**. Not yet investigated.

## Hypothesis

Reddit was a heavily-used pretraining source before its June 2023 API
closure. Reddit's per-language distribution was strongly English-
dominant with Western European tilt, possibly explaining Mistral-11's
Italian/Portuguese inclusion despite low absolute web rank.

## Sources to check

- **PushShift archives** (now archived publicly through Internet
  Archive) — Reddit submission/comment dumps with language labels.
  Per-language activity tabulations from academic studies.
- **Academic studies on Reddit linguistic distribution** —
  e.g., the Pushshift Reddit corpus papers.
- **Language identification on r/all / top subreddits** — some
  academic papers tabulate the top non-English Reddit communities.

## What to test

### Q1 — Top-30 Reddit languages (pre-2024)

If retrievable, build a top-30 list of Reddit by language. Compare
to Mistral-11 and HQ-20.

### Q2 — Does Reddit's Western tilt explain Italian/Portuguese inclusion?

Italian and Portuguese are HPLT-ranked 8 and 9. They might be
disproportionately represented on Reddit due to active
r/Italy / r/portugal communities, exceeding what their web share
alone would suggest. This is a long-shot test.

### Q3 — Does Reddit explain the Russian exclusion?

Russia has had a separate social-media ecosystem (VK / Telegram /
Yandex platforms) that captures Russian-language users away from
Reddit. Reddit being Russia-light would explain Mistral-11's Russian
omission if Reddit was a major data source.

## Why this matters for Greek

Reddit's Greek presence (r/Greek, r/Greece) is small. If Reddit
heavily shaped Mistral's training mix, Greek's low Reddit presence
would explain its absence from Mistral-11. This is a weak
data-availability rather than a stated policy.

## Output format

A markdown doc, ~500-1000 words. Likely sparse — Reddit per-language
data is hard to retrieve post-API-closure.

## Priority

**LOW.** Mistral / Apertus do not publicly confirm Reddit usage as a
pretraining source. The data is hard to get. The discriminatory power
is low. Most-likely outcome: this investigation generates "we don't
know what Mistral used Reddit-wise" and the hypothesis stays open
indefinitely.

Consider deferring or skipping this hypothesis unless other
investigations point toward Reddit specifically as the missing factor.

## Estimated effort

~60 min minimum, with high probability of inconclusive result.
