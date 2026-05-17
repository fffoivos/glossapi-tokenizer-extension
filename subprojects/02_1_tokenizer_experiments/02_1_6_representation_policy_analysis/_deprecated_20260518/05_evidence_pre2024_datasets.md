# Stub — Pre-2024 dataset landscape as Mistral-11 selection criterion

Status: **OPEN**. Not yet investigated.

## Hypothesis

When Mistral trained tekken v3 (Mistral-Nemo-Base-2407, July 2024),
HPLT 3.0 did not yet exist. The dataset landscape Mistral had access
to was different from 2025-2026. Mistral-11 may track the
per-language footprint of the pre-2024 datasets Mistral actually used,
which would explain its choices better than HPLT 3.0 does.

## Sources to check

### Web-derived per-language size, as of mid-2024

- **CommonCrawl-derived corpora** with published per-language stats:
  - **OSCAR-23.01** (`oscar-corpus/OSCAR-2301`) — per-language size in
    documents and bytes. HF dataset card with statistics.
  - **mC4** (`allenai/c4` multilingual config or `mc4`) — per-language
    sizes from the AllenAI C4 release.
  - **CulturaX** (`uonlp/CulturaX`) — August 2023 release combining
    OSCAR + mC4, per-language stats published.
  - **MADLAD-400** — Google's 400-language CommonCrawl-derived,
    January 2024 release. Per-language stats in paper.
  - **FineWeb-2 v1** if it existed at the time Mistral trained;
    otherwise FineWeb-1 (English only).

For each, harvest: per-language doc count / token count, sort by
size, compare top-30 to Mistral-11.

### Wikipedia per-language size, mid-2024

- **Wikipedia statistics** at https://en.wikipedia.org/wiki/List_of_Wikipedias
  for the snapshot closest to mid-2024.
- Wikipedia is a known Mistral training source (implied by tekken's
  domain coverage).

### Reddit per-language size (until June 2023 API closure)

- Reddit per-subreddit / per-language statistics from PushShift
  archives or academic studies.
- Likely English-dominant with strong Western European tilt.
- Note: Mistral may or may not have used Reddit; Reddit was widely
  scraped pre-closure and entered many models' pretraining mixes.

### Other large pre-2024 multilingual corpora

- **The Pile** (Gao et al., 2020) — primarily English but with
  multilingual subsets.
- **ROOTS** (BLOOM training corpus, 2022) — 46 languages, per-language
  share published.
- **MASSIVE** (Amazon, 2022) — 51 languages of NLU intent data.

## What to test

Build a side-by-side table:

| Language | Mistral-11? | HPLT 3.0 rank | OSCAR-23 rank | CulturaX rank | mC4 rank | Wikipedia rank | MADLAD-400 rank | ROOTS rank |
|---|---|---|---|---|---|---|---|---|

Then ask:

1. Does Mistral-11 fit any single pre-2024 dataset's top-11?
2. Does Mistral-11 fit the **union** of "top-N across pre-2024
   datasets"?
3. Are there languages in Mistral-11 that are top-N in one source but
   not another? (E.g., Hindi top in Wikipedia but bottom in OSCAR.)
4. Specifically: Russian's HPLT-2 rank — was Russian also massive in
   OSCAR-23 / mC4 / CulturaX as of mid-2024? If yes, the "Russian
   omission from Mistral-11" looks like a deliberate choice; if no,
   it might reflect Mistral's specific dataset choices.

## Why this matters for Greek

If Mistral's choices track pre-2024-dataset rankings rather than
HPLT 3.0, then the "fair share by data availability" frame should
also use pre-2024-dataset rankings, not HPLT 3.0. Greek's share in
CulturaX / OSCAR-23 / Wikipedia is different from its HPLT 3.0
share, and the answer might shift the principled budget.

## Output format

A markdown doc, ~1000-2000 words:

1. Per-language data tables across the pre-2024 sources
2. Comparison to Mistral-11 (does each Mistral-11 language land in
   each source's top-11? top-20? top-30?)
3. Comparison to HQ-20 (same)
4. Specific Greek position across the pre-2024 sources
5. Verdict: FIT / PARTIAL FIT / INVALIDATED for the hypothesis "Mistral-11
   tracks pre-2024 dataset top-N"

## Estimated effort

~30-60 min of focused research (one agent run + cross-checking).
Most of the relevant per-language stats are published on HF dataset
cards or in paper tables.

## Priority

**HIGH.** This is the most likely explanation for Mistral-11's
selection given that the speaker-count and HPLT-3.0 hypotheses don't
fully fit. Mistral built tekken v3 in mid-2024 from whatever data was
available then; that's where the answer most likely is.
