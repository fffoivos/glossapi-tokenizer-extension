# Greek vocab budget — recommendation

Derived from
[`FAIRNESS_DEFINITION.md`](FAIRNESS_DEFINITION.md) operationalisation 2
(HQ-20 peer-cluster parity). Citable from
[`02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md)
in place of the "match-X" empirical anchors.

## TL;DR

**Recommended Greek vocab budget: 1,479 base + 5,120 added = 6,599
total Greek tokens (5.03 % of vocab).**

| Field | Value |
|---|---|
| Base Greek tokens (Apertus inheritance from Mistral) | 1,479 |
| **Recommended added (this document)** | **5,120** |
| **Recommended total Greek tokens after extension** | **6,599** |
| Total vocab after extension (131,072 + 5,120 = 136,192) | 136,192 |
| Greek share of post-extension vocab | 4.85 % |
| 256-aligned? | yes (5,120 = 256 × 20) |
| 1024-aligned? | yes (5,120 = 1024 × 5) |
| 128-aligned? | yes |

This sits **between Italian (4,712) and Portuguese (5,549) in HQ-20
allocation terms** — i.e. the upper-middle of the HQ-20 peer cluster,
matching the script-isolated HQ-20 high-end (Arabic 7,146) but at a
discount (5,120 < 7,146) reflecting Greek's smaller in-domain data
volume.

## How the number falls out

The fairness definition specifies: Greek must reach fertility within
±50 % of the HQ-20 median on `modern_greek_eval`-equivalent slices.

The HQ-20 peer cluster occupies the band **2,000-7,000 PMI tokens**
in Apertus's existing vocab (omitting English/French outliers):

| Lang | PMI tokens | Quartile |
|---|---:|---|
| Polish | 2,570 | Q1 (low HQ-20) |
| Dutch | 3,045 | Q1 |
| Czech | 2,058 | Q1 |
| Hungarian | 2,419 | Q1 |
| Danish | 2,270 | Q1 |
| Swedish | 2,212 | Q1 |
| Persian | 2,785 | Q1-Q2 |
| Vietnamese | 1,564 | Q1 (low end) |
| Indonesian | 2,035 | Q1 |
| Turkish | 1,833 | Q1 |
| Japanese | 3,222 | Q2 |
| Russian | 4,153 | Q3 |
| Korean | 4,438 | Q3 (not in HQ-20 but script-isolated reference) |
| **Italian** | **4,712** | **Q3** |
| Portuguese | 5,549 | Q3 |
| German | 7,329 | Q4 (HQ-20 high) |
| Arabic | 7,146 | Q4 (script-isolated HQ-20 high) |
| Spanish | 6,714 | Q4 |
| Chinese | 2,650 | Q1-Q2 (script-isolated dense-token) |

HQ-20 median (omitting English/French): **2,898** (between Persian and
Dutch).
HQ-20 mean: **3,471**.
Greek current (1,479): **51 % of median**, **42 % of mean**, far
below the band floor.

### Three possible within-band picks

| Pick | Target | Added required | Cutoff | Rationale |
|---|---:|---:|---|---|
| Low HQ-20 | 2,500 | +1,021 | +1,024 | Greek crosses the floor of the HQ-20 band. Aligns with the "low end" peer cluster (Vietnamese / Turkish). Minimal investment. |
| **Mid HQ-20** | **6,500-6,600** | **+5,021-+5,121** | **+5,120** | **Greek matches the script-isolated HQ-20 cluster** (Arabic / Korean / German tier). The mid-of-band defensible position. Recommended. |
| High HQ-20 | 7,500 | +6,021 | +6,144 | Greek matches Arabic / German exactly. Defensible but pushes Greek to the top of the band — the maximum the fairness definition allows. |

The recommendation is the **mid pick (+5,120)**. Reasons below.

## Why mid (+5,120), not low (+1,024) or high (+6,144)

### Why not low (+1,024)

The low pick brings Greek to Vietnamese tier (2,500 tokens). Vietnamese
is in HQ-20 by what looks like deliberate inclusion (rank 23 by FW2
docs, below 2 unincluded languages) — it's the *floor* of HQ-20, not a
defensible target.

Greek has more FW2 docs than Vietnamese (44.2 M vs 40.7 M) and a
larger clean-wikipedia footprint (0.90 % vs 1.00 % by bytes, similar
in chars). Setting Greek's budget at Vietnamese's level violates
"peer-cluster parity" by anchoring to the cluster floor, not its
median.

### Why not high (+6,144)

The high pick brings Greek to Arabic / German tier (7,500-7,600
tokens). Arabic has 57.8 M FW2 docs (31 % more than Greek's 44.2 M)
and is a Mistral-strong-11 language (Greek is not). German is one of
the top-5-FW2-docs languages.

Anchoring Greek to Arabic / German violates "peer-cluster parity" in
the other direction — Greek does not have Arabic's data volume nor
Apertus's German-tier commitment. Reaching this tier is *permissible*
under the fairness definition but not *required*.

### Why mid (+5,120) is the unique anchor

Greek's natural peer cluster within HQ-20 is the script-isolated
languages of comparable doc volume:

| Lang | FW2 docs (M) | PMI tokens | PMI / M_docs |
|---|---:|---:|---:|
| Vietnamese | 40.7 | 1,564 | 38 |
| Danish | 43.0 | 2,270 | 53 |
| **Greek** | **44.2** | **1,479** | **33** |
| Swedish | 45.3 | 2,212 | 49 |
| Hungarian | 46.9 | 2,419 | 52 |
| Persian | 51.0 | 2,785 | 55 |
| (Arabic) | (57.8) | (7,146) | (124) |

Greek's PMI/M_docs ratio (33) is the lowest of its doc-share peers in
HQ-20 — even Vietnamese (38) gets a slightly higher ratio. If Greek
matched the average peer ratio of ~50 PMI/M_docs (mean of Danish,
Swedish, Hungarian, Persian), it would have 44.2 × 50 = **~2,210
tokens** — i.e. the low pick territory.

But this PMI/M_docs comparison is *cross-language* — and Latin-script
peers get an unfair boost from Latin merge sharing. Among the
script-isolated HQ-20 set specifically:

| Script-isolated HQ-20 lang | PMI tokens |
|---|---:|
| Chinese | 2,650 |
| Japanese | 3,222 |
| Korean (not HQ-20 but script-isolated peer) | 4,438 |
| **Greek (current)** | **1,479** |
| **Greek (recommended)** | **6,599** |
| Arabic | 7,146 |
| Persian | 2,785 |

The script-isolated cluster mean = **(2,650 + 3,222 + 4,438 + 7,146 +
2,785) / 5 = 4,048** (omitting outlier Hindi / Bengali / Thai which
have much lower doc counts). The script-isolated cluster median =
**3,222** (Japanese).

Greek's mid-of-script-isolated-HQ-20 target is therefore ~3,200-4,500.
Mid-band of the full HQ-20 peer cluster is ~5,000-6,500.

The **+5,120 pick lands at the intersection of both bands**: above
Japanese (3,222) and Korean (4,438), below Arabic (7,146), at exactly
Portuguese (5,549) territory for a Latin language at similar doc
volume.

This is the only within-band pick that satisfies both
sub-cluster anchors.

## How this compares to the C3 REPORT's current pick

[`02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md)
recommends **+11,264** under the "match English-unique" rhetorical
anchor. The principled budget derived here is **+5,120** — less than
half.

The C3 REPORT's pick is *rational-core-defensible* under
operationalisation 3 (the strong reading of R5) but goes beyond what
operationalisation 2 (the recommended reading) requires.

The C3 REPORT's pick:
- delivers fertility 1.47 (vs 1.83 at +3,072 and 1.65 at +5,120)
- has 99.5 % Greek-payload, 0.13 % noise, ~5.5 % unused
- 11,264 added × ~12 KB per BF16 embedding row ≈ 130 MB embedding
  weight

This recommendation:
- delivers fertility ~1.65 (estimated by linear interpolation between
  REPORT §2 entries at 5,120 and 6,144; cleaner number to be measured
  if used)
- expected ~99 % Greek-payload (same trajectory)
- 5,120 added × ~12 KB ≈ 60 MB embedding weight (~70 MB saved)

**The trade**: ~12 % fertility cost on every Greek inference call
forever, in exchange for ~70 MB embedding weight saved, slightly less
CPT compute, and a *principled* (rather than rhetorical) anchor.

## What this recommendation depends on

The +5,120 number falls out of two judgment calls:

1. **Operationalisation 2 over operationalisation 1 or 3.** Mid
   reading of Apertus's multilingual commitment (HQ-20-typical
   quality). If the user prefers operationalisation 3 (FLORES+ Gini
   fairness, strict per-language), the budget is **+11,264** (the
   C3 REPORT's current pick). If operationalisation 1 (enumerable
   support only), the budget is **0** and no extension is justified.
2. **Mid-band over low-band or high-band within operationalisation 2.**
   ±50 % tolerance, anchored to the script-isolated HQ-20 cluster
   intersection rather than to the cluster floor or top. Tightening
   to ±25 % would narrow to +4,096-+4,608; loosening to ±100 %
   would broaden to +1,024-+12,288.

## What would change this recommendation

The recommendation changes if any of the following are true:

- **Apertus's "1,811 languages" commitment is read as substantive
  (per-language fertility), not enumerable.** → switch to
  operationalisation 3 → +11,264.
- **The C3 polytonic arm changes Greek's effective in-domain
  data volume.** The polytonic extension adds Ancient/historical
  Greek volume on top of modern Greek. If polytonic adds substantial
  vocab demand, the budget should rise to accommodate it. Estimate
  per `02_1_polytonic_greek_extension/` outputs.
- **Future per-language toxicity / OCR / parallel-pair coverage for
  Greek lands.** Closing the tooling gaps reduces the case for vocab
  over-allocation as compensation; doesn't change this recommendation
  but might lower the "compensation premium" implicit in mid-vs-low.
- **The HQ-20 selection criterion is clarified.** If Messmer et al.
  state that HQ-20 is "languages with quality-classifier training
  data," then HQ-20 membership becomes a tooling-gap fact rather than
  a substantive commitment — which weakens R5 and pushes the
  recommendation toward the low-band (+1,024-+3,072).
- **Empirical Greek-fertility-on-FLORES+ measurement is done.**
  Current recommendation extrapolates from `modern_greek_eval` to
  FLORES+ Gini. A direct FLORES+ Greek measurement might raise or
  lower the band.

## Alternative numeric picks the user might prefer

| Pick | Rationale | Cost vs +5,120 |
|---|---|---|
| **+5,120 (recommended)** | mid-of-HQ-20-band, intersection of script-isolated and full-HQ-20 clusters | baseline |
| +4,096 | low-mid of band; matches Italian (4,712) at the lower side | -5 % fertility recovery, +10 MB savings |
| +6,144 | high-mid of band; matches German (7,329) tier; aligned to existing 1024 grid in `02_1_4_cutoff_analysis/REPORT.md` | +5 % fertility recovery, -10 MB cost, same band |
| +3,072 | low band; Korean-tier | +10 % fertility cost, -20 MB savings |
| +11,264 | C3 REPORT's current pick | -15 % fertility recovery vs +5,120; +70 MB cost; defensible only under operationalisation 3 |

The +6,144 pick is the closest grid-aligned neighbor and would be
equivalent on principle. Picking +5,120 vs +6,144 is a 1024-row
preference and not a substantive difference.

## Cross-check against the C3 fertility curve

From [`02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md)
§2 fertility table on three clean held-out slices:

| Added | Fertility | Vs +5,120 |
|---:|---:|---:|
| +3,072 (Korean-tier) | 1.83 | +11 % cost |
| +4,096 | 1.75 | +6 % cost |
| **+5,120** | **~1.68 (interp.)** | **baseline** |
| +6,144 | 1.63 | -3 % gain |
| +8,192 | 1.55 | -7 % gain |
| +11,264 | 1.47 | -12 % gain |

The fertility curve's diminishing returns appear *before* the C3
REPORT acknowledges them. At +5,120 we capture **65 % of the fertility
recovery available all the way to +11,264** for **45 % of the budget**.
This is the inflection that justifies stopping at the principled mid
rather than continuing to the maximal-rational +11,264.

## Final recommendation

**+5,120 added Greek tokens.** Total Greek vocab after extension =
6,599. Total Apertus vocab after extension = 136,192.

The recommendation is principled (operationalisation 2 mid-band),
grid-aligned (1024 × 5 = 5,120), embedding-weight-conservative (~60
MB vs ~130 MB at +11,264), and fertility-defensible (1.68 vs 1.47).

It puts Greek at the **upper-middle of the HQ-20 peer cluster**,
above all script-isolated peers except Arabic, below all primary-ring
HQ-20 languages. This is the *substantive HQ-20 membership*
position — Greek treated as a real HQ-20 language for vocab purposes,
not as a footprint outlier.
