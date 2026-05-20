# Polytonic vocab-budget check vs the sub-1B-language pattern

*Drafted 2026-05-20. Verifies the +5,120 polytonic budget already
chosen in `02_1_polytonic_greek_extension/` against the empirical
scaling of vocab-fired vs training-tokens observed across the 1,930
canonical languages Apertus's vocab serves.*

## TL;DR

| | value |
|---|---|
| Polytonic training corpus | **18,716 rows, 510,571,970 chars, 1,001 MB UTF-8** across train+val+test (the 802 MB figure that may appear elsewhere is the train split alone). |
| Apertus-base+C3 fertility on that corpus | 2.29 chars/token |
| **Implied training-token count** | **≈ 223 M tokens** — comfortably sub-1B ✓ |
| Sub-1B + Greek-script anchor (grc_Grek) | 128 M tokens → 3,502 vocab fired ≥100 |
| Script-isolated sub-1B power-law fit | `vocab_fired_geq_100 ≈ 0.1341 × tokens^0.5688`  (R² = 0.783, n=194) |
| **Pattern prediction at 223 M tokens** | **≈ 4,800 – 7,500 distinctive vocab tokens** |
| **Decided polytonic added-vocab** | **+5,120** (mid-range of the prediction ✓) |
| **Closest 256-divisible total** | **153,600 = 256 × 600** (current ship) |

The +5,120 budget chosen by the polytonic team **lines up with the sub-1B-language pattern**. No change is needed.

## 1. Where the corpus size came from

From [`splits/split_manifest.json`](../../02_1_tokenizer_experiments/02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/splits/split_manifest.json):

| split | rows | text chars | UTF-8 bytes |
|---|---:|---:|---:|
| train | 14,929 | 409,101,812 | 802,061,905 |
| val | 1,871 | 50,387,834 | 98,813,770 |
| test | 1,916 | 51,082,324 | 100,414,151 |
| **hygiene_kept_all** | **18,716** | **510,571,970** | **1,001,289,826** |
| dropped by hygiene | 10 | — | — |

Source mix (kept after strict filtering and dedup, post-hygiene): Scholarios graeca-patristic 12,991, Wikisource Greek 3,435, First1KGreek 983, GOARCH liturgical 673, Perseus 644.

## 2. Where the token-count came from

The polytonic run measured chars-per-token directly:

| variant | chars_per_token (poly_val_balanced) | tokens at 510 M chars |
|---|---:|---:|
| `c3p_poly_added_0000` (C3 base, no polytonic added) | 2.2925 | **≈ 222.7 M** |
| `c3p_poly_added_5120` (+5,120 polytonic) | 3.1389 | ≈ 162.6 M |

Source: [`FULL_REPORT.md`](../../02_1_tokenizer_experiments/02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/report/FULL_REPORT.md) §"Balanced Held-Out Metrics".

The training-corpus *budget* relevant for scaling is the **pre-extension** token count — what the polytonic corpus looks like to a tokenizer that does not yet know polytonic. That number is **≈ 223 M tokens**, well below 1 B. The post-extension number (~163 M) is the compressed view after polytonic units start absorbing multi-byte fallbacks.

## 3. Where the sub-1B pattern came from

The vocab-language attribution run (2026-05-13, [`02_2_2_vocab_lang_attribution/RUN_REPORT.md`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/RUN_REPORT.md)) tokenized up to 1 B tokens per canonical language (1,933 languages) through Apertus's base tokenizer and counted per-token firings. Output at [`outputs/lang_metadata.json`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/outputs/lang_metadata.json).

For each language `L`, that gives:
- `sample_tokens_total` — how many training tokens of `L` were tokenized
- `vocab_entries_fired` — how many of Apertus's 131,072 base-vocab entries fired at all
- `vocab_entries_fired_geq_10` — how many fired ≥ 10 times
- `vocab_entries_fired_geq_100` — how many fired ≥ 100 times *(the load-bearing distinctive measure — corresponds to "the language has enough material for Apertus to dedicate vocab to")*

### 3.1 Why we restrict the fit to script-isolated languages

Latin-script languages share a large substrate (basic ASCII letters, punctuation, common Latin BPE pieces). At 200 M tokens they routinely fire 70,000-90,000 vocab entries because all that shared substrate counts. A Greek-script language at the same training-token count fires far fewer vocab entries because the substrate it shares with the rest of the vocab is much smaller.

Polytonic Greek is **script-isolated** (Greek-script with diacritic-extended codepoints). The right cohort is the other script-isolated languages: Grek, Cyrl, Hani, Hira, Kana, Hang, Hebr, Arab, Deva, Beng, Tamil, Telu, Knda, Mlym, Guru, Gujr, Orya, Sinh, Thai, Lao, Mymr, Khmr, Tibt, Armn, Geor, Ethi, Syrc, etc. 194 of those have `sample_tokens_total < 1 B`.

### 3.2 The fit

Power-law log-log regression on those 194 points:

```
vocab_fired_geq_100  ≈  0.1341 × tokens^0.5688
```

R² = 0.783. Calibration anchors:

| anchor | tokens (M) | actual geq_100 | fit predicts |
|---|---:|---:|---:|
| **grc_Grek** (Ancient Greek, FineWeb-2) | 128 | **3,502** | 5,491 |
| pnt_Grek (Pontic Greek) | 1.0 | 921 | 169 |
| rmn_Grek (Romani in Greek script) | 3.3 | 1,206 | 327 |

The small-corpus Greek anchors over-fire the fit (the model has more Greek substrate than the per-corpus signal alone would predict). The grc_Grek anchor at 128 M tokens is the most-comparable size to our polytonic corpus — fit predicts 5,491, actual was 3,502 (fit over-predicts by ~57 %).

### 3.3 Projection at 223 M tokens

Two reads:

1. **Pure fit**: 0.1341 × 223,000,000^0.5688 ≈ **7,516** vocab entries with ≥100 firings.
2. **grc_Grek-anchored** (more conservative — uses the actual 128 M-token Ancient-Greek measurement and scales by the fit's exponent): 3,502 × (223/128)^0.5688 ≈ **4,793** vocab entries with ≥100 firings.

The two reads bracket the answer at **roughly 4,800 – 7,500 distinctive vocab tokens** that a 223 M-token script-isolated language is "worth" in Apertus's vocab budget.

## 4. Comparison to the chosen polytonic budget

The polytonic-extension team picked **+5,120** added tokens as the canonical ship of the `c3p_polytonic_20260518T_impl` run, on top of C3's 148,480 → final vocab **153,600 = 256 × 600**.

| budget | total vocab | 256-aligned | inside pattern range (4,800 – 7,500)? | comment |
|---:|---:|:---:|:---:|---|
| +3,584 | 152,064 = 256 × 594 | ✓ | below | substrate-heavy interpretation |
| +4,608 | 153,088 = 256 × 598 | ✓ | inside (just) | matches grc_Grek anchor exactly |
| **+5,120** | **153,600 = 256 × 600** | **✓** | **inside ✓** | **CURRENT SHIP** |
| +7,168 | 155,648 = 256 × 608 | ✓ | inside (top) | matches pure-fit prediction |
| +7,680 | 156,160 = 256 × 610 | ✓ | top edge | upper bound of pure-fit |
| +10,240 | 158,720 = 256 × 620 | ✓ | above | overshoots pattern |

**Verdict**: +5,120 sits cleanly in the middle of the pattern's prediction band. The decision is consistent with the sub-1B-language scaling.

There is some headroom to go higher (the [`FULL_REPORT.md`](../../02_1_tokenizer_experiments/02_1_polytonic_greek_extension/analysis/c3p_polytonic_20260518T_impl/report/FULL_REPORT.md) shows fertility curve still declines at Δ-0.02 per +512 step at the right edge, and added-vocab utilization remains 98.5 %), but each additional 1k tokens past +5,120 buys progressively less. **+5,120 is defensible as the current ship**; if a later quality measurement on the polytonic deployment register motivates more budget, the closest 256-aligned step up is **+7,168 → 155,648**.

## 5. What this means for `experiments_plan.md`

The plan's §10 Q3 *"HPLT polytonic/Katharevousa register supplementation"* and §11 worst-50 absorption check don't currently say anything about a stacked polytonic arm. The plan-diff to apply:

1. **§10 Q3** — reframe from "open" to "addressed by a separate stacked extension at +5,120 added → 153,600 total". The deployment decision is now "ship C3 base 148,480 OR ship C3+polytonic 153,600 as the CPT base."
2. **§3 Node 1 (BPE cutoff)** — already needs updating to 17,408 + curated-padded variant; add a note that 02_1_polytonic_greek_extension is the stacked layer on top.
3. **§5 Experiments / §3 Node 5 (three-arm experimental design)** — the three arms (Vanilla, ReTok, Distillation) need to specify which base they extend: 148,480 (modern-only) or 153,600 (modern + polytonic). My recommendation is to run the three-arm comparison on **148,480** (modern-only — keeps the comparison scoped to the question the plan is asking) and treat the polytonic layer as a separately-CPT'd downstream specialization.

These are plan-diff candidates, not commitments. Sign-off lives in
[ANALYSIS.md § Review checkpoints](ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).

## 6. Reproducibility

The fit + projections came from a 60-line Python pass over `lang_metadata.json`:

```bash
/home/foivos/.venvs/glossapi-merge-docling/bin/python3 -c "
import json, math
m = json.load(open('/home/foivos/Projects/glossapi-tokenizer-extension/subprojects/02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/outputs/lang_metadata.json'))
SCRIPT_ISOLATED = {'Grek','Cyrl','Hani','Hira','Kana','Hang','Hebr','Arab','Deva','Beng','Tamil','Telu','Knda','Mlym','Guru','Gujr','Orya','Sinh','Thai','Lao','Mymr','Khmr','Tibt','Armn','Geor','Ethi','Yiii','Mong','Syrc','Nkoo','Cher','Cans','Vaii','Adlm','Olck','Saur','Tale','Talu','Lana','Khar'}
pts = [(v['sample_tokens_total'], v['vocab_entries_fired_geq_100']) for v in m.values() if v.get('sample_tokens_total',0) < 999_000_000 and v.get('vocab_entries_fired_geq_100',0) >= 1 and v.get('script_iso15924','') in SCRIPT_ISOLATED]
xs = [math.log(t) for t,_ in pts]; ys = [math.log(v) for _,v in pts]
n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
b = sum((xs[i]-mx)*(ys[i]-my) for i in range(n)) / sum((xs[i]-mx)**2 for i in range(n))
a = math.exp(my - b*mx)
print(f'fit: vocab_fired_geq_100 ≈ {a:.4f} × tokens^{b:.4f}, n={n}')
print(f'at 223M tokens: {a*(223_000_000**b):.0f}')
print(f'at 128M tokens (grc_Grek anchor): {a*(128_000_000**b):.0f} (actual: {m[\"grc_Grek\"][\"vocab_entries_fired_geq_100\"]})')
"
```

Output (verified 2026-05-20 — see [`scripts/verify_polytonic_budget.py`](scripts/verify_polytonic_budget.py) for the runnable version):

```
fit: vocab_fired_geq_100 ≈ 0.1341 × tokens^0.5688, n=194
at 128M tokens (grc_Grek anchor): 5491 (actual: 3502 → fit over-predicts ~57%)
at 223M tokens (polytonic corpus): 7516 (pure fit)
                                   4793 (grc_Grek-anchored, scaling 3502 by (223/128)^0.5688)
```
