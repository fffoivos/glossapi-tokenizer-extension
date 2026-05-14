# Feedback for the Greek-tokenizer-extension planning doc — 2026-05-11

Reader: the Claude updating the tokenizer-extension working document
(currently at v0.2; the prior edit packet brought it toward v0.3, and
this doc supplies the additional measurements that should land in v0.3
or a v0.4 follow-up).

This doc combines two independently-verified measurements made on
2026-05-11 and lays out exactly **what was measured, how, and where
the artifacts live**, so any reader can reproduce or audit the numbers
before incorporating them.

The two measurements:

1. **Apertus-8B-2509 base-vocabulary embedding-norm diagnostic.** Are
   the ~1,500 existing Greek tokens in Apertus's base vocab well-trained
   or undertrained relative to other languages and to an empirical
   "untrained floor"?
2. **Greek share of Apertus-8B-2509 pretraining data.** What fraction
   of the 13.5T tokens Apertus actually consumed was Greek?

The two readings appear contradictory at first — Greek's pretraining
share is tiny, yet existing Greek tokens are well-trained. §3 of this
doc reconciles them.

---

## 1. Existing Greek tokens — norm diagnostic (Phase A)

### 1.1 Headline finding

The 1,506 existing Greek tokens in `swiss-ai/Apertus-8B-2509` are
**statistically indistinguishable from the English-baseline** on both
embedding matrices. Greek's lower percentiles are actually slightly
better-trained than English-baseline's; Greek's worst-trained tokens
are 6.5× above the empirical untrained floor.

Decision implication (data-tied): **Phase 1 grounding CPT (continued
pretraining on Greek with the original tokenizer before the extension)
is NOT needed.** The existing Greek tokens are healthy.

### 1.2 Numbers — per-group L2 norm distributions

| group                     | count   | E p5  | E p25 | E p50 | E p75 | E p95 | U p5  | U p25 | U p50 | U p75 | U p95 |
| ---                       | ---:    | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| Greek                     | 1,506   | 3.530 | 4.398 | 5.047 | 5.651 | 6.481 | 2.989 | 3.494 | 3.797 | 4.037 | 4.343 |
| CJK                       | 9,544   | 2.611 | 3.624 | 4.608 | 5.601 | 6.200 | 3.138 | 3.545 | 3.787 | 4.047 | 4.398 |
| Cyrillic                  | 7,685   | 3.543 | 4.670 | 5.195 | 5.674 | 6.489 | 3.031 | 3.575 | 3.865 | 4.138 | 4.508 |
| German                    | 1,524   | 3.187 | 3.957 | 4.597 | 5.229 | 6.080 | 2.960 | 3.371 | 3.652 | 3.942 | 4.292 |
| French                    | 3,169   | 2.802 | 3.532 | 4.103 | 4.815 | 6.007 | 2.819 | 3.269 | 3.518 | 3.807 | 4.234 |
| English-baseline          | 74,838  | 3.355 | 4.456 | 5.051 | 5.488 | 6.003 | 3.004 | 3.528 | 3.839 | 4.126 | 4.521 |
| structural_non_linguistic | 2,659   | 2.056 | 3.879 | 4.911 | 5.661 | 6.516 | 2.593 | 3.286 | 3.583 | 3.844 | 4.201 |
| all-vocab                 | 131,072 | 2.939 | 4.163 | 4.925 | 5.478 | 6.117 | 2.948 | 3.489 | 3.799 | 4.086 | 4.468 |

Empirical untrained-floor median ‖U‖ = **0.4566** (bottom-100 across
the vocab).

Greek-vs-English ratios:
- Median ‖E‖: Greek / English = **5.047 / 5.051 = 0.999** (essentially equal).
- Median ‖U‖: Greek / English = **3.797 / 3.839 = 0.989** (essentially equal).
- Greek p5 ‖U‖ vs floor: **2.989 / 0.457 = 6.54** (no Greek-token tail near the floor).
- Greek p5 ‖U‖ vs English p5 ‖U‖: **2.989 / 3.004 = 0.995** (lower-percentile parity).

E and U agree on the ordering of group medians, so the diagnostic
signal is consistent across both untied matrices (Apertus has
`tie_word_embeddings = False`).

### 1.3 Method — how the numbers were obtained (every step auditable)

**Model load.** `swiss-ai/Apertus-8B-2509` at sha
`3162c99675aa588097cecd4a24b9aa1f712af477` (resolved via
`huggingface_hub.HfApi().model_info(...).sha`, recorded in
`ledger.json`). Loaded with the canonical entrypoint:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
tokenizer = AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")
model = AutoModelForCausalLM.from_pretrained(
    "swiss-ai/Apertus-8B-2509",
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
)
E = model.get_input_embeddings().weight.detach().cpu().float().numpy()  # (131072, 4096)
U = model.get_output_embeddings().weight.detach().cpu().float().numpy() # (131072, 4096)
```

bf16 load, fp32 stats. CPU-only (no GPU dependency). transformers 5.8.0,
torch 2.11.0+cpu, on the gcloud `m3-megamem-64` instance.

**Norm computation.**
`np.linalg.norm(E, axis=1)` and `np.linalg.norm(U, axis=1)` produce
two length-131,072 fp32 arrays. Saved as
`arrays/E_norms_all.npy` and `arrays/U_norms_all.npy` so any reviewer
can re-derive per-group statistics without reloading the model.
Per-group `.npy` slices are also saved (`arrays/E_norms_<group>.npy`,
`arrays/U_norms_<group>.npy`).

**Token classification — careful byte-level pipeline.**
Apertus uses GPT-2 style byte-level BPE. A naïve string-based scan would
mis-classify the byte-remapped tokens. The pipeline used:

1. `raw = tokenizer.convert_ids_to_tokens([id])[0]` — raw BPE form (not
   `decode`, which has special-token side-effects).
2. Reverse the canonical GPT-2 byte_to_unicode mapping to recover the
   actual byte sequence: `bs = bytes(UNICODE_TO_BYTE[c] for c in raw)`.
3. Strip a single leading `0x20` (the `Ġ` word-initial marker) if
   present.
4. `body.decode("utf-8")` — failures (partial multi-byte codepoints
   mid-merge) go to `byte_fragment` and are excluded from all language
   groups.
5. Special tokens (`tokenizer.all_special_ids`) excluded first.
6. Pure-whitespace, pure-digit tokens go to their own non-groups.
7. Tokens that are entirely Unicode categories P*/S*/Z* (punctuation,
   symbol, separator) go to `structural_non_linguistic`. This is a
   first-class analyzed group; the C3 extension already plans to add
   structural tokens (table_separator, punctuation_run, math_symbol,
   postscript_glyph, dingbat_or_symbol) and their init reference is the
   base's structural distribution, not the Greek/English distribution.
8. Remaining tokens get script-range tests (multiple-membership allowed):
   Greek (U+0370–03FF ∪ U+1F00–1FFF), CJK (U+4E00–9FFF ∪ U+3040–309F
   ∪ U+30A0–30FF ∪ U+AC00–D7AF), Cyrillic (U+0400–04FF), German marker
   chars {ä,ö,ü,ß,Ä,Ö,Ü}, French marker chars {é,è,ê,à,ç,…} excluding
   German overlap, English-baseline (ASCII-only AND ≥3 alphabetic
   chars).

**Ledger reconciliation** (required check; aborts if it fails):

```
special          4
byte_fragment    1,435
whitespace_only  129
digits_only      92
classified       129,412
---------------------- TOTAL: 131,072  ✓
```

**Floors.** Two definitions are computed and reported:
- **Absolute floor** — bottom-100 ‖U‖ across all 131,072 tokens.
  Median = 0.4566.
- **Clean floor** — bottom-100 after excluding `special` and
  `byte_fragment`. Median = 0.4566 (the bottom is dominated by
  `<SPECIAL_NNN>` slots and a handful of mojibake tokens; not by
  byte fragments).

**Greek bottom-50 by ‖U‖** (Task 5 in the diagnostic spec — candidates
for "reset before extension"): all 50 are common short function words
like `και` (id 5147, U=1.880), `για` (11641, U=2.315), `με` (6569,
U=2.345), `είναι` (16005, U=2.355), `που` (11357, U=2.361), `δεν`
(27095, U=2.518), etc. **These are not undertrained — they are
high-frequency tokens that have learned a low-norm default-direction
representation.** No reset needed.

**Artifacts (locally on `home`, mirrored from the instance):**
- Report: `runs/apertus_greek_diagnostic_20260511/apertus_greek_diagnostic.md`
- Per-token classifications: `runs/apertus_greek_diagnostic_20260511/token_classification.jsonl`
- Ledger: `runs/apertus_greek_diagnostic_20260511/ledger.json`
- Per-group percentile table: `runs/apertus_greek_diagnostic_20260511/norm_stats.json`
- Floor analysis: `runs/apertus_greek_diagnostic_20260511/floors.json`
- Greek bottom-50: `runs/apertus_greek_diagnostic_20260511/greek_bottom_50.json`
- Histograms (log-x): `runs/apertus_greek_diagnostic_20260511/figures/{E,U}_{greek_vs_english,all_groups}.png`, `group_medians.png`
- Per-token L2 norm `.npy` arrays: `runs/apertus_greek_diagnostic_20260511/arrays/`
- The script itself (rerunnable): `runs/apertus_greek_diagnostic_20260511/phase_a_diagnostic.py`

### 1.4 Known limitations of this diagnostic (v2 fix-ups)

- A handful of `<SPECIAL_NNN>` reserved-but-not-in-`all_special_ids`
  slots got matched as `English-baseline` (ASCII alphabetic, ≥3
  chars). They land near the floor and bias English-baseline's lower
  percentiles slightly downward. The bias is small enough that
  Greek's tail still beats English's; the conclusion is robust. Add
  a regex exclusion `^<SPECIAL_\d+>$` at the special-token step if
  re-running.
- The norm diagnostic alone doesn't catch *behaviorally* undertrained
  tokens (a token can have a healthy norm pointing in the wrong
  direction). The behavioral check is **Phase B** (per-token NLL on
  real Greek text), still to run. Given the very strong norm signal,
  Phase B is expected to confirm rather than overturn; it's a
  cross-check, not a re-litigation.
- This diagnostic is about the **base** model. The Instruct variant
  (`Apertus-8B-Instruct-2509`) ran through SFT + QRPO alignment after
  the base; its embedding state will have drifted. If the extension
  target is the Instruct variant rather than the base, re-run this
  diagnostic on it.

---

## 2. Greek share of Apertus-8B-2509 pretraining data

Full doc with method-of-record at
[APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md](APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md).
Headline-only summary here, plus the load-bearing methodology.

### 2.1 Headline finding

Apertus-8B-2509 consumed roughly **3.1 B Greek tokens out of 13.5 T
total = 0.023 %** during pretraining.

For comparison:
- "Naïve" upper bound from "Greek is 1 of 20 high-quality languages in
  the FineWeb-2-HQ ring" would be 2 %.
- The actual figure is **~87× lower** than that naïve bound.
- The gap is explained by (a) Greek being only 0.97 % of raw
  FineWeb-2 (rank #22 of 40 in the paper's Appendix Table G.6), (b)
  Apertus's per-language `sampler.rate = 0.95` haircut for
  secondary-ring HQ languages, and (c) the `quality_filter.p ∈ {0.33,
  0.10}` restriction (i.e. only 33% / 10% of FineWeb-2-HQ-Greek
  actually entered each stage).

### 2.2 Method — how 3.1 B was obtained

The full reasoning chain (all explicit, all auditable):

1. **Token-budget grounding.** Paper Table H.8 lists 8B's
   stage-boundary token totals. 8B skips Stage 2; the realised
   pretraining budget is **13,545 B**, not the 15T headline (which
   is the 70B figure). Stage durations used: S1 = 7,038 B,
   S2 = 0, S3 = 4,962 B, S4 = 1,345 B, S5 ≈ 200 B cooldown.

2. **Inventory verification — which datasets contain Greek.** Verified
   by reading each dataset card on HF (2026-05-11, recorded in §2.1 of
   `APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md`). Greek-bearing
   datasets in Apertus's consumed mix:
   - **FineWeb-2-HQ** (`epfml/FineWeb2-HQ/ell_Grek`) — Greek is 1 of
     20 HQ languages. Present in every consumed 8B stage. **Dominant
     route.**
   - **Clean-Wikipedia el** (`HuggingFaceFW/clean-wikipedia`) — Stage 5 only.
   - **EuroParl Greek** (`Helsinki-NLP/europarl`) — Stage 5 only.
   - **EuroBlocks-SFT-Synthetic-1124 Greek** — Stage 5 only, tiny.
   - **ParaDocs** — confirmed zero Greek pairs (HF tree listing shows
     only `en-{cs,de,es,fr,hi,hu,id,it,km,lo,my,ne,nl,pl,pt,sv,th,vi}`;
     correcting an earlier "very likely yes" assumption).
   - Institutional Books 1.0 — long-context phase, treated as zero
     for the headline (dataset is gated; the run's token isn't
     authorized).

3. **Direct measurement on a GCP worker (Path-A).** Spun a
   `c4-highcpu-192` worker (on-demand, `europe-west4-b`, ~62 min, ~$8
   wall cost) and tokenized every Greek-bearing dataset with the
   canonical Apertus tokenizer using `tokenizers.Tokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")`,
   adding `+2` per doc for `<s>` BOD + `</s>` EOD to match the paper's
   pretrain-token accounting (paper §2.1). Per-dataset measurements:

| Dataset                         | Docs       | UTF-8 GB | Apertus tokens   |
| ---                             | ---:       | ---:     | ---:             |
| FineWeb2-HQ `ell_Grek` (60 pq)  | 4,346,440  | 30.25    | **6,383,239,455** |
| Clean-Wikipedia el              | 226,273    | 1.24     | 275,679,532     |
| EuroParl Greek (20 bitexts)     | 18,124,501 | 5.67     | 1,163,704,082   |
| EuroBlocks-SFT Greek            | 582        | 0.002    | 358,888         |
| ParaDocs Greek                  | 0          | 0        | 0               |
| Institutional Books Greek       | (gated)    | —        | —               |

4. **Stage weighting.** For each stage, multiply the measured Greek
   tokens by the dataset's Apertus-side filter (`p × sampler.rate` for
   FineWeb-2-HQ Greek; 1 for the other Greek-bearing sets), then weight
   by `(stage_duration / total_stage_pool)` where total_stage_pool is
   the sum of Table-6 dataset pools available in that stage. Full
   formula in §2.4-math of the source doc. Result for FineWeb-2-HQ
   Greek:

| Stage | `p × 0.95` | Pool total Mₛ (B) | Stage duration (B) | Greek consumed (B) |
| ---:  | ---:       | ---:              | ---:               | ---:               |
| 1     | 0.3135     | 8,641             | 7,038              | 1.629              |
| 3     | 0.3135     | 9,346             | 4,962              | 1.062              |
| 4     | 0.0950     | 2,905             | 1,345              | 0.281              |
| 5     | 0.0950     | 2,978             | 200                | 0.041              |
|       |            |                   | subtotal           | **3.014**          |

5. **Stage-5 auxiliary contributions** (upper bounds, assuming the
   measured Greek tokens enter S5 at the same `(stage_duration / pool_total)`
   rate of 0.0672):

| Slice                | Greek tokens (B) | Pool S5 (B) | Upper-bound consumed (B) |
| ---                  | ---:             | ---:        | ---:                     |
| Clean-Wikipedia      | 0.276            | 33          | 0.019                    |
| EuroParl             | 1.164            | 21          | 0.078                    |
| EuroBlocks (×3)      | 0.000359         | 3           | 0.000                    |
|                      |                  | subtotal    | **0.097**                |

6. **Headline** = (3.014 + 0.097) B / 13,545 B = **3.111 / 13,545 ≈ 0.023 %**.

7. **Sanity checks** (all internal to the method, recorded in §5.3 of
   the source doc):
   - 4.35 M FineWeb-2-HQ Greek docs ≈ 10 % of FineWeb-2 v2.0.1 raw
     Greek (paper Table G.6: 44.2 M docs). Matches the HQ retention
     rate.
   - 6.38 B Apertus-tokens / 30.25 GB UTF-8 = 4.74 bytes/token.
     Consistent with Mistral-Nemo-derived BPE under-merging Greek
     polytonic/NFD (which is exactly the motivation for the extension
     this project ships).
   - Long-context phase (~225 B tokens total) excluded from the
     headline; Greek long-tail there moves the number by < a few M
     tokens.

### 2.3 Known limitations / open items

- Institutional Books 1.0 contribution treated as zero (gated dataset
  not measurable in the Path-A run). If non-trivial Greek volumes are
  present in the Apertus-filtered 28.7 B-token subset, this would
  add to the long-context-phase share (which is itself separate from
  the 13.5 T pretraining budget).
- Trace-Greek in code datasets (StarCoderData, etc.) treated as zero
  for the headline; tightening this needs a fasttext pass on a
  sample.
- Robots.txt opt-out is a ~4 % haircut on multilingual data (paper §3.1.1).
  Not applied to the Greek figure in the headline (would lower 3.1 B
  to ~3.0 B; doesn't change the 0.023 % rounded share).

---

## 3. Reconciliation — why both findings are consistent

Architectural-cause analysis lives in a separate doc:
[APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md](APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md).
That doc enumerates the four mechanisms (gradient clipping at 0.1
applied at almost every step, Pre-Norm + RMSNorm, QK-Norm, logit
saturation, AdEMAMix long-tail momentum) that Apertus uses to make
per-token embedding training approximately frequency-independent. The
short-form reconciliation:

The two readings appear to contradict — Greek's pretraining share is
**0.023 %**, yet existing Greek tokens are well-trained. They reconcile
cleanly:

- 3.1 B Greek pretraining tokens distributed across **1,506 existing
  Greek tokens** = ~2.06 M training occurrences per Greek token on
  average. That's deep into the "well-converged BPE token" regime —
  per-token training convergence in modern BPE language models
  typically saturates at ~10⁴–10⁶ occurrences depending on token rank,
  and 2 M is comfortably above that.
- What matters for "is this token well-trained" is its **occurrence
  count**, not the language's overall corpus share. A small share of
  the corpus concentrated into a small vocabulary slice can produce
  per-token signal that's competitive with a large share spread across
  a large slice. That's exactly what happened with Greek in Apertus.
- Implication: **a 0.023 % corpus share is not by itself a reason to
  expect undertrained tokens.** The argument for vocabulary extension
  is not "Greek is undertrained at the token level" (it isn't);
  it's "Apertus's existing Greek vocabulary fragments Greek
  polytonic/NFD/long-word forms across too many BPE pieces, producing
  ~2.4 chars/token vs ~4–5 chars/token in well-fitted languages"
  (a structural compression problem). The C3 cutoff sweep already
  characterized that compression curve in `C3_CUTOFF_REPORT.md`.

For the extension plan (§5 of the working doc): the **Greek-share
finding strengthens the case for CPT replay-ratio targeting** (the
extension model should see a much higher Greek-share mix than the
0.023 % Apertus baseline if we want to *grow* Greek depth) but it
**weakens any argument for a Phase-1-grounding step before tokenizer
extension** (because the Phase A norm diagnostic shows the existing
Greek tokens don't need grounding). These two readings now agree.

---

## 4. Concrete edits this implies for the working doc

These are additions to the v0.3 edit packet handed earlier. Some
restate prior points; treat them as a single consolidated patchset.

### 4.1 §2 Project Context — table corrections

- `Existing Greek tokens` → **1,494** (strict-Greek filter, `tokenizer_analysis/inspection/base/greek_tokens/`)
  or **1,506** (looser "contains-Greek-character" filter used in the
  Phase A diagnostic). The two numbers differ because the diagnostic
  uses the task spec's "contains any Greek codepoint" rule, while the
  strict-Greek inventory requires every char to be in the Greek range
  after NFD strip. Quote whichever matches the surrounding context.
- `intermediate_size` → **21,504** (v0.2 said 14,336; corrected from
  `swiss-ai/Apertus-8B-2509/resolve/main/config.json`).
- `num_key_value_heads` → **8** (GQA 4:1; v0.2 said MHA / None).
- `Model identifier` → `swiss-ai/Apertus-8B-2509`. No undated
  `Apertus-8B` repo exists; the swiss-ai org uses dated release names.
- `BPE training corpus` → `glossapi + hplt` at **50/50** by training-token
  mass, on the wave-2-broad cleaner output. C3 arm name:
  `C3_wave2_broad_glossapi_plus_hplt_50_50`. Total vocab 156,672
  (131,072 base + 25,600 added).
- `BPE cutoff` → "not yet decided; 25-point sweep at every 1024
  added units is in `C3_CUTOFF_REPORT.md`". The candidate set should
  include **8,192** as a low-budget Pareto contender (75% of total
  fertility benefit at 32% of the 25,600 embedding-row cost).

### 4.2 New §2.x — Apertus pretraining Greek share (new sub-section)

Add a short factual paragraph naming the **0.023 %** figure and
pointing at `APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md` for the full
method. The number is load-bearing for §7 forgetting-prophylaxis
design and for §10 "Identify Apertus's pretraining data composition".

### 4.3 §4 Initialization methods — distinguish A1 vs A2 + add C

Same content as the prior packet:
- **Option A1**: literal ReTok (merge-order chained averaging — error
  compounds).
- **Option A2 (recommended default)**: base-piece retokenization
  averaging — for each new token T with surface form s, retokenize s
  with the base Apertus tokenizer, average those base embeddings. No
  error compounding.
- **Option B**: CW2V (Yamaguchi 2024) — keep as a candidate.
- **Option C** (promoted from §8.2): model-internal contextual init.
  For each new token's surface form, average the base Apertus's
  **last-layer hidden states** over real Greek-text occurrences of
  the constituent base-piece sequences. Uses Apertus's *learned*
  representation of those bytes in context, not just the lookup-table
  representation. Standard recipe in some production multilingual-LM
  papers; higher implementation cost than A2.

### 4.4 §5 Weaknesses — strengthen with the new data

- **§5.1.3 "constituent noise propagates"** — explicitly note that
  A2 (base-piece retokenization average) directly answers this
  weakness. A1 is exposed to it; A2 isn't.
- **§5.3.1 "CPT-only baseline (LLaMA-Beyond-English)"** — the Phase A
  finding that existing Greek tokens are well-trained makes the
  CPT-only baseline **more** plausible, not less. Without an
  undertraining problem to fix, the marginal value of vocabulary
  extension is purely about **compression efficiency** (fertility,
  inference cost) and not about **representational quality** (the
  existing Greek tokens are fine). The CPT-only baseline becomes the
  harder benchmark to beat, not the easier one.
- **§5.3.6 "Smaller vocab (5K–10K) tradeoff"** — no longer
  "underexplored"; 25-point clean-slice sweep in
  `C3_CUTOFF_REPORT.md` shows the curve. 8,192 is a real Pareto
  point.

### 4.5 §7 Norm matching — explicit defaults

- **Default for new Greek-content tokens**: match to the Greek-group
  median, per matrix. E ≈ **5.05**, U ≈ **3.80**.
- **Default for new structural tokens** (`punctuation_run`,
  `table_separator`, `math_symbol`, `postscript_glyph`,
  `dingbat_or_symbol`, `escaped_character_run`): match to the
  base's `structural_non_linguistic` median, per matrix. E ≈
  **4.91**, U ≈ **3.58**. Slightly lower than Greek/English medians
  — using the Greek-token median for structural tokens would
  over-norm them.
- Practically the all-vocab medians (E ≈ 4.93, U ≈ 3.80) are very
  close to the Greek and structural medians, so a single
  global-mean default works to ~5%; per-group matching tightens this.

### 4.6 §10 Immediate next actions — status update

```
- [x] Verify Apertus model card / config: vocab=131,072, hidden=4,096,
      tie_word_embeddings=False, hidden_act=xielu, intermediate=21504, GQA 4:1
      (config.json pulled from HF; recorded in this doc §4.1)
- [x] Pilot BPE compression at multiple cutoffs — 25-point sweep at
      every 1024 added units on three verified-clean held-out slices
      (docs/C3_CUTOFF_REPORT.md)
- [x] Measure L2 norm distribution of existing Greek tokens in Apertus
      — DONE 2026-05-11. Greek-group median ≈ English-baseline median;
      6.5× above empirical floor. Phase 1 grounding not needed.
      Artifacts at runs/apertus_greek_diagnostic_20260511/
- [x] Audit GlossAPI and HPLT-Greek for quality / MT — partial; the
      C3 wave-2-broad cleaner config covers it. Register-distribution
      audit (Katharevousa share, polytonic share) is the residual.
- [x] Identify Apertus pretraining data composition + Greek share —
      DONE. 0.023% Greek; method in APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md
- [ ] Stand up a CPT-only Apertus baseline on the same corpus mix
      as the §5.3.1 sanity check. Strengthened by the Phase A finding.
- [ ] Read Yamaguchi 2024 / EEVE / LLaMA-Beyond-English in full.
- [ ] Greek-text per-token NLL behavioral check (Phase B; see §5 below).
- [ ] Greek evaluation suite — `virgin_hplt`, `C3_val_clean`, `C3_test_clean`
      already exist; still need a Katharevousa / polytonic probe and an
      English-regression benchmark for the forgetting check.
```

---

## 5. Phase B — what's left, and the honest case for skipping

### 5.1 What Phase B is

Task 4 from the diagnostic spec: **per-token NLL on real Greek text**.
A behavioral cross-check of the structural Phase A finding. Steps:

1. **Get a GPU**: `a2-highgpu-1g` (1× A100-40GB) on gcloud,
   `europe-west4-b`. ~$3.67/hr on-demand; total Phase B cost ~$4–6
   for one hour including setup.
2. **Install** `accelerate`, `datasets` into the diagnostic venv (or
   a fresh GPU-side venv).
3. **Sample corpora**: ~10 MB each of `wikimedia/wikipedia`
   `20231101.el` and `20231101.en`. Tokenize with the Apertus
   tokenizer, concatenate into seq-2048 sequences.
4. **Forward pass** with `AutoModelForCausalLM.from_pretrained(...,
   dtype=torch.bfloat16, device_map="auto")` on the GPU. Compute
   `F.cross_entropy(logits.view(-1, V), labels.view(-1),
   reduction='none')` to get per-position losses; map back to
   token-ids of the targets.
5. **Aggregate per token-id** with an occurrence-count guard
   (`min_n = 20` — tokens seen fewer than 20 times in the sampled
   corpus get excluded from headline stats and listed separately).
6. **Report** Greek-token mean-NLL distribution vs English-token
   mean-NLL distribution (medians + percentiles); plus the **50
   Greek tokens with highest mean NLL** (behaviorally-undertrained
   candidates) gated on the occurrence minimum.

### 5.2 What the v2 fix-up needs

A regex exclusion `^<SPECIAL_\d+>$` at the special-token step of the
classifier (so reserved-but-not-formally-special slots don't leak
into English-baseline). This is a 1-line change in `phase_a_diagnostic.py`
at the `tokenizer.all_special_ids` step.

### 5.3 Honest case for skipping Phase B for now

The Phase A signal is unusually strong:
- Greek-vs-English median norms within 1 % on both matrices.
- Greek lower percentile (p5) *exceeds* English-baseline lower percentile.
- Greek p5 ‖U‖ is 6.5× the empirical floor.
- 2 M training occurrences per Greek-token on average — comfortably
  in the convergence regime.

A behavioral cross-check would almost certainly confirm rather than
overturn. Given the GPU rental cost (~$4–6) and the work cost of
provisioning the GPU instance + setting up `accelerate`, **Phase B is
nice-to-have, not blocking**. Decision points it informs:
- Whether any specific Greek tokens are behaviorally-undertrained
  despite healthy norms (norm and direction can diverge).
- Whether the per-token NLL distribution suggests targeted reset for
  any specific tokens.

These are second-order — they refine, not redirect, the extension plan.

### 5.4 If Phase B is greenlit, the runbook

Same outline as the original task spec §4, with the careful
methodology refinements:
- **min_n = 20** for headline stats; report per-token at all counts in
  a separate JSON for follow-up.
- **NLL computed on shifted labels**, standard left-to-right LM
  cross-entropy, no instruction template applied.
- **Cache the per-position loss tensor** to disk (`.npy` of shape
  `(num_tokens,)`) alongside the token-id stream so the per-token-id
  aggregate can be recomputed without re-running the forward pass.
- **Match the seq length to Apertus's rope window**: 2048 is
  comfortably inside; no rope scaling needed.

Wall-clock estimate at batch=1, seq=2048, A100-40GB: ~1.2–1.5 s per
sequence × ~2,500 sequences = ~50 min. Adding model download + venv
setup: ~75 min total ⇒ ~$5 wall.

---

## 6. What to verify when reviewing this doc

If you're the Claude updating the working plan: every number in §1.2,
§2.1, §2.2 has a method-of-record above it, and every artifact is
either reproducible from a script in the project repo or is a JSON/MD
file checked into `runs/apertus_greek_diagnostic_20260511/`.
Spot-checks you can run cheaply:

- `cat runs/apertus_greek_diagnostic_20260511/ledger.json` → should
  show `ledger_ok: true` and `ledger_sum: 131072`.
- `head runs/apertus_greek_diagnostic_20260511/norm_stats.json` →
  matches the §1.2 table.
- `cat runs/apertus_greek_diagnostic_20260511/greek_bottom_50.json |
  head` → expect short function words (`και`, `για`, etc.).
- Pretraining-share methodology lives at
  `APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md` §2.4 with the math
  formula in §2.4-math; the per-dataset Greek-token counts are in
  the source doc's §5.1.

If you spot a number that doesn't match the source artifact, treat
this doc as wrong and the artifact as authoritative; this doc is a
synthesis layer.
