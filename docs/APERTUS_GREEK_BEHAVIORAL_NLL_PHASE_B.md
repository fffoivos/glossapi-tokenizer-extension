# Apertus-8B-2509 — Greek-token behavioral NLL (Phase B v4 + v5 triangulation)

Date: 2026-05-12. Replaces (and corrects) the earlier Phase B v3
version of this doc; v3's headline finding was a sampling artifact,
and the corrected v4 picture is materially different. **v5 added a
3-language triangulation (Hindi, Hebrew, Arabic) to disentangle the
small-vocab mechanism from the Greek-specific lift** — see new §6.
Sister docs:
- [APERTUS_GREEK_DIAGNOSTIC_PHASE_A.md](../runs/apertus_greek_diagnostic_20260511_v2/apertus_greek_diagnostic.md) — norm-side diagnostic
- [APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md](APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md) — Greek share of Apertus pretraining
- [C3_CUTOFF_REPORT.md](C3_CUTOFF_REPORT.md) — fertility / compression analysis for the C3 cutoff decision

---

## 1. Headline finding (v4 + v5)

On diversified, register-matched held-out corpora, Apertus-8B-2509
predicts **modern Greek** (both web and academic) at **median NLL ≈
0.95** — about 3× lower than any of English / German / Russian on
their own native web corpora. The existing 1,506 Greek tokens are
not only well-trained (Phase A norm distribution) and well-utilised
(Phase A v2 ledger) but also **well-predicted in context** across the
two Greek registers that the C3 BPE extension actually targets.

**v5 update**: a 3-language triangulation (Hindi, Hebrew, Arabic)
confirms this lift is **not** purely a small-vocab metric artifact.
Hindi has the same vocab size as Greek (1,497 vs 1,496) and is 2×
harder (median 2.00); Hebrew is even smaller-vocab and still ~1.8×
harder (1.69); Arabic shares Greek's HQ-20 secondary training ring
but has a 6× larger vocab and is hardest (3.35). Greek's low NLL is
a real conjunction of small dedicated vocab + HQ-20-secondary training
filter + post-filter stylistic narrowness + Greek-specific orthographic
regularity. Details in §6.

| slice (v4, diversified)                                                       | native group       | median NLL | n token-ids (≥ min_occ) |
| ---                                                                            | ---                | ---:       | ---:                    |
| `hplt_el` (web Greek, 5,262 docs / 2,375 domains)                              | Greek              | **0.958**  | 1,493                   |
| `glossapi_el_modern` (modern monotonic Greek, 4,035 docs / many sources)       | Greek              | **0.942**  | 1,495                   |
| `hplt_en` (web English, 3,671 docs / 3,144 domains)                            | English-baseline   | 3.025      | 10,267                  |
| `hplt_de` (web German, 4,777 docs / 4,432 domains)                             | German             | 2.714      | 860                     |
| `hplt_ru` (web Russian, 4,135 docs / 3,688 domains)                            | Cyrillic           | 2.448      | 5,910                   |

**Decision implication**: the case for the C3 tokenizer extension is
purely about **compression economy** (chars/token, fertility,
inference cost) and not about lifting Greek prediction quality on the
target deployment register. The existing Greek tokens already predict
modern Greek as well or better than Apertus predicts English on
English. The earlier v3 finding that "GlossAPI register is harder
than HPLT register" was a register confound (v3's GlossAPI sample was
dominated by ancient / Katharevousa / polytonic Greek; once you
filter to modern monotonic, GlossAPI is essentially identical to
HPLT). **Out-of-scope**: ancient / polytonic Greek IS genuinely hard
for Apertus, but it is not in the C3 BPE training mix
(`wave-2-broad` cleaner + `greek_badness_score ≤ 60` strips most of
it) nor in the planned deployment register, so it is not a target of
this project.

---

## 2. Method (v4, auditable end-to-end)

### 2.1 Model and tokenizer

- Repo: `swiss-ai/Apertus-8B-2509`
- Resolved commit: `3162c9967…`
- Load: `AutoModelForCausalLM.from_pretrained(..., dtype=torch.bfloat16,
  device_map="auto")` on a single A100-40GB. Single device.
- Tokenizer: `AutoTokenizer.from_pretrained("swiss-ai/Apertus-8B-2509")`.

### 2.2 Token classification (carried over from Phase A v2)

Same byte-level pipeline as
`runs/apertus_greek_diagnostic_20260511_v2/phase_a_diagnostic.py`:
GPT-2 byte_to_unicode unmapping → UTF-8 decode → `<SPECIAL_NNN>`
regex exclusion + special-token exclusion → script-range tests for
Greek (U+0370–03FF ∪ U+1F00–1FFF), CJK, Cyrillic, German marker
chars, French marker chars excluding German, English-baseline
(ASCII-only + ≥3 alpha) → first-class `structural_non_linguistic`
group for tokens entirely in Unicode P*/S*/Z* categories.

### 2.3 Sampling (v4 — the load-bearing change from v3)

The samples are produced in a separate, CPU-only step on `home`
before any GPU forward pass. Three things changed vs v3:

1. **HF streaming shuffle**. `ds.shuffle(seed=20260512,
   buffer_size=100_000)` is applied before iterating. v3 took whatever
   the parquet shard order produced (which for Greek HPLT meant
   94.6% of docs came from a single domain in a single 2016
   crawl). v4 draws a randomised mix from the first 100k docs of
   each slice, then the next 100k, etc. — much more representative.
2. **GlossAPI register filter**. Drop docs where
   `is_historical_or_polytonic == True` OR `polytonic_ratio > 0.05`.
   v3 hit a Galen-era text in its first row and the sample was
   dominated by ancient / Katharevousa content; v4 isolates modern
   monotonic Greek (the register C3 actually trains for).
3. **Stratified sampling for GlossAPI**. GlossAPI's HF release has
   many source-tagged shards (`openarchives.gr`, `paper_*`, `opengov_*`,
   `wikisource_*`, `klassiki_*`, `ekkl_*`, plus chunked `train_chunks` /
   `test_chunks` shards). v4 round-robins parquet files with a
   per-source char cap (3 MB) so no single source dominates.
4. **Diversity audit before NLL**. Each sample is audited for
   domain / source variety before paying for GPU compute. The audit
   ran in 1–2 minutes per slice; only after all five passed did we
   start the GPU pass.

### 2.4 Held-out corpus + diversity audit (the five slices)

All five samples target ~10 MB of UTF-8 text each.

| slice                | dataset                                                  | config       | filter applied during streaming                                                                  |
| ---                  | ---                                                      | ---          | ---                                                                                              |
| `hplt_el`            | `fffoivos/hplt-greek-ge8-no-mt-clean60-wave4`            | (default)    | upstream pre-filtered; just shuffle + sample                                                     |
| `glossapi_el_modern` | `fffoivos/glossapi-greek-nanochat-pretraining-dataset`   | (default)    | upstream pre-filtered, **plus drop is_historical_or_polytonic + polytonic_ratio > 0.05**, **plus stratified across shards** |
| `hplt_en`            | `HPLT/HPLT2.0_cleaned`                                   | `eng_Latn`   | `max(doc_scores[:-1]) ≥ 8 AND doc_scores[-1] < 3` (approx. bin ≥ 8 + no-MT), then shuffle        |
| `hplt_de`            | `HPLT/HPLT2.0_cleaned`                                   | `deu_Latn`   | same gate                                                                                        |
| `hplt_ru`            | `HPLT/HPLT2.0_cleaned`                                   | `rus_Cyrl`   | same gate                                                                                        |

Audit results (per-slice diversity, recorded in
`runs/apertus_greek_phase_b_v4_20260512/<slice>_audit.json`):

| slice                | docs   | unique domains / sources | top-1 share        | crawls             |
| ---                  | ---:   | ---:                     | ---                | ---                |
| `hplt_el`            | 5,262  | 2,375 web domains        | 14.9% blogspot.com | 4 (2017–2022)      |
| `glossapi_el_modern` | 4,035  | 200+ doc-id prefixes (papers / opengov / wikisource / ekkl / openarchives / klassiki etc.); 80 file-level shards (after dropping the embedded HPLT shard) | ~25% per file-shard | n/a (mixed academic sources) |
| `hplt_en`            | 3,671  | 3,144 web domains        | 5.1% co.uk         | 1                  |
| `hplt_de`            | 4,777  | 4,432 web domains        | 0.8% blogspot.com  | 1                  |
| `hplt_ru`            | 4,135  | 3,688 web domains        | 1.5% com.ua        | 2                  |

For comparison, v3's Greek HPLT sample (203 docs unshuffled) had **11
unique domains, top-1 share 94.6% docplayer.gr, 1 crawl** — the
single-domain dominance is what produced v3's "Greek HPLT is easy"
(median 1.18) reading. v4's diversified sample lands at median 0.958
on the same metric, slightly *better* than v3's biased sample.

### 2.5 Forward pass

- Read each saved parquet on the GPU instance (no re-streaming).
- For each doc: tokenize, pack into seq=2048 windows, forward through
  Apertus in bf16, compute `F.cross_entropy(reduction='none')` on
  shifted positions in fp32.
- Aggregate per-token-id: `np.add.at(sum_loss, labels, loss)` and
  per-doc: mean of position-losses across the doc.
- Per-token mean NLL gated on `count ≥ 20` for headline stats.

### 2.6 Hardware + cost

- 1× NVIDIA A100-SXM4-40GB (`a2-highgpu-1g` in `europe-west4-a`).
- Total wall: ~37 minutes (model load + 5 slices × per-doc forward).
- Run cost: ~$2.30 GPU at $3.67/hr on-demand.
- Sampling step (CPU on home): ~5 minutes, free.
- Instance left **stopped, not deleted** — disk preserves the venv +
  model cache + arrays.

---

## 3. Detailed results (per group × slice)

Lower mean NLL is better predicted. Counts are token-ids in that
group meeting `min_occ = 20` on that slice.

### Native group on its own slice (headline rows)

| slice                | native group     | count | p5    | p25   | p50   | p75   | p95   | mean  |
| ---                  | ---              | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| `hplt_el`            | Greek            | 1,493 | 0.164 | 0.443 | **0.958** | 3.125 | 4.243 | 1.725 |
| `glossapi_el_modern` | Greek            | 1,495 | 0.155 | 0.432 | **0.942** | 2.950 | 4.026 | 1.661 |
| `hplt_en`            | English-baseline | 10,267 | 0.485 | 2.005 | **3.025** | 3.949 | 5.320 | 2.971 |
| `hplt_de`            | German           | 860   | 0.291 | 0.750 | **2.714** | 4.002 | 5.512 | 2.579 |
| `hplt_ru`            | Cyrillic         | 5,910 | 0.331 | 0.819 | **2.448** | 3.626 | 4.868 | 2.382 |

### Cross-group rows (English-baseline + structural on every slice)

These show how English-baseline tokens behave when they appear inside
non-English text, and how structural tokens (punctuation_run /
table_separator / symbol-only) behave per-slice.

| slice                | group                          | count  | p5    | p25   | p50   | p75   | p95   | mean  |
| ---                  | ---                            | ---:   | ---:  | ---:  | ---:  | ---:  | ---:  | ---:  |
| `hplt_el`            | English-baseline               | 312    | 0.187 | 0.927 | 2.174 | 3.332 | 5.340 | 2.290 |
| `hplt_el`            | structural_non_linguistic      | 124    | 0.975 | 1.754 | 2.823 | 4.094 | 6.646 | 3.117 |
| `glossapi_el_modern` | English-baseline               | 2,294  | 0.384 | 1.334 | 2.271 | 3.260 | 4.692 | 2.356 |
| `glossapi_el_modern` | structural_non_linguistic      | 232    | 0.464 | 1.162 | 2.050 | 3.144 | 5.047 | 2.355 |
| `hplt_en`            | structural_non_linguistic      | 149    | 0.693 | 1.466 | 2.357 | 3.650 | 5.022 | 2.598 |
| `hplt_de`            | English-baseline               | 8,165  | 0.312 | 0.997 | 2.813 | 4.063 | 5.662 | 2.727 |
| `hplt_de`            | structural_non_linguistic      | 152    | 0.713 | 1.501 | 2.582 | 3.549 | 5.106 | 2.684 |
| `hplt_ru`            | English-baseline               | 203    | 0.125 | 0.756 | 1.678 | 2.571 | 4.124 | 1.820 |
| `hplt_ru`            | structural_non_linguistic      | 143    | 0.661 | 1.446 | 2.453 | 3.620 | 5.051 | 2.569 |

Patterns:
- English-baseline tokens appearing inside non-English slices are
  predicted *better* than they are on English text itself (1.68 in
  Russian, 1.83 in Russian, 2.17 in Greek, 2.27 in modern Greek vs
  3.03 on English-English) — they're the high-frequency English
  tokens (brand names, URLs, code-switched tokens) which sit in easier
  contexts than the wider English vocabulary distribution.
- Structural tokens are consistently 2.0–3.1 median across all
  slices — no register surprises.

### Per-doc mean NLL (5,262 hplt_el docs as example)

| stat   | value |
| ---    | ---:  |
| p5     | 1.055 |
| p25    | 1.267 |
| median | 1.420 |
| p75    | 1.598 |
| p95    | 1.870 |
| mean / std | 1.437 / 0.257 |
| min / max  | 0.456 / 2.885 |

Per-doc CSVs for all 5 slices at
`runs/apertus_greek_phase_b_v4_20260512/per_doc/<slice>.csv` (≈22k
total docs × per-doc mean/std NLL). Within-slice variance is small
on `hplt_el` (std 0.257 around median 1.42) — confirming the
diversified sample doesn't have huge per-doc outliers, even with
2,375 unique domains.

---

## 4. Top-20 behaviourally-worst Greek tokens (combined modern Greek)

Combined sum-loss + count across `hplt_el` + `glossapi_el_modern`,
gated on combined `count ≥ 20`. These are the worst-predicted modern
Greek tokens despite their Phase A norms being healthy.

| rank | id     | count | mean NLL | decoded     |
| ---: | ---:   | ---:  | ---:     | ---         |
| 1    | 123557 | 50    | 5.745    | `Κων`       |
| 2    | 102014 | 97    | 5.536    | `Στις`      |
| 3    | 90797  | 76    | 5.447    | `λει`       |
| 4    | 88372  | 159   | 5.296    | `αποτέλε`   |
| 5    | 65759  | 182   | 5.216    | `δημι`      |
| 6    | 69161  | 39    | 5.193    | `χαρα`      |
| 7    | 94465  | 459   | 5.190    | `Στις`      |
| 8    | 51004  | 166   | 5.027    | `χρησιμο`   |
| 9    | 63665  | 470   | 5.001    | `Ω`         |
| 10   | 117742 | 376   | 4.970    | `απέ`       |
| 11   | 93612  | 282   | 4.947    | `βρίσκ`     |
| 12   | 59322  | 547   | 4.899    | `μεγά`      |
| 13   | 127461 | 725   | 4.885    | `επαν`      |
| 14   | 72840  | 147   | 4.831    | `υπάρχ`     |
| 15   | 107216 | 1,010 | 4.804    | `κυρίως`    |
| 16   | 130242 | 1,302 | 4.769    | `ακόμα`     |
| 17   | 114553 | 588   | 4.753    | `Μετά`      |
| 18   | 116236 | 318   | 4.752    | `άρχ`       |
| 19   | 89488  | 158   | 4.745    | `δεκα`      |
| 20   | 110515 | 1,621 | 4.720    | `εντ`       |

These are mostly mid-length stems / partial stems whose continuation
is locally underdetermined — `Κων` is the start of dozens of names
(Κωνσταντίνος, Κωνσταντίνου, Κωνσταντίνα, Κωνσταντινούπολη…), `λει`
prefixes many words, `μεγά` likewise. The worst-token list isn't
about register difficulty; it's about local-context ambiguity in a
heavily-inflected language. Full top-50 at
`runs/apertus_greek_phase_b_v4_20260512/greek_worst_50_combined.json`.

---

## 5. Reconciliation across measurements

| measurement                                                        | Greek reading                                                                | corpus + clean status                                       |
| ---                                                                | ---                                                                          | ---                                                         |
| Phase A v1 (norm distribution)                                     | Greek ≈ English-baseline                                                     | vocab-only                                                  |
| Phase A v2 (norm, with `<SPECIAL_NNN>` regex fix)                  | Greek ≈ English-baseline                                                     | vocab-only                                                  |
| Phase B v1 (Wikipedia)                                             | Greek > English (median 0.88 vs 1.88) — but corpus partially in Apertus    | `wikimedia/wikipedia` el + en                              |
| Phase B v3 on HPLT Greek                                           | Greek > English (1.18 vs 2.99) — single-domain artifact                     | unshuffled HPLT clean60                                     |
| Phase B v3 on GlossAPI Greek                                       | Greek ≈ English (3.11 vs 2.99) — polytonic-contaminated                     | unfiltered GlossAPI                                         |
| **Phase B v4 on HPLT Greek (diversified)**                         | Greek ≫ English (**0.96** vs 3.03)                                          | shuffled HPLT clean60, 5,262 docs / 2,375 domains          |
| **Phase B v4 on GlossAPI Greek (modern monotonic only)**           | Greek ≫ English (**0.94** vs 3.03)                                          | stratified GlossAPI, ancient/polytonic dropped              |

The v3 reading **directly contradicted itself across registers**
(easy on HPLT, hard on GlossAPI) for reasons that turned out to be
sampling biases, not real register differences:
- v3 HPLT made Greek look unusually easy because docplayer.gr (slide
  presentations) is more locally predictable than typical Greek web.
- v3 GlossAPI made Greek look unusually hard because the first
  parquet returned by the dataset's shard order was dominated by
  Katharevousa / Galen-era Greek that Apertus genuinely struggles
  with — but that register is not in the C3 training mix.

v4 controls for both. The corrected picture: modern Greek (web +
academic) is uniformly easy for Apertus, ancient/polytonic is
uniformly hard, and the C3 BPE training mix only targets the modern
register.

---

## 6. Triangulation: is Greek special, or is it just the small vocab?

A natural sceptical question: Greek's median NLL of 0.958 is much
lower than English-on-English (3.025). Is that just because Greek has
a small vocab (1,496 tokens vs English-baseline's 73,910 tokens) — so
the per-position prediction problem is mechanically easier? Or is
there something Greek-specific (training-data quality, BPE quality,
stylistic uniformity) that further reduces the NLL?

To disentangle, v5 measures three additional languages on their own
HPLT-clean slices, each picked to vary one structural axis at a time
relative to Greek.

### 6.1 Comparator selection

| candidate | vocab match | training-route match | dedicated script |
| ---       | ---         | ---                  | ---              |
| **Hindi** (Devanagari) | 1,497 vs 1,496 — essentially equal | **NOT in HQ-20**; long-tail `random-33%` route in Apertus | yes |
| **Hebrew** | 955 — smaller than Greek | **NOT in HQ-20**; long-tail route | yes |
| **Arabic** | 9,392 — 6.3× Greek's vocab (single Arabic-script vocab covers Arabic + Persian + Urdu + …) | HQ-20 secondary ring, same `p × 0.95` as Greek | yes |

Hindi tests "what if vocab is the same size but training-data route
is different". Hebrew tests "what if vocab is smaller but training is
weaker". Arabic tests "what if training route is the same but vocab
is much larger".

No single language matches both Greek's vocab size AND Greek's
training-data route — Greek is structurally unique among Apertus's
supported languages (smallest dedicated-script vocab in the HQ-20
secondary ring). The triangulation reads the pattern instead of any
single comparison.

### 6.2 Sampling for v5

Same shuffle + quality gate as v4. Each slice is ~10 MB UTF-8 text
sampled with `shuffle(seed=20260512, buffer_size=100_000)` and the
HPLT v2 `doc_scores ≥ 8 AND MT < 3` gate. Diversity confirmed
before GPU spend:

| slice    | docs   | unique TLD+1 domains | top-1 share | notes |
| ---      | ---:   | ---:                 | ---:        | ---   |
| `hplt_hi` | 3,520 | 2,124 | 1.5% co.in | Indian web (jagran, news18, indiatimes) |
| `hplt_he` | 4,636 | 823 | 66.0% co.il | Note: `co.il` is a ccTLD covering many distinct Israeli sites under one bucket; actual host-level diversity is much higher than 823 suggests. |
| `hplt_ar` | 3,422 | 2,587 | 1.0% alibaba.com | Pan-Arabic web; top hosts span e-commerce, news, blogs |

(Note: HPLT v2's Arabic config key is `ara_Arab`, not `arb_Arab`.)

### 6.3 Triangulation results

Native script tokens on the native HPLT slice (the "is this language's
own tokens easy to predict on its own text?" measurement):

| script-on-its-own-HPLT-slice | vocab tokens | n token-ids ≥ min_occ | median NLL | mean NLL |
| ---                          | ---:         | ---:                  | ---:       | ---:     |
| **Greek-on-`hplt_el`**       | 1,496        | 1,493                  | **0.958**  | 1.725    |
| Devanagari-on-`hplt_hi`      | 1,497        | 1,457                  | 2.003      | 2.090    |
| Hebrew-on-`hplt_he`          | 955          | 953                    | 1.688      | 2.130    |
| Arabic-on-`hplt_ar`          | 9,392        | 7,396                  | 3.350      | 3.239    |

For comparison, the v4 native-on-native medians on the same plot:

| slice (v4) | native group | median NLL |
| ---        | ---          | ---:       |
| `hplt_en`  | English-baseline | 3.025 |
| `hplt_de`  | German       | 2.714 |
| `hplt_ru`  | Cyrillic     | 2.448 |

### 6.4 What the triangulation says

**The "Greek is easy because its vocab is small" story is incomplete.**
Hindi has the same vocab size (off by 1 token) and lands at median
2.003 — more than 2× harder than Greek. Hebrew has a *smaller* vocab
than Greek (955 vs 1,496) and still lands at 1.688, ~75% harder than
Greek. Vocab-size-alone would predict Hebrew ≤ Greek ≤ Hindi; what we
observe is Greek « Hebrew < Hindi. Vocab size matters but is not
sufficient.

**HQ-20 ring membership alone also doesn't explain it.** Arabic is in
the same HQ-20 secondary ring as Greek, got the same `p × 0.95`
filter, and lands at the highest median NLL of the four (3.35). So
"HQ quality filter ⇒ low NLL" is also false on its own.

**The configuration that makes Greek easy is the conjunction:**
1. Small dedicated-script vocab (~1,500 tokens) → small per-position
   candidate set
2. HQ-20-secondary-ring training route (`p × 0.95` of FineWeb-2-HQ
   Greek) → per-vocab-slot training is well-filtered for quality
3. Stylistic narrowness of post-filter modern Greek HPLT (already
   noted in §3)
4. Greek's morphological + orthographic regularity making BPE-local
   context very informative
5. Possibly: Apertus's Mistral-Nemo-derived tokenizer (`tekken v3`)
   happens to allocate Greek BPE merges in an unusually efficient way

Drop any one and the median NLL jumps:
- Drop HQ-20 (Hindi): 0.96 → 2.00, ×2.1.
- Drop both vocab size AND HQ-20 (Hebrew is still smaller vocab, but
  no HQ-20): lands at 1.69, ×1.8 — confirms that HQ-20 quality is a
  real lift.
- Keep HQ-20 but drop small vocab (Arabic): 0.96 → 3.35, ×3.5 — large
  vocab dominates the loss budget.

So the corrected v5 reading: Greek is **measurably more predictable
than its closest structural analogues**, not just metric-favoured.
Part of the lift is small-vocab structural; part is HQ-quality-filter
training; part is Greek-specific (orthography / morphology /
tokenizer-construction). The combined effect is real, not artefactual.

### 6.5 Implication for the extension plan

This refines but does not overturn the v4 conclusion. v4 said: the
existing Greek tokens are well-predicted on the target register, so
the extension's value is compression economy, not representational
lift.

v5 adds:
- The well-predicted-ness of existing Greek tokens is not just
  measurement luck — it survives controlling for vocab size (Hindi
  comparator), and for training route (Arabic comparator). It is a
  real property of Apertus's existing Greek vocabulary on modern
  Greek.
- Therefore, **a baseline that the extended model must not regress
  against is "Apertus base on the v4 Greek slices"** (median NLL
  0.958 on hplt_el, 0.942 on glossapi_el_modern). The extension's
  added Greek tokens are *new* vocab slots and won't have the
  benefit of two years of pretraining yet; CPT needs to bring them
  close to the base's per-token prediction quality, not just close to
  some intuitive "Greek-is-easy" floor.
- The "small vocab ⇒ low NLL by metric construction" interpretation
  in my earlier message is not supported by the triangulation and
  should not propagate into the planning doc.

### 6.6 Artifacts (v5)

- `runs/apertus_greek_phase_b_v4_20260512/comparator_vocab_counts.json` — Devanagari / Hebrew / Arabic / English-baseline vocab counts (strict-script classification).
- `runs/apertus_greek_phase_b_v4_20260512/comparator_group_stats.json` — per-script × per-slice NLL percentile table for the 3 v5 slices.
- `runs/apertus_greek_phase_b_v4_20260512/hplt_{hi,he,ar}.parquet` — the v5 sample inputs.
- `runs/apertus_greek_phase_b_v4_20260512/hplt_{hi,he,ar}_audit.json` — per-slice diversity audits.
- `runs/apertus_greek_phase_b_v4_20260512/per_doc/{hplt_hi,hplt_he,hplt_ar}.csv` — per-doc mean NLL CSVs.
- `runs/apertus_greek_phase_b_v4_20260512/phase_b_v5_nll_triple.py` — the script.
- Per-token-id `.npy` arrays for the 3 v5 slices live on the GPU
  instance's persistent disk (preserved through stop). Re-sync if
  needed.

---

## 7. Implications for the extension plan

### 7.1 Recasting the value proposition (corrected)

The case for the C3 extension comes down to **compression economy**:

- Apertus already predicts modern Greek well per-token (median NLL
  ~0.95 across both web and academic Greek slices).
- The motivation for adding 8k–25k new Greek BPE units is therefore
  not "fix prediction quality on the deployment register" — that's
  already fine — but "reduce chars/token, which reduces inference
  cost, sequence length, and model-call latency on Greek workloads".
- The fertility curve in `C3_CUTOFF_REPORT.md` directly characterises
  this benefit; the cutoff decision should be driven by the elbow on
  that fertility curve and downstream economics, not by behavioural
  per-token NLL.

### 7.2 What this doc does NOT support

- "Lift the academic Greek register" — modern academic Greek doesn't
  need lifting per the v4 modern-monotonic GlossAPI slice. **The v3
  recommendation to rerun the cutoff sweep on a GlossAPI-register
  held-out slice is withdrawn.**
- "Fix undertrained Greek tokens" — Phase A v2 + Phase B v4 both
  show the existing Greek tokens are healthy.

### 7.3 Out-of-scope register flag

Ancient Greek / Katharevousa / polytonic registers are genuinely hard
for Apertus. We have not measured this directly in v4 (it was filtered
out), but the v3 GlossAPI signal (median 3.11 when ancient material
dominated the sample) suggests Apertus's per-token Greek-prediction
quality on polytonic material is more in line with English-on-English
than with modern-Greek-on-Greek. **If a future project decides to
target the ancient/polytonic register**, this is the gap to anchor on
and would call for separate evaluation, possibly a different
tokenizer-extension corpus, and a separate worst-token list.

### 7.4 Replay-ratio constraint for subproject 03

Updated from the v3 framing:

- Apertus pretraining: ~3.1 B Greek tokens (predominantly web register
  via FineWeb-2-HQ Greek), distributed over 1,506 existing Greek
  vocab slots ⇒ ~2 M training occurrences per Greek vocab token.
- Existing Greek tokens are well-predicted on modern Greek (median
  ~0.95 NLL on both HPLT and GlossAPI modern slices).
- The C3 extension adds N new Greek units (cutoff TBD). For those
  new units to be at least as well-trained as the existing 1,506,
  the CPT replay budget needs to bring each new unit's training
  exposure up to a comparable order of magnitude. With C3's training
  mix at `glossapi + hplt 50/50` and reasonable Greek-CPT volumes,
  hitting 100k+ exposures per new unit is feasible at all cutoffs in
  the C3 grid; tighter for the larger cutoffs.
- **Soft constraint**: pick a cutoff that fits the available Greek
  CPT budget such that each new unit gets ≥ 100k training occurrences
  during CPT. Below that, the new units risk being measurably less
  well-predicted than the existing ones.

---

## 8. Open items

- **Per-doc variance** is visible in the per-doc CSVs but not yet
  cross-correlated with `source_doc_id` prefixes. A follow-up could
  cluster the highest-NLL docs by source to see whether specific
  Greek registers (e.g. legal Greek, scientific Greek, code-heavy
  Greek pages) cluster around the per-doc-NLL p95 outliers.
- **Polytonic-register diagnostic**: if/when needed, an explicit
  polytonic+Katharevousa sample (the inverse of the v4 GlossAPI
  filter) would give a direct per-token NLL measurement on the
  ancient register. Not blocking for the C3 cutoff decision.
- **Post-extension behavioural re-test**: once the C3 extension is
  built at the chosen cutoff, repeat this Phase B test on the
  extended model. The headline metric should not regress on the modern
  Greek slices.

---

## 9. Artifacts

All on `home` at `runs/apertus_greek_phase_b_v4_20260512/`:

- Per-slice sample parquets (the inputs to the GPU pass):
  `hplt_el.parquet`, `glossapi_el_modern.parquet`, `hplt_en.parquet`,
  `hplt_de.parquet`, `hplt_ru.parquet`.
- Per-slice diversity audits:
  `<slice>_audit.json` (5 files, each with domain / crawl / source
  distribution).
- Per-slice per-doc mean NLL:
  `per_doc/<slice>.csv` (5 files, one row per doc with
  `mean_nll`, `std_nll`, `tokens_scored`, `chars`).
- Per-token-id arrays:
  `arrays/sum_loss_<slice>.npy`, `arrays/count_<slice>.npy`,
  `arrays/mean_nll_<slice>.npy` (per slice, vocab-length).
- Combined Greek sum + count + worst-50:
  `arrays/combined_greek_sum_loss.npy`, `arrays/combined_greek_count.npy`,
  `greek_worst_50_combined.json`.
- Cross-slice group×slice percentile table:
  `group_nll_stats.json`.
- Slice-level summary:
  `slice_summary.json`.
- Markdown report (auto-generated by the script, complementary to
  this curated doc):
  `apertus_greek_phase_b_v4.md`.
- Scripts (rerunnable):
  - sampling: `v4_sample_and_audit.py` (hplt_el), `v4_sample_remaining.py`
    (hplt_en/de/ru, glossapi initial), `v4_resample_glossapi.py`
    (stratified GlossAPI). All saved under
    `runs/apertus_greek_phase_b_v3_20260511/` for traceability.
  - NLL: `phase_b_v4_nll.py` saved alongside the v4 outputs.

GPU instance `apertus-greek-gpu-phaseb` (a2-highgpu-1g in
`europe-west4-a`) is **stopped, not deleted** — disk preserves venv,
model cache, intermediate arrays for a cheap restart.

---

## 10. Changelog vs v3 (and v4 → v5)

For traceability — what changed between the v3 version of this doc
and v4:

- **§1 headline**: v3 claimed "Greek HPLT is trivially easy (1.18),
  Greek GlossAPI is roughly as hard as English-on-English (3.11)".
  Both numbers were sampling artifacts (single domain on HPLT side,
  polytonic-dominated on GlossAPI side). v4 corrects both to
  ~0.95 on both Greek slices, ~3× better than English-on-English.
- **§2 sampling method**: v4 added shuffle + GlossAPI register filter
  + stratification. The HPLT v4 sample has 2,375 unique domains (vs
  v3's 11); the GlossAPI v4 sample drops ancient/polytonic content.
- **§3 detailed results**: replaced with v4 numbers across all 5
  slices.
- **§5 reconciliation**: v3's split picture is now understood as two
  separate sampling biases that cancelled the right reading.
- **§6 implications**: v3 argued the extension's value lives on
  GlossAPI register; v4 retracts this and re-anchors on compression
  economy. The recommendation to rerun the cutoff sweep on GlossAPI
  register is **withdrawn**.
- **§7 open items**: removed the "redo cutoff sweep on GlossAPI"
  item; added a polytonic-register diagnostic note as a separate
  potential future direction.

**v4 → v5** (this update):

- **New §6 (triangulation)**: added Hindi (Devanagari), Hebrew, and
  Arabic comparator slices to test whether Greek's low NLL is "just
  the small vocab" structurally. Hindi has the same vocab size as
  Greek and lands at median 2.00 (vs Greek's 0.96); Hebrew has a
  smaller vocab and lands at 1.69; Arabic is in the same HQ-20
  secondary ring and lands at 3.35. **Conclusion: vocab size alone
  does not explain Greek's lift; nor does HQ-20 ring membership
  alone. Greek's low NLL is a conjunction of small vocab + HQ-20
  training filter + post-filter HPLT-Greek stylistic narrowness +
  Greek's specific orthographic/morphological structure.**
- **§1 headline**: framing tightened — the v5 triangulation confirms
  the v4 numerical reading on Greek but corrects the casual
  explanation ("just the small vocab") that came up in a chat-side
  exchange. The Greek-is-easy result is **real**, not a metric
  artifact.
- **§7.5 (implicit)**: the comparator slices give the planning doc a
  concrete cross-language reference if it wants to reason about how
  Greek's per-token quality compares to similarly-positioned
  languages in Apertus. Hindi is the cleanest "what if vocab was
  matched but training route was different" counterfactual.
- **§9 artifacts**: added the v5 outputs (`comparator_*.json`,
  `hplt_{hi,he,ar}.parquet`, per-doc CSVs).
- **No section retractions vs v4** — the cutoff-decision-related
  recommendations in §7.1–§7.4 stand. v5 reinforces, not overturns,
  the v4 conclusions.
