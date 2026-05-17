# 02_1_7 Intrinsic Eval Sweep — Plan

**Status**: **IMPLEMENTED and shipped**. Original plan date 2026-05-17.
- Evidence + curves: see [`REPORT.md`](REPORT.md).
- Final decision + ship artifact: see [`CHOSEN_CUTOFF.md`](CHOSEN_CUTOFF.md).
- Canonical reproduction: `scripts/run_all.sh`.

This document is kept as the original plan record (what we said we'd
do); the report + decision documents are the source of truth for what
actually happened and what shipped.

---

Sub-subproject of `02_1_tokenizer_experiments`. **Downstream of
`02_1_4_cutoff_analysis` and `02_1_5_added_token_curation`** — both
have already done substantial work this sub-subproject builds on
rather than redoes.

```
[02_1_1 training]                  → full C3 arm at max vocab
[02_1_2 cutoff variant builder]    → emits tokenizer.json per cutoff (argparse-list)
[02_1_3 fertility evaluation]      → our internal metric harness on clean held-outs
[02_1_4 cutoff analysis]           → per-cutoff classification + distributions (1k grid)
[02_1_5 added-token curation]      → removal_list.jsonl (104 tokens; class B–F)
                  │
                  ▼
[02_1_7 intrinsic eval sweep]      ← THIS — TokEval suite on a 1k-spaced grid
                  │
                  ▼
[02_1_4 REPORT.md]                 ← updated to cite T1 + T2 results
[02_1_6 representation policy]     ← TFG curve cited as fairness evidence
```

## 1. Goal

Layer the **TokEval intrinsic-evaluation suite** (Meister 2025) on top
of the existing cutoff infrastructure to (a) match the 4-metric
surface area the Apertus paper uses for its own cutoff selection
(§2.2), (b) add a multi-criteria supplement that addresses the
reviewer objection that 4 metrics is too narrow, and (c) evaluate at
**1k cutoff granularity** instead of the original 4-point grid.

This sub-subproject **emits evidence**; cutoff selection stays with
the user.

## 2. What's already upstream (consumed, NOT re-built)

| upstream | what already exists | how 02_1_7 uses it |
|---|---|---|
| `02_1_2_cutoff_variant_builder/scripts/build_cutoff_variants.py` | `argparse`-based; takes a cutoff list and writes one HF tokenizer dir per cutoff. **Already parameterized — no extension needed.** | Call it once with the 13-point 1k grid; consume the emitted variant dirs. |
| `02_1_3_fertility_evaluation/scripts/run_tokenizer_fertility_suite.py` | Computes chars_per_token, tokens_per_byte, bytes_per_token, greek_word_space_fertility, single_token_greek_word_share, added_token_rate, eval_added_vocab_utilization_rate, eval_unused_added_tokens, unk_rate, byte_fallback_rate. Per-variant per-slice. | Run it on the 13-point grid for **continuity** with our existing metric definitions. T1 fertility + vocab utilization are produced by **two** harnesses (ours + TokEval) so we can document the methodology delta. |
| `02_1_3_fertility_evaluation/scripts/{build_virgin_hplt_eval,clean_holdouts}.py` + the resulting parquets at `/home/foivos/data/glossapi_work/...` | `virgin_hplt` (10k docs, source_doc_id anti-joined), `c3_val_clean` (val minus 30 train-overlap), `c3_test_clean` (test minus 36 train-overlap). Already integrity-checked. | Use as-is for the 3 in-house eval slices. **Do not re-derive.** |
| `02_1_4_cutoff_analysis/artifacts/cutoff_grid/distribution_at_{N}.json` (N = 1024, 2048, …, 25600 — already 1k-spaced) | Per-cutoff char-mask language distributions (analytical, not runtime tokenizers). | These already answer "what does the added vocab look like at each cutoff" at 1k granularity. 02_1_7 is the **runtime-evaluation** counterpart at the same granularity. |
| `02_1_4_cutoff_analysis/artifacts/per_cutoff_report.json` + `c3_added_lang_per_cutoff.csv` + `classified_added_tokens.jsonl` | Per-cutoff per-language counts; per-token char-mask classification. | Joined to the eval results in 02_1_7's report so each cutoff's metrics sit alongside its language-composition breakdown. |
| `02_1_5_added_token_curation/manifests/removal_list.jsonl` (104 tokens, 6 classes) | Curation policy: which added tokens to remove from the merge table. | Feeds the **pruned-variant arm** (§7) — apply removal_list per cutoff and re-evaluate; the delta vs the un-pruned variant is the curation gain. |

## 3. What's new in 02_1_7 (the actual delta)

1. **Runtime tokenizer variants at 1k granularity** — 02_1_2 has only
   built the original 4-point grid; we need 13 variants (per §4).
2. **TokEval suite** (`swiss-ai/tokenizer-intrinsic-evals`) wired in
   as a vendored submodule.
3. **Five metrics not in our existing harness**: TokEval-style
   compression ratio, TFG, Rényi-2.5 efficiency, UTF-8 Integrity Rate,
   MorphScore V2.
4. **A pruned-variant arm** that applies `02_1_5`'s `removal_list.jsonl`
   on top of each cutoff variant; UTF-8 Integrity is the regression
   guard.
5. **Unified per-cutoff report** that joins our existing metrics
   (02_1_3) and per-cutoff language composition (02_1_4) with the new
   TokEval metrics in one table.

## 4. Citation anchor

Apertus paper §2.2 selects on 4 intrinsic metrics: fertility,
compression ratio, vocab utilization, **Gini (TFG)** (Foroutan et al.
2025a, arXiv:2508.04796). Implementations live in TokEval (Meister
2025,
[`cimeister/tokenizer-analysis-suite`](https://github.com/cimeister/tokenizer-analysis-suite));
the Apertus team's working fork is
[`swiss-ai/tokenizer-intrinsic-evals`](https://github.com/swiss-ai/tokenizer-intrinsic-evals).
The fork bundles ~20 metrics; Apertus reports only 4 of them. T1
matches the paper; T2 uses the fork's broader catalogue.

## 5. Cutoff grid (13 raw variants + 2 curated twins = 15 tokenizers)

Per-1k from baseline to ~12k added tokens, including the currently-
leading 11,264 cutoff, plus curated twins at the two cutoffs most
likely to ship (per reviewer Medium-5: pruning regression guards
require the pruned variants to be in the grid):

| variant id | added tokens | total vocab | notes |
|---|---:|---:|---|
| `apertus_base` | 0 | 131,072 | reference; equivalent to Apertus shipping tokenizer |
| `add_1024`     | 1,024  | 132,096 | |
| `add_2048`     | 2,048  | 133,120 | |
| `add_3072`     | 3,072  | 134,144 | |
| `add_4096`     | 4,096  | 135,168 | |
| `add_5120`     | 5,120  | 136,192 | |
| `add_6144`     | 6,144  | 137,216 | |
| `add_7168`     | 7,168  | 138,240 | |
| `add_8192`     | 8,192  | 139,264 | |
| `add_9216`     | 9,216  | 140,288 | |
| `add_10240`    | 10,240 | 141,312 | original grid point |
| `add_11264`    | 11,264 | 142,336 | currently-leading cutoff |
| `add_12288`    | 12,288 | 143,360 | upper end of the sweep |
| `add_11264_curated` | 11,264 − 39 = 11,225 effective | 142,297 effective | applies `02_1_5/manifests/removal_list.jsonl` (39 tokens removable at 11,264) |
| `add_12288_curated` | 12,288 − 44 = 12,244 effective | 143,316 effective | applies the same removal list (~44 tokens removable at 12,288) |

All raw cutoffs are multiples of 1024 (preserves the 128/256 alignment
constraint from `docs/GLOBAL_DECISIONS.md`). Curated twins lose
alignment by design — the 39/44 removed tokens are scattered through
the merge order; the variant breaks append-only-vs-Apertus and the
implementer in `02_2_tokenizer_implementation` decides masking vs
pruning per class.

**Optional sensitivity extension** (off by default; user-gated):
`add_15360`, `add_20480`, `add_25600` — confirms the metric curves
flatten past 12k. Worth running once if the T1 / T2 curves look
non-monotonic in the first sweep. The 02_1_4 distribution_at_{N}.json
analytical sweep already covers this range; the question is only
whether to spend the runtime-eval compute.

## 6. Eval — three orthogonal axes (separated per reviewer High-1)

The reviewer correctly flagged that the previous draft braided
together "language" and "slice" — TokEval's `--language-config` is a
language-keyed map, and TFG is intrinsically a *cross-language*
metric. Running TFG on a single-language slice is meaningless, and
running fertility on a multi-lingual blob loses the per-language
detail. The three axes are now explicit:

### 6.1 Language axis (what TokEval calls `languages`)

| language config | langs | source | used by |
|---|---:|---|---|
| `apertus55` | **55** (incl. `ell_Grek` — per reviewer High-2; Apertus paper §2.2 reports TFG over 55 langs, our previous Apertus-13 was non-comparable AND excluded Greek) | filtered from TokEval `configs/flores+_lang_config.json` (~190 langs available) — the 55-lang list is built by `scripts/build_apertus55_config.py` (see §13) | TFG only |
| `greek_only` | 1 (`ell_Grek`) | TokEval `parallel/ell_Grek.txt` | per-language fertility / compression / utilization / UTF-8 / MorphScore |
| `apertus55_per_lang` | 55 single-language runs | same files as `apertus55`, but each language evaluated separately | per-language fertility / compression / utilization on every Apertus-55 language (so we can see the per-language cost of each cutoff) |

### 6.2 Slice axis (which corpus)

Slice = the *data*; language config = how to bucket it. For FLORES+
the slice is a single corpus that contains 55 languages; for our
held-outs the slice is single-language Greek.

| slice id | underlying corpus | provenance | role |
|---|---|---|---|
| `flores_plus_55` | TokEval `parallel/<55 codes>.txt` (built once; one combined-corpus eval-set) | **NEW — ships with TokEval** | paper-comparable multilingual + per-language eval |
| `virgin_hplt` | gcloud disk: `/home/foivos/data/glossapi_work/.../virgin_hplt_*` (10k docs, source_doc_id anti-joined) | **REUSED — built by 02_1_3's `build_virgin_hplt_eval.py`** | training-distribution proxy (Greek-only at the document level) |
| `c3_val_clean` | our cleaned val | **REUSED — built by 02_1_3's `clean_holdouts.py`** | in-domain Greek eval |
| `c3_test_clean` | our cleaned test | **REUSED — built by 02_1_3's `clean_holdouts.py`** | final unbiased Greek eval slice |

### 6.3 Metric axis (what's measured)

Spelled out in §7 with the validity constraints (which language
configs each metric requires).

### 6.4 Job matrix (which jobs actually run)

| job # | tokenizers | language config | slice | metrics |
|---:|---|---|---|---|
| 1 | all 15 | `apertus55` (multilingual) | `flores_plus_55` | **TFG only** |
| 2 | all 15 | `apertus55_per_lang` (per-lang) | `flores_plus_55` | fertility, compression ratio, vocab utilization, UTF-8 integrity, Rényi-2.5 efficiency |
| 3 | all 15 | `greek_only` | `flores_plus_55` (the `ell_Grek` slice) | + MorphScore Greek |
| 4 | all 15 | `greek_only` | `virgin_hplt` | fertility, compression, utilization, UTF-8, Rényi |
| 5 | all 15 | `greek_only` | `c3_val_clean` | fertility, compression, utilization, UTF-8, Rényi |
| 6 | all 15 | `greek_only` | `c3_test_clean` | fertility, compression, utilization, UTF-8, Rényi |

TFG runs once (job #1) — across 55 langs, that's where the
cross-language signal lives. Per-language metrics run on the same
multilingual data but bucketed per-language (jobs #2, #3) and on our
Greek-only held-outs (jobs #4–#6). Each cell of jobs #2–#6 produces
one row per (tokenizer, language, metric).

FLORES+ slices give us comparability with published baselines
(Apertus + Gemma + Qwen numbers all use FLORES+). Our held-outs
give us decision-relevant numbers on the actual training
distribution.

## 7. Metric tiers

Color key:
- 🟢 = already produced by 02_1_3 — re-emitted by TokEval for
  paper-config comparability + methodology delta check
- 🆕 = net-new to this sub-subproject

### Tier 1 — paper-aligned floor (mandatory)

Normalizations per reviewer Medium-3: TokEval's default fertility is
**words** (HF whitespace tokenizer); Apertus paper §2.2 reports
word-fertility. Per-line goes under T3 as `avg_tokens_per_line`, not
the primary fertility number.

| metric | TokEval module | normalization | validity scope | provenance | cutoff signal |
|---|---|---|---|---|---|
| Fertility (**words / HF whitespace**) | `metrics/basic.py` | `text_measurement_config_words_hf.json` | per-language only (job #2, #3, #4–#6) | 🟢 02_1_3 has `greek_word_space_fertility` + `tokens_per_byte`; TokEval adds word normalization for paper-config comparability | tokens-per-word; lower = better encoding |
| Compression ratio (**bytes**) | `metrics/basic.py` | `text_measurement_config_bytes.json` | per-language only | 🟢 02_1_3 has `chars_per_token` / `bytes_per_token`; TokEval reports bytes-per-token under its config | higher = better encoding |
| Vocabulary utilization (per reviewer Medium-4 — split into three metrics) | `metrics/basic.py` + custom | corpus-level | per-language for the custom variants; corpus-level for TokEval | 🟢 partial (see below) | see below |
|   — TokEval default `used / total_vocab` | TokEval | corpus | corpus | 🆕 | dominated by 131k base vocab; bystander |
|   — `used_added / added_total` (the actual cutoff-question metric) | custom in `04_aggregate.py` | per-language + corpus | per-language | 🟢 02_1_3 reports this | **the cutoff signal** — cutoff is wasteful if this drops |
|   — `used_curated_added / curated_added_total` | custom in `04_aggregate.py` | per-language + corpus | per-language | 🆕 | curation-arm signal (jobs #1-#6, curated rows only) |
| Tokenizer Fairness Gini (**TFG**) | `metrics/gini.py` | per-line cost across `apertus55` languages | **multi-language only** (job #1) | 🆕 | shift in multilingual fairness vs `apertus_base`; the cutoff's "cost to others" |

For the 🟢 rows we report BOTH our 02_1_3 number and TokEval's number
side-by-side in the first sweep; if they agree to within rounding we
pick TokEval as canonical going forward for paper comparability.
If they diverge materially we document the methodology delta and
choose deliberately. The first wet run includes a sanity check that
`apertus_base` fertility on the FLORES+ `ell_Grek` slice matches our
02_1_3 number within ~0.5 %.

### Tier 2 — multi-criteria upgrade (defends against reviewer "4 metrics is too narrow")

| metric | TokEval module | provenance | what cutoff signal it gives |
|---|---|---|---|
| Rényi-2.5 efficiency | `metrics/information_theoretic.py` | 🆕 | Zouhar et al. (ACL 2023): correlates with downstream LM perplexity better than fertility alone. Best single intrinsic predictor of LM behavior. |
| UTF-8 Integrity Rate + Character Boundary Split Count | `metrics/utf8_integrity.py` | 🆕 | regression guard for the **pruned-variant arm** (§8); binary go/no-go that applying 02_1_5's removal_list didn't cause Greek multibyte chars to leak across tokens |
| MorphScore V2 (Greek if data covers it) | `metrics/morphscore.py` | 🆕 | Greek morphology-aware quality; Apertus's 4-metric set has nothing on morphology |

### Tier 3 — diagnostics (free if the harness is already running)

reconstruction exact-match rate, CER, type-token ratio, token-length
distribution, average tokens-per-line, all bigram-entropy and Rényi-
α-other-than-2.5 outputs. Plus 02_1_3's `unk_rate`, `byte_fallback_rate`,
`single_token_greek_word_share`, `added_token_rate` carried through
for continuity. Not decision-driving; emit anyway.

## 8. Curated-variant arm (consumes 02_1_5)

At the **two cutoffs most likely to ship** (11,264 and 12,288), emit a
**curated twin** by applying `02_1_5/manifests/removal_list.jsonl` to
the merge table (Option 2 from `02_1_5/CURATION_REPORT.md` — the
merge-graph-validator-gated path). At 11,264 this removes 39 tokens;
at 12,288 ~44 tokens.

The curated twin is evaluated with the same metric pipeline as the
un-curated variant. The delta is the curation gain. **UTF-8 Integrity
Rate must stay at 100 %** on the curated twin — that's the regression
test Option 2 needs before adoption.

Why only at 11,264 and 12,288 and not the whole 1k grid: the per-1k
sweep's purpose is to map the cutoff curve; curation is a post-cutoff
surgery applied only to the cutoff that ships. We get the curation
signal where it matters (the two candidate ship cutoffs); we don't
inflate the sweep with 11 curated variants the project will never use.

Out of scope here: deciding masking-vs-pruning for production. That
choice stays in `02_1_5` / `02_2_tokenizer_implementation`. 02_1_7
just produces the evidence on what pruning costs (or doesn't).

## 9. Pipeline

```
┌────────────────────────────────────────────────┐
│ 01_build_variants  — REUSE 02_1_2 builder       │
│   call build_cutoff_variants.py with 12 cutoffs │
│   {1024…12288}; apertus_base symlinked.         │
│   IN:  C3 merge table (25,600 added)            │
│   OUT: variants/cutoff_<N>/  (13 dirs incl base)│
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ 01b_build_curated_variants — applies 02_1_5     │
│   for {11264, 12288} only, apply                │
│   removal_list.jsonl to merge table; emit       │
│   variants/cutoff_<N>_curated/                  │
│   (gated on merge-graph validator)              │
│   OUT: 2 extra dirs (total 15)                  │
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ 02_prep_eval — TokEval + Apertus-55 config      │
│   IN:  swiss-ai/tokenizer-intrinsic-evals       │
│        (vendored as submodule)                  │
│        + 02_1_3 held-out parquets (gcloud disk) │
│   OUT: configs/apertus55_lang_config.json       │
│        configs/greek_only_lang_config.json      │
│        configs/cutoff_sweep_tokenizers.json     │
│              (15 entries, no double-counting)    │
│        configs/our_holdouts_lang_config.json    │
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────┬───────────────────┐
│ 03a_run_tokeval            │ 03b_run_02_1_3    │
│   6 jobs × 15 tokenizers   │   our harness on  │
│   per §6.4 job matrix:     │   the same 15     │
│   - job#1: TFG on apertus55│   tokenizers ×    │
│   - job#2: per-lang metrics│   our 3 held-outs │
│     on apertus55 per-lang  │   (Greek-only)    │
│   - jobs#3-#6: greek_only  │                   │
│     on 4 slices            │                   │
│   OUT: tokeval_raw/        │   OUT:            │
│        {job_id}/{tok_id}/  │   our_suite_raw/  │
└────────────────────────────┴───────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ 04_aggregate — long-format parquet              │
│   joins TokEval + 02_1_3 + 02_1_4 + 02_1_5      │
│   classification; computes the three            │
│   utilization variants per reviewer Medium-4.    │
│   OUT: artifacts/results.parquet                 │
│        columns: variant_id, added_tokens,        │
│                 curated (bool), language,        │
│                 slice, metric, value, source,    │
│                 tier                              │
└────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────┐
│ 05_render — REPORT.md + plots                   │
│   per-tier tables, per-metric line plots        │
│   (x=added_tokens, y=metric, series=language    │
│   or slice, linestyle=curated-vs-not)           │
│   OUT: REPORT.md, plots/{metric}.png             │
│        manifests/per_cutoff_metrics.json         │
└────────────────────────────────────────────────┘
```

**Tokenizer count clarification** (per reviewer Low):
- 12 raw added-variants + `apertus_base` (symlink) = **13** raw
- + 2 curated twins = **15** total
- `cutoff_sweep_tokenizers.json` has 15 entries, no double-counting.

## 10. Outputs

- **`manifests/`** (git-tracked, consumed by `02_1_4` + `02_1_6`):
  - `per_cutoff_metrics.json` — canonical per-cutoff per-metric per-
    slice table (T1 + T2 only; small; <100 KB)
  - `eval_slices.json` — slice definitions (source paths, sizes,
    SHA-256 of input files for reproducibility)
- **`artifacts/`** (gitignored, regeneratable):
  - `results.parquet` — long-format raw output across all 15×6 (job matrix)×~25
    cells (a few MB)
  - `plots/{metric}.png` — one per metric
  - `tokeval_raw/` — raw TokEval JSON per (variant, slice)
  - `our_suite_raw/` — raw `02_1_3` harness JSON per (variant, slice)
- **`REPORT.md`** (git-tracked) — narrative + tables + plot
  references. Organized as:
  - § T1 paper-aligned summary (for citation against Apertus paper)
  - § T2 multi-criteria supplement (for the reviewer-response thread)
  - § T3 diagnostics
  - § Pruned-arm delta (curation gain per cutoff)
  - § Methodology delta (TokEval-fertility vs 02_1_3-fertility)
  - § Reproduction recipe

## 11. Compute posture

- **Where**: gcloud `apertus-greek-tokenizer` (64-vCPU
  `m3-megamem-64`, `europe-west4-b`, single-tenant). Per
  `gcloud_tokenizer_instance` memory: use all 64 vCPUs.
- **Parallelization**: 15 tokenizers × 5 slices = 140 (tokenizer, slice)
  pairs are independent. TokEval + our harness are both CPU-bound
  Python — saturate via process-level fan-out (one process per
  (tokenizer, slice) pair, capped at vCPU count), not threads.
- **Expected wall-clock**:
  - FLORES+ slices: O(seconds) per (variant, slice) pair
  - virgin_hplt (10k docs): O(minutes) per (variant, slice) pair
  - Full sweep total: O(1–2 h) wall-clock, dominated by virgin_hplt
    across the 28-variant set
- **Posture before launch** (per `feedback_utilize_available_compute`):
  state corpus shape + expected wall-clock in PR; document MorphScore
  data availability for Greek before the sweep starts (it may not
  cover Greek — degrade gracefully if not).
- **After**: stop the instance (per `feedback_instance_stop_decision`
  three-step rule) — sweep outputs are small and re-runnable.

## 12. Comparability invariants

Per `feedback_recipe_scope_source_allocation`:

- Same TokEval commit across all variants — pin the swiss-ai fork
  HEAD SHA in `manifests/tokeval_commit.txt`.
- Same `02_1_3` harness commit across all variants — record git SHA
  in `manifests/our_suite_commit.txt`.
- Same eval slice file SHAs across all variants — pin in
  `manifests/eval_slices.json`.
- Same `measurement_config` per metric across variants (lines for
  fertility / TFG, bytes for compression ratio).
- All numbers reported (a) absolute and (b) Δ vs `apertus_base` so
  the marginal cost of each 1k slice is visible.
- Pruned and un-pruned twins evaluated with identical configs;
  pruned-minus-unpruned delta is the curation gain per metric.

## 13. Dependencies + access

- **swiss-ai/tokenizer-intrinsic-evals**: vendor as a git submodule
  under `02_1_7_intrinsic_eval_sweep/vendor/tokenizer-intrinsic-evals`.
  Pin to a specific commit SHA in `manifests/`.
- **FLORES+ parallel data**: ships with TokEval under `parallel/`.
  No external download needed.
- **MorphScore data**: external; check `MorphScore` package readme
  for Greek availability. If unavailable, T2 → 2 metrics not 3.
- **Our held-outs**: already on disk at
  `/home/foivos/data/glossapi_work/...` and consumed by 02_1_3 today;
  mount the same path on the gcloud instance.
- **`02_1_2` and `02_1_3` scripts**: invoked as-is from this sub-
  subproject's wrappers; no fork.

## 14. Risks + open questions

| risk | mitigation |
|---|---|
| MorphScore lacks Greek data | T2 drops to 2 metrics (Rényi + UTF-8); flagged here; not blocking |
| TokEval's measurement_config differs from what `02_1_3` currently uses | Report BOTH side-by-side for the first sweep; document the delta in REPORT.md § "Methodology delta"; pick whichever the Apertus paper actually used as canonical going forward |
| Adding `apertus_base` (zero added tokens) to the sweep requires the original Apertus tokenizer.json | Already in `tokenizer_analysis/hf_snapshots/...` per CLAUDE.md; symlink it into the sweep config as variant #0 |
| Pruned arm: `removal_list.jsonl` removals may invalidate merge chains for kept tokens (the verified `/Ε`→`/ΕΕ` hazard from `02_1_5/CURATION_REPORT.md` §4) | Run merge-graph validator BEFORE building each pruned variant; skip-and-flag any cutoff where validation fails, do not proceed with a broken variant |
| Per `feedback_no_threshold_rules_unprompted` | This plan emits the metric tables; the cutoff is the user's call after reading them |

## 15. Reproduction recipe (target — once scripts land)

```bash
# (run on apertus-greek-tokenizer gcloud instance)
cd subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep
bash scripts/01_build_variants.sh         # invokes 02_1_2 with 1k grid
bash scripts/01b_build_pruned_variants.sh # applies 02_1_5 removal_list
bash scripts/02_prep_eval_slices.sh       # FLORES+ symlinks + 02_1_3 paths
bash scripts/03a_run_tokeval.sh           # TokEval suite (~30-60 min)
bash scripts/03b_run_our_suite.sh         # 02_1_3 harness on same grid
python scripts/04_aggregate.py            # collect parquet
python scripts/05_render_report.py        # emit REPORT.md + plots
```

## 16. Implementation order (when work starts)

1. **Wrapper script around `02_1_2_cutoff_variant_builder/scripts/
   build_cutoff_variants.py`** that invokes it with the 13-point
   cutoff list. Builder is already parameterized — no extension
   needed; just a thin caller.
2. **Vendor `swiss-ai/tokenizer-intrinsic-evals`** as a pinned
   submodule; install via `uv sync`.
3. **Pruned-variant builder**: applies `02_1_5/manifests/removal_list.jsonl`
   to each cutoff's merge table after running the merge-graph validator;
   produces the pruned twin tokenizer dirs.
4. **Wire up the 5 eval slices** in TokEval's config format
   (FLORES+ uses TokEval's own; our 3 held-outs need a TokEval-side
   adapter that points at the existing parquets without copying).
5. **Wrapper around `02_1_3/scripts/run_tokenizer_fertility_suite.py`**
   to run on the same 28-variant grid for continuity.
6. **Main TokEval entry script `03a_run_tokeval.sh`** — parallelize
   across (variant, slice) pairs.
7. **Aggregator** (parquet collector joining TokEval + 02_1_3 +
   02_1_4 sources into one long-format frame).
8. **Report renderer** — per-tier tables + per-metric plots.
9. **First wet run on the gcloud instance** — sanity-check the
   `apertus_base` row reproduces published Apertus paper numbers on
   the same FLORES+ slice. If it does not, the measurement-config or
   input pipeline diverges from the paper and we resolve before
   running the full sweep.
10. **Full sweep**; commit `manifests/` to git; stop instance.
11. **`02_1_4_cutoff_analysis/REPORT.md` + `02_1_6_representation_policy_analysis/`**
    updated to cite this sub-subproject's manifest.

## What this plan is not

- It is not the cutoff decision. It produces evidence; the user picks.
- It is not the implementation. Scripts get written in a follow-up.
- It is not a re-do of `02_1_3` or `02_1_4`. It calls 02_1_3's
  harness, joins 02_1_4's per-cutoff classification, and consumes
  02_1_5's curation manifest — without forking any of them.
- It is not a re-do of `02_1_3_fertility_evaluation`. That stays as
  our internal fertility infrastructure; this sub-subproject adds the
  TokEval-canonical metric surface alongside it for paper / fairness
  citation.
