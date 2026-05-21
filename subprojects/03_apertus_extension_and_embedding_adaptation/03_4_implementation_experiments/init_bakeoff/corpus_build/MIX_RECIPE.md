# Mix recipes — bulk and anneal

The bakeoff (and future production CPT) consumes a stream of documents
interleaved from several sources with explicit per-source weights. Per
[`../../cpt_plan.md`](../../cpt_plan.md) v0.7 §2 + §4:

- **Bulk phase**: one shuffled-mixture stream, all corpora at target weights from token 0. This is what the bakeoff actually trains on.
- **Anneal phase**: final 10–20 % of *production* training (out of scope for the bakeoff). Mixture shifts to highest-priority subsets; replay drops from ~30 % → ~15 %; LR decays. We define the recipe here for completeness; the bakeoff never executes it.

Both recipes are JSON, in `recipes/`. The mix builder (`mix_builder.py`) reads either and emits a JSON-lines stream.

## Three-step build path (reviewer round-2 Blocker 3 fix)

The CPT dataset build follows the canonical runbook at [`../../../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../../../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md). **Order matters**: if Apertus-overlapping docs are removed first, an internal-duplicate family can still keep a fresh alternate representative.

```
1. bash pull_greek_corpus.sh        # nanochat parquets + dedup_metadata + Apertus overlay
2. bash prepare_greek_pool.sh       # → $SELECTED parquet (Apertus-drop + drop_intra_and_inter)
3. SELECTED=$WORK/cpt/selected_after_apertus_and_internal_dedup.parquet \
   python3 mix_builder.py --recipe recipes/bulk.json --target-tokens 7_000_000_000 ...
```

All six Greek `bulk.json` sources point at `local_parquet: ${SELECTED}` — the single post-dedup parquet — and filter by `source_dataset` value to slice into HPLT / literary / dialogue / academic / legal / dictionary buckets. The Apertus-overlap-drop is applied **upstream** in step 2 (not per-source in mix_builder), so it applies uniformly to all six categories rather than only to HPLT (the previous bug). Internal dedup with `drop_intra_and_inter` also happens in step 2 (was missing entirely before).

## Composition (bulk recipe — what the bakeoff actually consumes)

Working defaults per v0.7 §2 + §4 + user answers 2026-05-20:

| Top-level bucket | Share | Sub-allocation |
|---|---:|---|
| Greek (post-Apertus-overlap-drop, post-internal-dedup) | **70 %** | see below |
| Non-Greek replay (24 languages, 3 tiers) | **24 %** | see below |
| Code | **4 %** | `bigcode/starcoderdata` (Apertus used StarCoder) |
| Math | **2 %** | `HuggingFaceTB/finemath` config `finemath-3plus` (Apertus stage-1 source per `submit_apertus_8b.sh:L29`) |

> **2026-05-21 rebalance.** Replay reduced from 26 % → 24 % to free 2 % for FineMath, taken entirely from English (FW-Edu) which went from 3.9 % → 1.9 %. The other 23 replay-language weights stay at their pre-rebalance values. Greek and code shares unchanged.

> **OPUS Greek-English parallel** is listed as optional in v0.7 §4.4 but **not yet in the bakeoff bulk mix** — it needs different schema handling in `mix_builder.py` (its rows are `{translation: {el: ..., en: ...}}` rather than `{text: ...}`). Deferred to a future iteration of corpus_build.

### Greek sub-allocation (70 % of total)

| Source | Share of Greek | HF id / filter |
|---|---:|---|
| **HPLT clean60 (fresh; Apertus-overlap overlay applied)** | 50 % of Greek = 35 % total | nanochat dataset, `source_dataset starts with HPLT__ell_Grek` + `apertus_overlap_drop_docs.parquet` exclude |
| Literary nanochat (artos-zois + Project_Gutenberg + Ekklisiastika + 1000_prwta + klasikh + Wikisource + dimodis_logotexnia + archetai) | 26 % of Greek = 18.2 % total | nanochat dataset, filter `source_dataset` ∈ literary list |
| Dialogue + textbooks (OPUS OpenSubtitles-el + openbook + Sxolika_vivlia + ert-press + istorima) | 9 % of Greek = 6.3 % total | nanochat dataset, filter |
| Academic (openarchives.gr + greek_phd + Apothetirio_Kallipos + Apothetirio_Pergamos) | 8 % of Greek = 5.6 % total | nanochat dataset, filter |
| Legal + civic (eurlex-greek-legislation + AI-team-UoA/greek_legal_code + opengov.gr-diaboyleuseis + ellinika_dedomena_europaikou_koinovouliou) | 5 % of Greek = 3.5 % total | nanochat dataset, filter |
| Dictionary + misc (modern-greek-dictionary capped + finewiki Greek half-weight + 95k_deigma + others) | 2 % of Greek = 1.4 % total | nanochat dataset, filter; modern-greek-dictionary explicitly capped per Claude-review note in `../../collegues_Apertus_plan.md` |

The HPLT half dominates by token mass (the underlying HPLT clean60 slice is ~50 B chars at v0.7 chars/token = ~14 B tokens; the GlossAPI sub-pools combined are smaller). The recipe weights are **probabilities for the interleaver**, which is what affects the per-step mix; effective token shares track these weights up to per-source exhaustion.

### Non-Greek replay sub-allocation (24 % of total, across 24 languages)

Tier weights per v0.7 §4.2 (40–50 % T1 / 35–45 % T2 / 10–15 % T3); working defaults at the midpoint, scaled to the **24 % outer share** (post-2026-05-21 rebalance: English's share absorbed the 2 % shift to FineMath):

| Tier | Languages | Share of replay | Total share |
|---|---|---:|---:|
| **T1** (8 langs, FW2-HQ where Apertus used HQ) | eng, fra, deu, ita, spa, rus, arb, cmn | ~46 % | 11.0 % |
| **T2** (11 langs, FW2 where Apertus used standard) | tur, bul, srp, ron, heb, por, pol, nld, pes, ukr, jpn | ~41 % | 9.88 % |
| **T3** (5 langs, FW2; small per-lang) | lat, hye, kat, sqi (or als), mkd | ~13 % | 3.12 % |

Within each tier, weights are roughly equal *except* English gets ~17 % of T1 (≈ 1.9 % of total, post-rebalance) — still the largest single non-Greek language, but trimmed from 3.9 % to 1.9 % to make room for FineMath at 2 %. The remaining 7 T1 langs each stay at 1.3 % of total.

Per-language sources:

- T1 (Apertus quality-filtered the 20 high-resource langs at top-10 % per-language XLM-R): **`epfml/FineWeb2-HQ`** with `config_name = <iso639-3>_<script>` (Messmer, Sabolčec, Jaggi 2025, [arXiv:2502.10361](https://arxiv.org/abs/2502.10361); top-10 %, NOT "Score-3" — that's FineWeb-Edu's filter).
- T2 / T3 (Apertus used unfiltered FW2 for these): **`HuggingFaceFW/fineweb-2`** with `config_name = <iso639-3>_<script>` (v2.0.1 per Apertus tech report footnote 18).
- English specifically: prefer **`HuggingFaceFW/fineweb-edu`** Score-3 (Penedo et al. [arXiv:2406.17557](https://arxiv.org/abs/2406.17557); matches Apertus stage-5 cooldown). FineWeb2-HQ English is the fallback.

> **Reviewer flag**: cpt_plan.md v0.7 §4.2's "24 replay languages" framing (T1+T2+T3 = 8+11+5) is **our internal grouping**, not Apertus's. Apertus's tech report (arXiv:2509.14233 Appendix G, p.88-89) enumerates **20 high-resource languages** that receive quality+toxicity filtering; the other ~4 languages we include (T3: lat/hye/kat/sqi/mkd) are added by us for cultural/regional coverage relevant to Greek. Token-share targets per language are also our derivation, not from Apertus (the tech report only publishes document counts and percentages of FineWeb-2 documents, not training-mixture token shares).

### Code (4 %)

| Source | HF id |
|---|---|
| Code | `bigcode/starcoderdata` (Apertus's pretraining source) |

Use the default "all-permissive" subset; no language filter (mixed-language source-code is fine for retention).

### Math (2 %)

| Source | HF id + config |
|---|---|
| Math (web-derived high-quality math text) | `HuggingFaceTB/finemath` config `finemath-3plus` (Apertus stage-1 source per `submit_apertus_8b.sh:L29`) |

FineMath-3plus is the higher-quality subset of FineMath, selected for math content (problem statements, solutions, derivations) from CommonCrawl. Apertus uses `finemath-3plus-merge` in pretraining stage 1; the bakeoff includes it at the same 2 % share Apertus's stage-1 allocates within the broader mix (Apertus stage proportions aren't published per-source — 2 % is a reasonable default).

## Composition (anneal recipe — defined but not run in the bakeoff)

Per v0.7 §3.2: in the final 10–20 % of production training, mixture shifts to highest-priority subsets and replay drops to ~15 %. Working pattern (between Llama 3 "narrow high-quality" and OLMo Dolmino "broad quality-curated"):

| Bucket | Bulk share | Anneal share | Reason |
|---|---:|---:|---|
| Greek high-quality literary (artos-zois + Project_Gutenberg + Wikisource + 1000_prwta) | 12 % | **30 %** | clean prose anchor |
| Greek academic + legal (openarchives + greek_phd + eurlex + greek_legal_code) | 9 % | **22 %** | domain depth |
| Greek dictionary (modern-greek-dictionary uncapped from anneal start) | 1 % | **15 %** | gap closure (Apertus_plan §"Final mix") |
| Greek dialogue + textbooks (OpenSubtitles + openbook + Sxolika + ert) | 6 % | **8 %** | maintained but de-emphasized |
| HPLT broad (kept reduced for register breadth) | 35 % | **10 %** | safety net against narrow-corpus overfit |
| Non-Greek replay (Tier 1 dominant) | 26 % | **12 %** | reduced per v0.7 §3.2 "30 % → 15 %" pattern; Tier 1 ≥ 70 % within replay during anneal |
| Code | 4 % | **3 %** | maintained at slightly reduced share |

The anneal recipe is `recipes/anneal.json`. It will be used in the production CPT (15–20 B tokens; final 10–20 % = ~2–4 B tokens of anneal) **after** the bakeoff picks an init winner. Not part of the bakeoff.

## Builder usage

```bash
# Generate the bakeoff bulk JSON-lines stream after the smoke passes
sbatch corpus_build/mix_builder_full.sbatch

# Same shape for the anneal recipe (not run in the bakeoff)
python3 mix_builder.py \
    --recipe recipes/anneal.json \
    --target-tokens 3_000_000_000 \
    --tokenizer /iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480 \
    --output /iopsstor/scratch/cscs/fffoivos/cpt_corpus/anneal_mix.jsonl \
    --seed 20260520
```

Output is JSON-lines. Each line is `{"text": "...", "source": "...", "doc_id": "...", "lang": "..."}`. Megatron-LM-Swiss-AI's `tools/preprocess_data.py` then converts JSON-lines → binary indexed dataset (`.bin` + `.idx`) for training.

Determinism: the `--seed` controls the interleave randomization. Same seed means the same JSONL text stream across runs. Vanilla and the extended arms then use different tokenizers/preprocessed Megatron binaries, so token IDs differ across tokenizer families even though the document order is shared.

## Why two stages (build JSONL → preprocess to .bin/.idx)

Megatron's training reads its native binary format; converting on the fly during interleaving is awkward. Splitting the pipeline:

1. `mix_builder.py` does the streaming interleave + budget-cap + writes JSONL (CPU job on `normal`; `xfer` is in maintenance during the 2026-05-21 run).
2. `tools/preprocess_data.py` (Megatron) does the tokenization + binary packing (CPU job on `normal`).
3. Training reads the binary on `normal`.

This way each stage has a single clear job, and we can re-run any stage independently.
