# 03_1 — Greek (and pan-language) embedding diagnostic

> **In one line:** a read-only geometric study of how Apertus-8B-2509 already represents Greek — and 74 other languages — on its input (`E`) and output (`U`) embedding matrices, run before any row was added; it produced the norm targets the ReTok and Centroid init arms used and killed two attractive-but-wrong hypotheses.
> **Period:** 2026-05-12 → 2026-05-18 (scripts landed in `002bddc5` 2026-05-14 and `7deea009` 2026-05-18; report bodies dated 2026-05-13 → 2026-05-15).
> **Status:** completed. v2 and v3 series complete; v4 stopped deliberately after step 3.
> **Came from / led to:** per-token language attribution in [`02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution`](../../02_2_tokenizer_implementation/02_2_2_vocab_lang_attribution/) → this → the init arms in [`../03_4_implementation_experiments/init_bakeoff/arms/`](../03_4_implementation_experiments/init_bakeoff/arms/README.md).

## Why this existed

Before deciding *how* to initialise 17,408 new embedding rows, it was worth knowing what the existing Greek rows look like: where they sit relative to the global centroid, how many directions they really span, whether Greek tokens cluster morphologically, and whether there is any usable geometric bridge between Greek and English for Greek-origin concepts (which would have justified a translation-mediated init). The design framing is [`docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md`](../../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md).

## History

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| 2026-05-12 → 05-13 | **v1 → v2.3 series**, single anchor: Greek vs ¬Greek. Centroid/PC geometry, hull occupancy, morphological clustering, families, cross-language semantic clusters | Greek is a coherent geometric subspace; Greek tokens cluster by morphology 5–9× tighter than random; en↔el cosine for Greek-origin concepts averages **+0.05** vs **+0.04** for non-Greek-origin pairs → **no etymology bridge**; Mikolov analogies unusable on byte-level BPE (7/8 candidates skipped) | `scripts/phase0_greek_vs_not_geometry.py` and siblings, landed `002bddc5` |
| 2026-05-13 | v2.1 / v2.3 self-corrections | The v2 "infiltrators" finding (non-Greek tokens inside the Greek hull) was traced to a projection-asymmetry artefact, not semantic overlap; a truncated-SVD bug that biased ¬Greek's `K_sig` low was fixed by the full d=4,096 eigendecomposition | `phase0_infiltrators_filtered.py`, `phase0_full_negreek_spectrum.py` |
| 2026-05-14 → 05-15 | **v3 series**, 11 PMI-attributed languages: per-language spectrum, Marchenko-Pastur edge, subspace overlap, shared dimensions, discriminant directions, pair-specific subspaces | Greek centroid displaced **0.676 (E) / 0.733 (U)** from the classified global; ¬Greek sits at 0.008 / 0.009 (it *is* the global by mass) | `scripts/*_v3.py`, landed `7deea009` |
| 2026-05-15 | **v3-corrected** — two bugs found and fixed in place | The MP edge used `q = min(d,n)/max(d,n)` instead of `c = d/n`: Greek `K_sig` fell **619 → 123** on `E`, Georgian 218 → 0. `var_C(d)` was computed from truncated `K_sig` reconstructions, inflating pair-specificity: counts dropped by orders of magnitude and **reordered** — tight-script cousins (Thai 2.23, Hindi 1.96, Georgian 1.93, Armenian 1.85) replaced the wide-Latin partners, and Greek↔Korean fell to **zero** pair-specific directions | in-place patches in `phase0_perlang_geometry_v3.py` and `phase0_pair_specific_shared_v3.py` |
| 2026-05-18 | **v4 series**, all 75 well-sampled PMI languages (12 dropped: 7 whose scripts Apertus byte-fragments — Amharic, Khmer, Sinhala, Lao, Tibetan, Oriya, Dhivehi — plus 4 `und_*` and Middle High German) | Steps 1–3 (geometry, subspace overlap, shared dims) run; **steps 4 and 5 deliberately not run at 75-language scale** — they were added during v3 methodological discussion and were not part of the reviewed canonical pipeline when the 75-language run was commissioned | `scripts/*_v4.py`, `build_groups_88lang_v4.py` |
| — | A Phase-2 leave-one-out benchmark was attempted and **archived as methodologically contaminated** | kept for traceability only, not used downstream | `report_phase2_preliminary.md` (in the gitignored artifacts tree) |

## Outcome

- **The two numbers that mattered downstream:** Greek-content token norm medians **E = 5.05, U = 3.80**. Both extension arms norm-match new rows to these; the arms' local smoke test reproduces them to within 1 % (`E[modern].norm.p50 = 5.047`, `U[modern].norm.p50 = 3.797`) — see [`../03_4_implementation_experiments/init_bakeoff/arms/README.md`](../03_4_implementation_experiments/init_bakeoff/arms/README.md).
- **Negative result that shaped the plan:** there is no Greek↔English etymological bridge in the static embedding view, which supports the v0.12 §4 hard constraint against translation-mediated init methods (WECHSEL / trans-tokenization / OFA).
- **Two self-corrections are the main methodological lesson:** both the infiltrator finding and the original `K_sig`/pair-specificity numbers were artefacts. Any number quoted from a v3-original artifact is wrong; only v3-corrected values stand.
- Left open: v4 steps 4–5 at 75-language scale; the 11-language v3 answers remain the only ones for those steps.

## Where things are

| What | Where |
|---|---|
| All scripts, by series (v1–v2.3, v3, v4) | [`scripts/`](scripts/) — `*_v3.py` / `*_v4.py` suffixes mark the series |
| Design framing | [`docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md`](../../../docs/EMBEDDING_DIAGNOSTIC_PLAN_V2.md) |
| Reports, figures, geometry arrays (~4.1 GB) | **Not in this repo** — `artifacts/` was gitignored at project level. `report_v2.md`, `report_v3_subspace_meaning.md`, `report_v4_full_panel.md`, `report_v4_vocab_and_training_inference.md`, `REVIEW.md` and the `E_fp32.npy` / `U_fp32.npy` foundation arrays lived only in the original run directory. Regenerable from `extract_embeddings.py` + the scripts; paths are hard-coded at the top of each script (`ROOT` / `SP`). |

## Working documents

None left in the directory — this sub-subproject is scripts plus this history. The prose reports it references are the gitignored artifacts above; the numbers quoted here are the ones the previous README carried forward from them.
