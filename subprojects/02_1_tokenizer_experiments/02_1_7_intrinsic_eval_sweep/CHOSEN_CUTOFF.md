# Chosen cutoff — 17,408 (curated + backfilled to alignment)

**Decision date**: 2026-05-18.
**Decided by**: user, based on the evidence in [`REPORT.md`](REPORT.md) and the
knee-analysis plot [`artifacts/plots/knee_analysis.png`](artifacts/plots/knee_analysis.png).
**Reviewer rounds addressed**: 2 (alignment + handoff-ambiguity).

## The decision

| field | value |
|---|---|
| **Added units** | **17,408** |
| **Total vocab (ship)** | **148,480** = 131,072 base + 17,408 added |
| **Alignment** | 148,480 = **128 × 1160 = 256 × 580** ✓ |
| **Apertus front-end contract** | Preserved — Apertus base ids 0..131,071 unchanged verbatim |
| **Curation mechanism** | **Built directly into the tokenizer.** 69 noise tokens (`02_1_5/manifests/removal_list.jsonl` in-cutoff scope) skipped during BPE merge selection; the next 69 valid merges from c3_full backfilled to maintain 17,408 useful added tokens |
| **Cascade-skips** | **0** — none of the 17,408 backfilled merges depended on a curated merge as a sub-component |
| **Merges walked from c3_full** | 17,477 (= 17,408 accepted + 69 skipped-for-removal) |
| **Ship artifact** | [`variants/c3_added_17408_curated_padded/tokenizer.json`](variants/c3_added_17408_curated_padded/tokenizer.json) |
| **Build manifest** | [`manifests/curated_padded_at_17408_manifest.json`](manifests/curated_padded_at_17408_manifest.json) |

## Why the backfilled tokenizer is the canonical ship

Two reviewer rounds rejected the obvious candidates:

| candidate | vocab | alignment | append-only | status |
|---|---:|:---:|:---:|---|
| Raw `c3_added_17408` (no curation) | 148,480 | ✓ 256-aligned | ✓ preserved | metrics good, but ships 69 noise tokens |
| Pruned `c3_added_17408_curated` (Option 2 of CURATION_REPORT) | 148,411 | ✗ NOT 128-aligned | ✗ ids renumbered | **rejected by reviewer round 2** — alignment + contract broken |
| **Backfilled `c3_added_17408_curated_padded`** | **148,480** | **✓ 256-aligned** | **✓ preserved** | **canonical ship artifact** |

The backfilled construction (built by [`scripts/01c_build_curated_backfilled.py`](scripts/01c_build_curated_backfilled.py))
walks c3_full's merge sequence in order, **skips** the 69 in-cutoff
ids from `02_1_5/manifests/removal_list.jsonl`, and **accepts** the
first 17,408 surviving merges. Because BPE merges only depend on
earlier merges as sub-components, the merge-graph validator runs
during build — no kept merge ends up depending on a skipped one
(measured cascade-skips at this cutoff: 0).

This makes the curation **structural**, not a runtime mask: the noise
tokens are *not in the vocab at all*, but vocab size + alignment +
Apertus base-id preservation are intact.

## Why 17,408 (the cutoff itself)

From the 0 → 25,600 sweep at 1k step:

| criterion | result at 17,408 |
|---|---|
| Greek fertility on C3_val held-out (the decision-relevant axis) | 1.345 (−44.2 % vs apertus_base 2.41) |
| % of theoretical-max fertility improvement captured | **82.4 %** (asymptote y∞ = 1.118; full sweep at 25,600 captures 88.4 %) |
| Marginal-Δ% per added 1k | first cutoff where the next 1k buys **< 1 %** |
| TFG @ Apertus-55 | 0.11613 (vs apertus_base 0.11606; +0.06 % — basis points; the metric is structurally biased against script-isolated extensions, see `REPORT.md`) |
| Apertus alignment | 17,408 = 1024 × 17; total 148,480 = 256 × 580 ✓ |

**The fertility / knee analysis is the load-bearing argument.** The
MorphScore recall bump at 17,408 (0.689 → 0.694) is real but small
(recall is "essentially flat" 0.686–0.695 across the entire sweep);
treat it as supporting color, not an independent pillar.

The real claim: pick the cutoff where each additional 1k of vocab buys
less than 1 % of fertility improvement — that's 17,408 — and it
also happens to be Apertus-aligned, captures >80 % of the maximum
achievable Greek-fertility gain, and sits at the second MorphScore
per-word milestone (clean inflectional splits for common Greek
nominal/verbal paradigms).

## What got curated out

The full `02_1_5/manifests/removal_list.jsonl` enumerates 104 noise
tokens. In-cutoff scope at 17,408 (ids < 148,480) is **69 ids**:

| removal_class | count | glossary category |
|---|---:|---|
| `mixed_script_artifact` | 50 | `mixed_script_token` (Greek-Latin homoglyphs + punct+Greek BPE fragments like `.Ε`, `,τι`, `/και`) |
| `pdf_postscript_glyph` | 9 | `postscript_glyph` (PDF font-glyph residue like `/Α`, `/η`) |
| `cleaner_linenewline_bpe_fragment` | 3 | `latin_acronym` (`LIN`, `ENEW`, `LINENEW` — BPE pieces of `LINENEWLINE`) |
| `latin1_utf8_mojibake` | 3 | `mojibake` (`ÉÉ`-family) |
| `cleaner_linenewline_placeholder` | 2 | `code_identifier` (`LINENEWLINE`, `NEWLINENEWLINE`) |
| `cleaner_extraction_tag` | 2 | `latin_fragment` (`-missing`, `-decoded`) |
| **total** | **69** | |

Per-id list with `decoded` strings: [`manifests/removal_mask_at_17408.jsonl`](manifests/removal_mask_at_17408.jsonl).

## Verification — directly measured (no extrapolation)

### Pairwise on Apertus-55 FLORES+ (TokEval, lines config)

| metric | raw 17,408 | **padded (ship)** | Δ vs raw |
|---|---:|---:|---:|
| TFG (Apertus-55, lower better) | 0.116128 | 0.116126 | −0.000002 |
| Rényi-2.5 efficiency (Apertus-55) | 8.428021 | 8.427630 | −0.000391 |
| UTF-8 completeness rate (Apertus-55) | 0.979127 | 0.979125 | −0.000002 |
| UTF-8 char split rate (Apertus-55) | 0.010320 | 0.010320 | +0.000001 |
| Fertility (Apertus-55, mean tokens/line) | 44.7535 | 44.7536 | +0.000093 |
| Compression rate (Apertus-55) | 0.022345 | 0.022345 | −0.000000 |
| Vocab utilization (Apertus-55) | 0.489278 | 0.489251 | −0.000027 |

### MorphScore Greek (UD-derived, n=693)

| metric | raw 17,408 | **padded (ship)** | Δ vs raw |
|---|---:|---:|---:|
| recall | 0.6940 | 0.6940 | 0.0000 |
| precision | 0.6912 | 0.6912 | 0.0000 |
| mean token-char ratio | 0.3364 | 0.3357 | −0.0007 |

### In-domain Greek held-outs (02_1_3 harness on gcloud)

| slice | raw 17,408 | **padded (ship)** | Δ vs raw |
|---|---:|---:|---:|
| C2_val — Greek word fertility | 1.3598 | 1.3591 | −0.00065 |
| C2_test — Greek word fertility | 1.4124 | 1.4118 | −0.00060 |
| C3_val — Greek word fertility | 1.3456 | 1.3450 | −0.00051 |
| C3_test — Greek word fertility | 1.4011 | 1.4007 | −0.00032 |

**Reading**: every directly-measured metric is either flat or **marginally
better** for the backfilled variant. The 69 swapped-in tokens (positions
17,409..17,477 in c3_full's merge order) fire slightly more usefully on
real Greek than the 69 noise tokens they replaced. Direction is
consistent across all 4 in-domain Greek slices on fertility — small
magnitude but a real, structural win, not noise.

## Embedding-budget cost

| component | per-token (BF16) | total at cutoff 17,408 |
|---|---:|---:|
| Embedding row (4096 × 2 bytes) | 8 KiB | 17,408 × 8 KiB = **136 MiB** new |
| LM head row (4096 × 2 bytes, `tie_word_embeddings=False`) | 8 KiB | 17,408 × 8 KiB = **136 MiB** new |
| **Δ vs apertus_base** | 16 KiB | **272 MiB** added (= 17,408 × 16 KiB) |
| Cumulative embed + LM head | — | 2.27 GiB (2.00 GiB Apertus base + 272 MiB added) |
| Δ as % of full 8 B model | — | ~1.7 % |

## Downstream handoff to `02_2_tokenizer_implementation`

**Canonical contract:**

1. Consume `variants/c3_added_17408_curated_padded/tokenizer.json`
   verbatim. Vocab is 148,480 (256-aligned), ids 0..131,071 are exactly
   the Apertus base, ids 131,072..148,479 are 17,408 BPE-merged Greek-
   extension tokens with the 69 noise tokens filtered out at build
   time.
2. **No runtime curation logic needed.** The pruning is structural, not
   a mask. `03_apertus_extension_and_embedding_adaptation` can warm-
   start every added-id embedding from byte-fallback averaging without
   any "skip these 69" branch.
3. The mask manifest `manifests/removal_mask_at_17408.jsonl` is still
   shipped for documentation — it tells the implementer **which 69
   noise tokens were filtered out** so they appear in any future audit
   trail, but it is NOT something the implementer needs to apply at
   runtime.

**Alternative (do not ship)**: the pruned `c3_added_17408_curated`
variant remains on disk under `variants/`. It is an ablation artifact
only — alignment-broken and append-only-broken.

## Build reproduction (deterministic)

```bash
cd subprojects/02_1_tokenizer_experiments/02_1_7_intrinsic_eval_sweep
$VENV/python scripts/01c_build_curated_backfilled.py
```

The builder:
1. Loads Apertus base (`tokenizers_local/apertus_base/tokenizer.json`)
   and full C3 (`tokenizers_local/c3_full/tokenizer.json`, 156,672 vocab).
2. Reads the 104-token removal list from `02_1_5/manifests/removal_list.jsonl`.
3. Walks c3_full added merges in order; skips those whose result is in
   the removal list and cascade-skips those depending on a skipped
   token; accepts the first 17,408 surviving merges.
4. Renumbers accepted merge results to ids 131,072..148,479.
5. Writes `variants/c3_added_17408_curated_padded/tokenizer.json` and
   the build manifest.

## Pinned ship artifact

```
variants/c3_added_17408_curated_padded/tokenizer.json
  sha256: 358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394
  vocab:  148,480  (128-aligned ✓, 256-aligned ✓)
  Apertus front-end contract: PRESERVED  (ids 0..131,071 verbatim)
  Curation: 69 noise-token ids structurally absent (filtered at build)
  Build manifest: manifests/curated_padded_at_17408_manifest.json
```

```
variants/c3_added_17408_curated/tokenizer.json    ← ablation only, do NOT ship
  vocab: 148,411 (not aligned, renumbered ids)

variants/c3_added_17408/tokenizer.json            ← raw, no curation
  vocab: 148,480 (aligned), but ships 69 noise tokens
```
