# Detector bake-off — Rust `reference_detector` vs `heading_lexgate` vs regex (2026-07-25)

## Why

We were about to port the Python `heading_lexgate` line model to Rust because it costs
**~9,300 CPU-hours** for the 202,792-doc academic slice (measured: 150 docs / 210,704 lines takes
16 min wall, ~99% single-threaded). Before writing any Rust, we scored the three candidate
detectors on the *same* labelled cohort with the *same* metric — a comparison that had never been
made. The Rust detector in `reference_detector/` predates the cohort and had never been scored
against it.

Cohort: `sealed_tests/bibliography_150_20260723_v2/30_consensus_silver` — 150 docs, 210,704 lines,
23,694 gold bibliography lines, dual-annotator consensus.

## Result

| detector | scope | char P | char R | body destroyed | bib chars per body char |
|---|---|---:|---:|---:|---:|
| heading_lexgate @0.9 | ALL | 0.9976 | 0.8437 | 0.00020 | 412 |
| **rust bib-spans** | ALL | 0.9942 | **0.8541** | 0.00049 | 172 |
| rust span_lr @0.9 (no hysteresis) | ALL | 0.9981 | 0.7826 | **0.00015** | **518** |
| regex heuristic | ALL | 0.9399 | 0.6901 | 0.00433 | 16 |

Per source:

| detector | greek_phd | openarchives | kallipos |
|---|---|---|---|
| heading_lexgate @0.9 | 0.9978 / 0.8548 / .00022 | **0.9982 / 0.8720 / .00013** | **0.9956 / 0.7590 / .00024** |
| **rust bib-spans** | **0.9987 / 0.9106 / .00014** | 0.9850 / 0.8759 / .00106 | 0.9831 / 0.5831 / .00070 |
| regex heuristic | 0.9235 / 0.8224 / .00799 | 1.0000 / 0.7831 / .0000060 | 0.0000 / 0.0000 / 0 |

(charP / charR / body-destroyed)

**Speed: the Rust detector processes the whole cohort in 0.76 s wall (14 s CPU) versus 16 minutes
for the Python pipeline — roughly 1,250× faster. Extrapolated to the academic slice that is
~4 CPU-hours against ~9,300.**

## Reading

- **greek_phd — Rust wins outright.** +5.6pp char recall *and* lower body damage than
  heading_lexgate (0.9106 / 0.00014 vs 0.8548 / 0.00022). Strictly dominates.
- **openarchives — heading_lexgate wins.** Recall is a wash (0.8720 vs 0.8759) but Rust destroys
  8× more body text (0.00106 vs 0.00013). Raising the Rust threshold to 0.95 fixes the damage
  (0.00009) but costs 12pp of recall.
- **kallipos — heading_lexgate wins clearly** (0.7590 vs 0.5831). Expected: Kallipos bibliographies
  are per-chapter and header-less, and this is whole-doc mode. The crate has a `--mode sections`
  path built for exactly this, unused here.
- **The hysteresis decoder earns its place** — it lifts Rust recall from 0.7826 (raw threshold) to
  0.8541 at a modest precision cost.
- **The regex heuristic is dominated** on every source and fails completely on Kallipos. Retire it
  as a candidate; it remains useful only as a fast structural probe.

## Decision

**Adopt the Rust `reference_detector`; do not port `heading_lexgate`.**

Overall it has *higher* char recall than heading_lexgate (0.8541 vs 0.8437) at 2,300× lower cost.
Its 2.5× higher body damage is 0.049% versus 0.020% of body characters — both negligible in
absolute terms for pretraining data, and the corpus-wide difference is ~20 MB of text in 48 GB.

Follow-ups, in value order:
1. **Kallipos / Pergamos** — wire `--mode sections`, or accept the lower recall on ~5% of the slice.
   Running heading_lexgate on Kallipos alone would cost only ~150 CPU-hours if we want its 0.759.
2. **openarchives body damage** — check whether the 0.00106 is concentrated in a few documents
   (fixable) or spread. It is 50% of the academic text by volume, so it is worth one look.
3. The threshold is a compile-time const (`span_line_model.rs: THETA_HI = 0.9`); expose it as a CLI
   flag so the operating point can be set per source without a rebuild.

## Reproduce

```
# build (Clariden, inside uenv — cargo is not on the login node)
sbatch scripts/build_refdet.sbatch          # cargo test --locked && cargo build --release

# export the cohort + gold labels
sbatch --export=ALL,SCRIPT=.../bakeoff_prep.py .../run_gaenv.sbatch

# run the detector (sub-second)
reference_detect --mode bib-spans --input cohort2_docs.jsonl \
  --out-spans rust_bib_spans.jsonl --out-counters rust_counters.jsonl \
  --source cohort2 --text-field text --id-field source_doc_id

# score everything on one metric
sbatch --export=ALL,SCRIPT=.../bakeoff_score.py .../run_gaenv.sbatch
sbatch --export=ALL,SCRIPT=.../rust_sweep.py   .../run_gaenv.sbatch   # threshold sweep
```

Artifacts: `/capstor/scratch/cscs/fffoivos/bib_cleaning_20260724/bakeoff/`
(`bakeoff_results.json`, `rust_bib_spans.jsonl`, `rust_line_prob.npy`, `gold.npy`).

## Caveat

Cohort-2 has been opened for triage and feature derivation earlier in this project, so absolute
values are development-grade. The *comparison* is sound — all three detectors saw identical inputs,
identical labels and identical scoring code, and none of them was tuned on this cohort.
