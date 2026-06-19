# Two-head line classifier — results (2000-doc gold)

Per-line structural tagger: two independent binary heads (bibliography + table-of-contents) over the
unified gold `units/STRUCT_2K_gold.jsonl` (2000 docs; labels 0 other 78% / 1 bib 15% / 2 toc 6%;
doc-grouped leak-free split train 1392 / test 608). Design = `MODEL_DESIGN_RESEARCH.md`. All numbers on the
held-out TEST. Pipeline: `struct_lines.py` (loader) → `train_struct.py` (heads) → `decode_struct.py`
(hysteresis) → `eval_struct.py` (prose-protection frontier).

## Heads
- **bib** = LR over the existing 22 features (`line_lr.FEATS`), retrained on the new gold.
- **toc** = LR over 27 features = the 22 (incl. `pos`, year/entry signals as anti-features) + 5 new ToC
  signals (`toc_leader`, `toc_secnum`, `toc_pagetail`, `toc_mdrow`, `toc_contents_hdr`). Probabilities
  **front-gated** to `abs_idx < min(300, 0.30·N)` before decoding.
Learned ToC weights (top): `pos −1.17` (front prior), `toc_mdrow 0.67`, `toc_secnum 0.44`,
`toc_pagetail 0.33`, `toc_leader 0.32`, `latin_frac −0.26`, `yd8 −0.23` (low-year = anti-bib).

## Raw per-line (pre-smoothing), recall @ line-precision 0.97
| head | recall@p0.97 |
|---|---|
| bib (new gold) | **0.876** |
| bib (deployed/old-data model, on new test) | 0.857 |
| toc (front-gated) | 0.380 |

**No-regression check:** the new-gold bib head beats the deployed model on the new test (+0.019) → promote.
ToC's low raw recall is expected (chapter-title lines inside a ToC block carry no own-line signal); the
smoother recovers them.

## Smoothed line-level (hysteresis), at line-precision target 0.97
| head | line-precision | recall | tuned θhi/θlo/gap/lmin |
|---|---|---|---|
| bib | 0.956 | 0.905 | 0.6/0.5/2/5 |
| toc | 0.972 | 0.602 | 0.8/0.3/8/2 |

Smoothing lifts ToC recall 0.38 → 0.60 at precision 0.97 (`gap=8` bridges chapter-title lines).

## Prose-protection frontier (THE precision-first metric — only running-main-text removal is costly)
Most bib "false positives" are citation/footnote mass, not prose; `failure_analysis.categorize_fp`
separates them. `results_struct_operating_point.json` has the full frontier.

| head | floor | test recall | prose-protection | genuine-prose lines removed |
|---|---|---|---|---|
| **bib** | 0.999 | 0.824 | 0.9990 | 61 / 60,814 |
| **bib** | 0.997 | **0.912** | 0.9972 | 197 / 70,336 |
| **toc** | (best) | **0.756** | **0.9997** | **9** / 28,609 |

Both heads remove their target structure while eating **almost no main text**. Frozen operating points
(conservative floor 0.999) → `struct_smooth_params.json`; bib floor 0.997 is available for +0.09 recall at
negligible prose cost — an operating-point choice for the user.

## Known ceilings / caveats
- **ToC front-gate** keeps 91% of gold ToC lines; the other 9% sit past `min(300, 0.30·N)` (long
  front-matter) — a structural recall cap. Loosening the gate is a tunable knob.
- TEST is reused for ToC feature design → treat these as an upper bound; a fresh holdout should give the
  one final number (per the design's validity note).
- Span-level IoU + doc-bootstrap CIs (the other two of the three views) are not yet run — line-level
  prose-protection is the headline; those are refinement.

## Next — Step 6, Rust deployment (gated on this eval; cleared). Precise scope:
1. **Promote bib + re-sync `span_line_model.rs`** from the new `span_line_lr_struct_model.json`:
   `MU/SD/W/BIAS` (22 feats) + `THETA_*` from `struct_smooth_params.json["bib"]`. **Coupled fix:**
   `py_latin_fraction` (`:67`) must add `'\u{1F00}'..='\u{1FFF}'` to match the now-polytonic Python `_GRK`
   (the head was retrained with it) — and flip the `latin_fraction_excludes_polytonic_like_python` test to
   assert inclusion (e.g. `py_latin_fraction("aἱ") == 0.5`). Re-run `rust_parity.py` (needs a struct-data
   variant since the harness currently loads the old `span_seq_data`).
2. **New `toc_line_model.rs`** mirroring `span_line_model.rs`: 27-feature vector (the 22 bib signals — reuse
   by exposing a feature-vector fn from `span_line_model`, don't duplicate — + the 5 ToC signals byte-mirrored
   from `span_signals.toc_signals`), `MU/SD/W/BIAS` from `toc_line_lr_model.json`, the front-gate
   `p *= (abs_idx < min(300, 0.30*N))`, `hysteresis` with `struct_smooth_params.json["toc"]`, emit
   `Span{kind:"toc_span"}`. Export in `lib.rs`; add `--mode toc-spans`/`toc-score-lines` in `main.rs`.
3. **`fold_char` polytonic-capital fix** (`reference_signals.rs:29`) + a polytonic-capital-header parity
   fixture — re-check the β-gate parity (fold_char feeds header detection).
4. **`rust_parity_toc.py`** mirroring `rust_parity.py`: per-line `max|Δp|<1e-3` + identical decoded spans on
   the struct test split; `cargo test` green.

NOTE: the eval `_GRK` fix already landed in Python; the Rust `py_latin_fraction`/`fold_char` are the
matching deployment-side edits, deferred to this step so the crate's parity invariants aren't half-broken.
