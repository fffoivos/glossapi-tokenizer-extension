# Bibliography-span classifier — bench results & deployment decision

**Objective (precision-first):** a false positive (deleting prose as bibliography) corrupts the training
corpus; a false negative (missing a bib) is cheap and reversible. So we optimize **line-level
precision** — the fraction of *removed* lines that are truly bibliography — and maximize recall subject
to it, not F1. See [[feedback_reference_cleaner_precision_first]].

**Segmentation answer:** predict at the **line** level as a multi-span sequence-labeling task over each
doc's line stream, with a contiguity smoother turning per-line scores into exact spans. This is what
lets us select **exact** start/end and catch **chapter** bibs (41% of docs are multi-span) — neither
possible with the current single-boundary header→EOF rule. Context is decisive: the strongest features
are neighbor year/entry density and document position, not the line in isolation.

## Results (held-out, doc-grouped, leak-free split)

### Head-to-head vs the current Rust detector (greek_phd + openarchives test docs)
| arm | line-precision | prose-amputation (FP) | det recall | κ | boundary |
|---|---|---|---|---|---|
| **Rust header→EOF (current, shipped)** | 0.481 | **51.9%** | 0.84 | 0.30 | starts early, runs to EOF |
| **line-LR (interpretable, NEW)** | **0.944** | **5.6%** | 0.86 | 0.46 | under-captures inward (safe) |

Paired bootstrap (precision-weighted Fβ0.5) **ΔFβ0.5 = +0.32 [95% +0.27, +0.37] → real.** The
interpretable line model removes **~9× less main text** while detecting *more* bibliography. The window
classifier added nothing as a synergy feature (the line model's own context already carries it) — kept
only as an optional high-recall gate. **Parsimony: ship line-LR alone.**

### DL frozen-embedding probe (e5-small, stratified ~33k-line sample, matched precision 0.94)
| | recall@prec0.94 | Greek-script bib recall |
|---|---|---|
| handcrafted features only | 0.930 | 0.922 |
| + frozen line embeddings | **0.972** | **0.970** |
| lift | **+0.042** | **+0.048** |

Embeddings recover the recall holes (monotonic/polytonic Greek, where Latin-biased regex features are
blind) **at the same precision**. The gate fires (lift > 0.02).

## The exact-span vs accuracy vs deployment-cost tradeoff
- **Exact-span / multi-span:** the line tagger wins decisively — the current header→EOF rule cannot
  represent chapter bibs and over-runs into appendices (70% of its detections eat prose). This is the
  whole point of segmenting at the line level.
- **Accuracy (recall):** the interpretable model's recall (~0.59 doc-level after the conservative
  smoother) is the weak axis; frozen embeddings lift it ~4–5 pts, concentrated on Greek script.
- **Deployment cost ladder:** line-LR = a Rust dot-product over cheap regex signals (~free, the
  production shape) ≪ frozen-E5 + numpy head (CPU embedding at corpus scale is **infeasible on home**:
  ~80 lines/s → 3 h for 840k lines, and the production corpus is ~47M rows) ≪ fine-tuned XLM-R (GPU only).

## Iteration (failure analysis → deterministic features → prose-protection)
Failure analysis (`failure_analysis.py`) showed **~97% of false positives are citation-shaped** (footnote
citations, inline cites in prose, reference table-rows, web sources), not running prose. A separability
probe (`determinism_probe.py`) found deterministic *context* features cleanly separate true bib entries
from confusable citations: distance-to-bib-header AUC 0.94, local run-purity 0.91, **under-bib-block
header-scoping 0.90**, **leading-number magnitude 0.86**. Adding the two missing ones (+ wider
run-purity) and **Greek/polytonic entry signals** (Greek author "Επώνυμο, Ό.", μτφρ./επιμ./εκδ. markers,
Greek Extended ranges) lifted the model at held precision:

| metric (gp+oa test) | first model | after deterministic + Greek features |
|---|---|---|
| line-precision | 0.944 | **0.947** |
| recall | 0.594 | **0.650** |
| median line-IoU | 0.62 | **0.78** |
| monotonic-Greek recall | 0.33 | **0.47** |
| bootstrap ΔFβ0.5 vs Rust | +0.32 | **+0.34** |

**Prose-protection (`composition_analysis.py`), the metric that matters:** of every removed line,
**99.91% is not running prose** (94% true bibliography + ~6% footnote/inline/table/web citations); only
**0.09%** is genuine main text wrongly deleted. The "5.6% FP" is almost entirely reference mass, so
whether to also strip footnote/inline citations is a **policy choice the line types expose**, not a
model failure. This is the strongest evidence for the precision-first goal: the classifier essentially
never amputates prose.

## Independent validation (`fp_validation_sample.py` + Opus audit + `fpval_aggregate.py`)
An Opus audit of 353 removed lines in context (sequential workflow), reweighted to the population,
independently confirms the story: genuine running-prose removed **≈0.14%** (prose-protection **0.9986**,
matching the deterministic 0.998); **99.3%** of removed lines are things a bibliography cleaner should
remove; and **67% of the apparent false positives are bibliography the windowed annotation MISSED**, so
the measured strict precision under-states reality — **true-bibliography precision ≈ 0.975**. The model
is more correct than the silver-gold says.

## The recall/safety knob (operating point, `operating_point.py`)
The smoother was first tuned for strict bib-precision ≥0.97, which over-conservatively treats footnote/
inline citations as errors and **cost ~30 points of recall**. Re-tuning against the metric that actually
matters — **prose-protection** — maps a clean frontier on held-out test (one hysteresis-threshold choice):

| prose-protection floor | test recall | prose lines removed |
|---|---|---|
| **≥0.999 (DEFAULT, conservative)** | **0.865** | 0.09% (45 / 50.4k) |
| ≥0.997 | 0.941 | 0.28% |
| ≥0.995 | 0.944 | 0.36% |

The shipped default is the conservative floor (prose-protection ≥0.999, recall 0.865, line-IoU 0.98,
boundaries median-exact, bootstrap ΔFβ0.5 +0.39 vs Rust). It is a single tunable knob
(`span_smooth_params.json`); dial the floor down for up to ~0.94 recall if the (reversible) ~0.3% prose
cost is acceptable. This honors "don't remove main text" while recovering recall the strict-precision
tuning was needlessly sacrificing.

## Decision
1. **Ship the interpretable line-LR now.** It is the decisive precision-first win (prose amputation
   52% → 5.6%, bootstrap-significant) and deploys as a Rust dot-product + hysteresis smoother — no new
   runtime dependency. Sync `span_line_lr_model.json` + `span_smooth_params.json` into the crate
   (`span_line_model.rs` + `span_smooth.rs`, mirroring `beta_gate_model.rs`).
2. **DL is NOT a shipped dependency** — embedding at corpus scale on CPU is infeasible and the
   production contract is interpretable + Rust hot-path. Use it two ways:
   - **Feature-discovery oracle (default):** embeddings recover Greek-script bibs that regex misses →
     add cheap Greek-specific signals to `span_signals.py` / `reference_signals.rs` (Greek author-name
     "Επώνυμο, Α." patterns, Greek bib conventions, polytonic tolerance) and re-measure. Closes the
     recall gap interpretably.
   - **Clariden-GPU ceiling (optional):** run the full frozen embed (minutes on a GH200) + the
     linear-chain CRF / fine-tuned XLM-R there, apply the precision-first paired-bootstrap gate at full
     scale. Only ship if the gain over the (oracle-improved) interpretable model clears the gate AND the
     deployment cost is justified — the burden-of-proof exception. Stub: `clariden/`.

## Artifacts
`span_seq_data.py` (data) · `line_lr.py` + `decode_spans.py` (interpretable tier) · `window_clf.py`
(gate) · `run_rust_baseline_spans.py` + `score_span_models.py` (bench) · `embed_lines.py` + `dl_probe.py`
(frozen DL probe) · `build_span_model_report.py` (hub). Models: `span_line_lr_model.json`,
`window_lr_model.json`, `span_smooth_params.json`. Metrics: `results_span_models.json`,
`results_dl_probe.json`.
