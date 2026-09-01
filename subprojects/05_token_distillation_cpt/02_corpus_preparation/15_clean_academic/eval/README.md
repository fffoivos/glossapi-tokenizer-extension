# eval — the measurement system

> **In one line:** the harness that decided every model in this stage — three annotation regimes, a bootstrap-gated iteration loop, a plateau broken by re-attaching document position, and a final bake-off that retired the incumbent *and* the metric it was scored on.
> **Period:** 2026-06-13 (`056396fd`) → 2026-07-23 (`445f984a`). **Status:** completed. Its output is [`RECOMMENDED_BIBLIOGRAPHY_MODEL.json`](RECOMMENDED_BIBLIOGRAPHY_MODEL.json).
> **Came from / led to:** [`../investigation`](../investigation/README.md) → this → [`../bib_line_model`](../bib_line_model/README.md) and [`../production`](../production/README.md). The sequence-model program is a sub-track: [`sequence_models/`](sequence_models/README.md).

## Why this existed

The stage's whole risk is asymmetric: removing a bibliography is cheap, amputating body prose is
not. So every question had to be answered by measurement against labelled data, and the *metric*
had to be argued about as much as the models. It was, three times — and the metric changed twice.

## History

### 2026-06-13 — the protocol and the pilot

[`ANNOTATION_PROTOCOL.md`](ANNOTATION_PROTOCOL.md) defines three evals: **B** = the β-gate on
Kallipos/Pergamos sections, **A** = the end-matter boundary on greek_phd/openarchives document
tails, **C** = footnote classification on body windows. Blind Opus-4.8 annotation, controlled
vocabulary, evidence-quote hallucination guard, 15% double-annotated for κ. **Eval C was
specified and never run** — no results file for it exists anywhere.

Eval B pilot, 64 sections, κ=1.00 on 9 double-annotated: **P 0.781 / R 0.862 / F1 0.820**
([`CONFUSION_MATRIX.md`](CONFUSION_MATRIX.md)). Reweighted to the 94,282-section population it
becomes P≈0.70 / R≈0.95 — and the first reversal:
[`EVAL_B_FINDINGS.md`](EVAL_B_FINDINGS.md) states plainly that *the investigation's 0.85–0.90
β-precision estimate was optimistic.*

### 2026-06-13 — Eval B at scale, and a second reversal

2,000 sections annotated by 63 Opus agents, 1,985 matched, a **30% test split frozen before
tuning**. Weighted P 0.762 [0.709–0.820] / R 0.914 / F1 0.831
([`CONFUSION_MATRIX_SCALE.md`](CONFUSION_MATRIX_SCALE.md)). CV publication lists explode from 2
false positives to 90 — a systematic trap, not noise.
[`EVAL_B_SCALE_FINDINGS.md`](EVAL_B_SCALE_FINDINGS.md) then compares three-way on held-out test:
current gate 0.831 F1, a hand-tuned deterministic rule **0.833**, a 13-feature logistic
regression **0.886**. It is explicit: *"a reversal of my earlier 'rules-first, model probably
unnecessary' advice… Your instinct to train a model was right."* Weights in
[`MODEL_COMPARISON.md`](MODEL_COMPARISON.md).

### 2026-06-13 — the adversarial iteration loop (rounds 1–4)

[`ITERATION_LOG.md`](ITERATION_LOG.md) runs a standing loop of three critics (explainability,
parsimony, idea exploration) behind a **paired-bootstrap gate: a change wins only if the ΔF1 CI
excludes zero.** What it produced:

| Round | What was tried | Result |
|---|---|---|
| R1 | 13-feature LR; plus an elegant `entry_density ≥ 0.5 + header deny` idea | LR 0.903/0.867/**0.885**; the idea **failed** (0.801) — CV lists are also entry lists, so the header is the discriminator |
| R2 | Drop the `url` shortcut; entry-count instead; a high-precision RULE | LR6* 0.888/0.860/0.874 — all model deltas **inside noise**; RULE (0.944/0.658) **bootstrap-rejected**, ΔF1 −0.084 [−0.133, −0.036] |
| R3 | Calibrate the sub-detectors | author recall 0.79→0.89, place 0.64→0.97, editor 0.55→0.87 — and model ΔF1 **+0.007 = noise**. Plateau confirmed a second way; R3b error analysis a third |
| R4 | Re-attach document position and section neighbours | **0.939/0.861/0.898, ΔF1 +0.046 [+0.017, +0.078] — real.** CV-list false positives 26 → 5 |

The conclusion is the reusable one: *the ~0.85 F1 plateau was an artifact of the unit of
classification (an isolated section), not the problem's ceiling.*

### 2026-06-13 — Eval A: the mirror image

497 document tails, 457 usable: **P 0.986 / R 0.619**
([`CONFUSION_MATRIX_A.md`](CONFUSION_MATRIX_A.md)). Localisation was already perfect — median
|error| 0 lines, signed median 0. Per source, recall is 0.78 openarchives against 0.50 greek_phd.
[`EVAL_A_FINDINGS.md`](EVAL_A_FINDINGS.md) diagnoses header-stem matching as the cause and
prescribes the R4 insight as the fix: header **or** entry-density at high position.

### 2026-06-16 — asking whether the pipeline achieves the goal

Two matrices reframe the question from per-component correctness to end-to-end outcome.
[`CONFUSION_MATRIX_GOAL_SECTIONS.md`](CONFUSION_MATRIX_GOAL_SECTIONS.md): the stacked
classifier→β→gate pipeline scores P 0.960 / R 0.811, and **1,927 of 6,753 misses (29%) are true
bibliographies in non-β sections the gate never saw** — invisible to a β-only eval.
[`CONFUSION_MATRIX_GOAL_WINDOWS.md`](CONFUSION_MATRIX_GOAL_WINDOWS.md), line-level over both
whole-doc sources: **tail recall 0.80, body recall 0.46.** End-of-chapter bibliographies are
structurally missed. This is what forces the multi-span pivot.

### 2026-06-13→15 — the span dataset, and two infrastructure dead ends

[`SPAN_DATASET_PLAN.md`](SPAN_DATASET_PLAN.md): three sources (Pergamos **dropped**, Kallipos
reconstructed doc-level), ~2,000 spans extended to ~2,800, line granularity. Two decisions worth
keeping: **Haiku was tested and rejected as annotator** (recall 0.63, fragments spans, κ 0.56
against Opus); and **mass-concurrent Opus annotation failed** on a server-side tokens/minute
throttle — 16 agents launched, 1–2 survived — so the pipeline went sequential (~5 min/batch,
16/16). One batch tripped the AUP content filter and was parked.

### 2026-06-16 — the model that replaced the header rule

[`MODEL_TRADEOFF.md`](MODEL_TRADEOFF.md) is the decision document. Head to head against the
shipped Rust header→EOF rule: **line precision 0.481 → 0.944, prose amputation 51.9% → 5.6%,
ΔFβ0.5 +0.32 [+0.27, +0.37]** — about 9× less main text removed. Then the metric changed for the
first time: strict bibliography precision was treating footnote and inline citations (reference
mass, fine to remove) as errors and costing ~30 points of recall, so the operating point moved to
**prose protection** — ≥0.999 → recall 0.865 (default), ≥0.997 → 0.941, ≥0.995 → 0.944. An
independent Opus audit of 353 removed lines confirmed it: genuine prose removed ≈0.14%, and
**67% of apparent false positives were bibliography the windowed annotation had missed**, so
true-bibliography precision is ≈0.975 against a measured 0.944. A frozen-embedding DL probe gave
+0.042 recall and was **deliberately not shipped** — ~80 lines/s on CPU is infeasible on a ~47 M-row
corpus, so it stayed an oracle for feature discovery.

### 2026-06-18/19 — v2 annotation, and a change of annotator

[`ANNOTATION_SPEC_v2.md`](ANNOTATION_SPEC_v2.md) redoes v1: span annotation on isolated windows
produced loose boundaries and had no notion of the rest of the document. The taxonomy collapses
from six kinds to **two** (`table_of_contents`, `bibliography`) with flags, after a 10-doc
experiment surfaced a merge bug in `merge_chunks.py` and a taxonomy question the owner resolved —
**abbreviation/glossary/figure lists are main text and are not marked**
([`STRUCT_REVIEW_ISSUES.md`](STRUCT_REVIEW_ISSUES.md)); a `greek_badness_score > 60` extraction
gate was added.

The annotator changed. [`CODEX_LIMITS.md`](CODEX_LIMITS.md) records a 10-doc bake-off in which
**gpt-5.5 via codex** scored accuracy 0.984, κ 0.937, exact boundaries and zero prose eaten
against Opus — *and recovered 7 real bibliographies Opus had missed*. With Opus weekly usage at
~82%, gpt-5.5 became the engine.
[`ANNOTATION_RUNBOOK.md`](ANNOTATION_RUNBOOK.md) — headed **"ARCHIVED HISTORICAL PROCEDURE — DO
NOT RERUN"** — carries six safety rules written after an unattended self-firing `/loop` spawned
**~1,872 agents in a day**.

[`MODEL_DESIGN_RESEARCH.md`](MODEL_DESIGN_RESEARCH.md) settles the architecture: two parallel
binary heads rather than a 3-way softmax (the classes are near-orthogonal), position as a score
gate on the ToC head only, CRF and transformers demoted to oracles (GROBID's own bound is +0.5–1
F1 at 2–3× runtime), no shipped embeddings, and the CV head **gated** in favour of a deny lexicon
because only 244 positive spans exist. It also found two real code bugs: `_GRK` omitting
polytonic Greek Extended, and Rust `fold_char` folding no polytonic capitals.
[`STRUCT_RESULTS.md`](STRUCT_RESULTS.md) reports the trained heads — bib recall 0.876 at line
precision 0.97 (deployed 0.857), ToC 0.380 raw → **0.602** smoothed at precision 0.972 — noting
the ToC front gate structurally caps recall at 91% and the test set was reused for ToC feature
design.

### 2026-07-11 — the honesty pass

`STRUCT_RESULTS.md` is retro-annotated: the Rust deployment is in (18 tests, Python↔Rust within
1e-12), but the importer emits only the 1,392 historical-train documents, **no import, ladder or
parity job has run, and the labels are LLM silver despite the `STRUCT_2K_gold.jsonl` filename** —
so the numbers are historical, not promotion evidence.

### 2026-07-20 → 07-23 — the NEXTGEN generation and the model decision

- **07-20** ([`BIBLIOGRAPHY_NEXTGEN_EXPERIMENT_LOG_20260720.md`](BIBLIOGRAPHY_NEXTGEN_EXPERIMENT_LOG_20260720.md)):
  five generations on 1,118 dev documents / 939,014 lines. The dev winner —
  position-aware HistGB + linear component scope @0.90 — scores 0.9974 line precision on
  development and only **0.9680 on the sealed 143-document test**, below the 0.98 gate.
  **Decision: keep `incumbent_entry`; component scope stays research.**
- **07-22, three documents in one day.** The worst-docs review reads the 30 worst sealed-test
  documents. The improvement review diagnoses the dev→test collapse three ways — winner's curse
  (max over ~192 decoder configs × 13 thresholds at the constraint boundary), a binding gate
  carrying **no test signal** (`spurious_blocks` = 0.0 for every test candidate), and a 4×
  distribution mismatch (dev 6.3% spurious components against test 26.9%) — decomposes the 1,704
  missed lines (360 Markdown headings the emitter was configured to skip, 501 in whole vetoed
  components, 808 never proposed) and the 623 false positives (44 long lines carrying 50.6% of the
  false-positive character mass), and ranks nine fixes with measured sizes. The devfix run
  implements them (dev line recall +2.34 pts) and records one **negative result**: a high-recall
  proposal pool raised scope negatives 152→485 and did not improve the operating point. The lexgate
  run removes two structural blockers — **a single mislabelled line** (`## ΒΙΒΛΙΟΓΡΑΦΙΑ` in one
  document) had disqualified all 256 gated-heading candidates, and the spurious-block gate was
  knife-edge (improving the decoder moved it 2→3 blocks and **cost 1.1 points of recall**),
  replacing the latter with a continuous body-character-damage criterion. It also **rejects two
  reviewer proposals on measurement**: a long-line guard (>330-char lines are 5,972 true against
  323 false; a relative rule removes 96,755 false characters and 276,011 true ones) and an
  asymmetric bar (precision +0.0001 for −0.8 to −1.3 points of recall).
- **07-23** ([`BIBLIOGRAPHY_NEXTGEN_COHORT2_BAKEOFF_20260723.md`](BIBLIOGRAPHY_NEXTGEN_COHORT2_BAKEOFF_20260723.md)):
  a fresh 150-document cohort, verified disjoint from both the 143-document cohort and the
  1,392-document dev source. The earlier cohort proves **optimistic by 2.3 points of precision
  and 8.2 of recall**. Same-cohort, same-metric comparison:

  | Model | char P | char R | bib removed | body destroyed | bib chars per body char |
  |---|---:|---:|---:|---:|---:|
  | `incumbent_entry` | 0.9127 | 0.5388 | 53.9% | 0.505% | 10.5 |
  | `position_hist_unscoped` | 0.9193 | 0.9254 | 92.5% | 0.796% | 11.4 |
  | `orig_component_scope` @0.90 | 0.9454 | 0.8840 | 88.4% | 0.500% | 17.3 |
  | `devfix_corrected` @0.85 | 0.9412 | 0.9033 | 90.3% | 0.553% | 16.0 |
  | **`heading_lexgate` @0.98** | **0.9703** | 0.8595 | **86.0%** | **0.258%** | **32.7** |

  `heading_lexgate` does **not** clear the 0.98 line-precision gate (it scores 0.95921) — and the
  argument is that the gate is denominated in the wrong unit: *"the incumbent scores 0.9788 line
  precision but 0.9127 character precision because its false positives are long lines, and
  corpus cleaning loses characters rather than lines."* Adopting the recommendation meant adopting
  that change of measure, recorded as an explicit owner decision (accepted 2026-07-27, see
  [`RECOMMENDED_BIBLIOGRAPHY_MODEL.json`](RECOMMENDED_BIBLIOGRAPHY_MODEL.json)). Error triage found
  annotation quality was **not** the limiting factor — lexicon-matched headings were 137/137 gold —
  and that one law thesis contributed 28% of all false-positive lines.

## Outcome

- **Model:** `heading_lexgate` @ scope threshold 0.98 supersedes `incumbent_entry` — 86.0% of
  bibliography characters removed for 0.258% body damage, against 53.9% and 0.505%.
- **Metric:** changed twice, both times because the incumbent metric was measuring the wrong
  thing — strict bibliography precision → prose protection (2026-06-16), then line precision →
  character body-damage budget (2026-07-23).
- **Method results that outlived their models:** the classification unit matters more than the
  classifier (R4); a discrete safety count can make a better model score worse; a dev-tuned
  operating point at a constraint boundary does not transfer; and the labels understate the model
  (67% of apparent false positives were missed bibliography).
- **Left open** in `RECOMMENDED_BIBLIOGRAPHY_MODEL.json`: Kallipos recall 0.677 at this operating
  point; n=150 with heavy tails (three documents moved pooled precision by 1.8 points); three
  sources only; **cohort 2 is now open**, so a future unbiased read needs a fresh seal; and the
  0.00025 body-damage level was back-calibrated on the 20260718 cohort rather than derived from a
  stated tolerance.

## Where things are

| Path | What |
|---|---|
| [`RECOMMENDED_BIBLIOGRAPHY_MODEL.json`](RECOMMENDED_BIBLIOGRAPHY_MODEL.json) | The machine-readable decision: config, six receipt SHA-256s, sealed one-shot metrics, same-cohort comparison, known limitations, conditions before corpus-scale use. |
| `span_line_lr_model.json`, `toc_line_lr_model.json`, `window_lr_model.json`, `beta_gate_model.json` (+ `*_struct_model.json`, `*_syn_model.json`) | The trained models as JSON weight vectors — numpy, no sklearn, Rust-portable by design. |
| `span_smooth_params.json`, `struct_smooth_params.json` | The frozen hysteresis operating points. |
| `span_dataset.jsonl` | The 3,186-span / 1,581-doc Opus dataset. |
| `units/` | Manifests that regenerate the (gitignored) annotation windows, plus `SPAN_split.json`, `SPAN_skip.json`, `SPAN_aup_dropped.log`. |
| `annotations*/` | Raw blind LLM labels per eval — span, scale, scale_A, goal_sections, goal_windows, and the rejected `goal_windows_haiku`. |
| `results_*.json` | One results file per experiment, matching the documents below. |
| Build/score/train scripts | `build_*_units.py`, `score_*.py`, `train_*.py`, `line_lr.py`, `window_clf.py`, `decode_*.py`, `operating_point.py`, `failure_analysis.py`, `composition_analysis.py`, `rust_parity*.py`, `iterate*.py`. |

## Working documents

Grouped, all historical:

- **Protocols and specs** — `ANNOTATION_PROTOCOL.md`, `ANNOTATION_SPEC_v2.md`, `STRUCT_PROMPT.md`, `SPAN_DATASET_PLAN.md`. `ANNOTATION_RUNBOOK.md` is explicitly marked do-not-rerun.
- **Confusion matrices** — `CONFUSION_MATRIX.md` (B pilot), `CONFUSION_MATRIX_SCALE.md` (B at scale), `CONFUSION_MATRIX_A.md` (boundary), `CONFUSION_MATRIX_GOAL_SECTIONS.md` and `CONFUSION_MATRIX_GOAL_WINDOWS.md` (goal-level).
- **Findings narratives** — `EVAL_A_FINDINGS.md`, `EVAL_B_FINDINGS.md`, `EVAL_B_SCALE_FINDINGS.md`.
- **The iteration loop** — `ITERATION_LOG.md` (the index), `ITERATION_ROUND1.md`, `ITERATION_ROUND2.md`, `ITERATION_ROUND4.md`. **There is no `ITERATION_ROUND3.md`** — round 3 exists only inside `ITERATION_LOG.md`.
- **Model decisions** — `MODEL_COMPARISON.md`, `MODEL_TRADEOFF.md`, `MODEL_DESIGN_RESEARCH.md`, `STRUCT_RESULTS.md`.
- **NEXTGEN run logs** — the four `BIBLIOGRAPHY_NEXTGEN_*_2026072*.md`. Three carry SUPERSEDED banners; `..._WORST_DOCS_REVIEW_20260722.md` does not, though it is from the same superseded line.
- **Operational notes** — `CODEX_LIMITS.md` (gpt-5.5 budget and auth traps), `STRUCT_REVIEW_ISSUES.md`.

### Known inconsistencies in this directory

- **β-precision is quoted three ways** — 0.85–0.90 (investigation), ≈0.70 (pilot reweighted),
  0.762 (at scale). The scale document reconciles them; the pilot files are not marked superseded.
- **Two different "final" β-gates coexist.** `MODEL_COMPARISON.md`/`EVAL_B_SCALE_FINDINGS.md`
  recommend the 13-feature LR (0.886 F1); `ITERATION_ROUND4.md` declares the final gate to be the
  6-internal + 7-position LR (0.898). `EVAL_A_FINDINGS.md` summarises the deployed gate as
  "P≈0.85/R≈0.91", which is the round-3 lock and matches neither published number.
- **`CODEX_LIMITS.md` reverses itself internally** — the 10-doc section says the weekly budget is
  the binding constraint and only ~800 of 2,000 fit; the 100-doc section says all 2,000 fit and
  the weekly cap is not the blocker. Both are left in place.
- **Canonical-prompt ownership conflicts**: `ANNOTATION_SPEC_v2.md` §9 names
  `wf_struct_annotate.js`; `ANNOTATION_RUNBOOK.md` and `STRUCT_PROMPT.md` name `STRUCT_PROMPT.md`.
- **`heading_lexgate` passes and fails the same gate** — 0.98503 line precision on the opened
  143-doc cohort (LEXGATE §4), 0.95921 on the sealed 150-doc cohort (COHORT2). Same model,
  opposite verdicts, driven entirely by the cohort.
- **"Gold" naming persists** throughout (`STRUCT_2K_gold.jsonl`, `build_gold_from_ann.py`) even in
  documents that carry a notice saying the labels are LLM silver.
