# Eval B at scale — confusion matrix, error modes, and the model decision

2000 β sections sampled (stratified, inclusion-weighted), annotated blind by 63 Opus 4.8 agents
(1985 matched, 15 dropped), with a **frozen 30% held-out test** (596) split before any tuning.

## 1. Trustworthy confusion matrix (held-out TEST, prevalence-weighted to 94,282 β)

precision **0.762** (95% CI 0.709–0.820) · recall **0.914** (0.888–0.940) · F1 0.831 · acc 0.799.

So the corpus β-gate precision is **~0.76**, recall ~0.91 — the investigation's 0.85–0.90 estimate
was optimistic; the pilot's 0.70 was slightly pessimistic. The gate **over-flags** (precision-limited).

## 2. Error modes (now statistically solid, n=2000)

- **False positives** (gate=bib, truly not): `not_reference` 116 + **`cv_publication_list` 90** +
  colophon 4 + footnote 5 + web 2. The CV-list FP mode is **much bigger than the pilot showed** (2→90)
  — the author's own publications (Δημοσιεύσεις / Ανακοινώσεις / Κατάλογος) are a systematic trap.
- **False negatives** (gate=kept, truly bib): **Greek 120 + mixed 40** + latin 11. Confirmed at scale:
  the gate misses **Greek-script bibliographies** (its rule needs Latin-fraction or dash bullets, which
  Greek hanging-indent bibliographies lack).
- Per-stratum: `bib_strong` FP-rate **0.274** (the dominant corpus stratum → drags precision),
  `kept_hasyear` FN-rate **0.308**.

## 3. Three-way comparison on the held-out test (all fit on train, scored on test)

| predictor | precision | recall | F1 | accuracy |
|---|---:|---:|---:|---:|
| current gate | 0.762 | 0.914 | 0.831 | 0.799 |
| tuned deterministic rule (deny CV/colophon + year/author include) | 0.757 | 0.927 | 0.833 | 0.799 |
| **logistic regression, 13 interpretable features** | **0.882** | 0.889 | **0.886** | **0.876** |

**The honest result — and a reversal of my earlier "rules-first, model probably unnecessary" advice:**
a hand-tuned single-threshold rule barely moved F1 (0.831→0.833), because the errors are **not**
separable by one or two thresholds — they need the multi-feature *interaction*. The logistic
regression over the same interpretable features lifts **precision 0.76→0.88 (+12 pts), F1 0.83→0.89**
on held-out data, well outside the bootstrap noise. **Your instinct to train a model was right.**

Crucially it stays in-bounds: the model is a **13-weight logistic regression over transparent
features** (year-line density, author-initial pattern, URL/DOI, year-in-parens, CV-marker, …), so it
deploys in Rust as a **deterministic dot-product** — auditable, fast, no NN. Standardised weights:
`year_line +1.34`, `url +1.14`, `author_init +0.82`, `year_paren +0.43`, `cv_marker −0.39`,
`digit −0.34`, `comma_ano +0.30` — all sensible.

## 4. Recommendation

Deploy the logistic regression as the β-gate (port the 13 weights + standardisation into the Rust
module), keeping the CV/colophon deny as hard guards. The 1389-train/596-test labelled set is the
regression harness; re-running this eval after any change re-measures honestly. Residual precision
0.88 leaves headroom — more labels (especially around the `bib_strong` FP boundary) and a couple more
features (a real Greek function-word/verb resource for the prose FPs) are the next lift.
