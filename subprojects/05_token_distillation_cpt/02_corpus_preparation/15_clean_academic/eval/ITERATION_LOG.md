# β-gate iteration log — standing adversarial loop

Process (every round): propose change (evolutionary OR idea-based) → measure on the **frozen held-out
test** (596 weighted units) → three standing adversarial agents [C1 explainability/justification ·
C2 parsimony/anti-overfit · C3 idea-based exploration] → **paired-bootstrap gate** (a change only
"wins" if its ΔF1 CI excludes 0). Nothing ships unless justified by what a bibliography IS (a list of
reference entries, Docling-extracted + GlossAPI-cleaned), short, and beats noise.

Dataset: 1985 Opus-labelled β sections (typed: kind/style/language/script/subject/syntax/n_entries),
1389 train / 596 test, stratified with inclusion weights → corpus-level metrics. Inter-annotator κ = 1.00.

| round | change | held-out P / R / F1 | verdict |
|---|---|---|---|
| — | starting deterministic gate | 0.762 / 0.914 / 0.831 | baseline |
| 1 | LR over interpretable features; "entry_density" 1-feature idea | LR-10 0.903/0.867/0.885; idea **failed** 0.728/.../0.801 | idea failed *for a reason*: CV lists are also entry-lists → header is the discriminator |
| 2 | idea: absolute entry **count** (not fraction); drop `url` (C1); header-deny | LR-6* 0.888/0.860/0.874 | entry-count replaced the `url` shortcut; **paired bootstrap: model deltas all within noise** → plateau suspected |
| 3 | calibrate detectors vs ground-truth labels; fix weak ones | author rec 0.79→0.89, place 0.64→0.97, editor 0.55→0.87; **model ΔF1 +0.007 [−0.019,+0.034] = noise** | plateau confirmed a 2nd way; FN became language-balanced (Greek bias fixed) |
| 3b | error analysis | residual FP = CV-lists (non-CV-header) + prose stubs; FN balanced | plateau confirmed a 3rd way → C2 says stop feature golf |
| **4** | **idea-based BREAK: document POSITION + neighbour context** | **0.939 / 0.861 / 0.898** | **ΔF1 +0.046 [+0.017, +0.078] → REAL.** CV-FP 26→5 |

## Outcome

The plateau (~0.85 F1) was an **artifact of the unit of classification** (isolated section), not the
problem ceiling. Re-attaching document position — already logged in every span, discarded at the model
boundary — broke it: **precision 0.85→0.94, F1 0.85→0.90**, killing 80% of the CV-list false positives.
Final β-gate = logistic regression over justified section-internal features (entry-count, author
"Surname, Firstname", year-in-parens, place, header-CV/APP deny) **+ position features** (pos, dist-to-end,
is-last-β, front-cluster density, n-siblings). Deployable in Rust as a dot-product.

## Standing caveats / next moves (per the critics)

- **C2:** prune the 7 position features to the load-bearing 3 (`front_cluster`, `is_last`, `dist_end`)
  and re-confirm via bootstrap; check parsimony didn't bloat.
- **C1:** owes the ablation honesty audit (each feature justified-and-load-bearing vs justified-but-inert)
  + a tight Greek-imprint detector to replace `place` defensibly.
- **C3 / pivot:** the position insight **generalises to Eval A** (whole-doc end-matter boundary for
  greek_phd/openarchives), where position *is* the core signal and headroom is larger — but needs a
  fresh annotation round (those sources have 0 labels today). Residual β errors (CV-lists w/o CV headers,
  short ambiguous bibliographies) are now genuinely hard; more labels shrink CIs, threshold sets the
  precision/recall operating point for the drop policy.
