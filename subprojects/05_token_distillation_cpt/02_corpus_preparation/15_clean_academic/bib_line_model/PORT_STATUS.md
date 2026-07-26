# Rust port of the `heading_lexgate` line model — status

**Bar:** decision-equivalent. The emitted line mask at threshold 0.9 must match the
Python pipeline document-for-document. Probabilities may differ in the last bits.

**Deployment target:** v3 feature contract (126 columns, no citation-grammar block),
academic sources only (202,792 documents).

---

## Verified so far

Everything below is checked against **what the reference pipeline actually produced
at scale**, not against fixtures regenerated from the same code.

| stage | gate | result |
|---|---|---|
| 35 count features | vs deployed `features.npy`, 210,704 lines | **bit-exact** |
| 34 `line_shape` values | same | **bit-exact** |
| 7 gap summaries | same | **bit-exact** |
| 5 structure flags | same | **bit-exact** |
| `probability:entry` | same | **bit-exact 210704/210704** |
| heading candidate mask | same | **exact** — 28,620 both sides, matching the run receipt |
| `bib_header` / `bib_subheader` / `non_bib_header` | same | max abs diff **1.19e-7**, 0 rows > 1e-6 |
| negative roles (8) + header kind | vs `dump_roles.py` reference matrix | **exact 210704/210704 (100.000%)** |
| `probability:signal_tcn` | vs deployed `features.npy` | max abs diff **1.49e-7**, mean 1.8e-9, 0 rows > 1e-6 |
| **columns verified** | | **121 / 126** |
| TF-IDF char_wb + word | vs *fitted* sklearn vectorizers, 4,000 real lines | **0 support, 0 value mismatches** (worst rel 1.7e-7 = float32 rounding) |

The three heading columns are not bit-exact and are not expected to be:
`HeadingTransform.apply` hstacks and L2-normalises in float32 where the port
accumulates in f64. That is one float32 ULP, with zero rows above 1e-6 — no
downstream tree split can resolve it. The binding check remains the end-to-end mask.

Reference artifacts (Clariden):

```
.../experiments/bib_nextgen_devfix_20260722/
    unseen_features_cohort2_v7/features.npy              210704 x 126   <- column gate
    unseen_predictions_cohort2_v7/
        heading_lexgate_scope.probability.npy            210704 f32     <- end-to-end gate
        heading_lexgate_scope.prediction.npy             210704 u8
```

Throughput: **20,346 lines/s at 64 threads** (full cohort-2 in 18.8 s). Scaling turns
over past 64 on a 4-socket Grace node, so the corpus run should use four tasks of 72
cores per node rather than one task of 288.

## Remaining — 5 connector columns, then the line model

**The one job left is the connector bundle.** Its gate is already dumped.

1. **Connector bundle** — the 5 remaining columns (`connector`,
   `continuation_specialist`, `continuation`, `filler`, `other`), from
   `_connector_probabilities` + `connector_feature_row` (**177 features**).
   - The gate is dumped by `fixtures/dump_connector.py` into `port/connector/`:
     `candidate_mask.npy`, `connector_rows.npy` (m x 177), `connector_index.npy`,
     `feature_names.json`. Port against the **feature matrix**, not the four output
     columns — a mismatch then names the feature instead of being a needle in a
     177-dimensional haystack. This is what made the role port land 100% first try.
   - Candidate gate is `candidate_window_mask`: lines within radius 30 of a seed,
     where a seed is entry >= 0.25 or a heading candidate. Receipt says 185,478
     candidates (~88% of lines).
   - Non-candidates are **not** zero: they default to `(0, 0, 0, 1)` — i.e.
     `other = 1.0`.
   - Each candidate needs its neighbours' *joined* text scored through P0D again
     (`score_counts`), deduplicated by the count vector's bytes. That dedup is
     per-document and is the stage the plan flagged as the main algorithmic waste.
   - Deployed arm is `hist` with `mean`/`scale` both null, so no scaling.
   - **Resolved:** `probability:continuation_specialist` is `connector[:, 1]`
     copied — `_load_specialist` returns its fallback unchanged when no specialist
     root is configured, which is what
     `continuation_specialist_policy: frozen_connector_continuation_fallback`
     means. Verified against the deployed table: columns 3 and 4 are bit-identical.
     So the stage produces **four** distinct values, not five.
   - **Confirmed against the deployed table:** 185,478 candidates and 25,226 rows
     sitting at the (0, 0, 0, 1) default, which matches the receipt exactly. That
     count is the first thing to check when the port runs.
2. **Line model** — HistGB x5 over the 126 columns, identity scaler, five-fold mean,
   threshold 0.9. Then diff the mask against
   `heading_lexgate_scope.probability.npy` — the contract, and the loop's stop
   condition.

Fold aggregation everywhere follows `_batched_predict`: sum `predict_proba[:, 1]`
over folds in **float64**, divide by the fold count, cast to **float32**.

---

## Design decisions worth not relitigating

**Regexes are compiled from Python's own pattern text.** `fixtures/dump_patterns.py`
reads the compiled `re.Pattern` objects out of the deployed modules and emits their
source; Rust compiles those exact strings. The patterns interpolate character classes
Python enumerates at import time from `unicodedata` over the European script ranges —
`_UPPER` alone is ~1,300 codepoints written longhand — so a hand-written Rust class
would be a guess about category tables that drift between runtimes. That drift is real
and was observed: **Clariden runs Unicode 15.0.0, this laptop 16.0.0**, and
`_NAME_INITIAL_PAIR` differs by 300-odd characters between them. The corpus run happens
on Clariden, so `patterns.json` and `unicode_tables.json` are generated **there**.

**fancy-regex, not `regex`.** 33 patterns use lookaround and two use named
backreferences. Keeping the pattern text verbatim is what makes the port equivalent by
construction; hand-restructuring 33 lookarounds would be 33 chances to diverge
silently.

**Python's character predicates come from Python's tables.** `unicode.rs` carries
run-length-encoded general categories, `str` methods, the combining class, a 1,530-entry
casefold map, and the GREEK/LATIN split that Python performs by inspecting each
character's *name* — which Rust has no table for at all.

**Offsets are bytes in the count features, characters everywhere else.** The count
arbitration only ever compares spans, and byte order is isomorphic to character order,
so it can stay in byte space. `line_shape` and the gap summaries divide by
`len(normalized)`, which is a code-point count, so they cannot.

## Semantic traps already found and fixed

Each of these was silent — no error, no crash, just different numbers — and each was
caught by widening a gate rather than by reading harder.

1. **`\w` means different things.** Rust's is `Alphabetic|M|Nd|Pc|Join_Control` and
   accepts combining marks; Python's is `str.isalnum()` plus underscore and does not.
   On OCR'd Greek maths a combining tilde before a token flipped `(?<!\w)` and dropped
   3 of 12 matches on one line. `\b` is defined in terms of the same class, so all 12
   patterns using it carried the bug. Now rewritten at dump time; `str.isalnum() ==
   category L* or N*` is asserted, not assumed.
2. **`_SENTENCE_TERMINAL` ends in U+0387 GREEK ANO TELEIA**, not the visually identical
   U+00B7 MIDDLE DOT — and since `line_shape` NFKC-normalizes first and NFKC maps
   U+0387 → U+00B7, the member can never match. The faithful port reproduces a dead set
   entry. All inline character sets are now codepoint escapes extracted from source.
3. **sklearn's `char_wb` break condition** fires at `n >= w_len`, not `n > w_len`. The
   strict inequality leaves the n-loop running, and `w[0:n]` saturates at the whole
   padded word, so extra iterations re-emit a string already present. Support unchanged;
   932,469 term weights wrong.
4. **NFKC happens at the entry point** (`bibliography_v2.py:813`) and is not cosmetic —
   it is what turns fullwidth forms and math-italic codepoints into the ASCII the
   detectors are written against.
5. **Python slicing clamps an inverted range**, Rust panics. Reachable on whitespace-only
   lines, where `_analysis_bounds` legitimately returns `start > end`.
6. **numpy sums pairwise**, not left to right, once an array reaches 8 elements.
7. **`export_scaler` inferred `with_mean` from `mean_ is not None`.** A
   `StandardScaler(with_mean=False)` still stores `mean_` — it needs it for the
   variance — and merely declines to subtract it. The heading bundle is fitted that
   way, so the artifact claimed the opposite and the port would have subtracted a mean
   the reference never subtracts.
8. **`[^\W\d_]` is the Python idiom for "letters"** and reduces to `L|Nl|No`, not to
   `L` — `\d` is Nd specifically, so Nl and No survive.
9. **The same glyph, opposite codepoints, in two files.** `line_shape`'s sentence
   terminals end in U+0387 GREEK ANO TELEIA (unreachable post-NFKC); the
   running-prose test in `deterministic_structure` ends in U+00B7 MIDDLE DOT
   (reachable, and the NFKC image of U+0387). Neither can be transcribed by eye.
10. **The TCN's chunking is part of the model.** 256 central lines with 32 of
    context, clipped to the physical segment; the convolutions are zero-padded at
    window edges, so a whole-segment pass gives different numbers.
11. **GELU is the erf form, not tanh** — 0.84134 vs 0.84119 at x=1, amplified by
    four stacked blocks.

## Gates

```bash
cargo test --release                    # unit tests + fixture parity + tfidf parity
PARITY_STRICT=1 cargo test --release    # turn the parity scoreboards into assertions
```

On Clariden, `scripts/parity_table.sbatch` rebuilds and diffs all 116 deterministic
columns against the deployed `features.npy`; `scripts/scale.sbatch` measures throughput
against thread count.
