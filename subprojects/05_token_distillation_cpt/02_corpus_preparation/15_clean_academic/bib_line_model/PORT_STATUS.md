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
| **columns 10..126 total** | same | **116/116 bit-exact** |
| TF-IDF char_wb + word | vs *fitted* sklearn vectorizers, 4,000 real lines | **0 support, 0 value mismatches** (worst rel 1.7e-7 = float32 rounding) |

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

## Remaining — 6 columns, then the line model

**The one large job left is `analyze_bib_line`'s negative-role taxonomy.** Everything
else is small and mechanical.

1. **`analyze_bib_line` hard-negative verdict + reason codes**
   (`deterministic_structure.py:638-873`, ~235 dense lines). Needed because
   `bibliography_deterministic_roles._analyze_document` classifies each hard-negative
   line into one of 8 mutually-exclusive roles by substring-matching the reason codes
   (`_role_index`, `bibliography_deterministic_roles.py:41`), and those 8 flags are 8
   of the TCN's 10 inputs.
   - All 81 patterns and 6 lexicons of that module are already dumped into
     `patterns.json`, so this is a control-flow port, not a transcription job.
   - Still to locate: `AUXILIARY_SCOPE_HEADINGS`, imported by the roles module and
     not a module-level string set in `deterministic_structure`.
   - Note `analyze_bibliography_line_v2` (`bibliography_v2.py:1017`) wraps it and adds
     a few v2-specific hard-negative returns plus two overrides
     (`override_running_prose`, `override_statistical_table`) — the roles module calls
     the **v2** function, so port that wrapper too.
2. **TCN forward pass** — no torch needed: `Linear(10→32)` masked → 4 residual blocks
   (`LayerNorm` → `Conv1d(32,32,k=3,dilation=d,padding=d)` → GELU → `+residual`,
   masked) → `LayerNorm` → `Linear(32→1)`. Dropout is identity at inference. Runs per
   **physical segment** (`_physical_segments`, split where `diff(abs_idx) >
   MAX_PHYSICAL_GAP`), not per document. Its 10th input, `exact_bibliography_header`,
   is `header_kinds > 0` — already available as `structure::is_heading_or_subheading`.
3. **Connector bundle** — 5 columns (`connector`, `continuation_specialist`,
   `continuation`, `filler`, `other`); see `_connector_probabilities`. Has its own
   candidate gate (receipt: `connector_candidate_count: 185478`, so ~88% of lines) and
   the deployed arm is `hist` with no scaler.
4. **Line model** — HistGB ×5 over the 126 columns, identity scaler, five-fold mean,
   threshold 0.9. Then diff the mask against
   `heading_lexgate_scope.probability.npy` — the contract.

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

## Gates

```bash
cargo test --release                    # unit tests + fixture parity + tfidf parity
PARITY_STRICT=1 cargo test --release    # turn the parity scoreboards into assertions
```

On Clariden, `scripts/parity_table.sbatch` rebuilds and diffs all 116 deterministic
columns against the deployed `features.npy`; `scripts/scale.sbatch` measures throughput
against thread count.
