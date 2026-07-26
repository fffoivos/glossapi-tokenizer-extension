# Rust port of the `heading_lexgate` line model — status

**Bar:** decision-equivalent. The emitted line mask at threshold 0.9 must match the
Python pipeline document-for-document. Probabilities may differ in the last bits.

**Deployment target:** v3 feature contract (126 columns, no citation-grammar block),
academic sources only (202,792 documents).

---

## Status: the port is decision-equivalent on cohort-2

**`LINE MASK @0.9: 210704/210704 agree (100.000000%)` — 19,117 positives on both
sides.** That is the contract.

| stage | gate | result |
|---|---|---|
| 35 count features | vs deployed `features.npy`, 210,704 lines | **bit-exact** |
| 34 `line_shape` values | same | **bit-exact** |
| 7 gap summaries | same | **bit-exact** |
| 5 structure flags | same | **bit-exact** |
| `probability:entry` | same | **bit-exact 210704/210704** |
| negative roles (8) + header kind | vs `dump_roles.py` matrix | **exact 210704/210704** |
| heading candidate mask | vs deployed | **exact** — 28,620 both sides |
| `bib_header` / `bib_subheader` / `non_bib_header` | same | max 1.19e-7, 0 rows > 1e-6 |
| `probability:signal_tcn` | same | max 1.49e-7, 0 rows > 1e-6 |
| connector features (177) | vs `dump_connector.py` matrix | **177/177 within 1e-6**, 185,478 candidates, index aligned |
| TF-IDF char_wb + word | vs *fitted* sklearn, 4,000 lines | **0 support, 0 value mismatches** |
| **line mask @ 0.9** | vs reference line probability | **210704/210704 (100.000000%)** |

### Why the mask match is not luck

The line probabilities are *not* bit-identical: max |dp| 2.50e-2, with 645 rows above
1e-6. That is inherent to the bar. Feature differences of one float32 ULP feed a
250-tree ensemble, and a line whose feature sits exactly on a split threshold takes a
different branch, moving the probability by a whole leaf value. The error is therefore
bimodal — near zero almost everywhere, occasionally a discrete jump.

What matters is whether any such jump could cross 0.9:

```
lines where |p - 0.9| < |dp|  (a flip was possible):   0
closest margin to threshold:                          2.30e-05
largest errors sit at margins:                        0.44 - 0.78
```

The two populations are disjoint. The large errors land on mid-range probabilities
where the ensemble is genuinely uncertain and trees disagree; lines near the threshold
are ones the ensemble is confident about, and there the port agrees to far better than
the margin. Three lines sit within 1e-4 of 0.9 — on this cohort none was at risk, and
that residual is the irreducible cost of decision-equivalence rather than a defect.

### Throughput

20,346 lines/s at 64 threads for the deterministic block; the full chain scores
cohort-2 (210,704 lines) in 142 s on one node. Scaling turns over past 64 threads on a
4-socket Grace node, so the corpus run should use four tasks of 72 cores per node
rather than one task of 288. Against the ~15 days of single-stream Python that
motivated the port, the 285M-line academic slice is now a few node-hours.

Reference artifacts (Clariden):

```
.../experiments/bib_nextgen_devfix_20260722/
    unseen_features_cohort2_v7/features.npy   210704 x 126   <- column gate
    line_hist_v3/models/fold*.pkl                            <- 232-feature line model
```

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
