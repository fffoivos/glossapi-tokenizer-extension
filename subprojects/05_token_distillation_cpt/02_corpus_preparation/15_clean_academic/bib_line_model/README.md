# bib_line_model — the Rust port of `heading_lexgate`

> **In one line:** sixteen commits in a single day took the Python bibliography line model to Rust at *decision-equivalence* — the emitted line mask matched 210,704/210,704 — turning a ~9,300 CPU-hour corpus pass into a few node-hours.
> **Period:** 2026-07-25 (`911abb83`, the citation-grammar features) → 2026-07-26 (`62e94175` … `87f866e7`). **Status:** completed and superseded *in place* — the canonical copy moved into glossAPI (`rust/glossapi_rs_bib/`); this tree is the research original.
> **Came from / led to:** [`../eval`](../eval/README.md) (which chose `heading_lexgate`) → this → [`../production`](../production/README.md)

## Why this existed, and the decision it contradicts

[`../BIB_DETECTOR_BAKEOFF_20260725.md`](../BIB_DETECTOR_BAKEOFF_20260725.md) had concluded, one
day earlier, *"Adopt the Rust `reference_detector`; do not port `heading_lexgate`."* The port
happened anyway and is what shipped. The cost argument the bake-off was answering remains the
motivation: the Python pipeline takes 16 minutes wall for 150 docs / 210,704 lines at ~99%
single-threaded — about **9,300 CPU-hours** for the 202,792-document academic slice.

## The bar, and why it is not bit-exactness

Stated in [`PORT_STATUS.md`](PORT_STATUS.md): **decision-equivalent**. The emitted line mask at
threshold 0.9 must match document-for-document; probabilities may differ in the last bits. They
do — max |Δp| 2.50e-2, 645 rows above 1e-6 — because a one-ULP feature difference feeds a
250-tree ensemble and a line sitting on a split threshold takes a different branch, moving the
probability by a whole leaf value. The error is bimodal.

So the match was checked for robustness rather than accepted:

```
lines where |p - 0.9| < |dp|  (a flip was possible):   0
closest margin to threshold:                          2.30e-05
largest errors sit at margins:                        0.44 - 0.78
```

The two populations are disjoint. Three lines sit within 1e-4 of 0.9; none was at risk.

## Verified gates

| Stage | Result |
|---|---|
| 35 count features, 34 `line_shape`, 7 gap summaries, 5 structure flags | **bit-exact** over 210,704 lines |
| `probability:entry` | **bit-exact 210,704/210,704** |
| negative roles (8) + header kind | **exact 210,704/210,704** |
| heading candidate mask | **exact** — 28,620 both sides |
| heading probabilities (3), `probability:signal_tcn` | max 1.19e-7 / 1.49e-7, 0 rows > 1e-6 |
| connector features (177) | **177/177 within 1e-6**, 185,478 candidates, index aligned |
| TF-IDF char_wb + word vs *fitted* sklearn | **0 support, 0 value mismatches** |
| **line mask @ 0.9** | **210,704/210,704 (100.000000%)**, 19,117 positives both sides |

Throughput: 20,346 lines/s at 64 threads for the deterministic block; full chain scores the
cohort in **142 s on one node**. Scaling turns over past 64 threads on a 4-socket Grace node, so
the corpus run wants four tasks of 72 cores per node, not one of 288.

## The traps — why this took a day and not an hour

Each was silent (no error, no crash, just different numbers) and each was caught by widening a
gate rather than by reading harder. The full list is in [`PORT_STATUS.md`](PORT_STATUS.md);
the ones worth knowing:

- **`\w` means different things.** Rust's accepts combining marks; Python's is `str.isalnum()`
  plus underscore and does not. On OCR'd Greek maths a combining tilde flipped `(?<!\w)` and
  dropped 3 of 12 matches on one line — and `\b` is defined on the same class, so all 12
  patterns using it carried the bug.
- **`_SENTENCE_TERMINAL` ends in U+0387 GREEK ANO TELEIA**, and since `line_shape` NFKC-normalizes
  first and NFKC maps U+0387 → U+00B7, that member can never match. The faithful port reproduces
  a **dead set entry**. The same glyph appears with the opposite codepoint in `deterministic_structure`,
  where it *is* reachable. Neither can be transcribed by eye.
- **sklearn's `char_wb` break fires at `n >= w_len`, not `n > w_len`** — support unchanged,
  **932,469 term weights wrong**.
- **`export_scaler` inferred `with_mean` from `mean_ is not None`.** A `StandardScaler(with_mean=False)`
  still stores `mean_`; the heading bundle is fitted that way, so the artifact claimed the
  opposite and the port would have subtracted a mean the reference never subtracts.
- **Unicode tables drift between runtimes.** Clariden runs Unicode 15.0.0, the laptop 16.0.0, and
  `_NAME_INITIAL_PAIR` differs by ~300 characters between them — so `patterns.json` and
  `unicode_tables.json` are generated **on Clariden**, and regexes are compiled from Python's own
  pattern text rather than hand-written.

## Where things are

| Path | What |
|---|---|
| [`PORT_STATUS.md`](PORT_STATUS.md) | The gate table, the robustness argument, the design decisions "worth not relitigating", and all eleven traps. |
| `src/` | `features.rs`, `shape.rs`, `gaps.rs`, `structure.rs`, `roles.rs`, `connector.rs`, `tcn.rs`, `predict.rs`, `chain.rs`, `context.rs`, `table.rs`, `patterns.rs`, `unicode.rs`, `artifacts.rs`. |
| `fixtures/dump_*.py` | Extract the reference's own compiled patterns, roles, connector matrix, TF-IDF cases and Unicode tables — the port is equivalent *by construction*, not by transcription. |
| `tests/fixture_parity.rs`, `tests/tfidf_parity.rs` | `cargo test --release`; `PARITY_STRICT=1` turns the parity scoreboards into assertions. |
| `patterns.json`, `unicode_tables.json` | Generated on Clariden; do not regenerate locally. |

Note: [`../BIB_CLEANING_HANDOVER_20260727.md`](../BIB_CLEANING_HANDOVER_20260727.md) records
that this directory is *"superseded; glossAPI is canonical"* — the shipped copy lives at
`rust/glossapi_rs_bib/` in the glossAPI repository, on branch `codex/bibliography-hardening`,
final commit `284c120ae2ae891fa57bfa5a30d83b815755aaec`.
