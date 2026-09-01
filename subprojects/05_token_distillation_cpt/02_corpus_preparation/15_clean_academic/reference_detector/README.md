# reference_detector — the Rust detection crate

> **In one line:** the hot-path detector built on 2026-06-13 and extended through 2026-07-11; it won the 2026-07-25 bake-off on speed and greek_phd recall, and then was not what production shipped.
> **Period:** 2026-06-13 (`056396fd`) → 2026-07-11 (`30f1fd71`). **Status:** superseded as the production model by `heading_lexgate`; still the fastest path and still the only `--mode sections` implementation.
> **Came from / led to:** [`../investigation`](../investigation/README.md) → this → [`../bib_line_model`](../bib_line_model/README.md) (the port that replaced it in production)

## Contract

DETECT → SEGMENT → EMIT auditable spans → COUNT per family. **Never** hard-delete, **never**
bake a deletion threshold, fail closed (keep text on disagreement). Counters are split per
signal family and never aggregated, because the four in-text citation regimes co-exist.

## Modules

| File | What |
|---|---|
| `src/reference_signals.rs` | Regex / label / codepoint inventory. Accent + homoglyph folding, with U+0387 (ano teleia) kept distinct from U+00B7. |
| `src/reference_module.rs` | `detect_doc` (end-matter boundary + footnote stream + in-text counters) and `detect_sections` (the `predicted_section==β` path); split counters. |
| `src/span_line_model.rs` | The promoted conservative bibliography line head — the 22-feature standardized LR + hysteresis decoder ported from `eval/line_lr.py`. `THETA_HI = 0.9` is a compile-time constant. |
| `src/toc_line_model.rs` | Frozen ToC line head plus the front gate. |
| `src/beta_gate_model.rs` | The trained β-gate deployed as a dot product. |
| `src/structural_rules.rs` | Structural rules, with an adversarial test file (`tests/structural_rules_adversarial.rs`). |
| `src/main.rs` | JSONL streaming, rayon batches, modes `wholedoc` / `sections` / `spans` / `score-lines` / `structure-spans` / `bib-spans`. |

## Verified

- 2026-06-13: `cargo test` green on the first landing (the commit message says 9/9 crate tests;
  [`../REFERENCE_CLEANING_INVESTIGATION.md`](../REFERENCE_CLEANING_INVESTIGATION.md) §6 says
  7 unit tests in `reference_module` — the two counts refer to different scopes).
- 2026-06-16 (`3de740fd`): span-model parity against the Python reference — **max per-line
  |Δp| 2.4e-5 over 38 k lines / 60 docs, 60/60 documents producing identical decoded spans**,
  12 cargo tests.
- 2026-07-11 (`30f1fd71`): 18 passing tests; both heads checked across every U+0370–U+03FF edge
  code point at **<1e-12** Python↔Rust probability difference.

## How it ended

The 2026-07-25 bake-off ([`../BIB_DETECTOR_BAKEOFF_20260725.md`](../BIB_DETECTOR_BAKEOFF_20260725.md))
scored it against `heading_lexgate` and a regex heuristic on the same 150-document cohort:
overall **char P 0.9942 / char R 0.8541 / body destroyed 0.00049**, against 0.9976 / 0.8437 /
0.00020 for `heading_lexgate` — better recall, 2.5× more body damage, and **0.76 s wall against
16 minutes (~1,250×)**. Per source it wins greek_phd outright (0.9987 / 0.9106 / 0.00014) and
loses Kallipos clearly (0.5831 vs 0.7590, expected: Kallipos bibliographies are per-chapter and
header-less, and the bake-off ran whole-doc mode while `--mode sections` exists for exactly that
case). The stated decision was *"Adopt the Rust `reference_detector`; do not port
`heading_lexgate`"* — reversed the next day. Production runs `heading_lexgate` via
[`../bib_line_model`](../bib_line_model/README.md).

Its recorded follow-ups were never done: wire `--mode sections` for Kallipos/Pergamos, check
whether openarchives body damage is concentrated in a few documents, and expose `THETA_HI` as a
CLI flag instead of a compile-time constant.
