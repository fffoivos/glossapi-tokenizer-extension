# Bibliography cleaning hardening — implementation record, 2026-07-27

This record supersedes the operational recommendations in
`BIB_CLEANING_HANDOVER_20260727.md`. The old mixed dry-run directory remains
forensic evidence only and was not edited or reused.

## Outcome and stop boundary

- All 202,792 documents in the 13 academic ranks are analyzed in a fresh
  receipt-bound **dry-run**.
- Future apply scope remains the 175,242 Greek PhD, OpenArchives, elocus and
  libduth documents. Kallipos is not promoted into that scope automatically.
- The libduth exception is recorded only for this run-scoped private variant;
  public redistribution is prohibited.
- Size columns may be recomputed only where their source values are non-null.
- No apply, cleaned-corpus materialization, token count or publication was
  performed or authorized.

The user's original dirty worktrees were not modified. The changes are isolated
on local-only branches and were not pushed.

## Implemented code

### GlossAPI

Final local branch: `codex/bibliography-hardening`

Final commit: `284c120ae2ae891fa57bfa5a30d83b815755aaec`

The final tree includes:

- artifact schema v2 with exact stage names, byte sizes, SHA-256 hashes,
  feature schema, threshold and five folds per stage;
- fail-closed regex evaluation with privacy-safe pattern name, input byte
  length and input SHA-256 diagnostics;
- a 100-million-step `fancy-regex` safety guard plus semantics-preserving
  atomic rewrites for the corpus-proven `_PLACE_PUBLISHER_SHAPE`,
  `_VOLUME_MARKER`, `_PAGE_MARKER` and `_MONTH_DATE` ambiguity classes;
- bounded automatic parallelism for nonpositive Python thread counts;
- exact accounting for removed content, separator characters, spans and lines;
- wheel-content validation for bundled model artifacts;
- Rust and Python CI gates.

An attempted proactive rewrite of the three author-name patterns changed 9 of
210,704 sealed decisions and was completely reverted by `284c120`. The final
tree is identical to the exact-parity tree at `ffa4d72`.

Relevant commit chain:

- `74a959ef` — harden bibliography runtime and artifacts
- `c86b1cb` — raise the fail-closed backtracking guard
- `75cc65ac` — atomic publisher word bodies
- `e7f2f908` — tolerate only floating-point noise in the CI gate
- `88f06f59` — atomic volume-marker whitespace
- `ffa4d72a` — atomic optional-punctuation whitespace
- `1cdb89b6` / `284c120a` — rejected author rewrite and its full revert

### Corpus workflow

Final local branch: `codex/bib-cleaning-production-hardening`

Code commit: `0c6117f94fa49b080b557fddbaf9e3714daaecc5`

The final tree includes:

- immutable v2 run contracts and stable row-group unit IDs;
- exact bindings for code archives, commits, wheel, model artifacts, input
  shards, preflight, parity and work plan;
- atomic per-unit Parquet document ledgers and completion receipts;
- explicit and separate `dry-run` and `apply` modes;
- exact aggregation with duplicate, missing and post-aggregation ledger
  mutation rejection;
- a deterministic QA packet rehydrated from source rows and verified span
  hashes;
- a QA review bound to that packet and a frozen fail-closed gate;
- the owner-approved character-damage criterion;
- the private, non-redistributable libduth exception.

Relevant commits:

- `74bd44dd` — add the receipt-bound production workflow
- `4adfdfd2` — lock unit execution and harden worker logging
- `0c6117f9` — bind QA construction to the aggregated ledger set

## Local validation

GlossAPI final tree:

- `cargo fmt --check`
- 65 Rust unit tests passed
- two Rust parity-fixture tests passed
- the strict parity-fixture mode passed
- 18 Python integration tests passed
- `maturin build --release` passed

Corpus workflow:

- five synthetic production tests passed
- Ruff passed
- Python byte-compilation passed
- all production Bash scripts parsed successfully

## Immutable Clariden inputs

Preflight job `2912077` passed:

- 431 files
- 51,839,746 rows
- 141,797,094,485 bytes
- zero local drift, missing files or Hub mismatches
- Hub commit `c368d37c474bbef3d603d111f13997551c8cd2e0`

Receipt:
`/capstor/scratch/cscs/fffoivos/bib_cleaning_bootstrap_74bd44dd_74a959e/evidence/preflight.json`

Final bootstrap:
`/capstor/scratch/cscs/fffoivos/bib_cleaning_bootstrap_0c6117f9_284c120`

Exact source archives:

- train archive SHA-256:
  `5a455a06cc486ea86fbb365d51099e06a5a3798ea1979cb31bac193ef7360e61`
- GlossAPI archive SHA-256:
  `0795db01600bde33f7eeb29ba17f45e6e333cdeba546445fec9ffe304f4dee9a`
- Linux wheel SHA-256:
  `4f6064dbbf84d323248f44a0ed6e6b1fc24b721951b9c59ea8a9b0a52735e77a`

Sealed parity job `2912714` passed:

- 210,704/210,704 line masks equal
- 19,117/19,117 positives
- candidate probability SHA-256:
  `048edfad9a6f7d7f87e25df2f370e7d8513f0e22c32b8c2347fa20aee428982b`
- cohort SHA-256:
  `33e316671a2fd648f3ea4360e6ec0c96739c44da900907dceb57b252e314dfe4`
- reference probability SHA-256:
  `581c781d89a786ed85fa17438ef5aac8c2871330b13b2a13df134de1734cadb5`

Parity receipt:
`/capstor/scratch/cscs/fffoivos/bib_cleaning_bootstrap_0c6117f9_284c120/evidence/parity.json`

## Corpus-proven regex failures and fixes

The production dry-run deliberately failed closed on previously unseen long
lines. Each diagnostic contract was abandoned; no partial ledger from it was
reused.

1. The default one-million-step guard failed on a long Pergamos line.
2. A 100-million-step rerun isolated `_PLACE_PUBLISHER_SHAPE`; atomic bounded
   word bodies fixed it without making the suffix-yielding repetition atomic.
3. The next complete attempt isolated `_VOLUME_MARKER` on a 2,228-byte
   near-match; audited whitespace runs were made atomic.
4. The same safe optional-punctuation rewrite was applied proactively to
   `_PAGE_MARKER` and `_MONTH_DATE`.
5. A broader author rewrite failed sealed parity by 9 masks and was reverted.

This sequence is why exact parity is a contract gate rather than an informal
test.

## Complete dry-run

Run root:
`/capstor/scratch/cscs/fffoivos/bib_cleaning_runs/20260727T193808Z-ddf94a84b8b7`

- contract SHA-256:
  `e3dd2695cb6a42e7affda1a0e0a488e040a6ad154ae3718de4b12d7f277283b8`
- work-plan SHA-256:
  `1ac9d325a4adbdba8c9aca8e1273dc6d87aa3f2767f28ad40109af8c8b59cc0f`
- 157 stable units
- 202,792 expected rows
- Slurm array: `2912781`

All three array tasks completed with exit code zero. Aggregation verified:

- 157/157 atomic unit receipts
- 157/157 document ledgers
- 202,792/202,792 documents
- zero partial files
- zero `would_empty` documents
- zero worker stderr output

Final summary SHA-256:
`07fa570a27f16653b31bdd40d03cea3b0603f415002fd064fb8adce0680294be`

Final CSV SHA-256:
`af88a297f631322560b722f433fff0ba277fbaa03a5e0aafb17ee433d052ab97`

Overall dry-run result:

- 159,142/202,792 documents had at least one bibliography cut (78.475482%)
- 48,373,473,465 input characters
- 2,933,770,472 characters removed (6.064833%)
- 28,190,335/401,650,438 lines removed
- 1,980,170 removed spans
- document removal-fraction p50 0.052067 and p95 0.210334
- 1,540 documents over 30% removal
- 192 documents over 50% removal

| Source | Documents | Cleaned | Characters removed | Removed % | Over 50% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kallipos | 4,784 | 3,048 | 42,044,384 | 5.5971 | 0 |
| Pergamos | 11,071 | 7,527 | 153,404,499 | 6.8172 | 3 |
| ELLAK articles | 5,690 | 15 | 13,986 | 0.0564 | 0 |
| elocus | 7,699 | 7,286 | 145,071,103 | 7.4619 | 0 |
| libduth | 9,254 | 8,210 | 115,795,828 | 6.1565 | 2 |
| libiep | 6,005 | 701 | 818,062 | 0.0488 | 0 |
| Greek PhD | 31,692 | 31,048 | 1,243,169,342 | 7.9609 | 8 |
| OpenArchives | 126,597 | 101,307 | 1,233,453,268 | 5.0903 | 179 |

## QA

The QA packet contains:

- 30 deterministic median-sized Kallipos cuts;
- every OpenArchives document with removal over 50%;
- every document that would become empty.

Every packet item is reviewed from the removed text and its surrounding source
context. The gate requires:

- a complete decision and non-empty rationale for every item;
- zero catastrophic, body-only or uncertain items;
- at least 27/30 Kallipos cuts primarily bibliography;
- every OpenArchives removal over 50% acceptable and primarily bibliography.

Packet SHA-256:
`c8ef4f0404a6b77a525bb2ff9980d94dcba745be3e3cc0e57bc7938d79492738`

The completed item-by-item review passed the frozen gate:

- 209/209 decisions complete and acceptable
- zero catastrophic, body-only or uncertain decisions
- 30/30 Kallipos items primarily bibliography
- 179/179 OpenArchives removals over 50% acceptable and primarily bibliography
- zero empty-document items

Review SHA-256:
`c28753b19a64816cffc4147194c56079d5aae08c4919df57ac03c0cdb41770bc`

Gate receipt SHA-256:
`d6d5702caaae7df7b5bc179cc3fcd979bca97a0f2f6cb35d260457b97afd293d`

QA artifacts:
`/capstor/scratch/cscs/fffoivos/bib_cleaning_runs/20260727T193808Z-ddf94a84b8b7/dry-run`

## Remaining authority boundary

The 2026-07-27 implementation ended at dry-run and QA.

On 2026-07-28 the owner authorized a new apply contract for the existing
175,242-document scope, explicitly including libduth, and directed that the v2
Hugging Face target remain public. Live verification found the canonical target,
`fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`, already public and
manually gated. The owner decision is recorded without changing or concealing
the existing libduth source-rights warning: it is not represented as
rightsholder permission.

The apply contract remains non-publishing. Publication is a later action after
apply fragments, release reconstruction, whole-release validation and token
counts pass.
