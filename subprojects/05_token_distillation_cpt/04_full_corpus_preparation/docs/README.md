# 04/docs — design contracts and prototype results

> **In one line:** the six documents that fixed the rules of the full-corpus build — what a quality claim may say, what a licence permits, what a release must prove, how cleaning is split, how HTML became Markdown, and how the lane that actually ran was operated.
> **Period:** 2026-07-11 → 2026-07-22. **Status:** complete; five of the six describe designs, one (`agent1_v4_gfm_normalization.md`) reports a finished audit.
> **Came from / led to:** [`../REVIEW_20260711.md`](../REVIEW_20260711.md) → these contracts → the [`../scripts`](../scripts) and [`../schemas`](../schemas) that enforce them.

## Why this existed

The parent phase had four parallel lanes and a Clariden run that could not be re-done cheaply. Each document here pins one thing that had to be decided *before* jobs were submitted, so that a receipt could later be checked against a written rule rather than against memory.

## History

| Date | Document | What it fixed | Evidence |
|---|---|---|---|
| 2026-07-11 | [`source_license_adjudication.md`](source_license_adjudication.md) | Default-deny, per-source technical evidence review. Noncommercial local CPT for 7 source IDs; public redistribution for only `diavgeia`, `eellak_articles`, `open_council`, `opengov_deliberations_v2`; 16 sources excluded with a stated reason. HF gating counts as access control, never as a reuse grant. | `26162a1c`, `43fcdde2` |
| 2026-07-12 | [`two_pass_cleaning.md`](two_pass_cleaning.md) | Stage 50 (source decisions, source-specific cleanup, high-confidence PII masking) and Stage 58 (terminal admission + optional structural spans) are different passes, not a replayed script. Structural removal is a hard no-op in Stage 50. Token ledger keeps ToC-only, bibliography-only and union counterfactuals separately because BPE boundaries make them non-additive. | `01cba0ee`, `14d803fb` |
| 2026-07-12 | [`release_integrity.md`](release_integrity.md) | Fail-closed release boundary: every dedup row carries `input_text_sha256` and materialisation requires `input_text_sha256 == cleaned_text_sha256 == sha256(text)`. The public tree is an explicit allowlist that never inherits new canonical columns; validation proves the public/private relation in both directions. | `8a9efebd`, `01cba0ee` |
| 2026-07-12 | [`dataset_quality_review.md`](dataset_quality_review.md) | Separates "what do the datasets look like" (Stage 35 representative sample) from "what are the population rates" (optional Stage 15 full scan). A sample statistic must be labelled `review_sample` with its denominator and may never be promoted to a corpus-wide claim. Names what could not be evaluated at all (MDC archives, one metadata-only repo, two empty scaffolds). | `ed552eba`, `72026703` |
| 2026-07-14 → 07-15 | [`agent1_v4_gfm_normalization.md`](agent1_v4_gfm_normalization.md) | **Results**, not a plan: 348 documents, 177 changed / 171 byte-identical, 117,875 HTML start tags handled with zero recognised HTML left, 1,722 tables converted, 40 repetition spans (168,230 chars) removed, 4,799 image artifacts removed with 4,621 provenance comments retained, idempotent on all outputs, Luna 100/100 critical regions passed. Its "porting gate" into GlossAPI production was never executed. | `e6c289c4`, `b4ac157c`, `ed6c2e84`, `f26fe1ed` |
| 2026-07-15 | [`agent1_v5_eiger_pipeline.md`](agent1_v5_eiger_pipeline.md) | The executable handoff for the lane that built the shipped corpus: five ordered steps, the frozen pins (GlossAPI `a2aace04`, DataTrove 0.9.0 / `87f7bad5`, NanoChat `e1d54136`), the dedup parameters, and the explicit decision to run production on Clariden `debug` against the repo's usual policy. | `c144116c`, `33a3be81` |
| 2026-07-22 | `dataset_quality_review.md` rewritten again during worktree consolidation | Added the Agent-3 presentation-site handoff contract (role/byte/SHA/schema/run-ID/producer-commit binding, `127.0.0.1`-only server, strict CSP, no CDN or web fonts) and the public-sample packet distinction. | `33653540`, `84b6ab63` |

## Outcome

- The licence matrix, release-integrity contract and two-pass split were all enforced in code and are still the authority for what the private and public trees may contain.
- The quality-review document's central rule — never report a sample as a population — survived into the schemas (`dataset_quality_summary.schema.json` carries the `review_sample` / `full_scan` distinction).
- The GFM prototype is the only document here reporting completed measurements. Its five-step porting gate is open work, not history.
- `dataset_quality_review.md` is the one file rewritten across three separate passes (2026-07-12, 07-12 hardening, 07-22 consolidation); read it as the final state of a lane that grew a presentation site late.
