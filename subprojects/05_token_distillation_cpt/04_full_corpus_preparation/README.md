# Phase 04 — Full CPT corpus preparation

This phase turns the reusable cleaning components in `02_corpus_preparation`
and the decisions from `03_training_experiments` into one reproducible
production-corpus build.

It deliberately separates two activities:

1. **Script preparation** — code, tests, source registry, cleaning policy and a
   rendered execution plan. This is safe to do on the development Mac and must
   not download datasets or submit Slurm jobs.
2. **Clariden execution** — pinned source acquisition, audit-only passes,
   policy sign-off, materialization and final token accounting. Runtime data and
   reports live outside Git.

No destructive cleaner is enabled merely because it exists. Every cleaner first
emits a reversible ledger and an exact ModernGreek-148k counterfactual token-loss
report. A later materializer must consume the approved, run-bound ledgers rather
than re-running detection implicitly.

## Implementation status

This commit is **script preparation**, not a completed corpus build.
The evidence, recommendations and sign-off boundary are summarized in
[`REVIEW_20260711.md`](REVIEW_20260711.md).

- Ready: immutable HF source resolution/download, staged Parquet schema checks,
  source-quality/PII/template audits, reversible Diavgeia boilerplate candidates,
  the promoted bibliography + ToC detector, and exact structural token-loss
  accounting.
- Still to implement before production: canonical normalizers for nested and
  non-Parquet sources, exact rule/document token accounting for Diavgeia
  signing/ADA spans, exclusions and PII replacements, source-specific overlay
  application, full-run sharding/resume, final cross-source
  dedup/decontamination/anonymization integration, and release materialization.
- Execution-blocked pending artifact recovery: the private 2,000-document
  structural gold is absent locally and on Clariden. The detector launcher
  requires its pinned hash and a full 608-held-out-document parity receipt.
- Intentionally blocked: every `enabled_for_materialization` flag is false and
  the policy status is `audit_only`.

The live Clariden audit on 2026-07-11 found that the old
`$SCRATCH/cpt_corpus/nanochat/data` directory is empty. Existing Megatron binaries
and small detector counters are not substitutes for the raw source corpus. The
tracked HF registry currently resolves to about 168 GB after narrowing the prior
Apertus-overlap repository to its four final overlay/validation artifacts.

## Production DAG

1. Resolve the Hugging Face repositories in `configs/sources.json` to immutable
   revisions and selected file/LFS hashes.
2. Download on Clariden and normalize to the canonical source-preserving schema.
3. Apply the already validated Apertus-overlap overlay to the nanochat base and
   run incremental overlap checks only for genuinely new sources.
4. Run source-specific quality and structural detectors in **audit-only** mode.
5. Measure exact counterfactual token loss with the pinned ModernGreek-148k
   tokenizer for bibliography-only, ToC-only and combined removal.
6. Review the stratified samples and freeze `configs/cleaning_policy.json`.
7. Apply only approved cleaning overlays.
8. Run cross-source and post-clean exact/near deduplication.
9. Run GreekMMLU decontamination, including the short-question fallback.
10. Run high-confidence PII masking.
11. Run a final exact-dedup check and exact token accounting.
12. Materialize immutable Parquet shards plus a complete provenance manifest.

Megatron preprocessing belongs to the later production-training phase, not here.

## Structural-cleaning routing

The bibliography/ToC classifier is an academic-document model. Routing is an
allowlist, never a global default.

- `apply_after_review`: Greek PhD, OpenArchives, Kallipos and Pergamos.
- `shadow`: Psepheda, E-Locus, LibDUTH, LibIEP and technical/book-like sources.
- `disabled`: HPLT, Diavgeia, legal corpora, news, blogs, parliament, dialogue,
  comments and subtitles.

For enabled academic sources, ToC removal is the simpler policy decision.
Bibliography removal remains conservative (`prose_protection=0.999`) and its
Greek, Latin and polytonic removed-character mass is reported alongside the
exact tokenizer delta before approval.

## Diavgeia

Diavgeia does **not** use the academic structural remover. Its profile is:

- remove only the deterministic Ministry of Digital Governance signing footer;
- report and initially exclude `privateData=true` records;
- mask validated direct identifiers and quarantine PII-heavy personnel tables;
- identify stamp-only, OCR-bad and table-loop records;
- fingerprint templates after variable identifiers are normalized, then cap or
  deduplicate dominant `(decisionTypeId, organizationId, template)` groups;
- preserve substantive statutes, legal citations, recitals, agendas and ordinary
  administrative prose.

`profile_source_quality.py` writes two audit ledgers. The span ledger contains
only complete Ministry signing blocks and isolated `ΑΔΑ:` watermark lines; the
document-action ledger records `privateData`, structured-PII, personnel-table and
correction-version candidates without changing text. Both require review and an
exact token delta before any rule is enabled.

The current pinned repository has more rows than its dataset card, so counts in
the card are not accepted as build inputs. The resolved lock and the full audit
are authoritative.

## Source backlog

`configs/source_backlog.json` records pinned metadata for organization datasets
that were reviewed but are not acquisition inputs. Every entry is deliberately
`acquisition_eligible=false`; the validator also rejects any backlog repository
that appears in `configs/sources.json`. Moving an entry into acquisition requires
an explicit source-contract review, removal from the backlog, and a separate
registry change.

The five genuinely additive HF-resident cultural candidates contribute only
about 21.5 million card-reported BERT tokens. The larger School-books and
Openbook V2 artifacts are overlap/replacement audits, not additive volume. They
therefore do not materially change the roughly 60-billion-token planning case;
their value, if licensing and extraction checks pass, is register diversity.

## Exact token-loss contract

Token loss is calculated on complete text variants, not by tokenizing removed
spans in isolation. For every document the audit records:

- original, bibliography-cleaned, ToC-cleaned and combined-cleaned token counts;
- exact per-policy deltas and the non-additive interaction term;
- original/output hashes, characters, bytes and affected lines;
- span counts, document-loss fraction and whether a document became empty;
- tokenizer hash, detector policy, source revision and input path.

Aggregates include source totals, affected documents, p50/p90/p99/max loss,
Greek/Latin/polytonic removed-character mass and top-loss review examples. EOD
is reported separately; when documents are retained it contributes zero token
loss.

The pinned tokenizer is `fffoivos/apertus-tokenizer-extension` at revision
`a4826df7f76b54cdd6dc21d09fe97283c466999b`, with `tokenizer.json` SHA-256
`358ae3f29ac17c99769d6d437339e28657d5fcaed3486f8550feed3d6adfc394`.

## Clariden execution policy

Clariden has no CPU-only production compute partition. `normal` and `debug`
allocate exclusive 288-core GH200 nodes; `xfer` is transfer-only. The `low`
partition is visible but is not currently available to project `a0140`'s normal
QoS.

- Use `xfer` only for `cp`/`mv`/`rsync` between CSCS filesystems. External
  Hugging Face acquisition runs as a bounded `normal` job.
- Use `debug` only for genuine short smokes.
- Use `normal` for production cleaning, request no GPU/GRES, and keep roughly
  256 CPU cores busy so the exclusive node is not mostly idle.
- Keep first audits bounded to explicit input shards. The present launchers
  preserve failed outputs under `.partial-*` but do not resume within a shard;
  add row-group checkpoint/resume before a measured run approaches 12 hours.

The thin launchers under `clariden/` default to dry-run. Submission requires
`CONFIRM_LAUNCH=1`.

## Repository/runtime boundary

Committed:

- scripts, tests and schemas;
- source registry and frozen cleaning policy;
- small aggregate reports and provenance locks.

Not committed:

- downloaded datasets, normalized shards and cleaned shards;
- span ledgers and per-document token ledgers;
- caches, virtual environments and full review packs.
