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
tracked HF registry is therefore a reacquisition plan, not a claim that the raw
inputs survive on Clariden.

## Production DAG

1. Resolve the Hugging Face repositories in `configs/sources.json` to immutable
   revisions and selected file/LFS hashes.
2. Download on Clariden and normalize to the canonical source-preserving schema.
3. Apply the already validated Apertus-overlap overlay to the nanochat base and
   build the source-key and exact-identity graph for new/replacement candidates.
4. Run source-specific quality and structural detectors in **audit-only** mode.
5. Measure exact counterfactual token loss with the pinned ModernGreek-148k
   tokenizer for bibliography-only, ToC-only and combined removal.
6. Review the stratified samples and freeze `configs/cleaning_policy.json`.
7. Apply only approved cleaning overlays.
8. Run GreekMMLU decontamination, including the short-question fallback.
9. Run high-confidence PII masking.
10. Resolve same-source replacements and hybrids at canonical work/document
    granularity, retaining base-only and eligible candidate-only documents.
11. Run post-transform exact deduplication, then within-family and cross-family
    near deduplication, then one final exact check.
12. Compute the exact token waterfall and materialize immutable Parquet shards
    plus a complete provenance manifest.

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

The source-lineage anchor is the first commit that actually added corpus data,
not the latest Nanochat processing commit. `configs/nanochat_initial_roster.json`
pins commit `500b8bf577e1e70f4902b77edce2cda02a2559cb`, its 18 exact
`source_dataset` values, 717,265 rows and the SHA-256 of `row_counts.csv`. It
also records OPUS and HPLT as the only later source-name additions. Current
repository names are reviewed in `configs/source_lineage_aliases.json` as
direct, replacement or hybrid lineages. An alias never establishes snapshot
equivalence: refreshed and successor artifacts still require canonical document
keys and cross-version content hashes before replacement or addition.

`source_dataset` is a real corpus column, not a filename guess: the Nanochat
release builder derives `row_counts.csv` from that field. The normalized schema
therefore freezes its exact value when present (falling back to the pinned HF
repository ID only when an upstream source field is absent) and stores the
reviewed `source_family_id` separately. The latest Nanochat roster has 19 names:
the initial 18 plus OPUS and HPLT, minus the later-removed FinePDFs source.

Repository creation and last-modified timestamps are not source cutoffs. For
example, the current Greek-PhD and Openbook V2 Parquets were uploaded after the
first Nanochat data commit, while their source families were already represented
under `greek_phd` and `openbook_gr`. School-books now also contains multiple
overlapping editions. Candidate discovery therefore uses the initial names plus
per-artifact history, not a comparison against Nanochat's latest commit.

The earlier 21.5-million-token figure described only five small backlog entries.
It was **not** the total published after the Nanochat source roster was composed.
The complete pinned inventory is in `configs/post_december_inventory.json`:

- 25 current organization repositories were created on or after 2026-01-01;
- 22 represent names/source families absent from the first Nanochat roster;
- 16 of those have usable full text on HF, totalling 15,644,950,021 artifact
  bytes, 4,188,366 raw Parquet-footer rows and 4,324,586,953 card-reported tokens;
- three are external-only cards (153,584,939 reported tokens), one is
  metadata-only and two are empty scaffolds;
- three are same-source replacements already represented in Nanochat; and
- four older organization repositories have material post-cutoff payload
  changes, including new School-books editions and reprocessed OpenArchives.

Those token numbers are inventory sizing only. They mix tokenizer scopes and
include stale Diavgeia and Archetai card counts. They are neither a
ModernGreek-148k total nor net-additive mass. `configs/sources.json` now tracks
23 acquisition candidates: the 16 usable new-name families plus seven
replacement/overlap routes. Their selected payload is about 30.78 GB; the 17
remaining organization repositories stay non-acquiring in the backlog.

A metadata-only resolution on 2026-07-11 verified all pinned revisions and
selected 26 registry entries (base, overlap evidence, tokenizer and 23
candidates), 521 files and 168,623,515,496 bytes. Candidate payload accounts for
30,781,620,482 of those bytes. No dataset payload was downloaded by that check.

## Logical union and source-aware deduplication

The published Nanochat release remains an immutable physical base. The output
is a new `full_corpus_v2` release; it is not an in-place Nanochat rewrite and it
is not a raw concatenation of every current repository.

The first-upload `source_dataset` value is the primary lineage signal. Reviewed
name aliases identify same-source refreshes, replacements and hybrids, while
document keys and content hashes decide actual identity. Every normalized row
must preserve the exact upstream source name, source-family ID, repository,
revision, artifact path, upstream row/document ID, original hash, stable corpus
UID and representation generation.

For each source family:

1. Normalize alternate representations without concatenating them; group
   sectioned sources to comparable work/document granularity.
2. Match canonical source keys and original/normalized exact hashes.
3. Retain base-only documents; retain candidate-only documents after policy
   review; choose the better complete extraction for matched-but-different
   documents; quarantine ambiguous matches.
4. Prefer an eligible, decontaminated, PII-safe and higher-quality extraction.
   The base wins otherwise, including unresolved cross-family ties. Record every
   loser as a provenance alias with a reason.
5. After approved source cleaning, GreekMMLU decontamination and PII masking,
   run exact deduplication and then near deduplication within each family and
   across the retained base/candidate union. Finish with another exact check.

The final report must show an exact ModernGreek-148k token waterfall per source:
raw candidate, normalization, source cleaner, GreekMMLU, PII, replacement
candidate added/base retired/net, exact dedup, near dedup and final retained
tokens. Until that Clariden CPU audit runs, the gross 4.32B card number must not
be reported as training contribution.

### Lineage and source-review tooling

`scripts/build_source_lineage.py` turns canonical base/candidate JSONL
envelopes into three text-free, deterministic artifacts: a route-level registry
manifest, a row provenance manifest and exact/work relationship memberships.
Every candidate route has `blind_append_allowed=false`. The row envelope must
contain `text`, `source_artifact_path`, `source_row_id` and `source_doc_id`;
candidate rows also contain the registered `source_id`. Preserve an upstream
`source_dataset` byte-for-byte. When it is absent, and only then, the pinned HF
repository ID becomes the explicit fallback name. Resegmented sources must
provide a work-level `work_id` rather than allowing a section ID to masquerade
as a document identity.

`scripts/build_source_review_packet.py` samples each exact `source_dataset`
value independently. The tracked policy fixes the ordinary sample at 100 unique
documents (60 deterministic-random, 20 high-risk and 20 cluster
representatives), and large/heterogeneous sources at 200 (100/50/50).
`privateData=true` rows are excluded, direct identifiers are redacted, long
documents use front/middle/end excerpts, and a deterministic 10% is duplicated
for independent review. A normalizer may provide `review_cluster_id`,
`minhash_cluster_id` or `template_cluster_id`; otherwise the exact normalized
text hash is the conservative fallback. Replacement samples may also carry a
`base_comparison_text` and `base_comparison_uid` for paired review.

Reviewers return JSON matching
`schemas/source_review_response.schema.json`. The aggregator rejects missing or
identity-drifted responses, requires adjudication for low confidence or
primary/secondary disagreement, and emits `include`,
`include_after_cleaning`, `quarantine`, `exclude` or
`pending_adjudication`. A source admitted only after cleaning must receive a new
post-clean packet; pre-clean reviews cannot be reused as proof of cleaning.

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
