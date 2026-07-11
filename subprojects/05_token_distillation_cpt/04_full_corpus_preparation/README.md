# Phase 04 — Full CPT corpus preparation

Source-quality review and license/admission evidence are independent gates.
See [`docs/source_license_adjudication.md`](docs/source_license_adjudication.md)
for the checksum-bound local-training versus public-redistribution matrix.

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

The final private/public materialization and publication boundary is specified
in [`docs/release_integrity.md`](docs/release_integrity.md). It binds completed
stage manifests and content-hashed dedup decisions, proves public/private row
parity, and permits publication only into a manually gated empty repository.

## Implementation status

This is an implemented, dry-run-first CPU pipeline, not yet a completed corpus
build. The evidence, recommendations and sign-off boundary are summarized in
[`REVIEW_20260711.md`](REVIEW_20260711.md).

- Ready: immutable source acquisition and schema verification, scalable
  normalization, exact source lineage, redacted source review, source admission,
  Stage50 source/PII cleaning, optional post-clean review, structural-last
  Stage58 finalization, GreekMMLU freeze/decontamination, content-bound exact and
  near deduplication, private/public materialization, validation and a standalone
  manually gated publisher. Per-file/shard receipts and exact inventories make
  the expensive stages resumable and fail closed on drift.
- Staged on Clariden: all 26 pinned source directories, occupying 158 GiB under
  `$DATA_ROOT/hf`, with no `.incomplete` files. Acquisition job `2735391`
  downloaded and payload-verified the complete 168,623,515,496-byte selection;
  it failed only because that checkout's School-books schema expectations were
  stale. It is not a passed acquisition receipt.
- Next: run one fresh `ACQUISITION_EXISTING_ONLY=1` acquisition from the corrected
  exact commit. That run creates a new lock, download manifest, schema audit and
  receipt while refusing to download a missing byte. No downstream stage depends
  on job `2735391`.
- Structural application is deliberately a deterministic no-op in this CPT run.
  The tracked policy is `audit_only` and both materialization flags are false.
  Existing supervision is LLM silver, never human gold: BIB text/coordinates can
  be rehydrated, but the raw joint ToC+BIB `STRUCT_2K` corpus is absent. No new
  2,000-item annotation effort is planned.

## Production DAG

1. Recover the already staged Clariden payload with a fresh existing-only
   acquisition receipt, then normalize it to the canonical source-preserving
   schema.
2. Apply the Apertus-overlap actions and build exact/work-level source lineage.
3. Build a redacted review packet for every exact `source_dataset` (at least 100
   documents per source; the frozen policy uses 200 for named large or
   heterogeneous sources). Copy only the small packet to the authenticated Mac,
   review it with Codex `gpt-5.6-luna` at low effort, and aggregate the returned
   schema-valid responses.
4. Manually inspect and checksum-confirm source admission.
5. Stage50 applies source decisions, narrow source-specific cleaning and
   high-confidence direct-identifier masking. It does not apply ToC or
   bibliography spans.
6. If any source is `include_after_cleaning`, build and review a new post-clean
   packet for those sources and freeze the terminal admission.
7. Stop before Stage58, then always run it through one separately confirmed
   finalization path. It either applies promoted structural spans last or, under
   the current `audit_only` policy, copies Stage50 text unchanged while recording
   an explicit structural no-op decision. A future application run would require
   an approved policy frozen before Stage10 and passed Stage54 evidence; policy
   cannot be changed mid-run.
8. Freeze the exact GreekMMLU query set, decontaminate, run post-transform exact
   and near deduplication, and materialize/validate the private training tree and
   the license-limited public redistribution tree.
9. Optionally publish only the redistributable delta after a separate manual
   gate. The default target is
   `fffoivos/glossapi-greek-cpt-redistributable-delta-v2`; publication is never
   part of the production chain.

Megatron preprocessing belongs to the later production-training phase, not here.

## Structural-cleaning routing

The bibliography/ToC classifier is an academic-document model. Routing is an
allowlist, never a global default. These are research/audit routes; the current
`audit_only` policy means Stage58 removes no structural text from this CPT run.

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
value independently. Every source receives at least 100 unique documents (60
deterministic-random, 20 high-risk and 20 cluster representatives); the frozen
policy raises named large/heterogeneous sources to 200 (100/50/50).
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
- Keep first audits bounded to explicit input shards. Normalization,
  structural prediction, deduplication and materialization use atomic
  file/shard checkpoints; retry only through the documented resume path so
  every reused byte is revalidated.

The thin launchers under `clariden/` default to dry-run. Submission requires
`CONFIRM_LAUNCH=1`. Production corpus stages also require one operator-chosen
`PIPELINE_RUN_ID`; it resolves only to
`$RUN_ROOT/pipeline_runs/$PIPELINE_RUN_ID`. A completed stage has both a
hash-bound `stage_receipt.json` and `COMPLETED` marker. An incomplete directory
is never accepted downstream and can only be re-entered with the explicit
`submit.sh resume <stage>` path.

`chain-to-review` submits normalization, lineage and packet construction with
`afterok` dependencies, then stops. It does not invoke a reviewer.
`chain-after-admission` requires the exact SHA-256 of the inspected admission
file. If any source is `include_after_cleaning`, it runs the reviewed cleaning
pass, builds a fresh post-clean packet and stops again. Otherwise it stops after
Stage50. Neither cleaning path submits Stage58. After the structural audit track
has finished (or the operator intentionally elects not to wait), a separate
`chain-finalize-noop` or `chain-finalize-promoted` command binds the immutable
choice and launches Stage58 plus the local downstream release chain. The no-op
requires `CONFIRM_STRUCTURAL_NOOP=1`; promoted application requires the exact
manually confirmed Stage54 model-receipt SHA-256 and fails rather than silently
falling back. The Hugging Face publisher is always standalone and
requires `CONFIRM_PUBLISH` equal to the target gated repository ID. No token is
written to a run receipt or printed as a command argument.

## Repository/runtime boundary

Committed:

- scripts, tests and schemas;
- source registry and frozen cleaning policy;
- small aggregate reports and provenance locks.

Not committed:

- downloaded datasets, normalized shards and cleaned shards;
- span ledgers and per-document token ledgers;
- caches, virtual environments and full review packs.
