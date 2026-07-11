# Release integrity contract

The technical source-license audit is a first-class, checksum-bound release
input. Validation independently checks every materialized source against its
private-training or public-redistribution decision, and publication includes
the same matrix as provenance. See
[`source_license_adjudication.md`](source_license_adjudication.md).

The release boundary is fail-closed. `materialize_release.py` requires the
completed cleaning, GreekMMLU-decontamination and dedup manifests as explicit
arguments. The release manifest records the absolute path and SHA-256 of each
one. Their declared input/output paths must form one exact chain, the dedup
manifest must be `status=completed`, and the selected decision artifact must be
the manifest's `full_cpt_dedup_decisions_content_bound_v1` receipt.

Every dedup row carries `input_text_sha256`. Materialization and independent
validation join it to `(source_dataset, stable_uid)` and require
`input_text_sha256 == cleaned_text_sha256 == sha256(text)`. A filename, row
count, or stable ID alone is not accepted as a content binding. Decisions are
loaded and indexed once; the full corpus/decision join is materialized once and
then written shard by shard.

Each completed training/redistribution shard pair receives an atomic
`full_cpt_materialization_checkpoint_v1` receipt bound to the input-shard
checksum, all three upstream manifest checksums, the decision checksum and the
exact private/public schemas. `--resume` rereads every checkpointed input and
output byte before reuse. An uncheckpointed generated pair is rebuilt; an
unknown file or a drifted checkpoint fails closed.

## Private and public outputs

`training/data` is the private, complete canonical output. The public
`redistribution/data` tree is generated from an explicit allowlist. It never
inherits newly added canonical columns automatically.

The public release omits `title`, `author`, `source_metadata_json`, raw text
hashes and all raw upstream document/work/storage identifiers. Where an audit
link is useful, `source_doc_id`, `source_row_id`, `source_artifact_path` and
`work_id` are replaced with domain-separated SHA-256 values. The policy and
the exact allowed, dropped and hashed columns are frozen in the release
manifest as `full_cpt_public_metadata_v1`.

Validation proves both directions of the public relation: every kept training
row with `eligible_for_redistribution=true` occurs exactly once publicly, and
every public row comes from such a training row. Text, cleaned hash, safe
provenance and dedup metadata must be equal; the four public hashes are
recomputed from the private values. File inventories, row counts and checksums
must also match exactly.

## Hugging Face publication

Publication accepts only the real, non-symlinked
`<release>/redistribution/data` path for payload and only the Parquet inventory
frozen in a passed validation receipt. Materialization also generates
`<release>/publication/README.md`; its bytes, size and SHA-256 are frozen in the
release manifest and independently checked by validation. The card explicitly
names the artifact as the **GlossAPI Greek CPT Redistributable Delta v2**, states
that it is not the full private corpus, and gives per-source pinned provenance,
license/attribution conditions and exact public row counts.

The publisher rereads and hashes every local byte immediately before upload,
including a fresh comparison of the current token-waterfall bytes to the hash
bound in the release manifest. The only production mode is manually gated and
requires a **new and empty** dataset repository. Any payload left by a failed
attempt is rejected. The publisher never deletes it or tries to infer that it is
a safe partial upload; the operator must inspect and delete/recreate the
repository or choose a new empty repository. Xet is disabled
so the remote API exposes Git LFS SHA-256 metadata for every large Parquet
object. The configured default repository is
`fffoivos/glossapi-greek-cpt-redistributable-delta-v2`.

The exact permitted remote inventory is:

- `README.md`, matching the manifest-bound generated card;
- `data/**/*.parquet`, matching the validated redistribution inventory;
- the explicit `provenance/**` release, validation, token-waterfall,
  source-license and upstream-stage receipts.

The uploader uses temporary same-filesystem hardlinks for the Parquet data, so
its cache cannot add files beneath the immutable release. That temporary cache
is intentionally discarded after the attempt, which is why a partial remote
upload requires explicit repository cleanup instead of an automatic resume.
After upload, the publisher compares the complete final remote path inventory, byte
sizes and SHA-256 values at the returned commit. Any extra, missing or drifted
README, Parquet or provenance file fails publication. The immutable local
receipt records the verified commit/inventory, and the Clariden stage checks its
schema, status and commit before creating the completion marker.

Validation receipts are first written under an attempt-specific immutable
path. Only a zero-failure `passed` receipt is atomically promoted to the
canonical stage path; failed receipts remain diagnostics and are never
overwritten or treated as completion.

Machine-readable schemas live in:

- `schemas/full_cpt_release_manifest.schema.json`
- `schemas/full_cpt_release_validation.schema.json`
- `schemas/full_cpt_publication_receipt.schema.json`
