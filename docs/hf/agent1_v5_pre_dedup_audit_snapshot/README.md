---
pretty_name: Greek NanoChat plus 18 sources - Agent 1 v5 verified pre-dedup build
language:
- el
- en
license: other
task_categories:
- text-generation
tags:
- greek
- dataset-audit
- parquet
- private
- intermediate
- pre-dedup
configs:
- config_name: default
  data_files:
  - split: train
    path: data/*.parquet
---

# Greek NanoChat plus 18 sources — Agent 1 v5 verified pre-dedup build

> **OPERATIONAL STATUS: VERIFIED — DEDUPLICATION IN PROGRESS**
>
> Dataset construction, acquisition integrity, metadata lineage, Parquet/schema
> closure, compression, and private HF storage passed. This repository is the
> complete input to the active deduplication run. It is not the final
> deduplicated corpus because deduplication is still running.

Quality-score findings are retained for later adjudication and are
**non-blocking for the current construction and dedup operation**. They do not
instruct an execution agent to stop, rebuild, refilter, cancel, or restart the
active chain.

## Immutable identity

| Property | Value |
|---|---|
| Repository | `fffoivos/greek-nanochat-plus-new-pre-dedup-agent1-v5-clariden-debug-20260715T111552Z-30c72e9` |
| Visibility | Private |
| Release kind | Verified pre-dedup input |
| Immutable base-data revision | `362d99a79dece48e2c54a8924cf4419e02e3bee0` |
| Build implementation revision | `30c72e99f6949b85b4db176e4753f69b932d2f7a` |
| Run identifier | `agent1-v5-clariden-debug-20260715T111552Z-30c72e9` |

Always pin the base-data revision when loading corpus shards. Later commits
change only this card and `manifests/provenance.json`; they do not rewrite data
objects.

## Verified construction and storage

At the base-data revision:

| Property | Verified value |
|---|---:|
| Total rows | 53,046,533 |
| NanoChat-base rows | 49,474,947 |
| Candidate rows | 3,571,586 |
| Parquet data shards | 431 |
| NanoChat-base shards | 273 |
| Candidate shards | 158 |
| Parquet payload bytes | 149,746,029,389 |
| Authored release files | 435 |
| Authored release bytes | 149,746,154,488 |
| Complete Hub tree entries, including `.gitattributes` | 436 |
| Complete Hub-reported bytes | 149,746,156,992 |

The passed base-publication receipt compared all 435 authored local and remote
objects. It verified 432 objects through matching LFS SHA-256 values and three
small files by downloading and hashing them. The Hub-managed `.gitattributes`
is the additional tree entry. HF Dataset Viewer reports the expected single
`train` split, 53,046,533 rows, 44 columns, and 149,746,029,389 Parquet bytes.

Integrity anchors:

| Artifact | SHA-256 |
|---|---|
| Base publication receipt | `ebad0b3374b7863c558b9f1ac6eb128fe8a1abaf4e89b8bd46aea67ad62bd681` |
| `manifests/combined_manifest.json` | `4ff9f598f0e592324ae08c139e7f241344bdd180497b39f011d810a990ffdacf` |
| `manifests/dedup_input_inventory.parquet` | `065ec2d92b761d89ad8ec397bb0cbf70aee4f0571b1e848bae6f207a9c719521` |
| `manifests/license_override_receipt.json` | `efb4de6179f5580611cc26ac31d0b748f98e10fa0b66628b8e92eaf5ea060b77` |
| Acquisition-integrity audit | `7b975687e0a8e3f3869a8c5cef4600cf55865523472ef150303c860e598f4f5c` |
| Candidate quality/lineage audit | `4fdeaa54283fcb90d30302bf3b73a412bb949db035300d9265e743c39f134ad0` |

## Construction accounting

The construction manifests close the candidate row flow:

```text
3,576,290 candidate inputs
-    4,703 recorded transform quarantines
=3,571,587 transform survivors
-        1 recorded GlossAPI quarantine
=3,571,586 released candidate rows
```

These are recorded pipeline outcomes, not silent loss. Candidate text passed
through structural repetition cleanup, generated-image artifact cleanup,
HTML-to-GitHub-Flavored-Markdown normalization, and the pinned GlossAPI
cleaner/evaluator. The NanoChat base was schema-cast under a text-preservation
contract rather than re-cleaned.

Pinned components include:

- NanoChat base `fffoivos/glossapi-greek-nanochat-pretraining-dataset` at
  `e1d54136a880ed1df2ed95a5445dabd230453207`;
- GlossAPI commit `a2aace04fbae61ed58931be1a1237a52d1b8ddb3`;
- DataTrove `0.9.0`, commit `87f7bad5c4a56ec648265fbf0b91d7d226bad428`;
- Rust toolchain `1.85.1`.

## Metadata and document-lineage evidence

The exhaustive candidate audit verified all 158 tasks and every released
candidate row. It checked:

- exact candidate row and shard closure;
- one 44-column ordered schema and ZSTD compression;
- transform→GlossAPI preservation of source row UID, source identity,
  candidate document ID, title, author, and source metadata;
- GlossAPI→release preservation of source identity, collision-resolved document
  ID, cleaned-text SHA-256, title, author, and source metadata.

There are 52 duplicate `(source_dataset, source_doc_id_candidate)` keys. The
envelope deterministically suffixed 105 affected row IDs; every derived ID and
row attachment matched the release.

The acquisition audit independently rehashed 293 / 293 files and
153,031,751,003 / 153,031,751,003 bytes. Every hash, filesystem-identity,
path-containment, manifest, and task-binding mismatch counter is zero.

## Deferred quality observations

All 3,571,586 candidate rows were scanned. The audit recorded:

- 275,816 rows with `filter="greek>60"`;
- 10,728 rows with forbidden C0 controls;
- 389 rows with U+FFFD;
- 29 rows with a complete recognized residual HTML tag.

The audit receipt says `status="blocked"` because it was authored with
conservative release-grade quality gates. Its operational disposition for this
run is `deferred_nonblocking`: construction passed and deduplication should
continue unchanged.

`greek>60` means `greek_badness_score > 60`, not “more than 60% Greek.” The
flags include both true corruption and legitimate mixed, polytonic,
dictionary, and non-Greek material. No blanket deletion is approved or applied
in this pre-dedup build. Quality policy will be handled later as a separate
operation.

## Schema

The release has 44 ordered fields. The document envelope is:

```text
source_dataset
source_doc_id
text
title
author
source_metadata_json
```

`title` and `author` remain null where a source has no appropriate value.
`source_metadata_json` is compact JSON containing non-null upstream fields
after the selected text, title, and author paths are removed.

## Deduplication status

Exact-index preparation passed. Near deduplication is healthy and in progress,
so no final retained-row count should be inferred from this pre-dedup input.

The frozen recipe uses 5-token shingles, 128 MinHash permutations, 32 buckets
of four hashes, seed 1, 64-bit hashes, and verified Jaccard threshold 0.85.
Greek Unicode diacritics are preserved and NanoChat-base representatives are
protected.

The remaining operational work is to finish signatures and downstream dedup,
materialize the deduplicated dataset, publish the final private HF repository,
and verify its complete object/checksum and row/schema inventories.

## License and access

The dataset uses `license: other` because it combines source-specific terms.
The repository remains private under the current run-specific publication
configuration. This operational card makes no broader rights determination.

## Immutable loading

```python
from datasets import load_dataset

repo_id = (
    "fffoivos/greek-nanochat-plus-new-pre-dedup-"
    "agent1-v5-clariden-debug-20260715T111552Z-30c72e9"
)
data_revision = "362d99a79dece48e2c54a8924cf4419e02e3bee0"

dataset = load_dataset(
    repo_id,
    split="train",
    revision=data_revision,
    token=True,
    streaming=True,
)
```

Do not embed a Hugging Face token in code. Before reproduction, download the
combined manifest at the pinned revision, verify its SHA-256 above, and verify
each shard against it.

This revision is the verified input to deduplication. After deduplication
finishes, use the separately published and checksum-verified final private
deduplicated repository for downstream training workflows.
