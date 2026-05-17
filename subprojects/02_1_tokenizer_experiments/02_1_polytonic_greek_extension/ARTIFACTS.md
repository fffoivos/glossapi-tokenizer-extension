# Polytonic Greek Extension Local Data

This subproject keeps the current post-dedup kept-text training parquet
beside the scripts:

```text
data/strict_w050_c010/20260517T131514Z/
  polytonic_greek_training_kept_strict_w050_c010_20260517T131514Z.parquet
```

This file is the usable corpus for the next tokenizer-extension step. It
contains `18,726` kept rows, including text, source metadata, and dedup
metadata columns. The compressed parquet is about `250 MiB`.

Checksum:

```text
2b89e098de95501734446b5f767205286eb709ac58c6fb7d2eca6ceb2d873001
```

The `data/` directory is local experiment data, not source code. It is
ignored by the repository-level `.gitignore` under the normal
`subprojects/**/data/` rule.

The following are intentionally not kept in this subproject:

- raw source corpora
- filtered text/input parquets used by the dedup runner
- separate kept/dropped decision parquet exports
- dedup state SQLite files
- transient LSH bucket shards
- stage-2 signature matrices and other large intermediates

The full run remains on the tokenizer-extension worker disk under:

```text
/home/foivos/data/glossapi_work/polytonic_extension/strict_w050_c010/
```

The worker-side source for the local training parquet is:

```text
/home/foivos/data/glossapi_work/polytonic_extension/strict_w050_c010/
  training_data/polytonic_greek_training_kept_strict_w050_c010_20260517T131514Z.parquet
```
