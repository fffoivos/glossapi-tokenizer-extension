# Token Count Audit - 2026-05-26

## Question

Do we already know the token mass for
`fffoivos/apertus-c3-dedup-audit-dedup-20260519t010924z`, or do we need a
Clariden CPU job?

## Current Answer

We now have the exact HPLT source count as well as the earlier CPT-mix counts:

- Final CPT mix: `5,754,172` rows, `9,831,704,774` base-tokenized Megatron
  tokens, and a `7,000,141,612` extended-tokenizer mix-builder budget.
- Full staged HPLT clean60 Wave4 source slice, tokenized with
  `ModernGreek-148k`: `48,728,774` rows, `44,195,950,025` tokens without EOD,
  and `44,244,678,799` tokens with one EOD per row.
- Apertus dedup audit: `98,203,721` HF source-pool docs audited and
  `2,223,781` matched docs.

HPLT token-count artifacts:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/hplt_clean60_full_staged/summary.json
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/hplt_clean60_full_staged/per_file.csv
```

The HF audit repo is an overlap-artifact repo, not the exact sampled CPT text
stream. Its public payload intentionally excludes the large source hash tables
and raw text, so the report gives document counts/fresh chars but not a clean
training-token count for the hard-dropped Apertus-overlap documents.

## CPU-Only Job Added

Script:

```text
count_apertus_overlap_tokens.py
```

Slurm wrapper:

```text
count_apertus_overlap_tokens.sbatch
```

The job streams:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/nanochat/data/*.parquet
```

and joins against:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/apertus_overlap_overlay/artifacts/dedup_20260519T010924Z/cpt_final_overlay/apertus_overlap_drop_docs.parquet
```

Outputs:

```text
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/apertus_overlap_hard_drop/summary.json
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/apertus_overlap_hard_drop/by_source.csv
/iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/apertus_overlap_hard_drop/by_source.json
```

It uses the Clariden `xfer` partition and the local CPU-only Slurm guard. It
does not request GPUs.

Submit from Clariden:

```bash
cd /iopsstor/scratch/cscs/fffoivos/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build
sbatch count_apertus_overlap_tokens.sbatch
```

## Notes

By default the job tokenizes only dropped docs and reports total/fresh rows,
chars, and bytes from the full source scan. Add `--count-all-tokens` to the
Python command only if we also need exact token counts for every fresh source
row; that is slower and currently unnecessary for estimating the omitted
Apertus-overlap mass.
