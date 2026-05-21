# Takeover log - 2026-05-21

This file records Codex takeover actions after the CSCS overnight run. It is the local, commit-tracked copy of operational changes, checks, job starts/stops, and errors encountered during the handoff.

## Initial live state

- Clariden user/account: `fffoivos` / `a0140`.
- Active Slurm job at takeover: `2334880` (`prepare_greek_pool`) RUNNING on `nid006899`.
- Failed prior `prepare_greek_pool` attempts verified by `sacct`: `2334476`, `2334826`.
- R1 roundtrip verified by `sacct`: `2333864` COMPLETED `0:0`.
- V4-HF baseline job `2334245` had eval artifacts, but Slurm marked `FAILED 13:0` because of the documented `ls | head` SIGPIPE after eval completion.

## Local fixes made

- Added missing `global_mmlu` to `eval/run_eval.sbatch` retention tasks so corrected V4 runs cover Apertus Table 14.
- Marked `v4_baseline_20260521` as a partial baseline for its listed tasks, not the final threshold source.
- Corrected `mix_builder.py` determinism wording: all arms share the same JSONL text stream, but base-vs-extended token IDs differ.
- Corrected `corpus_build/recipes/bulk.json` metadata to match the actual `70/24/4/2` Greek/replay/code/math weights.
- Fixed `normalize_nfc.sh` to include the `cpt/` directory, so the runbook-produced selected parquet is normalized when normalization runs after `prepare_greek_pool`.
- Clarified R17 q/k norm evidence and added q/k max-diff printing to future `r1_roundtrip.sbatch` reruns.
- Updated `CSCS_OVERNIGHT_STATE.md` and `SESSION_LOG_20260521.md` with takeover state and known corrections.

## Errors encountered

- GCP active-instance check failed from `home` because `gcloud` requires non-interactive reauthentication:

```text
ERROR: (gcloud.compute.instances.list) There was a problem refreshing your current auth tokens: Reauthentication failed. cannot prompt during non-interactive execution.
```

No GCP instance state was verified during takeover.
- Local `pytest` verification could not run because this checkout has no pytest installed on the default `python3`:

```text
/bin/bash: line 1: pytest: command not found
/usr/bin/python3: No module named pytest
```

Static shell/Python/JSON checks did pass.

## Process decisions

- Did not stop `prepare_greek_pool` job `2334880`: DuckDB temp grew during checks, so it was actively spilling rather than stuck.
- First `rsync` of `glossapi_corpus_cli/` failed with an SSH connection-close error while other rsyncs were running in parallel; retried it by itself and it succeeded. Local/deployed checksums matched after retry.
- A later multi-file `rsync` briefly copied `normalize_nfc.sh` to the remote subproject root instead of `corpus_build/`; removed that accidental remote copy and resynced the script to the correct path.
- Submitted corrected V4 evals after syncing:
  - `2335100` = V4-HF corrected baseline, output `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_baseline_v4_corrected_20260521_121639`
  - `2335101` = V4-post-conversion corrected baseline, output `/capstor/scratch/cscs/fffoivos/runs/eval/apertus_postconv_v4_corrected_20260521_121639`
- Verified both eval jobs entered RUNNING state on normal partition and the logs show `global_mmlu` in `Selected Tasks`.
- Continue monitoring `2334880`; after success, run normalize -> smoke mix -> full mix -> preprocess x2 -> init checkpoints -> bakeoff arms.
