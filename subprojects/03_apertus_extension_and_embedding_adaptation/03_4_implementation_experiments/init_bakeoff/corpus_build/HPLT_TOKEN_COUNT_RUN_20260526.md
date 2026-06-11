# HPLT Token Count Run - 2026-05-26

Goal: count extended-tokenizer firings/tokens for the staged HPLT clean60 parquet files on Clariden CPU nodes only.

## Live Run

- Slurm job: `2399397`
- Partition/account: `xfer` / `a0140`
- Node: `nid001154`
- Script: `count_hplt_tokens.sbatch`
- Counter: `count_hplt_tokens.py`
- Tokenizer: `/iopsstor/scratch/cscs/fffoivos/tokenizers/apertus_greek_modern_only_148480`
- Inputs: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/nanochat/data/HPLT__ell_Grek_ge8_no_mt_clean60*.parquet`
- Output: `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/hplt_clean60_full_staged`
- Final status: completed at 2026-05-27 00:11 Europe/Athens.
- Slurm verification: job `2399397` completed with exit code `0:0` after `03:29:01` on `nid001154`.
- Final count: 250 parquet files, `48,728,774` rows, `44,195,950,025` tokens without EOD, `44,244,678,799` tokens with one EOD per row.
- Throughput: final stdout rate `3.52M` tokens/sec; worker-wall average `3.53M` tokens/sec.
- Output artifacts: `summary.json`, `per_file.csv`, and 250 file-level partial JSONs under `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/hplt_clean60_full_staged`.

## Fixes Before Successful Launch

- `2399296` failed because `uenv` was not available on the xfer batch node.
- `2399306` failed because `lm_eval` on `PYTHONPATH` exposed an incompatible `regex` install.
- `2399310` and `2399326` exposed that the existing preprocessing env had ARM wheels, while the xfer node is `x86_64`.
- Created xfer-built env: `/iopsstor/scratch/cscs/fffoivos/python_envs/preprocess_xfer_py311`.
- Switched the count job to:
  - xfer-built Python 3.11 env,
  - `WORKERS=1` to avoid Python multiprocessing import/path fragility,
  - tokenizer-internal parallelism across the 64 allocated CPU cores,
  - `BATCH_SIZE=1024`.

## Health Check Commands

```bash
squeue -j 2399397 -o "%.18i %.9P %.40j %.8T %.10M %.6D %R"
tail -120 /capstor/scratch/cscs/fffoivos/runs/preprocess/count_hplt_tokens_2399397.out
tail -80 /capstor/scratch/cscs/fffoivos/runs/preprocess/count_hplt_tokens_2399397.err
find /iopsstor/scratch/cscs/fffoivos/cpt_corpus/token_count_audit_20260526/hplt_clean60_full_staged/partials -type f | wc -l
```
