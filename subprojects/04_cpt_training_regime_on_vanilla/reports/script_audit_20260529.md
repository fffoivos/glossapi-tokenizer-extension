# Script audit — 04 Vanilla CPT pipeline (sidecar safety)

Date: 2026-05-29
Trigger: silent comma-truncation bug in `submit_checkpoint_sidecars.sh` L156
(`--export=ALL,...,BENCHMARKS="$NATIVE_BENCHMARKS",...`). Fix landed
(`7eb4667e…` → `e865c65a…`) by hoisting `BENCHMARKS=` onto the sbatch shell
line so `--export=ALL` carries it.

Scope: every `.sh` and `.sbatch` under
- `subprojects/04_cpt_training_regime_on_vanilla/scripts/`
- `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/`
- `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/bakeoff_training/`

Constraints: read-only; no jobs submitted, no files edited.

## Verdict

The 04 sidecar chain that fires next on iter 834 and iter 1192 is **safe
from a repeat of the comma-truncation bug**. The original `BENCHMARKS=` case
is the only `--export` value in 04 that carried a comma-bearing list, and it
has been moved to the sbatch environment.

There remain two **critical** risks unrelated to commas that could block or
silently corrupt the iter-834/iter-1192 sidecars:

1. **Heavy reliance on `--partition=xfer`** despite the active 2026-06-11
   Apertus maintenance reservation on xfer. The watcher itself, the
   checksum sidecar, the code/math heldout build, and the home-side `xfer`
   submitter for the 5B td-vs-vanilla run all queue against xfer and will
   sit pending until the reservation lifts. The user memory notes Clariden
   xfer is drained and CPU jobs should be routed to `normal`
   (`--cpus-per-task=64 --mem=400G`).
2. **Unquoted heredoc bodies in two GPU eval sbatches** (`run_eval.sbatch`
   and `run_greek_nlp_benchmark_hf.sbatch`) which interpolate
   `$MODEL_PATH`/`$TASK_LIST`/`$MODEL_SPEC` from the outer shell. Today's
   values are well-behaved (no spaces, no `$`, no backticks); a future
   stray character in a checkpoint label or HF dir name would crash the
   eval or silently substitute the wrong model. These do not threaten the
   iter-834/iter-1192 run because the values used by
   `submit_checkpoint_sidecars.sh` are sanitized, but the fragility is the
   same class as the bug just fixed and should be tracked.

Everything else is hygiene / minor.

The remainder of this report is per-finding, file:line first.

---

## Critical findings

### C1. xfer-bound sidecar chain during xfer maintenance reservation (2026-06-11)
Category: scheduling / blocking.

- `subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_and_submit_checkpoint_sidecars.sbatch:6`
  `#SBATCH --partition=xfer` — the watcher itself.
- `subprojects/04_cpt_training_regime_on_vanilla/scripts/watch_and_submit_checkpoint_sidecars.sbatch:139`
  self-resubmit also requests `--partition=xfer`.
- `subprojects/04_cpt_training_regime_on_vanilla/scripts/submit_checkpoint_sidecars.sh:259`
  checksum manifest sidecar uses `--partition=xfer`.
- `subprojects/04_cpt_training_regime_on_vanilla/scripts/build_code_math_heldouts.sbatch:6`
  `#SBATCH --partition=xfer`.
- `subprojects/04_cpt_training_regime_on_vanilla/scripts/hplt_b1_dataset_build.sbatch:6`
  and `hplt_b1_dataset_smoke.sbatch:6` use xfer.
- `subprojects/03_apertus_extension_and_embedding_adaptation/.../eval/cache_native_greek_benchmarks.sbatch:6`,
  `build_cpt_heldout_jsonl.sbatch:4`, `watch_td_checkpoint_evals.sbatch:10`,
  the eval-submitter in
  `bakeoff_training/submit_5b_td_vs_vanilla_chain.sh:221` all hit xfer.

Per the active memory entry "Clariden xfer maintenance till 2026-06-11"
xfer is drained for the Apertus reservation. Route CPU-only jobs to
`normal --cpus-per-task=64 --mem=400G` until the reservation lifts.

Fix suggestion: parameterize partition on the watcher and its children
(`PARTITION="${PARTITION:-xfer}"`), and submit iter-834/iter-1192 with
`PARTITION=normal CPUS_PER_TASK=64 MEM=400G`. The conversion + native_mcq
+ greek_nlp + bpb + retention jobs already use `--partition=normal`, so
only the watcher + checksum + code/math heldout need re-routing.

### C2. `run_eval.sbatch:127` — unquoted heredoc body interpolates `$MODEL_PATH`/`$TASK_LIST` from outer shell
Category: silent corruption / fragile shell quoting.

```
uenv start "$UENV_IMAGE" --view=default --ignore-tty <<INNER
...
--model_args pretrained=$MODEL_PATH,dtype=$DTYPE,trust_remote_code=True \
--tasks $TASK_LIST \
--batch_size $BATCH_SIZE \
--output_path $OUTPUT_DIR/results.json \
$LIMIT_ARG
INNER
```

The heredoc terminator is `INNER` (unquoted), so the outer shell expands
every `$var` into the heredoc body before passing it to `uenv start` over
stdin. `$MODEL_PATH`, `$OUTPUT_DIR`, `$DTYPE`, `$BATCH_SIZE`, and
`$LIMIT_ARG` are all left unquoted on the inner Python command line. Any
space, backtick, `$`, `*`, or apostrophe in any of those variables would
break or corrupt the eval invocation. `$MODEL_PATH` also rides next to the
literal `,` separator in `--model_args` — a comma inside `$MODEL_PATH`
would silently rebind `dtype=` to a piece of the path. Same class as the
bug just fixed.

The values used by today's 04 sidecars (HF checkpoint paths) cannot
contain those characters, but the contract is unenforced.

Fix suggestion: switch to `<<'INNER'` (quoted) and pass values through
`export` from the outer body (the script already exports
`MODEL_PATH`/`OUTPUT_DIR`/etc. for run_metadata.json on lines 110–125); or
quote the inner variables (`--model_args "pretrained=$MODEL_PATH,..."`).

### C3. `run_greek_nlp_benchmark_hf.sbatch:60` — unquoted heredoc body interpolates `$MODEL_SPEC`/`$TASK_NUM_PREDICT_OVERRIDES`
Category: silent corruption / fragile shell quoting.

```
uenv start "$UENV_IMAGE" --view=default --ignore-tty <<INNER
...
python "$SCRIPT_DIR/run_greek_nlp_benchmark_hf.py" \
  --benchmark-root "$BENCHMARK_ROOT" \
  ...
  --models "$MODEL_SPEC" \
  ...
  --task-num-predict-overrides "$TASK_NUM_PREDICT_OVERRIDES" \
  ...
INNER
```

Same shape as C2: terminator `INNER` is unquoted. Outer shell expands
`$MODEL_SPEC` (label=path) and `$TASK_NUM_PREDICT_OVERRIDES`
(comma-separated kv list) into the heredoc text. Today's 04 sidecar value
for `MODEL_SPEC` is `"Vanilla-3.5B=/capstor/.../iter_0000834_hf"` and the
override list is the default — both safe. But a checkpoint label with a
`$`, `"`, or backtick would inject into the inner shell.

Note the matching `run_greek_nlp_benchmark_hf_packed.sbatch:57` and the
two MCQ variants correctly use `<<'INNER'`, so this is asymmetric — likely
an oversight when the single-spec variant was branched.

Fix suggestion: change `<<INNER` to `<<'INNER'`. All `$vars` used inside
are already `export`-ed on lines 53–55.

---

## Major findings

### M1. `submit_5b_td_vs_vanilla_chain.sh:228` — values with spaces inside `--export`
Category: fragile Slurm export.

```
--export=ALL,...,ITER_LIST="1013 1192",TASK_GROUP=full,EVAL_ARMS="vanilla td_layer11",DIAG_ARMS="td_layer11",...
```

The bash double quotes are stripped before sbatch sees the argv; sbatch
sees `--export=ALL,...,ITER_LIST=1013 1192,...,EVAL_ARMS=vanilla td_layer11,...`.
Slurm tolerates spaces inside `--export` values, so this works in
practice, but the same `--export` line is right next to the bug class
that just bit us: commas inside values would not be tolerated. The
defensive form is to hoist multi-token values onto the sbatch shell line.

Not in the iter-834/iter-1192 critical path (this script governs the
parallel 5B td-vs-vanilla chain), so flagged as Major rather than
Critical.

Fix suggestion: hoist `ITER_LIST` / `EVAL_ARMS` / `DIAG_ARMS` onto the
sbatch shell line (`ITER_LIST="1013 1192" EVAL_ARMS="..." DIAG_ARMS="..." sbatch ...`)
and pass them through via `--export=ALL`.

### M2. `submit_all_arms.sh:94,109` — unquoted variables inside `--export`
Category: fragile Slurm export.

```
--export=ALL,ARM=$arm,INIT_CKPT=$init_ckpt,OUTPUT_DIR=$output_dir,SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR",...
```

`INIT_CKPT=$init_ckpt` and `OUTPUT_DIR=$output_dir` are unquoted. Today's
paths have no commas, but if any path ever contains a comma the export
list silently truncates exactly as it did for `BENCHMARKS=`. Compare with
the surrounding `SCRIPT_DIR_OVERRIDE="$SCRIPT_DIR"` which is quoted.

Fix suggestion: quote all values uniformly:
`INIT_CKPT="$init_ckpt",OUTPUT_DIR="$output_dir"`.

### M3. `run_tokenizer_fair_metrics.sbatch:66-76` — `bash -c "..."` with interpolated `$MODEL_PATH` between single quotes
Category: shell quoting.

```
uenv run "$UENV_IMAGE" --view=default -- bash -c "
set -euo pipefail
python3 '$METRICS_SCRIPT' \
    --model-path '$MODEL_PATH' \
    --eval-jsonl '$EVAL_JSONL' \
    --output-json '$OUTPUT_JSON' \
    --max-context $MAX_CONTEXT \
    --device cuda \
    --dtype bfloat16 \
    $MAX_DOCS_ARG
"
```

Outer double-quoted string interpolates the variables into single-quoted
inner positions. Apostrophe inside `$MODEL_PATH`/`$EVAL_JSONL`/`$OUTPUT_JSON`
would break the inner `bash -c` parsing and silently mis-bind the next
flag. `$MAX_DOCS_ARG` is intentionally unquoted (word-splitting for
`--max-docs N`).

Today's values are checkpoint/heldout/JSON paths — clean — but the contract
is unenforced. Identical pattern at
`run_new_token_diagnostics.sbatch:78-90`.

Fix suggestion: emit the inner script via a `<<'INNER'` heredoc and pass
data through env vars (consistent with the `MAX_DOCS_ARG` array-building
pattern already used in `run_td_pilot_intrinsics_packed.sbatch`).

### M4. `preprocess_data.sbatch:101-113` — odd `\"$UENV_BIN\"` quoting plus unquoted `$MEGATRON_LM_SWISSAI_DIR`/`$WORKERS`
Category: shell quoting.

Line 101: `\"$UENV_BIN\"` is syntactically a literal `"` followed by the
expansion, which works but is unusual — the intended pattern is
`"$UENV_BIN"`. Inside the `bash -c "..."` body, line 103 has
`cd $MEGATRON_LM_SWISSAI_DIR` unquoted (`$WORKERS` on line 111 is numeric
and safe).

Today's Megatron-LM path has no whitespace. Not in the iter-834/iter-1192
critical path (this sbatch is the dataset preprocessor).

Fix suggestion: replace `\"$UENV_BIN\"` with `"$UENV_BIN"` and quote
`cd "$MEGATRON_LM_SWISSAI_DIR"`.

### M5. `watch_and_submit_checkpoint_sidecars.sbatch:105` — checkpoint readiness uses only `.metadata`
Category: precondition robustness.

```
if [ -d "$ckpt_dir" ] && [ -f "$ckpt_dir/.metadata" ]; then
```

torch_dist with `--async-save` writes `.metadata` near the end of save,
but does not document `.metadata` as the durable "save complete" marker;
in practice the `latest_checkpointed_iteration.txt` tracker advances only
after the save is durable. The downstream watcher at
`watch_and_submit_checkpoint_evals.sh:50-61` cross-checks the tracker;
the 04 watcher does not.

If a checkpoint were partially flushed when the watcher polls, the
conversion sbatch would inherit a half-written dist-checkpoint dir.

Fix suggestion: add `[ -f "$TRAIN_RUN_DIR/checkpoints/latest_checkpointed_iteration.txt" ]`
and `[ "$(tr -dc 0-9 < "$tracker")" -ge "$iter" ]` checks before submission,
mirroring `watch_and_submit_checkpoint_evals_packed.sh:40-61`.

### M6. `monitor_5b_status.sh:5` — `set -uo pipefail` missing `-e`
Category: error swallowing.

`subprojects/03_apertus_extension_and_embedding_adaptation/.../bakeoff_training/monitor_5b_td_vs_vanilla_status.sh:5`
omits `-e`. Failed `ssh` calls inside the polling loop are intentionally
soft (the script wants to keep polling on transient SSH failures), but a
typo / arithmetic-bug inside the loop body would not abort. The 04 sibling
`subprojects/04_cpt_training_regime_on_vanilla/scripts/monitor_5b_status.sh:7`
uses full `set -euo pipefail` correctly.

Fix suggestion: leave the SSH calls under `|| true`, but enable `set -e`
for the rest of the body. Otherwise document the asymmetry.

### M7. Concurrent eval cache dirs key off `${SLURM_JOB_ID:-manual}`
Category: race / cross-run interference.

- `run_eval.sbatch:74` `EVAL_CACHE_ROOT=…/eval_cache_${SLURM_JOB_ID:-manual}`
- `run_eval_packed_arms.sbatch:39` `EVAL_CACHE_ROOT=…/eval_cache_${SLURM_JOB_ID:-manual}_packed`
- `run_greek_nlp_benchmark_hf.sbatch:33` `CACHE_ROOT=…/greek_nlp_benchmark_${SLURM_JOB_ID:-manual}`

In Slurm this is safe (per-job IDs are unique). Manual reruns from a
shell (the `:-manual` fallback) reuse `eval_cache_manual` and would race
if two manual runs co-execute. Not a current threat (manual reruns are
not in the 04 plan). Hygiene only.

Fix suggestion: fail closed if `SLURM_JOB_ID` is unset, or salt the manual
fallback with `${RANDOM}_$(date +%s)`.

---

## Minor findings

### m1. `submit_checkpoint_sidecars.sh:127-128` and friends — redundant `--gpus-per-node` + `--gres=gpu:N`
Category: hygiene.

Every sidecar submission specifies both. Slurm accepts both, and the
values match (`gpus-per-node=1` / `gres=gpu:1`). Redundant but harmless;
if either diverges in a future edit, Slurm rejects.

Fix suggestion: pick one (`--gpus-per-node=N` is the Apertus convention)
and drop `--gres=gpu:N`.

### m2. `submit_5b_td_vs_vanilla_chain.sh:228` — `OUT_ROOT="$EVAL_OUT_ROOT",STATE_DIR=...` overrides callee defaults
Category: hygiene.

Same `OUT_ROOT` name is overloaded between the training submitter and the
eval submitter. Intentional (the eval-side ROOT differs from the training
ROOT) but easy to misread.

Fix suggestion: rename to `EVAL_OUT_ROOT` end-to-end.

### m3. `run_eval.sbatch:144-151` — embedded Python heredoc that interpolates `$(date -u …)` via outer shell
Category: hygiene.

The post-eval `run_metadata.json` updater is a Python `-c` script wrapped
in `"`, with `$(date -u +…)` substituted by the outer shell into a Python
string literal. The substituted timestamp has no special characters; if
that ever changes, the Python literal would explode.

Fix suggestion: pass the timestamp through an env var and read it in
Python.

### m4. `submit_training_5b_chain.sh:204` — large single-line `--export` (16 keys)
Category: hygiene.

Long single-line `--export` is hard to audit by eye for the same comma-bug
class. The values today are all clean (numeric, paths without commas).

Fix suggestion: when next touched, split into shell-line env + `--export=ALL`,
matching the new pattern in `submit_checkpoint_sidecars.sh:145`.

### m5. `watch_and_submit_checkpoint_sidecars.sbatch:147` — self-resubmit `--export=ALL,...` carries forward `CODE_HELDOUT_JSONL` / `MATH_HELDOUT_JSONL`
Category: hygiene.

These are absolute paths today; same comma-truncation class were they
ever to contain `,`. Harmless given current values.

### m6. `pull_benchmarks.sh:108` — non-fatal `pip install … || echo "may have failed"`
Category: hygiene / silent fallback.

Out of the iter-834/iter-1192 critical path.

### m7. `run_apertus_baseline.sh:24` — `--export=ALL,MODEL_PATH="$APERTUS_BASE",OUTPUT_DIR="$OUTPUT_DIR",TASK_GROUP=full`
Category: hygiene.

Clean today. Mentioned for completeness of the `--export` audit.

### m8. `run_native_greek_mcq_eval.sbatch` line 8 — sbatch directive lacks `--gres=gpu:1`
Category: hygiene.

`#SBATCH --gpus-per-node=1` only. Submitters in 04 add `--gres=gpu:1` on
the CLI, so the merged spec is consistent. Direct manual sbatch
submission of this file without CLI overrides would lack `--gres`.

### m9. Mixed convention: `--ntasks-per-node` vs `--ntasks`
Category: hygiene.

`build_code_math_heldouts.sbatch:8` uses `--ntasks=1` while
`hplt_b1_dataset_build.sbatch:6-7` uses `--ntasks-per-node=1`. Both are
correct for a single-node CPU job; just inconsistent.

---

## Summary of `--export` audit (comma-bug class)

All `--export=ALL,…` invocations across the three directories were
examined. The only value that contains a comma (the original bug) was
`BENCHMARKS="$NATIVE_BENCHMARKS"` in `submit_checkpoint_sidecars.sh:159`,
already fixed. No other `--export` value across the three dirs contains
a top-level comma; the audit therefore clears the iter-834/iter-1192
sidecar chain for that specific bug class.

Values that contain **spaces** inside `--export`: M1 only
(`submit_5b_td_vs_vanilla_chain.sh:228`). Slurm tolerates spaces in
values, so these are not actively broken today.
