# Required upstream edits to the deployed trainer + watcher

Three small, backward-compatible, env-gated edits. Each is a no-op unless the new env var is set,
so they do NOT change pilot behavior. Apply to the DEPLOYED copies on Clariden (and the local mirror),
then re-confirm the pilot config still dry-runs unchanged.

Files — apply to the EXACT copies `paths.env` resolves to, not a sibling mirror:
- `$TRAIN_DIR/bakeoff_train.sbatch` — verified this is the **glossapi-tokenizer-extension-nested**
  `init_bakeoff/bakeoff_training/bakeoff_train.sbatch` (md5 `424ba84…`), which differs from the
  top-level legacy mirror (`c45b34d…`). Editing only the mirror does nothing. `echo $TRAIN_SCRIPT`
  after `source paths.env` prints the precise file.
- `$SUB/scripts/watch_and_submit_td_checkpoint_sidecars.sbatch`  (the deployed sidecar watcher)

---

## EDIT 1 — bakeoff_train.sbatch: 9-set extra-valid from `EXTRA_VALID_SETS`

Today (lines ~284–300) it hardcodes exactly 3 NAME→prefix pairs (hplt/openarchives/greek_phd).
Make it build the list from `EXTRA_VALID_SETS` (space-separated names), mapping new-Greek names to
`val_<name>_<tok>` and forgetting names to `val_forget_<name>_<tok>`. Default = the 3 pilot sets, so
the pilot is unchanged.

Replace the `EXTRA_VALID_PREFIXES=(...)` block + the `EXTRA_VALID_ARGS=( --extra-valid-data-path ... )`
block with:

```bash
EXTRA_VALID_SETS="${EXTRA_VALID_SETS:-hplt openarchives greek_phd}"
NEW_GREEK_SETS=" hplt openarchives greek_phd "   # these use val_<name>_<tok>; all others use val_forget_<name>_<tok>
EXTRA_VALID_ARGS=()
if [ "$ENABLE_EXTRA_VALID" = "1" ]; then
    EXTRA_VALID_ARGS=(--extra-valid-data-path)
    for name in $EXTRA_VALID_SETS; do
        case "$NEW_GREEK_SETS" in *" $name "*) base="val_${name}";; *) base="val_forget_${name}";; esac
        prefix="$VAL_DATA_DIR/${base}_${ARM_VAL_TOKENIZATION}_text_document"
        if [ ! -s "${prefix}.bin" ] || [ ! -s "${prefix}.idx" ]; then
            echo "WARN: extra-valid binary missing, skipping $name: ${prefix}.{bin,idx}" >&2; continue
        fi
        EXTRA_VALID_ARGS+=("$name" "$prefix")
    done
    [ "${#EXTRA_VALID_ARGS[@]}" -le 1 ] && EXTRA_VALID_ARGS=()   # none found -> disable cleanly
fi
```
(Keep the surrounding `ENABLE_EXTRA_VALID` validation + the `validation:` echo line. Missing sets are
skipped with a warning rather than aborting, so de/ru/zh/code can land later.)

---

## EDIT 2 — bakeoff_train.sbatch: optional trainer wrapper (the reset guard)

Today `TRAINING_CMD` invokes `python3 "$RUNTIME_PATCH_DIR/pretrain_gpt_te_guard.py" "$MEG/pretrain_gpt.py" ...`.
Prepend an optional `$TRAINER_WRAPPER` so our `reset_data_index_guard.py` wraps the TE guard. Default
unset = unchanged.

At the `TRAINING_CMD=( python3 ... pretrain_gpt_te_guard.py ... )` construction, change to:

```bash
PRE_WRAPPER=()
[ -n "${TRAINER_WRAPPER:-}" ] && PRE_WRAPPER=("$TRAINER_WRAPPER")
TRAINING_CMD=( python3 "${PRE_WRAPPER[@]}" "$RUNTIME_PATCH_DIR/pretrain_gpt_te_guard.py" "$MEGATRON_LM_SWISSAI_DIR/pretrain_gpt.py" "${ARGS[@]}" )
```
`TRAINER_WRAPPER` is set by `curriculum_common.env` to `reset_data_index_guard.py`; the guard is a
transparent passthrough unless `RESET_DATA_INDEX=1` (only the first phase-2 segment sets it). It must be
inside the same `uenv run` shell as the trainer (it imports megatron) — it is, because `TRAINING_CMD`
runs there.

---

## EDIT 3 — watch_and_submit_td_checkpoint_sidecars.sbatch: keep GreekMMLU-only across resubmit (OPTIONAL / hardening)

NOTE (corrected): this is **belt-and-suspenders, not a real bug-fix today.** The watcher's per-checkpoint
submit and its ~23 h self-resubmit both use `--export=ALL`, which re-propagates the already-exported
`NATIVE_BENCHMARKS=greekmmlu` / `SUBMIT_*` (the eval watcher sets them via `--export`), so GreekMMLU-only
already survives the resubmit under standard Slurm semantics. There is no `SLURM_EXPORT_ENV` restriction
set today. Apply this edit only to harden against a future restrictive `SLURM_EXPORT_ENV`: add the four
vars explicitly to BOTH (a) the per-checkpoint submit of `submit_*_checkpoint_sidecars.sh` and (b) the
self-resubmit `--export=ALL,...` line. Skipping EDIT 3 does not break the GreekMMLU-only policy.

---

After applying: dry-run the pilot's `submit_two_arm_full_run.sh` with `DRY_RUN=1` and confirm the
extra-valid list is still the 3 pilot sets and `TRAINER_WRAPPER` is empty (no behavior change), then
run the curriculum smoke (RUNBOOK step 11).
