# Per-set held-out validation loss — patch + wiring

**Goal:** log validation loss on each of the 3 held-out sets (hplt, openarchives,
greek_phd) **separately** at the eval cadence. The stock swiss-ai Megatron fork
*blends* `--valid-data-path` into one loss, so we add named extra-valid
iterators. Apply this to the fork at
`$SC/code/training/Megatron-LM-Swiss-AI` (keep a copy under
`init_bakeoff/megatron_patches/` for provenance). **Smoke-test before the full
run** (a 50-iter run with `--eval-interval 5`).

The hook point is already convenient: `evaluate_and_print_results(prefix,
forward_step_func, data_iterator, model, …)` takes a single `data_iterator` +
a `prefix`. Stock Megatron keys TensorBoard validation scalars only by loss
name, so this patch also prefixes TensorBoard/W&B scalar keys when the eval
prefix ends in `[name]`. We build one iterator per held-out set and call it once
per set with a per-set prefix, giving distinct per-set scalars and `.out` lines.

## 1. `megatron/training/arguments.py` — add the arg (near line 2036, the data group)

```python
group.add_argument('--extra-valid-data-path', nargs='*', default=None,
                   help='Named extra validation sets, evaluated + logged '
                        'SEPARATELY at --eval-interval. Format: NAME PATH [NAME PATH ...] '
                        '(PATH = a Megatron .bin/.idx prefix, no _text_document suffix needed '
                        'if --data-impl handles it; pass the full *_text_document prefix).')
```

## 2. `pretrain_gpt.py` — build one eval-only iterator per set

Mirror `core_gpt_dataset_config_from_args` (pretrain_gpt.py:253) and
`train_valid_test_datasets_provider` (`:282`). Add:

```python
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset
from megatron.training.training import cyclic_iter   # or replicate get_train_valid_test_data_iterators' iterator wrap

def build_extra_valid_iterators(args):
    """Return {name: data_iterator} for --extra-valid-data-path, eval-only."""
    pairs = args.extra_valid_data_path or []
    assert len(pairs) % 2 == 0, "--extra-valid-data-path must be NAME PATH pairs"
    out = {}
    eval_samples = args.eval_iters * args.global_batch_size  # enough for the eval pass
    for i in range(0, len(pairs), 2):
        name, path = pairs[i], pairs[i+1]
        cfg = core_gpt_dataset_config_from_args(args)
        # single-path, valid-only blend (train=0, valid=eval_samples, test=0)
        cfg.blend = ([path], [1.0]); cfg.blend_per_split = None; cfg.split = "0,1,0"
        valid_ds = BlendedMegatronDatasetBuilder(
            GPTDataset, [0, eval_samples, 0], is_dataset_built_on_rank, cfg
        ).build()[1]
        dl = build_pretraining_data_loader(valid_ds, 0)      # same helper training.py uses
        out[name] = iter(cyclic_iter(dl)) if dl is not None else None
    return out
```
Build them once in `setup_model_and_optimizer`/`pretrain` setup (right after the
standard valid iterator is built) and stash on `args` or pass through.

## 3. `megatron/training/training.py` — evaluate each in the eval block

Right after the existing `evaluate_and_print_results(... valid_data_iterator ...)`
call (training.py:1653), add:

```python
for _name, _it in (getattr(args, "_extra_valid_iterators", {}) or {}).items():
    if _it is None:
        continue
    evaluate_and_print_results(
        f'iteration {iteration} [{_name}]', forward_step_func, _it, model,
        iteration, process_non_loss_data_func, config,
        verbose=False, write_to_tensorboard=True, non_loss_data_func=non_loss_data_func)
```
With the scalar-key patch in `evaluate_and_print_results`, this logs
`'[{name}] lm loss validation'` (and ppl) per set to TensorBoard/W&B plus
distinct `.out` lines such as
`validation loss at iteration 5 [hplt] | lm loss value: ...`.
(Also call it once in the `do_train` final-eval block ~training.py:451-459.)

## 4. Wire into the launcher

In `bakeoff_train.sbatch`'s `DATA_ARGS` (and `configs/common_cpt.env`), add:
```bash
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"     # 2026-06-10 smoke: 3-set eval ~145s
EVAL_ITERS="${EVAL_ITERS:-1}"
EXTRA_VALID_FLAGS="--extra-valid-data-path \
  hplt         $STAGE/megatron/val_hplt_${TOK}_text_document \
  openarchives $STAGE/megatron/val_openarchives_${TOK}_text_document \
  greek_phd    $STAGE/megatron/val_greek_phd_${TOK}_text_document"
```
`TOK=base` for arm1 (131072), `TOK=ext` for arm2 (148480) — set in the arm
config. Add `--eval-interval $EVAL_INTERVAL --eval-iters $EVAL_ITERS
$EXTRA_VALID_FLAGS` to the train command, and set `--do-valid` (it's implied
when an eval path is given). The 3 curves appear in tensorboard as
`[hplt] lm loss validation`, `[openarchives] …`, `[greek_phd] …`. The same
per-set `.out` lines are parsed by `scripts/collect_metrics.py` into CSV rows
with `metric_type=valid` and `validation_set=<name>`.

> **Cost:** `--eval-interval 1 --eval-iters 1` = 3 extra single-batch forward
> passes per training step (~mild but non-zero slowdown). If too slow, raise
> `--eval-interval` to 10–25 for a dense-enough curve. Filenames above match
> `tokenize_vals.sbatch` output (`val_<name>_<tok>_text_document`).

## Verification (the execution agent MUST do before the full run)

1. `make_recipe_13b.py` → `bulk_13b.json`; confirm greek bucket = hplt_70 +
   openarchives_30 and both carry `drop_doc_keys_parquet=val_holdout_ids.parquet`.
2. After `build_holdout_vals` + `tokenize_vals`: the 6 `val_*_text_document.{bin,idx}` exist and are non-empty.
3. Apply this patch; run a 50-iter smoke (`EVAL_INTERVAL=5`, tiny `TRAIN_TOKENS`)
   and confirm 3 distinct `[name] lm loss validation` scalars appear.
4. Confirm the held-out docs are absent from the train mix (spot-check a few
   `val_hplt` source_doc_ids are not in `bulk_mix_final.jsonl`).
