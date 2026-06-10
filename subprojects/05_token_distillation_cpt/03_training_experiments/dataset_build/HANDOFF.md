# Dataset build + held-out validation — detail doc (Phase 1)

> Part of the general handoff at [`../HANDOFF.md`](../HANDOFF.md) — start there.
> This file is the Phase-1 detail: building the 13.5B 2-arm dataset and the 3
> held-out validation sets. Corpus jobs are **CPU-only on `xfer`**; reserve
> `normal`/GH200 for training.

## 0. What changed (2026-06-09, this spec)

- **10B unseen (new) Greek = 70% HPLT + 30% openarchives** (was a 6-source Greek
  mix). Both live in `SELECTED` (`source_dataset` = `HPLT/ell_Grek_ge8_no_mt_clean60`
  and `openarchives.gr`).
- **Final stream order:** replay/code/math/Greek-replay rows stay in their
  original interleaved line positions; only HPLT/openarchives slots are rewritten
  so the new-Greek subsequence is HPLT first, then openarchives.
- **3 held-out validation sets, 0.5B tokens each — HPLT, openarchives, greek_phd**
  — disjoint from the training data; **validation loss measured per-set at the
  eval cadence** (current full-run default `EVAL_INTERVAL=25`; see §5 cost note).
- Carried over from prior decisions (keep unless told otherwise): replay = **24%
  multilingual / 4% code / 2% math / 5% Greek-replay OF the 10B new** (→ ~13.5B
  total); decontaminate (`correct_only`, covers Greek-replay) + anonymize
  (email/IP/IBAN over the full mixed stream) + HPLT confident-only E001 clean.

## 1. Final mixture (shares of total, ≈13.5B)

| bucket | source | share | ≈tokens |
|---|---|---|---|
| new Greek | HPLT `^HPLT/ell_Grek` (70% of new) | 0.5185 | 7.0 B |
| new Greek | openarchives `openarchives.gr` (30% of new) | 0.2222 | 3.0 B |
| replay | multilingual (24 langs) | 0.1778 | 2.4 B |
| replay | code (codeparrot) | 0.0296 | 0.4 B |
| replay | math (finemath) | 0.0148 | 0.2 B |
| replay | Greek-replay (Apertus-original) | 0.0370 | 0.5 B |

new-Greek total = 10B (HPLT 7B + openarchives 3B); replay = 3.5B. The held-out
0.5B×3 val docs are **excluded** from the train mix via `drop_doc_keys`.
The post-anonymization stream is then slot-reordered: replay/non-new-Greek rows
are byte-identical and remain at the same line numbers; new-Greek slots draw
from the HPLT queue until exhausted, then from the openarchives queue.

## 2. Inputs already on Clariden (verified)

- `SELECTED` (deduped Greek, 121 GB, 47,061,862 rows, cols incl. `text`,
  `source_dataset`, `source_doc_id`): `$SC/cpt_corpus/cpt/selected_after_apertus_and_internal_dedup.parquet`
- Greek-replay parquet (**built**, reuse): `$SC/cpt_corpus/greek_replay/greek_replay.parquet`
  (2,224,446 docs, ~2.85B tok — nanochat ∩ apertus_overlap_drop).
- Replay/code/math staged parquets: `$SC/cpt_corpus/{replay,math}/…`
- Tokenizers: base `$SC/models/apertus-8b-2509` (131072), ext
  `$SC/tokenizers/apertus_greek_modern_only_148480` (148480).
- TD init checkpoint (arm2): `$SC/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched`.
- Decontam queries: `/capstor/.../05_decontam_5b_20260602T011447Z/queries/native_greek_mcq_decontam_queries.jsonl`
- `$SC` = `/iopsstor/scratch/cscs/fffoivos`.

## 3. ⚠ Environment gotchas (hit + solved; the execution agent must reuse)

1. **Use the xfer-native build venv**
   `$SC/python_envs/cpt_build_xfer_py312` for CPU-only corpus jobs. Build or refresh
   it with `build_xfer_env.sbatch`. The older `$SC/python_envs/cpt_build_py312`
   links to an ARM `uenv` Python and cannot run on `xfer`.
2. **The mix and preprocessing need transformers WITH Apertus + deps.**
   `build_xfer_env.sbatch` pins the packages used by the known-good ARM/uenv env
   (`datasets==5.0.0`, `datatrove==0.8.0`, `pyarrow==24.0.0`,
   `tokenizers==0.22.2`, `transformers==5.10.2`) plus CPU `torch`/`numpy` for
   Megatron preprocessing. Verify
   `AutoTokenizer.from_pretrained('$SC/tokenizers/apertus_greek_modern_only_148480')`
   loads (vocab 148480). **Missing Apertus tokenizer support is a hard blocker.**
3. **`mix_builder.py` lives at the LEGACY mirror**, not under glossapi-tokenizer-extension:
   `$SC/repo/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/corpus_build/mix_builder.py`.
   The decontaminate/anonymize scripts ARE under
   `$SC/repo/glossapi-tokenizer-extension/subprojects/05_token_distillation_cpt/02_corpus_preparation/{30_decontaminate,40_anonymize}/scripts/`.
4. CPU-only corpus jobs run on `xfer`, which has no `uenv` and no GPU GRES.
   Use `build_xfer_env.sbatch` first; it creates
   `$SC/python_envs/cpt_build_xfer_py312` from the xfer-visible standalone
   Python 3.12 plus pinned dataset/tokenizer/Megatron-preprocess deps.
5. Avoid `normal` for corpus jobs: it allocates GH200 nodes and can bill
   `gres/gpu=4` even when the Python workload is CPU-only.
6. nanochat join key is **`(source_dataset, source_doc_id)`** — there is no `doc_key`
   column in nanochat (drop-list `doc_key` is a hash absent there).

## 4. Build runbook (dependency chain; all afterok so failures block downstream)

```
E  build_xfer_env.sbatch           # x86_64 Python env for xfer CPU-only corpus jobs
V  build_holdout_vals.sbatch        # 0.5B×3 val sets (HPLT/openarchives/greek_phd) + val_holdout_ids.parquet
J2 mix_13b.sbatch  (array 0-7%2)    # 70/30 new + replay, EXCLUDING val_holdout_ids  → bulk_mix_part_NN.jsonl   [dep: V + greek_replay(done)]
A  stageA_clean_decontam.sbatch     # concat → E001 clean → decontaminate(correct_only)                          [dep: J2]
B  stageB_anon_preprocess.sbatch    # anonymize full mix → bulk_mix_final.jsonl (+ legacy interleaved tokenization) [dep: A + E]
C  stageC_order_replay_fixed_preprocess.sbatch # preserve replay slots, order new-Greek slots, tokenize ×2          [dep: B + E]
T  tokenize_vals.sbatch (array)     # tokenize each val set ×2 (base+ext)                                           [dep: V + E]
```
Then training (§5). The recipe (`bulk_13b.json`) is generated by
`make_recipe_13b.py` (run it first; it points the greek bucket at HPLT 70% +
openarchives 30% and adds the `drop_doc_keys_parquet=val_holdout_ids.parquet`).

Submit order (after `make_recipe_13b.py`; `build_greek_replay` already done
unless replay has to be rebuilt):
```bash
E=$(sbatch --parsable build_xfer_env.sbatch)
V=$(sbatch --parsable --dependency=afterok:$E build_holdout_vals.sbatch)
J2=$(sbatch --parsable --array=0-7%2 --dependency=afterok:$V mix_13b.sbatch)
A=$(sbatch --parsable --dependency=afterok:$J2 stageA_clean_decontam.sbatch)
B=$(sbatch --parsable --dependency=afterok:$A:$E  stageB_anon_preprocess.sbatch)
C=$(sbatch --parsable --dependency=afterok:$B:$E  stageC_order_replay_fixed_preprocess.sbatch)
TV=$(sbatch --parsable --array=0-2 --dependency=afterok:$V:$E tokenize_vals.sbatch)
```

## 5. Training with per-set held-out loss

Apply the patch `megatron_patches/extra_valid_per_set.patch` (or the equivalent
in `EXTRA_VALID_README.md`) to the Megatron fork: it adds
`--extra-valid-data-path NAME PATH [NAME PATH ...]`, builds one eval-only
`GPTDataset` iterator per NAME, and at every `--eval-interval` evaluates each and
logs `NAME validation lm loss` to tensorboard + the `.out`. (The stock fork
*blends* `--valid-data-path` into a single loss — insufficient here.)

Per arm, pass the three val binaries tokenized with **that arm's** tokenizer:
```
--eval-interval 25 --eval-iters 1 \
--extra-valid-data-path \
   hplt        $STAGE/megatron/val_hplt_<tok>_text_document \
   openarchives $STAGE/megatron/val_openarchives_<tok>_text_document \
   greek_phd   $STAGE/megatron/val_greek_phd_<tok>_text_document
```
`<tok>` = `base` for arm1, `ext` for arm2.

> **Cost note.** `--eval-interval 1` (loss every training iteration) runs 3 extra
> forward passes per step and materially slows training. The 2026-06-10 smoke
> measured about 145s for one three-set eval event, so the full-run default is
> `EVAL_INTERVAL=25`, `EVAL_ITERS=1`: dense enough to track per-set loss while
> keeping overhead modest. The config exposes both for deliberate overrides.

Then launch both arms in parallel + auto-benchmarks via `../scripts/launch_all.sh`
(wire the `--extra-valid-data-path` args into `configs/common_cpt.env` /
`bakeoff_train.sbatch`'s `DATA_ARGS` first — see `EXTRA_VALID_README.md`).

## 6. Files in this handoff dir

| file | role |
|---|---|
| `make_recipe_13b.py` | generates `bulk_13b.json` (70/30 new + replay + holdout exclusion) |
| `build_xfer_env.sbatch` | builds the xfer-native CPU Python env |
| `build_holdout_vals.py` / `.sbatch` | 3 held-out val sets (0.5B each) + `val_holdout_ids.parquet` |
| `build_greek_replay.py` / `.sbatch` | Greek-replay source (already run; rerun if needed) |
| `hplt_clean.py` | E001 confident-only residue clean |
| `mix_13b.sbatch` | sharded mix → `bulk_mix_part_NN.jsonl` |
| `stageA_clean_decontam.sbatch` | concat → clean → decontaminate |
| `stageB_anon_preprocess.sbatch` | anonymize → `bulk_mix_final.jsonl` (+ legacy interleaved tokenization) |
| `stageC_order_replay_fixed_preprocess.sbatch` | preserve replay positions, order new-Greek slots, tokenize ×2 to launch prefixes |
| `reorder_new_greek_slots.py` | deterministic Stage-C reorder with a manifest proving non-new-Greek line positions are preserved |
| `tokenize_vals.sbatch` | tokenize the 3 val sets ×2 |
| `EXTRA_VALID_README.md` + `megatron_patches/` | per-set validation-loss patch + wiring |
| `../{configs,scripts,docs}` | the training configs, `launch_all.sh`, hyperparameter docs |
