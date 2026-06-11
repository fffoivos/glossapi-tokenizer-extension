# Artifacts

This release is meant to tell the short artifact story first, with the
technical details in manifests and provenance.

```text
swiss-ai/Apertus-8B-2509
  |
  | continuous BPE on GlossAPI + HPLT 50/50
  v
C3 candidate tokenizer
  base 131072 + 25600 = 156672
  |
  | cutoff sweep, curation, backfill, alignment
  v
ModernGreek-148k
  base 131072 + modern Greek 17408 = 148480
  |
  | optional polytonic stack
  v
ModernGreek-Polytonic-154k
  base 131072 + modern Greek 17408 + polytonic Greek 5120 = 153600

fffoivos/glossapi-greek-nanochat-pretraining-dataset
  + nanochat internal dedup metadata
  + Apertus overlap drop overlay
  |
  v
selected_after_apertus_and_internal_dedup.parquet
  |
  | 70/24/4/2 Greek/replay/code/math mix
  v
CPT-7B-mix
  |
  | token-count audit with ModernGreek-148k
  | HPLT clean60 Wave4: 48,728,774 rows -> 44,195,950,025 tokens
  |
  | Megatron indexed dataset preprocessing
  v
bulk_mix_base_nfc_megatron/bulk_mix_text_document.{bin,idx}

ModernGreek-148k + CPT-7B-mix
  |
  | Vanilla, ReTok, Centroid, TokenDistil
  v
4-arm init bakeoff
  |
  +-> 2B endpoint (all 4 arms)
  |     -> 2B-iso checkpoints + iter-476 evals
  |
  +-> 3.5B continuation (Vanilla / ReTok / TokenDistil)
  |     -> iso-token checkpoints + benchmark-evals/3.5B-comparison
  |
  +-> 5B continuation (Vanilla / TokenDistil only)
        -> bakeoff-final endpoint
        -> Vanilla-5B (BPB leader) + TokenDistil-5B (downstream leader)
        -> benchmark-evals/bakeoff-final
        -> provenance/decisions/CPT_MASTER_20260526.md (synthesis)
        -> provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md
```

The pre-commit decision-rule thresholds from the v0.12 experimental plan (`old_experiments_plan.md` §10 Q8: X / M_progress / M_ext / M_van / T) were never locked before bakeoff results came in. The 5B headline is therefore an honest description of the numbers — TokenDistil-5B wins downstream, Vanilla-5B wins tokenizer-fair BPB — not a rule-bound winner. See `provenance/decisions/PLAN_VS_RESULTS_RECONCILIATION_20260526.md` for the 14-entry discrepancy log.

`experiment-checkpoints/` is the top-level checkpoint area. It contains one
folder per public experiment checkpoint; source paths and exact technical
details live in each checkpoint manifest.

`cpt-training-dataset/token-counts.json` records the source-token accounting:
the full staged `HPLT/ell_Grek_ge8_no_mt_clean60` slice tokenizes to
`44,195,950,025` `ModernGreek-148k` tokens without document separators, or
`44,244,678,799` tokens when adding one EOD per row.

Loss evidence follows the tokenizer-fair policy in
`provenance/evals/LOSS_MEASUREMENT_POLICY.md`: heldout BPB and downstream evals
decide cross-arm comparisons. Raw Megatron `lm loss` is a training health trace
because the compared arms do not all use the same tokenizer. Older files may
call BPB `BPC`; that is a legacy bits-per-byte label.

Checkpoint weights were uploaded from Clariden through the non-GPU `xfer`
partition:

```text
Slurm job: 2382635
state: COMPLETED
exit code: 0:0
log: /users/fffoivos/apertus_hf_upload_checkpoints_20260525_2382635.log
```
