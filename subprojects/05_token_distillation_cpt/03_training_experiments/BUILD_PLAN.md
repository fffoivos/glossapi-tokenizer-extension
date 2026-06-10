# BUILD_PLAN — two-arm 13.5B Greek CPT of Apertus-8B

End-to-end runbook. Each stage calls an **established tool**; bespoke pieces are
flagged in [`TOOLING_DECISIONS.md`](TOOLING_DECISIONS.md). Paths are Clariden
scratch (`/iopsstor`, `/capstor`); adjust to your mirror. Account `a0140`,
partition `normal` (xfer in maintenance till 2026-06-11), 4×GH200/node, 12 h cap.

Build order (arm-2 has the extra TD leg; otherwise identical):

```
§0 convert+revert+R17   →  §1 TD init (arm2 only)  ─┐
§2 data: dedup→decontam→PII→mix→NFC→preprocess ×2  ─┼→ §3 train (full-run WSD)  →  §4 convert→eval
```

---

## §0 · Convert Apertus-8B-2509 → Megatron, revert geometry, R17-patch

Tool: fork `tools/checkpoint/convert.py` (loader `apertus_hf`, saver `core`) +
`patch_apertus_extras.py`. Template: `…/init_bakeoff/megatron_patches/td_layer11_r17_roundtrip.sbatch`.

1. One-time: `bash megatron_patches/install.sh $MEGATRON_DIR` (symlinks the Apertus HF loader into `tools/checkpoint/`).
2. **Revert geometry on the HF side first** (we start from `main`, which carries
   the post-long-context geometry): set `config.json` `rope_theta 12000000→500000`,
   `max_position_embeddings 65536→4096`, keep `rope_scaling` = llama3 factor 8
   (`original_max_position_embeddings=8192, high=4.0, low=1.0`). This is the
   Apertus main-pretraining geometry (paper Table C.4; `submit_apertus_8b.sh:188-192`).
3. Forward convert (TP=2/PP=1), inside `uenv run pytorch/v2.9.1:v2`:
   ```bash
   python3 tools/checkpoint/convert.py --model-type GPT \
     --loader apertus_hf --saver core \
     --load-dir "$HF_DIR" --save-dir "$RAW" \
     --tokenizer-model "$HF_DIR" --bf16 \
     --loader-transformer-impl transformer_engine \
     --target-tensor-parallel-size 2 --target-pipeline-parallel-size 1
   mv "$RAW/iter_0000000" "$RAW/release"; echo release > "$RAW/latest_checkpointed_iteration.txt"
   python3 patch_apertus_extras.py --hf-dir "$HF_DIR" --megatron-dir "$RAW" \
     --out-dir "$PATCHED" --max-current-diff 1e-3 --overwrite
   ```
4. Gate: `python3 verify_hf_roundtrip.py --require-r17-match --logits` (R17 = saver_core has
   no slots for xIELU α_p/α_n + QK q/k_norm; skipping the patch silently resets 128 params).

Output: `…/megatron_tp2_r17patched` per arm. **Arm 1** uses the BASE-131072
checkpoint; **arm 2** uses the TD checkpoint from §1.

> Traps: `--target-tensor-parallel-size 2` is mandatory (omitting → TP=1, untrainable);
> `--loader-transformer-impl transformer_engine` required; `--bf16` is a loader-group
> flag; pin `pytorch/v2.9.1:v2` (4.48 transformers lacks `ApertusForCausalLM`). Do **not**
> use `convert_init_checkpoints.sbatch` (defaults TP=1).

## §1 · Token-Distillation init (arm 2 only) — E and U

Tool: Dobler `token-distillation` `train_embeddings`. Full mechanism + command:
[`docs/TOKEN_DISTILLATION_E_AND_U.md`](docs/TOKEN_DISTILLATION_E_AND_U.md).
Produces an HF checkpoint with the 17,408 new rows of **both** `embed_tokens`
(E, MSE-distilled at layer 11) and `lm_head` (U, CE) trained; originals frozen.
Then run §0 step 3-4 on that HF dir to get the Megatron init checkpoint.
**Shortcut:** reuse `td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched`
if its tokenizer == `apertus_greek_modern_only_148480` (verify first).

## §2 · Data

**Corpus-content prep is owned by the canonical pipeline** — see
[`../02_corpus_preparation/PIPELINE.md`](../02_corpus_preparation/PIPELINE.md)
(consolidated 2026-06-09). Do NOT re-specify ordering here. Canonical spine:

```
clean (10_clean_hplt, confident-only residue overlay)
  → dedup (20_dedup: VALIDATE the existing 129 GB SELECTED, not a re-run)
  → decontaminate (30_decontaminate: drop GreekMMLU-leaking docs, correct_only)
  → anonymize LAST (40_anonymize: mask email/IP/IBAN)
  → training-shard build
```
Settled there (not open): **mask last**; **no global NFC** (HPLT stays Apertus
`normalizer:null`; decontamination uses NFKC internally for matching only); PII
is stage 4 (masker validated 2026-06-05). The deduped `SELECTED` already exists
(129 GB); stages 1/3/4 still need their **full-corpus runs** (execution, not
decisions). HPLT cleaning = confident-only residue removal; the broad
destructive overlay stays unapplied per its own gate audit.

**This subproject owns only the last steps** (training-data shaping):

1. **Decontaminate the Greek replay too.** The 5% Greek-replay = Apertus-original
   Greek (`apertus_overlap_drop_docs.parquet`, 2,223,742 docs) — it is Greek and
   can carry `greekmmlu`/ilsp/plutus content, so run it through
   `30_decontaminate` (`correct_only`) exactly like the new Greek. (Multilingual/
   code/math replay are non-Greek → the Greek-MCQ benchmarks can't leak there; no
   decontamination needed for those.)
2. **Mix** to the resolved budget — new Greek = 10B; replay layered at 24% ML /
   4% code / 2% math / **5% Greek-replay** (of new) → ~13.5B. (`mix_builder.py` +
   `recipes/bulk.json`.) The builder interleaves buckets for token fairness.
3. **Anonymize the FULL mixed stream LAST.** Run `40_anonymize` (email/IP/IBAN
   masking) over the *mixed* stream so **every bucket — new Greek + Greek-replay
   + multilingual + code + math — is masked**, not just the new Greek. PII is
   language-agnostic; masking the post-mix stream is the single "mask last" pass
   that covers replay. (Idempotent on already-masked Apertus-origin data.)
4. **Slot-order the new-Greek subsequence while keeping replay fixed.** Run
   `stageC_order_replay_fixed_preprocess.sbatch`: all non-new-Greek rows
   (multilingual/code/math/Greek-replay replay) remain byte-identical and at the
   same line positions; HPLT/openarchives slots are filled by HPLT first, then
   openarchives. The Stage-C manifest must prove `non_new_positions_preserved`.
5. **Tokenize the one shared Stage-C mix TWICE** (same seed → byte-identical doc stream;
   never cross-tokenize):
   ```
   preprocess_data.sbatch  TOKENIZER_DIR=$BASE_TOKENIZER_DIR  → bulk_mix_ordered_replay_fixed_base_text_document  (131072, arm1)
   preprocess_data.sbatch  TOKENIZER_DIR=$EXT_TOKENIZER_DIR   → bulk_mix_ordered_replay_fixed_ext_text_document   (148480, arm2)
   ```
   Set the two `*_DATA_PREFIX` in the arm configs accordingly.

## §3 · Train — full-run WSD (both arms, in parallel)

Tool: `bakeoff_train.sbatch` (Megatron fork). Configs: `configs/*.env`.
Submitter: `scripts/submit_two_arm_full_run.sh` (anchors WSD on the full run;
walltime-bounded segments; no stale overrides).

```bash
bash scripts/launch_all.sh                            # dry-run, both arms
NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn bash scripts/gate_cpt2arm_artifacts.sh
NCCL_NET=Socket NCCL_SOCKET_IFNAME=hsn DRY_RUN=0 CONFIRM_LAUNCH=1 bash scripts/launch_all.sh
```

Socket/HSN passed the 16-node one-iteration smoke, but only at
`30046.7 ms/iter` (~`26.9h` raw training per arm before eval/checkpoint
overhead). Use it only if that walltime is acceptable, or keep working on the
AWS Libfabric/CXI `NET/OFI ... NO_SPACE` blocker / alternate parallelism path.

Schedule numbers: [`docs/SCHEDULER_MATH.md`](docs/SCHEDULER_MATH.md). Checkpoints
every 119 iters (~0.5B). `run_metadata.json` records the actual `ADEMA_*`/`LR_*`
used (reviewer trail).

## §4 · Convert trained checkpoints → HF → eval

Tools: fork `convert.py --loader core --saver swissai_hf` via
`run_megatron_convert_with_pg.py` (`CONVERT_FAKE_SHARDING_WORLD_SIZE=2` for the
trained TP=2 back-leg); lm-eval (swiss-ai fork). Orchestrator pattern:
`…/04_…/scripts/submit_checkpoint_sidecars.sh` (convert → retention `run_eval.sbatch`
+ native Greek MCQ + tokenizer-fair BPB). `greekmmlu` = **dascim/GreekMMLU**
(native), not ilsp/mmlu_greek. Set dtype explicitly on load (saver writes
`torch_dtype=float16` into config even though weights are bf16).

---

## §5 · Pre-launch checklist (decisions + verifications)

**Decisions / confirmations:**
1. **Per-step mixture — resolved 2026-06-09.** Use 10B new Greek + replay at
   24/4/2/5% of new-Greek (=13.5B total; per-step ≈74/18/3/1.5/3.7).
2. **NFC** on the corpus — yes/no (conflicts with Apertus `normalizer:null`).
3. **Warmup** — accept the 400-iter `2/(1-β2)` floor (vs 1% = 32)?

**Builds (work, not just config):**
4. **5% Greek-replay source** — locate/build the `apertus_overlap_drop` Greek parquet; add the `greek_replay` recipe bucket.
5. **PII masking** — wire `../anonymization/` over `${SELECTED}`, or confirm upstream.
6. **EXT/148480 binary** — never built; run Stage-3 a second time (only the BASE binary exists, and it's the old 9.83B mix).

**Verifications:**
7. **TD reuse** — `td_full25_layer11_r17_roundtrip` tokenizer == `apertus_greek_modern_only_148480`? else re-run §1.
8. **Geometry** — confirm `USE_ROPE_SCALING=1` on the Megatron side reproduces HF llama3 (factor 8, orig_max_pos 8192, hi 4 / lo 1). The Vanilla-5B run OMITTED scaling (`USE_ROPE_SCALING=0`); these runs turn it ON to match Apertus.
9. **R17** — every convert through `patch_apertus_extras.py` + `verify_hf_roundtrip.py --require-r17-match`.
10. **Submitter hygiene** — confirm the chain does NOT export `LR_SCHEDULE_STYLE`/`ADEMA_*_WARMUP_STEPS`/`LR_WARMUP_INIT` (config must win); confirm `LR_WSD_DECAY_SAMPLES` (20%) and whole-run `ADEMA_*_WARMUP_STEPS` take effect via `run_metadata.json`.
11. **Goldfish** — *(verified, no action)* the fork computes the mask **in the dataloader** (`GPTDataset.__getitem__` → `apply_goldfish`); no offline pass exists or is needed, and the hash is uniform over the 148480 vocab.
12. **Determinism** — both arm binaries from the same JSONL + seed; rebuilding the JSONL between the two preprocess runs breaks cross-arm comparison.
