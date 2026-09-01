# scripts — TD 5 B diagnostic launcher and shared eval sidecars

> **In one line:** the leftovers of the pre-pilot Task-2 work — a checkpoint-bounded launcher
> for the 5 B TD diagnostic and the eval-sidecar machinery that the pilot and the sweeps then
> reused.
> **Period:** 2026-06-11 → 2026-06-17. **Status:** completed / historical; the paths inside
> point at Clariden scratch and at the legacy bakeoff trainer in
> [`../../03_apertus_extension_and_embedding_adaptation`](../../03_apertus_extension_and_embedding_adaptation).

## History

Most of this landed in one catch-up commit, `a19c136f` (2026-06-11), which committed work that
had already run: the TD-17k layer-11 5 B diagnostic chain and its watcher. [`../LOG.md`](../LOG.md)
records the verdict on those runs — *"the earlier 5 B runs were diagnostics, not the full
training recipe; the corrected hyperparameter regime is the central finding from them"*.

Three later commits are repairs made while larger runs were live:
`699dd181` (2026-06-11) made the sidecar submitter fail early unless the tokenizer directory
contains both `config.json` and `tokenizer.json` — the live watcher had inherited the raw
extended tokenizer directory, which is correct for training but invalid for Megatron→HF
conversion; `fcaf3dd0` (2026-06-12) added the home-side GreekMMLU poller used when Clariden's
`xfer` partition went down mid-sweep; `f8587b91` (2026-06-17) shortened the GreekMMLU-only
sidecar walltime.

## What is here

| File | Role |
|---|---|
| `submit_td_17k_5b_chain.sh` | Submits the 5 B TD diagnostic as checkpoint-bounded segments; drives the legacy `bakeoff_train.sbatch`. |
| `train_config_td_path_a.env` | Its **Path-A** geometry config (`max_pos 65536`, `rotary_base 12M`, RoPE scaling) — the geometry subproject 04 recommended for Task 2. The pilot did not use it: it launched at max positions 4,096 and RoPE base 500,000 with Apertus's `rope_scaling` preserved (see [`../RUNBOOK.md`](../RUNBOOK.md)). The file is a compatibility shim over the public `greek-apertus` cpt.env. |
| `watch_and_submit_td_checkpoint_sidecars.sbatch` | CPU-only (`xfer`) watcher that fires eval sidecars when a checkpoint appears. |
| `submit_td_checkpoint_sidecars.sh` | Submits one checkpoint's sidecar fan-out: convert → native MCQ, Greek-NLP, retention, Greek/code/math BPB, checksum. |
| `home_poll_curriculum_greekmmlu_sidecars.sh` | The `xfer`-outage fallback: polls Clariden over SSH from home and submits GreekMMLU-only sidecars. |
| `write_checkpoint_checksum_manifest.py` | Streams a checkpoint + its HF conversion into a SHA-256 manifest. |
| `run_td_checkpoint_adversarial_review.sh` | Runs a local Codex adversarial review of one checkpoint against read-only Clariden. |
