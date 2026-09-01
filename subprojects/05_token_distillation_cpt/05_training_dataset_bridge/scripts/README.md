# 05/scripts — the bridge tooling

> **In one line:** ten CPU tools that take a cleaned Parquet corpus to verified Megatron binaries; two of them (`bridge_common.py`, `build_binary_shard.py`) outlived this subproject and became the shared tokenisation layer for subprojects 06, 07 and 08.
> **Period:** 2026-07-12 → 2026-08-07. **Status:** the full chain was never run here; the two shared modules are still in production use.
> **Came from / led to:** invoked by [`../clariden`](../clariden) → outputs consumed by [`../train`](../train), and by [`../../06_25b_midtraining_probe`](../../06_25b_midtraining_probe) and later subprojects.

## The chain, in order

| Script | Role |
|---|---|
| `acquire_replay_sources.py` | Receipt-bound Clariden restaging of every replay prerequisite. Accepts only full 40-hex Hugging Face revisions, stages the exact matching file inventory, deterministically rewrites the pinned StarCoder subset, and can restore a clean detached Megatron checkout. Never resolves a moving `main`. |
| `build_old_greek_replay.py` | Rebuilds `greek_replay.parquet` from the receipted NanoChat shards and the exact Apertus overlap overlay, emitting a receipt that binds inputs, implementation bytes, composite identity policy, counts and output hash. |
| `freeze_inputs.py` | The expensive exact receipt pass: accepts only a **passed Phase-04 local release**, hashes every private training shard and replay shard, checks the Phase-04 tokenizer identity, and binds the clean repository commit and complete Megatron source tree. This is the input contract the corpus that actually shipped never satisfied. |
| `build_heldouts.py` | Builds the nine deterministic, source-disjoint LM-loss heldouts and their exclusion lists. Later gained `selector_not_regex`, GreekMMLU decontamination during selection (`8dbb6d25`) and set-union merging of overlapping components (`86c1b8fe`). |
| `build_binary_shard.py` | Encodes one Megatron indexed-dataset shard: reads a frozen Parquet file, drops heldout identities, applies the Phase-04 GreekMMLU policy to replay pools, optionally applies a receipt-bound phase partition, and writes `.bin`, `.idx`, ledgers and finally the manifest — manifest last, so Slurm-array retries are safe. |
| `finalize_bridge.py` | Validates every shard and freezes the 79/20/1 blend using a disk-backed exact-uniqueness audit rather than holding identities in RAM. Writes `bridge_capacity_failure.json` and stops if unique capacity is short. |
| `freeze_training_assets.py` | Freezes launch-time code, the TD layer-11 init checkpoint and the semantic layer-11 roundtrip evidence bundle. |
| `verify_launch_assets.py` | Fail-closed launch validation over bridge, binaries, code, init and resume state. Run twice: once before submission and again at job start inside the pinned uenv. |
| `freeze_resume_checkpoint.py` | Freezes and validates one completed segment checkpoint before a relaunch; a missing, altered, wrong-iteration or already-submitted segment fails closed. |
| `bridge_common.py` | The shared primitives: SHA-256 helpers, atomic JSON writes, file/tree/tokenizer receipts, frozen-repository validation, composite `document_key`, deterministic heldout hashing, `.idx` reading and writing, and launch-dependency receipts. |

## Outcome

- **`bridge_common.py` + `build_binary_shard.py` are the durable artifacts.** They are loaded by path (with a declared SHA-256 checked first) from `../../06_25b_midtraining_probe/dataset/freeze_inputs.py`, `../../../06_dataset_scheduling_experiments/dataset/*`, `../../../07_full_8b_cpt/dataset/anonymization/*` and `../../../08_targeted_8b_cpt_experiments/scripts/run_parallel_task_batch.py`.
- **The phase-partition hook** in `build_binary_shard.py` is the mechanism behind the probe's two-phase disjoint document allocation: it refuses to load an implementation whose hash drifts, and counts excluded rows explicitly instead of dropping them quietly (`8dbb6d25`, merged 2026-09-01 as `600148e6`).
- **`freeze_inputs.py` is the reason this chain stalled.** It is bound to the Phase-04 Stage-80 materialisation output; the corpus that shipped came from the Agent 1 v5 lane on Hugging Face, so [`../../06_25b_midtraining_probe`](../../06_25b_midtraining_probe) wrote its own `freeze_inputs.py` and reused only the two modules above.
