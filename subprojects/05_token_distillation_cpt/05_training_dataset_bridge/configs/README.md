# 05/configs — the frozen recipe and the replay pins

> **In one line:** two JSON files that fixed what the 25B probe would train on and where its replay data would come from.
> **Period:** 2026-07-12, replay config revised 2026-07-31. **Status:** frozen; never consumed by an executed build.

## `frozen_25b_td.json`

Recipe `full_corpus_td_25b_79_20_1_v1`, seed `20260609`. Nominal 25,000,000,000 tokens become **5,960 steps / 6,103,040 samples / 24,998,051,840 effective tokens** at sequence 4,096 and global batch 1,024, leaving a 1,948,160-token floor residual. Mix numerators 79 / 20 / 1 over denominator 100, with a `minimum_unique_capacity_ratio` of 1.005 and one extra boundary sample per physical prefix. Tokenizer `fffoivos/apertus-tokenizer-extension` at `a4826df7…`, `tokenizer.json` SHA-256 `358ae3f2…`, vocab 148,480, identity `ModernGreek-148480`.

New Greek is declared to come from the `phase04_validated_private_training_release` keyed on `stable_uid` and already GreekMMLU-decontaminated. Foreign replay inherits the historical 13.5B recipe [`../../03_training_experiments/dataset_build/bulk_13b.json`](../../03_training_experiments/dataset_build/bulk_13b.json), pinned by SHA-256 `a3f9591b…`, restricted to the `replay`, `code` and `math` buckets and re-decontaminated.

Nine heldouts are declared here with a 2 GB character budget each and a 0.25 maximum pool fraction, selected by `domain_separated_sha256_threshold_v1`: `hplt`, `openarchives`, `greek_phd` (new Greek); `english`, `de`, `ru`, `zh`, `code` (foreign replay); `old_greek`.

## `replay_acquisition.json`

The pinned restaging plan. The historical FineWeb-Edu, FineWeb-2/HQ, FineMath and StarCoderData commits were recovered from retained acquisition-day cache refs, the StarCoder snapshot inventory and a completed staging log, because the payload copies had been deleted; the config records those evidence paths and hashes. It deliberately does not expand the pinned FineWeb2-HQ globs to all 5,285 matching shards (5.36 TB): the full 10BT English sample and old-Greek inputs are kept, while multilingual web and FineMath are capacity-sized by a seed-20260609 domain-separated SHA-256 file ranking, with the matched count, selected count and selected remote paths recorded per source (`62d4aac3`, 2026-07-31).

## Outcome

- The recipe's arithmetic (5,960 steps, 24,998,051,840 tokens) carried forward unchanged into [`../../06_25b_midtraining_probe`](../../06_25b_midtraining_probe), which kept the horizon and replaced the single 79/20/1 blend with two randomized phases.
- The replay-selection bound stated here is explicitly *pre-build*: the authoritative check remains the post-tokenisation unique-capacity gate in `finalize_bridge.py`, which must pass before any launch.
