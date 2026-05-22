# Iter-130 Bakeoff Digest

- run tag: `bakeoff_1node_chain_20260522_005620`
- checkpoint: `iter_0000130` (~0.545B tokens at checkpoint, 130 / 476 planned steps)
- reading: still an early checkpoint, not the final bakeoff decision. Vanilla remains strongest on downstream Greek/retention; ReTok is clearly ahead of Centroid on new-token integration and Greek downstream metrics; Centroid remains weak on Greek use despite okay retention-style scores.

| arm | BPC | NLL/char | el ARC | el Belebele | el XNLI | el XQuAD F1 | el MMLU | el Base44 | el PIQA | HellaSwag | ARC-C | MMLU |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vanilla | 0.5432 | 0.6426 | 0.4275 [0.3993, 0.4556] | 0.5556 [0.5211, 0.5856] | 0.4137 [0.3956, 0.4325] | 0.3524 [0.3319, 0.3733] | 0.4459 [0.4375, 0.4539] | 0.4819 [0.4384, 0.5217] | 0.5900 [0.5000, 0.6900] | 0.7648 [0.7567, 0.7729] | 0.5614 [0.5333, 0.5904] | 0.5572 |
| retok | 0.7561 | 0.8943 | 0.3157 [0.2875, 0.3413] | 0.4678 [0.4355, 0.5000] | 0.3916 [0.3731, 0.4096] | 0.2737 [0.2544, 0.2935] | 0.3693 [0.3611, 0.3769] | 0.3859 [0.3424, 0.4239] | 0.6200 [0.5298, 0.7100] | 0.7494 [0.7412, 0.7574] | 0.5290 [0.5017, 0.5572] | 0.5538 |
| centroid | 1.1318 | 1.3387 | 0.2483 [0.2235, 0.2722] | 0.3211 [0.2911, 0.3533] | 0.3679 [0.3482, 0.3871] | 0.0253 [0.0187, 0.0329] | 0.2807 [0.2731, 0.2880] | 0.2862 [0.2482, 0.3225] | 0.5400 [0.4498, 0.6300] | 0.7613 [0.7535, 0.7697] | 0.5614 [0.5333, 0.5913] | 0.5580 |

CI brackets are 95% bootstrap intervals where sample-level data is available. `BPC` and `NLL/char` come from the 500-document tokenizer-fair heldout; task metrics come from lm-eval result JSONs.

Artifacts copied locally alongside this digest: per-arm `results.json`, `bootstrap_cis.json`, `run_metadata.json`, tokenizer-fair metrics, and new-token diagnostics.
