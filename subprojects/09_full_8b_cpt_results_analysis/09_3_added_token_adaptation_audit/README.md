# 09.3 — Added-token adaptation audit

> **In one line:** asked whether the 17,920 added vocabulary entries adapted and whether they explain the GreekMMLU peak at update 9,536 — the answer is **yes** and **no**: added-token behaviour improves monotonically all the way to the terminal checkpoint, so the extension cannot be blamed for the post-peak regression.
> **Period:** 2026-08-23 (one day, `newtok-adaptation-audit-20260823`). **Status:** completed; **never committed by its author and not independently reviewed** — recovered from an uncommitted working tree on 2026-09-01 (`2aec4a66`).
> **Came from / led to:** the peak in [`../RESULTS.md`](../RESULTS.md) → this → an open question about layer-30 divergence (§3 of [`RESULTS.md`](RESULTS.md)).

## Why this existed

The CPT run took the vocabulary from 131,072 to 148,992 (17,408 modern Greek + 512 polytonic entries). GreekMMLU peaks at 9,536 and is worse at 18,284, and an obvious hypothesis was that the added tokens were the weak part. No artifact in the repository answered that. The existing D1–D7 new-token diagnostics owned by [`03_apertus_extension_and_embedding_adaptation`](../../03_apertus_extension_and_embedding_adaptation/) had only ever been run on the 2 B-token init bakeoff arms at vocabulary 148,480, never on a full-8 B checkpoint — and D6/D7 are weak by construction, since [`docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md`](../../../docs/APERTUS_ARCHITECTURE_FOR_EMBEDDING_NORM_ANALYSIS.md) already shows that Apertus's 0.1 gradient clip, Pre-Norm/RMSNorm, QK-Norm and logit saturation force per-token norm parity regardless of language share. Norm parity proves saturation, not that a merge is well targeted.

## What was measured

Three per-token tests, all **paired on identical text** and run on **held-out** documents, so the comparison is contamination-free and confound-free:

| Test | Statistic | What it detects |
| --- | --- | --- |
| T1 merged-vs-split likelihood | `logP(added token \| ctx) − Σ logP(base pieces \| ctx)` | a token that is alive but not worth its vocabulary slot |
| T2 hidden-state agreement | `cos(h_L[merged], h_L[last base piece])` at layers 11 and 30 | whether the token occupies the same representational slot as the phrase it replaced |
| T3 echo probe | rank and log-probability after a repetition prompt | behaviourally dead tokens |

Layer 11 is the layer the production Token-Distillation initialization was fitted at (`initialization.target_layer = 11` in the 8 B recipe); layer 30 tests whether the merged and split paths reconverge downstream. Inputs are the CPT bridge's held-out sets (six Greek-relevant panels: historical polytonic, forget-old-Greek, Greek PhD, OpenArchives, non-HPLT, HPLT), which are excluded from training, GreekMMLU-decontaminated and PII-masked upstream — not the training parquets.

Document selection is greedy **supply-first**: a document is kept only if it supplies an added token still below a 16-occurrence floor. A uniform sample would have drowned the rare tail in head tokens; this reached 96.3 % of added tokens at the floor from 9,559 documents in 10.6 s ([`evidence/coverage.json`](evidence/coverage.json): 670 below floor, 39 with zero occurrences).

## History

The whole audit ran on 2026-08-23. Its own [`evidence/launch_timeline.json`](evidence/launch_timeline.json) records first debug submission at 13:06:20Z, three reported blockers, and production submission about 59 minutes after the request. [`evidence/deviations.json`](evidence/deviations.json) is unusually candid and worth reading: the readiness plan was **not** green at submission — the canonical campaign runner has no verb for a post-hoc read-only diagnostic on released checkpoints (`apertus-cscs-efficiency#153`) — and the run proceeded under a recorded deviation on the owner's explicit in-session approval, explicitly *not* reported as green. Two process deviations are also recorded: an early weight-space pass that downloaded ~9.8 GB of `embed_tokens`/`lm_head` slices to the Mac before the intake existed, and a 200-document tokenisation smoke on a login node; both were corrected by moving all subsequent work to bounded Clariden jobs.

Throughput was measured rather than assumed ([`evidence/throughput_evidence.json`](evidence/throughput_evidence.json)): 3.19 docs/s for the slowest of three concurrent single-GPU tasks at batch 8, which is what sized the two-hour wall time. Batch 24 is unusable — the full fp32 log-softmax over a 148,992-token vocabulary needs 31.5 GiB at 3,072-token sequences and OOMs a 96 GB GH200. Production was Slurm `3162910` on `normal`, 00:38:31, scoring 2,348,881 occurrences per checkpoint across 9,558 held-out documents with 0 unaligned, at updates 400, 9,536 and 18,284.

## Outcome

Full numbers in [`RESULTS.md`](RESULTS.md). The four findings:

1. **Adaptation is monotone through the terminal checkpoint.** Modern added tokens (17,171 scored at ≥4 occurrences): Δlogp median +4.294 → +9.070 → +9.586 and echo top-1 44.3 % → 71.0 % → 85.3 % across 2 B / 40 B / 77 B. By update 9,536 **not one** measurable added token is net-negative. Polytonic tokens follow the same shape from a worse start.
2. **The peak is not an added-token phenomenon.** From 9,536 to the terminal, 79.8 % of the 17,171 modern tokens and 88.2 % of the 439 polytonic tokens improve. Token-level adaptation and benchmark accuracy decouple after 9,536, and late training buys the rare tail: tokens with 4–19 occurrences gain +0.810 nats on average while 35.1 % of the 500+-occurrence tokens regress.
3. **One metric prefers an earlier checkpoint.** Layer-11 merged/split cosine agreement holds flat (0.966 → 0.974 → 0.973), so the Token-Distillation objective survives 77 B tokens of CPT — but layer 30 declines (median 0.977 → 0.974 → 0.965) and its p5 tail declines three times faster. Whether that is healthy specialisation or drift is explicitly **not decidable** from this measurement; it was named as the next thread.
4. **The embedding-geometry reading was retracted.** A weight-space pass found input embeddings staying where TD put them while the output head spreads (participation ratio 0.314 → 0.531) — mechanically expected, since `lm_head` rows get gradient at every position and `embed_tokens` rows only when the token appears as input. An earlier reading of this project treated that asymmetry as a candidate explanation for the peak; finding 2 refutes it.

Coverage caveat: of the 152 tokens with zero scored occurrences, 39 are genuinely absent from the held-out corpus and 113 are single-character tokens whose base decomposition is a single token — for those the test is **inapplicable, not failed**, and must not be reported as a null finding. Δlogp magnitude scales with how many base pieces a merge replaces (2 pieces ≈ +8.12 nats, 7 pieces ≈ +26.84), so only its sign and trajectory are interpretable.

## Where things are

| What | Path |
| --- | --- |
| The result | [`RESULTS.md`](RESULTS.md) |
| Compact payload (with SHA-256 pointers to the ~16 MB raw per-token files on CSCS) | [`presentations/ADDED_TOKEN_ADAPTATION.data.json`](presentations/ADDED_TOKEN_ADAPTATION.data.json) |
| The audit as executed | [`evaluation/new_token_behavioral_audit.py`](evaluation/new_token_behavioral_audit.py), [`evaluation/build_coverage_corpus.py`](evaluation/build_coverage_corpus.py), [`evaluation/stageA_build_corpus.sbatch`](evaluation/stageA_build_corpus.sbatch), [`evaluation/stageB_audit.sbatch`](evaluation/stageB_audit.sbatch) |
| Reduction and cross-checkpoint comparison | [`analysis/build_payload.py`](analysis/build_payload.py), [`analysis/compare_checkpoints.py`](analysis/compare_checkpoints.py) |
| Coverage, throughput, smoke, readiness and deviation receipts | [`evidence/`](evidence/) (manifest: [`evidence/ARTIFACT_MANIFEST.json`](evidence/ARTIFACT_MANIFEST.json)) |
| Contract test | [`test_added_token_contract.py`](test_added_token_contract.py) |

Ownership boundary: [`07_full_8b_cpt`](../../07_full_8b_cpt/) remains authoritative for the recipe, checkpoints and raw receipts, and [`03_apertus_extension_and_embedding_adaptation`](../../03_apertus_extension_and_embedding_adaptation/) for the tokenizer extension and the TD initialization. This directory owns only the post-hoc conclusion about how those tokens behave in the released checkpoints.
