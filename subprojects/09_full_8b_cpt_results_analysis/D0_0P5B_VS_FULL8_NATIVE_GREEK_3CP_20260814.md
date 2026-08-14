# D0 0.5B versus sanitized 8B: native-Greek three-checkpoint replication

Date: 2026-08-14

## Verdict

The 0.5B D0 run **partially reproduces early learning, but does not reproduce the
8B late-training trajectory**. On strict-filtered choice NLL, the direction
from initialization to approximately 40B token slots agrees on
**6/9** comparable benchmarks.
From approximately 40B to the endpoint it agrees on only
**3/9**, and the exact
best-checkpoint identity agrees on **2/9** benchmarks with
choice NLL.

The practical result is narrow but useful: a Mini checkpoint near 40B is a
credible screening point for whether mainstream Greek MCQ capability has
emerged. It is not a reliable proxy for whether the 8B model will peak there or
continue improving afterwards.

## Choice NLL trajectories

Lower is better. Each cell is initialization → approximately 40B → final. All
non-GreekMMLU rows use the strict contamination filter; GreekMMLU uses its
separate frozen 16,159-question decontaminated subset.

| Benchmark | D0 0.5B NLL | 0.5B best | Sanitized 8B NLL | 8B best |
| --- | --- | --- | --- | --- |
| GreekMMLU | 1.5545 → 1.3002 → 1.2869 | final | 1.4586 → 1.0740 → 1.1221 | mid |
| ASEP MCQA | 1.4918 → 1.3545 → 1.3525 | final | 1.4807 → 1.2548 → 1.2604 | mid |
| DemosQA | 1.4067 → 1.2819 → 1.2836 | mid | 1.3688 → 1.2855 → 1.2854 | final |
| GPCR | 0.7514 → 0.6968 → 0.6931 | final | 0.7541 → 0.6698 → 0.6677 | final |
| Medical MCQA | 1.7657 → 1.6015 → 1.6045 | mid | 1.7192 → 1.5030 → 1.5057 | mid |
| OYXOY metaphor | 0.8698 → 0.8388 → 0.7231 | final | 0.6444 → 0.8420 → 1.2008 | initial |
| OYXOY NLI binary | 0.7607 → 0.8328 → 0.8643 | initial | 0.7288 → 0.6663 → 0.7640 | mid |
| OYXOY WiC | 0.7758 → 0.7704 → 0.7418 | final | 0.5673 → 0.7444 → 0.7659 | initial |
| OYXOY WSD | 1.4547 → 1.3082 → 1.3029 | final | 1.4480 → 1.2974 → 1.3319 | mid |
| OYXOY NLI exact set | — → — → — | — | — → — → — | — |

## Accuracy trajectories

Higher is better. Accuracy is secondary to choice NLL, particularly for the
imbalanced OYXOY binary tasks.

| Benchmark | D0 0.5B accuracy | Sanitized 8B accuracy |
| --- | --- | --- |
| GreekMMLU | 34.61% → 41.82% → 42.13% | 35.78% → 56.81% → 54.85% |
| ASEP MCQA | 26.95% → 33.90% → 33.90% | 28.47% → 54.83% → 55.08% |
| DemosQA | 29.88% → 45.08% → 44.74% | 33.06% → 46.41% → 46.58% |
| GPCR | 53.61% → 58.25% → 58.76% | 53.09% → 61.34% → 62.89% |
| Medical MCQA | 24.58% → 26.25% → 23.63% | 24.58% → 40.81% → 38.42% |
| OYXOY metaphor | 34.13% → 33.89% → 38.05% | 64.94% → 35.46% → 33.89% |
| OYXOY NLI binary | 43.69% → 35.98% → 36.02% | 51.16% → 62.22% → 38.73% |
| OYXOY WiC | 36.20% → 25.28% → 32.29% | 76.14% → 39.31% → 33.64% |
| OYXOY WSD | 35.33% → 36.96% → 37.72% | 35.87% → 38.67% → 38.06% |
| OYXOY NLI exact set | 4.98% → 0.06% → 0.06% | 10.35% → 18.31% → 1.72% |

## What actually replicates

- GreekMMLU, ASEP, DemosQA, GPCR, Medical MCQA and OYXOY WSD all improve in
  choice NLL from initialization to approximately 40B at both scales.
- GPCR continues improving after 40B at both scales.
- Medical MCQA worsens slightly after 40B at both scales, making it the clearest
  replicated mid-run peak.
- The other late trajectories do not transfer. The 0.5B GreekMMLU, ASEP and WSD
  NLLs continue improving slightly, while their 8B counterparts worsen. DemosQA
  changes by very little in either model but with opposite exact signs.
- OYXOY metaphor, NLI and WiC begin from very different label-bias regimes at
  the two scales. Their raw accuracies and NLL shapes therefore should not be
  interpreted as a clean scale-replication result.

## Why a mismatch is unsurprising

This is a token-aligned scorer replication, not a controlled scale study:

- **Architecture:** Mini is 20 × 1,024 with tied embeddings; 8B is 32 × 4,096
  with untied embeddings.
- **Embedding adaptation:** Mini selected tied layer-7 TD with MSE plus
  auto-weighted CE. The 8B initialization used untied layer-11 TD plus separate
  polytonic output calibration.
- **Optimization:** both use WSD-10, but Mini peaks at `1.5e-4` and 8B at
  `5.5e-5`; their global-token batches are 2.097M and 4.194M respectively.
- **Data:** Mini consumed the pre-sanitation 80.730B-token schedule. The 8B run
  used PII masking followed by exact post-mask deduplication, dropping
  2,386,676 documents and reducing active tokens to 76.685B—a 4.044B-token
  difference.
- **RoPE:** both use theta 500,000 and 4,096 context, but Mini uses the native
  no-scaling geometry while the 8B recipe uses scaling factor 8.

These differences prevent a causal claim that model scale alone caused the
trajectory mismatch.

## Execution and evidence

- Clariden scoring jobs: `3079741` preserved 39 completed shards before its
  planned wall-time exit; resume job `3079936` completed the remaining 24,
  aggregated all 63, and finished in 13:06.
- Each checkpoint contains 83,970 newly scored examples across the frozen suite.
- The strict filter applied exactly 10,076 frozen exclusions at every checkpoint.
- D0 matrix receipt SHA-256:
  `86eecdefdcbb32717773a096f0afffb27ad907129f7e88e70afd09dda17c6849`.
- D0 contamination-filter receipt SHA-256:
  `6e8b3b07ad4c450685cf3c30d4f9562acc4307ded55d14779d2522571126b2af`.
- Machine-readable comparison: `D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.data.json`.

Greek Protipa Exams remains intentionally unscored because its owner-side
manual access gate was not available. It is not counted as evaluated.
