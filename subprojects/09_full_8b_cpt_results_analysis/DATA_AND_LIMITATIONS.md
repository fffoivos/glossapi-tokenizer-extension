# Data and interpretation boundaries

## Executed data contract

The completed 8B run used:

- 41,512,804,679 HPLT Modern-Greek tokens;
- 19,068,732,797 non-HPLT/GlossAPI Modern-Greek tokens;
- 15,337,098,095 foreign replay tokens;
- 766,854,905 Old-Greek replay tokens.

The stream was stationary D0 mixing. It used the 148,992-token Modern and
Polytonic Greek tokenizer, untied layer-11 Token Distillation initialization,
4,096-token sequences and the WSD-10 learning-rate schedule.

## Sanitation difference

The executed dataset was GreekMMLU-decontaminated and Apertus-standard PII
masked. After masking, the pipeline also ran a second global exact-content
deduplication. Its receipt reports 2,378,595 duplicate documents removed and
8,081 validation-collision documents removed from 97,136,622 input documents.

The user requested anonymization, not a second global deduplication. Therefore:

- this run remains internally valid as an evaluation of the dataset it
  actually consumed;
- it is not a data-identical replication of the earlier run or the 0.5B study;
- the second deduplication must not silently become release policy;
- the public Hugging Face dataset should be anonymized without changing its
  already-approved document multiset unless a separate deduplication decision
  is made.

## Validation interpretation

The final package contains 13 corrected panels and exact per-document endpoint
measurements at initialization, cooldown start and terminal checkpoint. The
earlier contaminated Old-Greek panel is not part of the canonical result.

Frequent validation curves are useful trajectories but are aggregate packed
losses. The per-document endpoints are the stronger evidence for endpoint
comparisons. Neither substitutes for a broad behavioral evaluation.

## Causal limitation

This is one D0/WSD-10 trajectory. It cannot isolate the effect of sanitation,
post-mask deduplication, replay proportion, model scale, tokenizer extension,
Token Distillation or learning-rate shape. Differences aligned with any of
those factors are descriptive until a matched control changes that factor only.
