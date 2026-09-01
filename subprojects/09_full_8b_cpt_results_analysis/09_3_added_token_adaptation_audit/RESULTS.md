# 09.3 Results — the added tokens adapted, and that is not why 9,536 wins

Evidence: Slurm 3162910 (`normal`, 00:38:31), 2,348,881 scored occurrences per
checkpoint across 9,558 held-out documents, 0 unaligned. Payload:
[`presentations/ADDED_TOKEN_ADAPTATION.data.json`](presentations/ADDED_TOKEN_ADAPTATION.data.json).

## 1. Adaptation is monotone through the terminal checkpoint

Modern added tokens (17,171 scored at ≥4 occurrences):

| update | Δlogp p50 | tokens net-negative | echo top-1 | cos L11 | cos L30 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 400 (2B) | +4.294 | 0.46% | 44.3% | 0.966 | 0.977 |
| 9,536 (40B, GreekMMLU peak) | +9.070 | 0.00% | 71.0% | 0.974 | 0.974 |
| 18,284 (77B, terminal) | **+9.586** | **0.00%** | **85.3%** | 0.973 | 0.965 |

Polytonic added tokens (439 scored) follow the same shape from a worse start:
Δlogp p50 +5.406 → +6.764 → +8.138, net-negative 5.24% → 0.00% → 0.00%, echo
top-1 6.8% → 60.6% → 75.9%.

By update 9,536 **not one** measurable added token is net-negative: every one
predicts its own surface string better than the base pieces it replaced.

## 2. The peak is not an added-token phenomenon

From the GreekMMLU peak (9,536) to the terminal (18,284):

| group | shared tokens | improved | regressed | mean change |
| --- | ---: | ---: | ---: | ---: |
| modern | 17,171 | **79.8%** | 20.2% | +0.640 nats |
| polytonic | 439 | **88.2%** | 11.8% | +1.561 nats |

Token-level adaptation and downstream benchmark accuracy **decouple** after
update 9,536. The GreekMMLU regression reported in
[`../RESULTS.md`](../RESULTS.md) must not be attributed to the vocabulary
extension: on every per-token behavioural metric the terminal checkpoint is the
best one.

Late training buys the tail. Change from peak to terminal, by occurrence count:

| occurrences | tokens | mean change | regressed |
| --- | ---: | ---: | ---: |
| 4–19 | 882 | +0.810 | 19.7% |
| 20–99 | 8,967 | +0.683 | 19.1% |
| 100–499 | 6,763 | +0.588 | 20.4% |
| 500+ | 559 | +0.324 | **35.1%** |

The rarest tokens gain most; the most frequent ones stall, and a third of them
regress.

## 3. The one metric that moves against the terminal checkpoint

Hidden-state agreement between the merged and split paths, modern tokens:

| update | L11 p5 | L11 p50 | L30 p5 | L30 p50 | L30 p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 400 | 0.915 | 0.966 | 0.873 | 0.977 | 0.992 |
| 9,536 | 0.924 | 0.974 | 0.851 | 0.974 | 0.989 |
| 18,284 | 0.921 | 0.973 | **0.824** | **0.965** | 0.984 |

Layer 11 — the layer Token Distillation was fitted at — holds flat, so the
initialization's objective survives 77B tokens of CPT. Layer 30 declines, and its
p5 tail declines three times faster than its median: a growing minority of tokens
depart sharply from their base decomposition downstream.

This is the only measured quantity that prefers an earlier checkpoint. Whether it
is healthy specialisation (the token acquiring its own semantics) or drift is
**not decidable from this measurement**, and this subproject does not claim it is.
It is the thread worth pulling next.

## 4. Coverage and what is not measured

| category | modern | polytonic |
| --- | ---: | ---: |
| scored at ≥4 occurrences | 17,171 | 439 |
| under-sampled (<4 occurrences) | 237 | 73 |
| zero scored occurrences | 116 | 36 |

Of the 152 tokens with zero scored occurrences, **39** are genuinely absent from
the held-out corpus and **113** are present but have no merged-vs-split contrast:
they are single-character tokens (`Ρ`, `Θ`, `ὶ`, `ῦ`, `ὰ`, `ῶ`, `ῆ`) whose base
decomposition is a single token. For those the test is **inapplicable, not
failed**, and they must not be reported as null findings.

Read the **sign and trajectory** of Δlogp, not its magnitude: the absolute value
scales with how many base pieces the merge replaces — 2 pieces gives a median of
+8.12 nats, 7 pieces +26.84.

## 5. Relation to the embedding-space picture

A separate weight-space pass over the same checkpoints found that the added
block's **input** embeddings stay geometrically where Token Distillation put them
(mean pairwise cosine +0.0794 at 2B → +0.0781 at 77B, against +0.020 for the
1,501 pre-existing base Greek tokens), while the **output** head spreads steadily
(participation ratio 0.314 → 0.531 of its maximum).

That asymmetry is real, and it is mechanically expected: `lm_head` rows receive
gradient at every position through the softmax, `embed_tokens` rows only when the
token appears as input, and Apertus's 0.1 gradient clip caps per-step movement.

It is **not**, however, evidence of a capability deficit. The behavioural results
above show the tokens working better at every checkpoint. Input-embedding
anisotropy is a fact about representation geometry, not about whether the model
can use the vocabulary. An earlier reading of this project treated the geometry as
a candidate explanation for the 9,536 peak; §2 refutes that reading.
