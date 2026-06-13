# Peak LR Decision Table - 2026-06-13

Replay mix fixed at 20% foreign multilingual replay + 1% old-Greek replay, with 79% new Greek. Lower LM-loss deltas are better for forgetting; positive foreign deltas mean held-out old-data loss rose from the first observed validation point. Code remains a non-Apertus proxy.

| peak LR | final GreekMMLU | best GreekMMLU | best point | foreign mean Δloss | foreign max Δloss | old Greek Δloss | code proxy Δloss | read |
|---:|---:|---:|---|---:|---:|---:|---:|---|
| `2.75e-5` | 0.5721 | 0.5738 | curr-11.0B | -0.0900 | -0.0675 | -1.9487 | -0.1111 | Most conservative retention, but weakest adaptation. |
| `5.5e-5` | 0.5850 | 0.5868 | curr-13.0B | -0.0579 | -0.0435 | -1.9296 | -0.0824 | Solid retention, but no longer adaptation leader. |
| `8.25e-5` | 0.5874 | 0.5885 | curr-13.0B | -0.0279 | -0.0175 | -1.8376 | -0.0564 | Best no-regression compromise among higher-performing arms. |
| `1.1e-4` | 0.5921 | 0.5921 | curr-final | +0.0011 | +0.0174 | -1.7774 | -0.0300 | Best adaptation; small en/de/ru loss increases are the tradeoff. |

## Final foreign deltas

| peak LR | English | de | ru | zh |
|---:|---:|---:|---:|---:|
| `2.75e-5` | -0.0959 | -0.0678 | -0.0675 | -0.1287 |
| `5.5e-5` | -0.0546 | -0.0448 | -0.0435 | -0.0888 |
| `8.25e-5` | -0.0175 | -0.0196 | -0.0187 | -0.0557 |
| `1.1e-4` | +0.0174 | +0.0057 | +0.0043 | -0.0232 |

## Decision

User decision on 2026-06-13: pick **`5.5e-5`** as the best overall loss-first
balance. GreekMMLU alone points to `1.1e-4`, and `8.25e-5` is the higher-MMLU
no-positive-foreign-delta arm, but `5.5e-5` is the preferred compromise between
new-Greek held-out loss, old-data retention, and still-strong GreekMMLU.
