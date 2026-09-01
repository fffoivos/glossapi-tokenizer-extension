# Essential results

## 1. Adaptation succeeded, while replay limited but did not eliminate forgetting

The completed run consumed 76.685B active tokens over 18,284 updates. Its
stationary stream contained 60.582B Modern-Greek tokens, 15.337B foreign replay
tokens and 0.767B Old-Greek replay tokens. Training completed with zero skipped
or non-finite updates.

All six Greek learning panels improved from their first frequent measurement
to the endpoint: HPLT, aggregate non-HPLT, OpenArchives, Greek PhD material,
historical/polytonic Greek and neutral external Modern Greek. The independent
per-document evaluation also improved from initialization through the terminal
checkpoint on all 13 panels.

Retention was less monotonic. English, code, mathematics, German, Russian and
Chinese all ended above their own best earlier BPB, although every one remained
far better than at initialization. Old Greek finished at its best frequent-panel
BPB. The correct summary is therefore **substantial Greek adaptation with
measurable late foreign-language forgetting**, not either “no forgetting” or
“catastrophic forgetting.”

The cooldown helped every exact per-document panel relative to cooldown start.
That observation does not prove that the cooldown caused the recovery because
this run has no matched no-cooldown control.

## 2. GreekMMLU peaked before heldout Greek loss stopped improving

On the fixed decontaminated 16,159-question GreekMMLU subset:

| Checkpoint | Token slots consumed | Accuracy | Choice NLL | Correct-answer BPB |
| --- | ---: | ---: | ---: | ---: |
| Initialization | 0 | 35.78% | 1.4586 | 0.6645 |
| Best observed | 39.997B | **56.81%** | **1.0740** | **0.1701** |
| Terminal | 76.689B | 54.85% | 1.1221 | 0.1926 |

The slot count includes the small loss-inactive packed tail; the corpus itself
contains 76.685B active tokens.

The benchmark made most of its initial gain very early, fluctuated thereafter,
and reached its best point at update 9,536. The terminal checkpoint was 1.96
accuracy points below that peak. Meanwhile, source-conditioned Greek BPB kept
improving through the run.

This is a real metric separation: continued language-model adaptation did not
translate monotonically into the narrower multiple-choice capability measured
by GreekMMLU. It is why future checkpoint selection needs more than one native
Greek benchmark.

## 3. The 0.5B screen did not establish scale predictivity

The 0.5B study provisionally favored stationary mixing, and the production 8B
run used that D0 schedule. At comparable endpoints, 0.5B D0 reached 42.25%
GreekMMLU accuracy while the 8B D0 run reached 54.85%.

That does not validate the small model as a predictor of the 8B trajectory.
The comparison differs in model architecture, tied versus untied embedding
adaptation, peak learning rate, sanitation, exact post-mask deduplication and
total token count. Moreover, the five 0.5B arms were close on GreekMMLU and no
formal winner was selected.

The defensible conclusion is narrower: D0 was a reasonable production choice,
but the completed evidence does not show that schedule rankings or checkpoint
shapes transfer from 0.5B to 8B. A controlled same-data replication across
0.5B, 1.5B and 8B is still required.

A subsequent frozen-scorer replication evaluated the D0 0.5B initialization,
39.728B checkpoint and final 80.732B checkpoint on the same native-Greek panel.
Strict-filtered choice-NLL direction agreed with 8B on 6/9 tasks from
initialization to about 40B, but only 3/9 from about 40B to the endpoint; the
exact best-checkpoint identity agreed on 2/9. This strengthens the distinction:
early capability emergence partially transfers, while late checkpoint timing
does not. See
[`D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md`](D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md).

## Decisions supported by this run

- Preserve the update-9,536 checkpoint as the observed GreekMMLU leader.
- Do not select future checkpoints from GreekMMLU accuracy alone.
- Retain source-conditioned Greek and replay loss as continuous diagnostics.
- Add orthogonal native-Greek benchmarks before interpreting the late
  GreekMMLU plateau as a general capability plateau.
- Treat the second post-mask deduplication as a run-specific data difference,
  not as an approved default for the next dataset release.
