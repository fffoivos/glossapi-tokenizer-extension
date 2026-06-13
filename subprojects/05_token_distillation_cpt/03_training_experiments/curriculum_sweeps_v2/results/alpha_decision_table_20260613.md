# Alpha Sweep Decision Table - 2026-06-13

Runs: `FOREIGN_REPLAY_R=0.253164557`, `OLD_GREEK_REPLAY_R=0.012658228`, `LR_PEAK=5.5e-5`, AdEMAMix alpha in `{0,4,8}`.

| alpha | final GreekMMLU | best GreekMMLU | last-3 avg | foreign loss avg final | foreign delta avg | old Greek final loss | old Greek delta | new Greek loss avg final | new Greek delta avg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.5663 | 0.5734 | 0.5670 | 1.5594 | -0.1101 | 1.9409 | -1.8620 | 1.7500 | -1.6662 |
| 4 | 0.5948 | 0.5948 | 0.5906 | 1.6078 | -0.0616 | 1.9063 | -1.8813 | 1.6988 | -1.7104 |
| 8 | 0.5782 | 0.5782 | 0.5731 | 1.6570 | -0.0133 | 1.9084 | -1.8940 | 1.6870 | -1.7313 |

Readout: alpha `4` has the strongest GreekMMLU trajectory and final score. Alpha `0` has the lowest foreign held-out loss / strongest foreign retention. Alpha `4` is the best compromise on adaptation plus old-Greek loss; alpha `8` gives no GreekMMLU win and worse foreign retention.

Sources: `alpha_greekmmlu_trajectory.csv` and `alpha_forgetting_loss.csv` collected from Clariden run root `/capstor/scratch/cscs/fffoivos/runs/curriculum_v2`.
