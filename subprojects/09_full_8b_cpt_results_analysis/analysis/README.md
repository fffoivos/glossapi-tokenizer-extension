# analysis/ — the post-hoc analysis scripts

> **In one line:** the six scripts that turned the completed run's raw receipts into the conclusions in [`../RESULTS.md`](../RESULTS.md); preserved byte-for-byte, not a second pipeline.
> **Period:** 2026-08-12 → 2026-08-14. **Status:** completed.

## History

Five scripts were copied out of the completed-run workspace on 2026-08-12 with the rest of the package (`2a413042`) and pinned by SHA-256 in [`../evidence/ARTIFACT_MANIFEST.json`](../evidence/ARTIFACT_MANIFEST.json), so `../verify_bundle.py` fails closed if any of them drifts. Nothing else from subproject 07's much larger analysis workspace was copied: synthetic drift simulations, alternative displacement statistics and exploratory prefix reports were left behind on the stated grounds that they did not change any of the three decisions.

One script was added later: `compare_d0_0p5b_native_3cp_to_full8.py` on 2026-08-14 (`20230c1c`, revised the same day by `2a7eb9d8`), when the token-aligned 0.5 B replication needed a comparison that produced both the prose verdict and the machine-readable payload for [`../D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md`](../D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md). It is not covered by the artifact manifest, which was frozen on 08-12.

## What each script answers

| Script | Question it served |
| --- | --- |
| `analyze_retention_snapshot.py`, `forecast_retention_slope.py` | the learning-versus-forgetting trajectory across the 13 validation panels |
| `analyze_per_document_endpoints.py` | exact document-local BPB at initialization, cooldown start (14,627) and terminal — the strongest endpoint evidence |
| `analyze_greekmmlu_answer_drift.py` | how many individual GreekMMLU answers change between checkpoints (the input to the later instability work in [`../09_1_downstream_task_instability/`](../09_1_downstream_task_instability/)) |
| `analyze_checkpoint_source_exposure.py` | which corpus sources each checkpoint had actually seen |
| `compare_d0_0p5b_native_3cp_to_full8.py` | direction and best-checkpoint agreement between the 0.5 B D0 arm and the 8 B run |

Raw CSCS prediction payloads, packing catalogs and receipts were never copied here; they remain under subproject [`07_full_8b_cpt`](../../07_full_8b_cpt/) at the paths embedded in the canonical JSON payloads.
