# evidence/ — pinned artifacts and raw score files

> **In one line:** the hash manifest that makes the copied package verifiable, plus the two raw evidence sets the reports read directly.
> **Period:** 2026-08-12 → 2026-08-19. **Status:** completed.

## Contents

**`ARTIFACT_MANIFEST.json`** (2026-08-12, `2a413042`) records the completed run root, the copy date and a SHA-256 for each of the eleven artifacts lifted out of subproject 07 — three HTML reports, their three payloads, and the five analysis scripts. `python3 ../verify_bundle.py` fails closed if any of them has drifted. It was frozen on 08-12 and does not cover anything added later.

**`d0_0p5b_vs_full8_native_greek_3cp_20260814/`** (2026-08-14, `20230c1c`) holds the strict-filtered metric CSVs and receipts behind [`../D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md`](../D0_0P5B_VS_FULL8_NATIVE_GREEK_3CP_20260814.md): three 8 B checkpoints (`iter_0000000`, `iter_0009536`, `iter_0018284`), three D0 0.5 B checkpoints (`d0_iter_0000000`, `d0_iter_0018944`, `d0_iter_0038496`), their GreekMMLU receipts, the matrix receipt and the contamination-filter receipt. The three `full8_filtered/` CSVs are load-bearing beyond that document: the 19-checkpoint report of 2026-08-19 reads them as its anchor points and joins the other sixteen from Clariden.

**`retention_lm_eval_20260819/`** (2026-08-19) holds the raw lm-eval result files for the five retention checkpoints (`iter_0000000`, `iter_0002384`, `iter_0009536`, `iter_0014627`, `iter_0018284`) behind [`../evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md`](../evaluation/RETENTION_LM_EVAL_RESULTS_20260819.md). These were uncommitted working-tree files until `2aec4a66` (2026-09-01) and are not covered by `ARTIFACT_MANIFEST.json`.

Everything else — packing catalogs, per-shard receipts, prediction payloads, the contamination audit's 18-million-row match table — stayed on CSCS at the paths recorded inside these receipts and inside subproject [`07_full_8b_cpt`](../../07_full_8b_cpt/).
