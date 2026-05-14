# Legacy Material

This folder is for material that should not drive new execution.

It includes:
- the deprecated `add_tokens(...)` baseline workflow
- old monolithic status and plan files
- exploratory review/sample scripts that do not yet satisfy the stricter current constraints
- alternative research-direction notes that the project considered and did not pursue
- pre-C3-convergence sub-pipelines whose outputs were consumed by the wave-2 broad cleaner

Subfolders:
- `legacy/monolithic_docs/`
- `legacy/add_tokens_baseline/`
- `legacy/exploratory_hplt/`
- `legacy/baseline_reports/`
- `legacy/corpus_clean_normalization/` — rule-discovery sub-pipeline that produced cleaning + normalization rule candidates. Stages 1–4 + Phase 6 built; Phases 7–8 (landing in `Corpus.clean`, before/after diff, regression generation, token-impact, negative controls) never built. Outputs were consumed manually into the wave-2 broad cleaner. See `corpus_clean_normalization/PLAN.md` and `NORMALIZATION_DESIGN_20260420.md`.
- `legacy/analysis/` — dedup analysis outputs (exact-stage analysis scripts + audit artifacts).
- `legacy/nanochat_glossapi_en_vs_el/` — older nanochat en-vs-el experiment harness (prepare-data script). The fertility evaluator that this dir originally exposed is now run via `~/runs/c2_c3_analysis_tools_20260506/run_tokenizer_fertility_suite.py` on the gcloud instance.

Root-level notes moved into this folder:
- `legacy/MORPHBPE_APPLICATION_NOTES.md` — research-track note on applying morphology-aware BPE (MorphPiece / MorphBPE lineage) to Greek. Competes with the shipping merge-rule extension on the critical path. Not pursued.
- `legacy/NORMALIZATION_PENDING_IDEAS.md` — cleaner-level normalization ideas (polytonic preservation, dot-leader target choice, ligatures, homoglyphs, soft-hyphen de-hyphenation). Triaged with a "Resolution (2026-04-20)" mapping each item to `promoted` / `deferred` / `off-scope` / `already shipped` / `resolved`.

Rule:
- keep for reference
- do not treat as the active project stack
