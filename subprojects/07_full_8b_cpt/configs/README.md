# 07 · configs — the machine authority

> **In one line:** five JSON files that decided what the 8B run was: the scientific recipe, the two candidate execution profiles, the owner's explicit decisions, the deadline forecast, and the per-document validation estimate.
> **Period:** 2026-08-05 → 2026-08-07. **Status:** frozen; the recipe here is the pre-sanitization geometry, superseded in execution (see below).
> **Came from / led to:** the D0 result in [`../../06_dataset_scheduling_experiments/`](../../06_dataset_scheduling_experiments/) → these files → every gate in [`../scripts/`](../scripts/) and [`../clariden/`](../clariden/).

## The files

### [`recipe_8b_full_mixed.json`](recipe_8b_full_mixed.json) — `recipe_id: full8b-mixed-79-20-1-wsd10-v1`

The scientific contract. Data: `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2@3f97cec4…`, complete eligible training pass, 79/20/1 stationary mix, seed 20260801, sequence length 4096 with cross-document mask/position reset and EOD loss masking. Tokenizer `fffoivos/apertus-tokenizer-extension@fcd33ec0…` (131,072 base + 17,408 modern + 512 polytonic = 148,992, `tokenizer.json` SHA-256 `bbb08e71…`). Initialization: `swiss-ai/Apertus-8B-2509@3162c996…` with `verified_untied_layer11_token_distillation_plus_polytonic_output_calibration`. Model: 32 layers, hidden 4096, FFN 21504, 32 heads / 8 query groups, xIELU, RMSNorm, QK-layernorm, RoPE base 500,000 with scaling factor 8.0. Optimization: AdEMAMix `(0.9, 0.999, 0.999, α=4)`, WD 0.1, clip 0.1, bf16 with fp32 main gradients, NaN/Inf checks **enabled**, Goldfish `k=h=50`, WSD peak `5.5e-5` → floor `5.5e-6`, 400-update warmup, cooldown from update 15,398. Evaluation: 13 panels every 25 updates, 20 GreekMMLU checkpoints, per-document validation at `[0, 15398, 19248]`, checkpoint averaging false. Fifteen named launch gates close the file.

Four things it is careful to say out loud, in `provenance_disclosures`: WSD-10 is a *settled baseline*, not a sweep-selected winner (the T10/T20/T30 study never selected a floor); the 20% pool replays vetted source *families*, not documents proven to be in Apertus's original stream; the 1% pool and its `old_greek` validation ID are dominated by Modern-Greek HPLT/FineWiki rows and must be reported as **Greek replay retention**; and the complete-v2 directive includes the small `openarchives.gr` `needs_ocr` subset, recorded as data identity and not as an OCR-quality endorsement. `data.libduth` similarly records the licence conflict with `legal_conclusion_claimed: false`.

**Superseded in execution.** Sanitization changed the horizon: the run consumed 76,685,490,476 active tokens over 18,284 updates in five segments, not 80.73B over 19,248 in six. The derived sanitized contracts were produced by [`../scripts/derive_sanitized_contracts.py`](../scripts/derive_sanitized_contracts.py); see [`../SANITIZED_RESTART_RUNBOOK_20260807.md`](../SANITIZED_RESTART_RUNBOOK_20260807.md).

### [`owner_decisions_20260805.json`](owner_decisions_20260805.json)

Recorded `2026-08-05T11:17:54Z`. Four decisions, each with an explicit basis: D0 accepted on `explicit_point_estimate_acceptance` (before per-document confidence intervals existed); `libduth` included on `explicit_recorded_risk_acceptance`, qualified as "does not assert that the legal conflict was resolved"; production launch authorized; checkpoint averaging excluded. Validated by [`../scripts/validate_owner_decisions.py`](../scripts/validate_owner_decisions.py) as a launch gate.

### [`execution_profiles.json`](execution_profiles.json)

Two candidates over identical scientific invariants: `dp32_16node` (64 GPUs, DP=32, accumulation 16, six boundaries) marked `proven_fallback`, and `dp64_32node` marked `candidate_requires_benchmark_promotion`. The promotion block sets the 288-update / 32-burn-in benchmark, the trajectory-drift thresholds, a ≥1.6× median-throughput requirement, and the restart tolerances `restart_gradient_norm_atol 0.001` / `rtol 0.02` — the two values whose post-hoc addition is disclosed in [`../evidence/DP32_RESTART_ACCEPTANCE_DISCLOSURE_20260806.md`](../evidence/DP32_RESTART_ACCEPTANCE_DISCLOSURE_20260806.md). **Outcome: DP64 rejected on trajectory drift; DP32 selected.**

### [`eta_16node_to_20260809.json`](eta_16node_to_20260809.json)

The "can it finish by Sunday" forecast, calibrated on five completed 16-node jobs (`2972672`, `2972674`, `2975267`, `2975269`, `2975271`) at a median 8.6655–8.7288 s/update and 10.493–11.053 observed wall seconds/update. Answer, as recorded in the historical README: plausible, not safe to promise — training alone barely fits the conservative bound and complete evidence does not.

### [`per_document_validation_estimate.json`](per_document_validation_estimate.json)

The costed plan to add document-cluster uncertainty to the five **0.5B** endpoints (13 panels, 113.9M tokens/model, 569.4M total, 20 nodes for five concurrent models), explicitly `planning_only_until_one_panel_smoke`. This is the gap that kept subproject 06 from declaring a winner. It was never executed for those endpoints.

### `prequeue_schedule_8b.json`

Added during the run (2026-08-09): the approved `measured_safe_idle_overlap_v1` holder policy over the **executed** five-segment boundaries `[0, 4000, 8000, 12000, 14627, 18284]` at 9.0 measured seconds/update, with a 43,200-second allocation, a 1,200-second unspendable reserve and per-target hold/trigger budgets.

## Outcome

The recipe plus the owner-decisions file are the two documents that make the run auditable: the first says what was intended, the second says which of its premises were accepted rather than proven. Read them together with the runbook, which records what actually changed.
