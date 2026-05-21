# V4-HF vs V4-post-conversion — Apertus-8B-2509 benchmark comparison

Companion to [`v4_baseline_corrected_20260521/README.md`](v4_baseline_corrected_20260521/README.md) and [`v4_postconv_retry_20260521/README.md`](v4_postconv_retry_20260521/README.md). Quantifies the R17 risk empirically: same Apertus weights, run through HF → Megatron → HF, then re-evaluated.

## Per-task comparison

Top-level group scores per swissai lm-evaluation-harness. Stderrs are the harness defaults (analytic, not bootstrapped — see `compute_bootstrap_cis.py` for per-sample CIs if needed).

### English retention (Apertus Table 14)

| Task | Metric | V4-HF | V4-postconv | Δ | Chance |
|---|---|---:|---:|---:|---:|
| `arc_challenge` | acc_norm | 0.5870 ±0.0144 | 0.2619 ±0.0128 | **−0.3251** | 0.25 |
| `arc_easy` | acc_norm | 0.8363 ±0.0076 | 0.2614 ±0.0090 | **−0.5749** | 0.25 |
| `hellaswag` | acc_norm | 0.7884 ±0.0041 | 0.2675 ±0.0044 | **−0.5209** | 0.25 |
| `winogrande` | acc | 0.6930 ±0.0130 | 0.5107 ±0.0140 | −0.1823 | 0.50 |
| `piqa` | acc_norm | 0.7992 ±0.0093 | 0.5212 ±0.0117 | −0.2780 | 0.50 |
| `mmlu` | acc | 0.5923 ±0.0039 | 0.2295 ±0.0035 | **−0.3628** | 0.25 |

`mmlu = 0.5923` on V4-HF matches the Apertus tech report Table 14 (≈0.59), confirming the eval pipeline is wired correctly.

### Multilingual retention

| Task | Metric | V4-HF | V4-postconv | Δ | Chance |
|---|---|---:|---:|---:|---:|
| `global_mmlu` (15 langs) | acc | 0.5246 ±0.0063 | 0.2381 ±0.0054 | **−0.2865** | 0.25 |
| `xcopa` (11 langs) | acc | 0.6575 ±0.0063 | 0.5185 ±0.0067 | −0.1389 | 0.50 |
| `xnli` (15 langs) | acc | 0.4400 ±0.0026 | 0.3321 ±0.0024 | −0.1079 | 0.33 |

### Greek slice

| Task | Metric | V4-HF | V4-postconv | Δ | Chance |
|---|---|---:|---:|---:|---:|
| `global_mmlu_full_el` | acc | 0.5155 ±0.0040 | 0.2295 ±0.0035 | **−0.2861** | 0.25 |
| `include_base_44_greek_few_shot_en` | acc | 0.5054 ±0.0208 | 0.1975 ±0.0169 | **−0.3080** | 0.25 |
| `belebele_ell_Grek` | acc | 0.6367 ±0.0160 | 0.2289 ±0.0140 | **−0.4078** | 0.25 |
| `arc_challenge_mt_el` | acc_norm | 0.4795 ±0.0146 | 0.2637 ±0.0129 | **−0.2159** | 0.25 |
| `xnli_el` | acc | 0.3984 ±0.0098 | 0.3333 ±0.0094 | −0.0651 | 0.33 |
| `xquad_el` | f1 | 0.5172 ±0.0117 | **0.0000 ±0.0000** | **−0.5172** | n/a |
| `global_piqa_completions_ell_grek` | acc_norm | 0.6200 ±0.0488 | 0.5400 ±0.0501 | −0.0800 | 0.50 |

## Reading

1. **V4-HF reproduces Apertus's published baseline.** Eval rig validated.
2. **R17 is catastrophic, not cosmetic.** R1's "standard max abs diff = 0.0" was bit-exact only on the standard transformer tensors — the 128 reset xIELU αp/αn/β/ε parameters (per layer × 32 layers × 4 params) destroy the model's learned representations. Closed-form MC tasks fall to or below chance; extractive QA (`xquad_el`) collapses to zero f1 because the model cannot produce coherent text spans at all.
3. **xIELU is the load-bearing R17 component.** xIELU's α-parameters are multiplicative slope gates: resetting them to the `XIELU.__init__` default (αp = αn = 0.8) destroys the learned activation shape across all 32 layers. The QK-Norm `q_norm`/`k_norm` weights happened to drift inside bf16 noise in the R1 diff (within 1e-3 threshold), but the xIELU resets are larger and they propagate through every forward pass.
4. **Tasks that "survive" R17 are easy-baseline tasks.** `winogrande`, `piqa`, `xcopa`, `xnli`, `global_piqa_completions_ell_grek` retain near-chance behaviour because the random init still produces *some* output and binary/3-way MC is forgiving. The information-rich tasks (MMLU, ARC, HellaSwag, Belebele, XQuAD) collapse.

## Bakeoff selection implications

The three bakeoff arms all start from approximately the V4-postconv baseline (R17-reset Apertus + the embedding-resize init choice they differ on), not from V4-HF. The §5.6 hard-gate plan in [`EVAL_RECIPE.md`](EVAL_RECIPE.md) was originally framed with V4-HF as the retention reference; that needs to change:

- **Old framing (V4-HF reference):** "Arm X regresses > 3 p.p. from V4-HF on mmlu" → trivially fails for every arm at iteration 0 (each is 36 p.p. below V4-HF by construction).
- **New framing (V4-postconv reference):** "Arm X recovers > Y p.p. from V4-postconv on mmlu after 2B tokens, with bootstrap CI separating it from the other arms" → measurable signal.

The bakeoff is **recovery**, not retention. Concrete questions per arm:

- Which arm climbs off the chance floor fastest on the closed-form MC tasks (mmlu, hellaswag, arc_easy)?
- Which arm preserves the partially-surviving signals (winogrande, piqa, xcopa) without backsliding?
- Which arm recovers `xquad_el` extractive QA — i.e. produces non-empty answers?
- Greek-specific recovery is the primary bakeoff signal: which arm recovers `belebele_ell_Grek` and `global_mmlu_full_el` fastest?

If after 2B tokens no arm has recovered meaningfully past V4-postconv (e.g. all three sit at chance ±0.01 on mmlu), the bakeoff doesn't differentiate at this budget; either extend the budget or block on the production patcher (`megatron_patches/patch_apertus_extras.py`) that fixes R17 before training.

## Production CPT prerequisite

R17 quantification confirms the existing risk note in [`RISKS.md`](../../../RISKS.md): **the post-conversion xIELU patcher is required before production CPT**, not just nice-to-have. Recovering 30-50 percentage points across MMLU/HellaSwag/ARC inside a 15-20B-token production CPT budget is unrealistic; the model must start from intact xIELU values.

The bakeoff's three arms differ in *embedding* init (Vanilla baseline / ReTok / Centroid). They do *not* differ on xIELU init, so the bakeoff cannot evaluate the patcher. Patching is independent of arm selection and must be validated separately (write the tensors, re-run the V4-postconv eval, confirm it tracks V4-HF).

## Artifacts

- `v4_baseline_corrected_20260521/results.json` — V4-HF full lm-eval-harness output.
- `v4_postconv_retry_20260521/{retention,greek}_results.json` — V4-postconv split runs (retention on job 2338020, Greek on job 2338021).
- Comparison script: `/tmp/v4_compare_full.py` (paths in script point at this dir).
