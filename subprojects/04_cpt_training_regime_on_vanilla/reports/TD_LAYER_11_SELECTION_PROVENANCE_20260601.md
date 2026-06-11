# TD layer-11 selection — provenance report

**Date.** 2026-06-01.
**Purpose.** Precise record of how `target_layer=11` was selected for the
4-arm bakeoff TD arm. Written to inform the Task-2 planning agent. This
report supersedes my earlier audit notes (in `reports/PLANNING_AGENT_REPLY_20260601.md §4`
and `cpt-plan.md §3.2.1` as written 2026-06-01 morning) which incorrectly
claimed only Candidate B was trained. **Both candidates A and B were
trained at two scales, and the selection was empirical.**

---

## 1. Headline

Layer 11 was selected on the basis of **intrinsic-eval BPB on a
500-document Greek heldout + new-token recall diagnostics**, measured on
two TD runs that trained both `target_layer = -1` (paper default) and
`target_layer = 11` (package README suggestion) in the same Slurm job
allocation, on the same selected token set, on the same student model
init.

**Pilot (1,024 tokens × 50 snippets) selected layer 11 by 0.008 BPB.**
**Full run (17,377 tokens × 25 snippets) reconfirmed by 0.040 BPB +
better new-token top-1 / top-5 recall.** The selection criterion is
documented in `summarize_td_pilot_intrinsics.py:194-203`: pick the arm
with the minimum `global.bpb_bits_per_byte` on the heldout.

The pilot ran 2026-05-23 ~07:00–08:30 UTC, Slurm job 2349898 (token
selection prepass) → `retok_td_layer_pilot_20260523T082252Z` (training).
The full run ran 2026-05-23 ~09:00–14:00 UTC, Slurm job `2353960`
(training) → `2355714` (intrinsic eval). Both candidates trained as
arms within the same job's GPU allocation.

---

## 2. The two candidates that were tested

From `subprojects/03_apertus_extension_and_embedding_adaptation/TOKEN_DISTILLATION_PLAN.md §16`:

| Candidate | `target_layer` | Source | Trained at pilot scale? | Trained at full scale? |
|---|:---:|---|:---:|:---:|
| A | `-1` (last layer = layer 32) | TD paper §5.3 default; the value defined in `tokdist.py:67` as `target_layer: int = -1` | yes — `td_last` arm | yes — `td_full25_last` arm |
| B | `11` (≈ one-third depth = `ceil(32/3)`) | TD package README hint that one-third-depth target layers can work better than the last layer | yes — `td_layer11` arm | yes — `td_full25_layer11` arm |
| C (optional) | `L*` from a logit-lens / tuned-lens probe (paper §6.1) | `TOKEN_DISTILLATION_PLAN.md §6.1` proposed an Apertus-specific probe of where the residual stream resolves multi-token Greek words | **NO — not run** | **NO — not run** |

So the empirical layer comparison was a two-way A-vs-B with the
package-README hint as the contrarian challenger to the paper default.
Candidate C (logit-lens probe) was the missing third leg — it was
proposed as an optional Apertus-specific data point and was not
executed.

---

## 3. The selection criterion

The TD arm with the **lowest `global.bpb_bits_per_byte` on
`cpt_greek_heldout_500_20260522.jsonl`** wins. Heldout is the same
500-document Greek set used throughout subsequent Task-1 BPB tracking
(SHA-256 `3487a53f…`, mtime `2026-05-22`, no regeneration; documented
in `5B_REPORT.md §8`).

Selection logic in `summarize_td_pilot_intrinsics.py`:

```python
def _bpb(global_metrics):
    return global_metrics.get(
        "bpb_bits_per_byte",
        global_metrics.get("bpc_bits_per_byte"),
    )

# Picks the arm with min BPB on heldout; reports per-arm absolute BPB +
# delta vs retok (no-TD baseline) + new-token diagnostics.
best_bpb = sorted(
    [(arm["arm"], _bpb(arm["tokenizer"])) for arm in arms],
    key=lambda kv: kv[1],
)
summary["best_bpb_arm"] = best_bpb[0][0] if best_bpb else None
```

**Secondary diagnostics** (recorded but not part of `best_bpb_arm`
gate; useful for sanity-check):
- `d1_top1_rate`, `d1_top5_rate`, `d1_top10_rate`, `d1_top50_rate` —
  fraction of new-token target positions where the model's top-k
  predictions include the correct new token.
- `d1_mean_rank` — mean rank of the correct new token across positions
  with new-token targets.
- `d2_avg_prob_mass_new_per_pos` — total probability mass the model
  puts on *any* new-token logit at positions whose target is new (a
  sanity check that new vocabulary isn't dead).
- `d4_new_rate`, `d5_greedy_utilization_rate` — at how many positions
  the model produces a new token (teacher-forced and free-generation
  respectively).
- `E_new_to_existing_norm_ratio`, `U_new_to_existing_norm_ratio` —
  embedding-row norm ratios of new tokens to existing tokens (target
  ~1.0, so new rows don't dominate or vanish).
- `new_E_cos_p95` — 95-th-percentile pairwise cosine similarity among
  new-token rows (should be small).
- `new_E_participation_ratio` — effective rank measure of the new-token
  subspace.

The decision is BPB-driven; the other diagnostics confirm or override
in cases where BPB is too close to call.

---

## 4. Scripts that did the selection

All paths relative to
`subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/`:

| Script | Role |
|---|---|
| `token_distillation/select_td_pilot_tokens.py` | Picks the N tokens to TD-train, ranked by observed firings + document coverage from the CPU coverage prepass JSONL. Filters by `status_rank` (`enough_100` → 4, `enough_25` → 3, …); pilot used `--max-selected-tokens 1024`, full used `--max-selected-tokens 0` (all). |
| `token_distillation/external/token-distillation/token_distillation/tokdist.py` | Vendored upstream TD package (Dobler & de Melo 2025). Defines `target_layer: int = -1` as the function-signature default. |
| `token_distillation/train_retok_td_layer_pilot_packed.sbatch` | Trains both candidate layers in ONE Slurm allocation (Clariden `normal` packs the 4-GPU node either way, so we pack independent layer candidates onto separate GPUs of the same job). Accepts `TARGET_LAYERS="-1 11"`. |
| `eval/run_td_pilot_intrinsics_packed.sbatch` | Intrinsic eval (heldout BPB + new-token diagnostics) packed across all arms in one job. Same packing logic as the training script. |
| `eval/summarize_td_pilot_intrinsics.py` | Aggregator that emits `td_pilot_intrinsics_summary.json` with `best_bpb_arm` field. |

The packing pattern in `train_retok_td_layer_pilot_packed.sbatch`
(verbatim, lines 92-107):

```bash
mapfile -t LAYERS < <(printf '%s\n' $TARGET_LAYERS)
if [ "${#LAYERS[@]}" -gt 4 ]; then
    echo "ERROR: packed layer pilot supports at most 4 layers" >&2
    exit 2
fi
…
run_one() {
    local layer="$1"
    local gpu_id="$2"
    label="$(layer_label "$layer")"   # "-1" → "last", "11" → "layer11"
    out_dir="$OUTPUT_ROOT/$label"
    …
    cat > "$inner_script" <<INNER
    export CUDA_VISIBLE_DEVICES="$gpu_id"
    …
INNER
}
```

Both candidates trained on identical inputs (token set, snippets,
student model, seed) — only `target_layer` differs.

---

## 5. Token-selection criterion (separate from layer choice)

The layer pilot tested layer choice on a *fixed* selected token set
(both arms train the same tokens; only target_layer differs). Token
selection itself ranked the candidate new tokens by document coverage
and firing count from the CPU coverage prepass:

```python
STATUS_RANK = {
    "enough_100": 4,    # ≥100 firings + good coverage
    "enough_25":  3,    # ≥25 firings
    "low_20_24":  2,
    "low_lt20":   1,
    "zero":       0,
    "mismatch": -1,
}
```

Eligibility predicate: `min_status >= 3` (`enough_25` or better). Rank
by `firings DESC, coverage DESC`. Pilot bounded by `--max-selected-tokens
1024` for the bounded smoke; full run used all eligible tokens (17,377
of 17,408 candidates; 15 were skipped for not meeting the eligibility
gates).

This is the criterion that determines *which* 17,392-token subset of
the 17,408 modern Greek extension tokens gets TD'd. Both layer
candidates trained on exactly this token set; the layer decision is
orthogonal.

---

## 6. Pilot results (1,024 tokens × 50 snippets/token, 2026-05-23)

Bounded smoke at the most-stable end of the token distribution
(`status>=enough_100`). Output root on Clariden:
`/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_layer_pilot_20260523T082252Z`.

Intrinsic eval output:
`/iopsstor/scratch/cscs/fffoivos/token_distillation/td_pilot_intrinsics_20260523T091637Z`.
Local mirror summary: `eval/td_pilot_intrinsics_20260523T091637Z/td_pilot_intrinsics_summary.json`.

| Arm | `target_layer` | Heldout BPB | NLL/token | Δ BPB vs `retok` |
|---|:---:|---:|---:|---:|
| `retok` (no TD) | — | 2.9503 | 13.865 | (baseline) |
| `td_last` (Candidate A) | −1 | 2.7830 | 13.078 | −0.167 |
| **`td_layer11` (Candidate B)** | **11** | **2.7753** | **13.042** | **−0.175** |

**Selection: `best_bpb_arm: td_layer11`** by `0.008 BPB`. Very small
margin at this scale — within the noise floor for a 500-doc heldout —
but consistent direction on the secondary new-token diagnostics
(td_layer11 had slightly higher `d1_top5_rate`).

The pilot also confirmed both TD arms beat the no-TD `retok` baseline
by ~0.17 BPB, sanity-checking that TD itself is doing something useful
(rather than just providing noise).

---

## 7. Full-scale results (17,377 tokens × 25 snippets/token, 2026-05-23)

Production-scale TD run on the same student model with all eligible
tokens. Snippets-per-token dropped from 50 (pilot) to 25 (full) to
keep wall-clock feasible (~5h 45m on 4 GPUs). Same two candidates
(`-1` and `11`), same packing scheme.

Output roots on Clariden:
- Training: `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z` (Slurm `2353960`)
- Eval: `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_intrinsics_20260523T124000Z` (Slurm `2355714`)

Local mirror summary:
`eval/td_full25_intrinsics_20260523T124000Z/td_pilot_intrinsics_summary.json`.

| Arm | `target_layer` | Heldout BPB | New-token d1_top1 | New-token d1_top5 | Δ BPB vs `retok` |
|---|:---:|---:|---:|---:|---:|
| `retok` (no TD) | — | 2.9503 | 0.65 % | 2.31 % | (baseline) |
| `td_full25_last` (Candidate A) | −1 | 1.4249 | 3.81 % | 15.96 % | −1.525 |
| **`td_full25_layer11` (Candidate B)** | **11** | **1.3846** | **4.15 %** | **17.22 %** | **−1.566** |

**Selection: `best_bpb_arm: td_full25_layer11`** by `0.040 BPB`. The
margin widened from pilot (0.008) to full (0.040), and the secondary
diagnostics also moved in the same direction (layer 11 better on both
d1_top1 and d1_top5 recall — meaning, when the model is at a position
whose target is a new token, it ranks the correct new token in its
top-1 / top-5 predictions more often).

This was the gate that locked layer 11 for the 4-arm bakeoff.

---

## 8. Preservation verification

Both TD arms were verified to have changed only the new-token rows and
left everything else byte-exact, per
`token_distillation/full_td_20260523T092602Z/README.md`:

| Arm | Slurm job | State | Elapsed | Report |
|---|---|---|---|---|
| `last` | `2355706` | COMPLETED | 00:01:36 | `last/td_preservation_report.json` |
| `layer11` | `2355707` | COMPLETED | 00:01:39 | `layer11/td_preservation_report.json` |

Both reports identical on the verifications:
- No non-embedding tensor changed.
- No xIELU tensor changed.
- No QK-Norm tensor changed.
- No shape or dtype mismatches.
- All 17,377 trained `model.embed_tokens.weight` rows changed.
- All 17,377 trained `lm_head.weight` rows changed.
- All preserved embedding and output rows stayed byte-exact.

Then R17-patched into Megatron TP=2 via the `td_full25_layer11_r17_roundtrip_2357565`
roundtrip (Slurm `2357565`, COMPLETED, 00:02:06, standard tensors max
abs diff = 0.0, R17 tensors max abs diff = 0.0). That's the checkpoint
that became the 4-arm bakeoff TD arm.

---

## 9. What was NOT done (and the implications)

### 9.1 Candidate C — Apertus-specific L* via logit-lens / tuned-lens probe

`TOKEN_DISTILLATION_PLAN.md §6.1` proposed a quick probe: project
Apertus's hidden states at every layer through `lm_head` (logit lens)
or a learned per-layer probe (tuned lens, Belrose et al. 2023, arXiv
2303.08112), find the layer `L*` where the top-1 projection first
reliably matches the whole-word identity. For Llama-class 32-layer 8B
models L* is reportedly in the first third of layers; for Apertus it
could be shifted by xIELU + QK-Norm.

This was never run. If `L*` is materially different from 11, our
choice may be sub-optimal. The cost would be minutes-not-hours and
should be added to Task 2's TD-prep step.

### 9.2 Downstream MCQ comparison between layer −1 and layer 11

The bakeoff trained only the layer-11 TD arm at scale; layer −1 was
never carried forward into the multi-billion-token CPT, so there is no
downstream MCQ comparison of layer −1 vs layer 11 *after CPT*. The
intrinsic-eval pilot established a 0.040 BPB advantage at iter 0 of
CPT (i.e., just after TD, before any continued pretraining). Whether
that advantage transfers to downstream Greek MCQ post-CPT is not
empirically established by Task 1.

### 9.3 Other layer candidates in the suggested sweep

`cpt-plan.md §3.2` (Task 2 plan) proposes a broader sweep over `{4, 8,
11, 16, 20}`. Only layers `{−1, 11}` were actually compared in Task 1.
The sweep recommendation in §3.2 is for *Task 2*, not a retroactive
gap in Task 1's selection.

---

## 10. The validity caveat the planning agent should hold

The intrinsic eval used to select layer 11 measures **heldout BPB +
new-token recall right after TD, before any CPT**. This is a fast
proxy for "did TD initialize the new embedding/output rows in a way
that lets the model already represent Greek tokens reasonably well?"

The bakeoff's actual *downstream* outcome (after 2B / 3.5B / 5B Path-B
CPT) had layer-11 TD lagging Vanilla on the native MCQ aggregate at 5
B (TD-5B headline = 0.4109 vs Vanilla-5B = 0.4305). That doesn't
indict the layer choice specifically — the whole bakeoff also paid the
rope re-adaptation cost (TASK2_HANDOFF §2.3 + the Path-A probe
finding) — but it does mean **the intrinsic-eval-BPB-at-iter-0 metric
did not predict downstream Greek MCQ ordering after a multi-billion-
token CPT under Path B**. The layer-11-vs-layer-−1 intrinsic advantage
(0.040 BPB) may or may not survive CPT; we don't have direct evidence
either way.

### Implications

- **Don't treat the intrinsic eval as a definitive layer-selection
  criterion**. It's evidence that "TD with target_layer=11 produces
  cleaner new-token initialization than target_layer=−1 on the
  retok-init checkpoint." It's not evidence that "target_layer=11
  produces a better Apertus-Greek CPT endpoint."
- **The Task-2 layer sweep recommendation in cpt-plan §3.2 still
  stands**, but for a different reason than I originally framed.
  Rather than "we never validated layer 11" (false), it's "we don't
  know if intrinsic-BPB-at-iter-0 predicts post-CPT downstream
  outcome, and the sweep tests that correspondence directly."
- **The logit-lens probe (Candidate C)** is genuinely missing and
  cheap to add. If it indicates `L*` significantly different from 11
  on Apertus specifically, that's a sharper alternative to a brute-
  force sweep.

---

## 11. Compute used (audit trail)

| Step | Slurm job | Elapsed | GPU-h (4× normal node) |
|---|---|---|---|
| Pilot training (1024 × 50, both layers) | (pilot, no fixed JobID logged) | ~30 m | ~2 |
| Pilot intrinsic eval | `2349898`-area | ~10 m | ~0.7 |
| Full training (17,377 × 25, both layers) | `2353960` | 05:45:53 | ~23 |
| Preservation per arm | `2355706`, `2355707` | ~01:36 each | ~0.2 |
| Full intrinsic eval | `2355714` | ~10 m | ~0.7 |
| R17 roundtrip (layer-11 only) | `2357565` | 00:02:06 | ~0.1 |
| **Total for layer selection (both candidates)** | | | **~27 GPU-h** |

About 12 % of Task 1's 217 GPU-h, comparable to the Path-A probe's
24 GPU-h cost.

---

## 12. Implications for Task 2 (concrete)

The planning agent should:

1. **Do not redo the Candidate A vs B comparison at iter 0 (intrinsic).**
   It's done. Layer 11 wins on intrinsic-BPB by 0.040 over layer −1,
   and on new-token recall by ~0.34 pp top-1 and ~1.26 pp top-5.

2. **Do run the deferred logit-lens / tuned-lens probe (Candidate C).**
   Minutes-not-hours of compute. If `L*` lands in a region not covered
   by the existing two-candidate comparison (e.g., `L* ≈ 5` or `L* ≈
   24`), include that layer as an explicit Candidate D in the Task-2
   sweep. Cite `TOKEN_DISTILLATION_PLAN.md §6.1` for the recipe.

3. **Run the deferred broader sweep (`cpt-plan §3.2`: 4 / 8 / 11 / 16 /
   20`) ONLY if budget permits**, framed as "test the
   intrinsic-vs-downstream correspondence" — not as "validate layer
   11 from scratch." 5 layers × ~0.5 B Path-A CPT each = ~125 GPU-h
   plus intrinsic + sidecar evals; total ~150 GPU-h. The sweep gives
   downstream MCQ + retention numbers at matched tokens for each
   layer, which the intrinsic eval cannot.

4. **Keep `target_layer=11` as the Task-2 default if the sweep is
   deferred.** It's the empirically-selected best of the two
   candidates that were actually compared. The package-README hint
   that motivated the choice ("one-third-depth target layers can work
   better than the last layer") is supported by both the pilot and
   the full-scale intrinsic results on this Apertus-Greek setup.

5. **Document the choice transparently** in the Task-2 v1.x spec:
   "Layer 11 selected per `reports/TD_LAYER_11_SELECTION_PROVENANCE_20260601.md`;
   intrinsic-BPB advantage of 0.040 over layer −1 on `cpt_greek_heldout_500_20260522`;
   downstream-vs-intrinsic correspondence not yet validated."

---

## 13. Retraction of earlier claims

Three documents I wrote on 2026-06-01 morning contained the same
incorrect statement (paraphrased): "Plan called for two-candidate
pilot; only Candidate B was run." Correction:

1. `reports/PLANNING_AGENT_REPLY_20260601.md §4` — said "Only Candidate
   B (layer 11) was actually trained." **Wrong; both A and B were
   trained at two scales (pilot + full).**
2. `cpt-plan.md §3.2.1` — same statement, same correction.
3. `TASK2_HANDOFF.md` — did not literally make the claim but pointed
   at the cpt-plan claim.

The corrected reading: layer 11 selection was *empirical at the
intrinsic-eval layer, not heuristic*. The remaining unresolved item is
whether intrinsic-eval-at-iter-0 predicts downstream-MCQ-post-CPT;
that's a real Task-2 question, not a Task-1 gap.

I'll patch documents 1 and 2 with pointers to this report when the
corrections land.

---

## 14. Artifacts

### Local (in the repository, mostly tiny audit artifacts):

- `subprojects/03_apertus_extension_and_embedding_adaptation/TOKEN_DISTILLATION_PLAN.md` — the original two-candidate-plus-logit-lens plan.
- `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/token_distillation/`
  - `select_td_pilot_tokens.py` — token selection script.
  - `train_retok_td_layer_pilot_packed.sbatch` — both-layers-in-one-job training sbatch.
  - `external/token-distillation/token_distillation/tokdist.py` — vendored upstream TD package.
  - `full_td_20260523T092602Z/README.md` + `last/td_preservation_report.json` + `layer11/td_preservation_report.json` — preservation reports for both arms.
- `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/eval/`
  - `run_td_pilot_intrinsics_packed.sbatch` — intrinsic eval sbatch.
  - `summarize_td_pilot_intrinsics.py` — aggregator.
  - `td_pilot_intrinsics_20260523T091637Z/td_pilot_intrinsics_summary.json` — pilot results JSON.
  - `td_full25_intrinsics_20260523T124000Z/td_pilot_intrinsics_summary.json` — full-scale results JSON.
- `subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/init_bakeoff/megatron_patches/td_layer11_r17_roundtrip_2357565/README.md` — the R17 roundtrip evidence for the selected layer-11 checkpoint.

### Clariden (large artifacts, not in repository):

- Pilot training: `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_layer_pilot_20260523T082252Z/{last,layer11}/`
- Full training: `/iopsstor/scratch/cscs/fffoivos/token_distillation/retok_td_full25_layers_20260523T092602Z/{last,layer11}/`
- Pilot intrinsic eval: `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_pilot_intrinsics_20260523T091637Z/`
- Full intrinsic eval: `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_intrinsics_20260523T124000Z/`
- R17-patched Megatron TP=2 init: `/iopsstor/scratch/cscs/fffoivos/token_distillation/td_full25_layer11_r17_roundtrip_2357565/megatron_tp2_r17patched/`
- HF roundtrip: `…/td_full25_layer11_r17_roundtrip_2357565/hf_roundtrip/`

### HuggingFace (published):

- `fffoivos/apertus-tokenizer-extension` manifest.json `target_layer: 11` records (TokenDistil-Init / 2B / 3.5B / 5B entries) — each citing the matching Clariden source path.

---

**End of provenance report.** Hand to the Task-2 planning agent as input
for the layer-choice section of v1.x.
