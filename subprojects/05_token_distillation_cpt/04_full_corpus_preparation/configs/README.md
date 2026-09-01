# 04/configs — frozen registries and policies

> **In one line:** the seventeen tracked JSON/Markdown files that answer "which sources, on what terms, sampled how, cleaned how far" — the inputs every Clariden stage rehashes before it will run.
> **Period:** 2026-07-11 → 2026-07-15. **Status:** frozen; `cleaning_policy.json` deliberately ended the phase still at `audit_only`.
> **Came from / led to:** the inventory work in [`../REVIEW_20260711.md`](../REVIEW_20260711.md) → these registries → [`../scripts/validate_configs.py`](../scripts/validate_configs.py) and every sbatch launch gate.

## Why this existed

"Add the newer GlossAPI datasets" is not a build instruction. Turning it into one required separating four independent questions — is this source *new*, is it *legally usable*, is it *good enough*, and is it *net-additive after dedup* — and pinning each answer to an exact repository revision so a later stage could fail closed on drift.

## History

| Date | What was frozen | Content | Evidence |
|---|---|---|---|
| 2026-07-11 | `nanochat_initial_roster.json` | Lineage anchor: NanoChat's first data commit `500b8bf5…`, its 18 exact `source_dataset` names, 717,265 rows, the SHA-256 of `row_counts.csv`, and OPUS + HPLT as the only later name additions. | `02b4cb50`, `cd2f168b` |
| 2026-07-11 | `post_december_inventory.json` | 25 repositories created on/after 2026-01-01, 4 older repositories with material post-cutoff payload changes, and 6 explicit warnings (stale Diavgeia/Archetai card counts, mixed tokenizer scopes). | `02b4cb50` |
| 2026-07-11 | `sources.json`, `source_backlog.json` | 26 acquisition candidates (19 new-name families + 7 replacement/overlap routes) against 14 reviewed-but-not-acquired backlog entries, every one `acquisition_eligible=false`. The validator rejects any backlog repository that also appears in `sources.json`. | `26162a1c`, `d8d09d14` |
| 2026-07-11 | `source_lineage_aliases.json` | 17 reviewed aliases classified as direct, replacement or hybrid, plus 3 unaliased initial sources. An alias never establishes snapshot equivalence. | `43fcdde2` |
| 2026-07-11 | `source_review_policy.json` | ≥100 unique documents per exact `source_dataset` (60 random / 20 high-risk / 20 cluster), raised to 200 (100/50/50) for five named large or heterogeneous sources; 10% double review; admission thresholds. | `43fcdde2` |
| 2026-07-11 | `cleaning_policy.json` | `status: audit_only`; both structural materialisation flags false; bibliography prose-protection floor 0.999; the application gates that a future run would need (≥100 reviewed deletions, ≤0.001 prose deletion, ≥0.999 main-text retention, 0 catastrophic deletions); the Diavgeia-specific profile. **Never changed** — mid-run policy edits are forbidden. | `1d3b71f4` |
| 2026-07-11 | `training_eligibility_policy.json` | Five coarse categories as *upper bounds only*; the per-source licence matrix is always the final machine decision. | `26162a1c` |
| 2026-07-12 | `source_license_adjudication.json` | Default-deny matrix over the same 26 sources, pinned to the exact `sources.json` SHA-256, recording the card URL and revision plus upstream terms per candidate. | `01cba0ee` |
| 2026-07-12 → 07-22 | `dataset_review_evaluations.json` | 29 presentation entries = 25 post-cutoff repositories + 4 older repositories with material changes. An inventory universe, not a claim that all 29 have normalised text. | `ed552eba`, `84b6ab63` |
| 2026-07-13 | `agent1_v3_policy.json`, `agent1_v3_candidate_roster.json`, `agent1_v3_codex_review_prompt.md` | The v3 lane's own review (`gpt-5.6-luna`, low effort), dedup, GreekMMLU, anonymisation and structural policy; 26 candidate source IDs and 3 inventory-only exclusions. | `3a887c36`, `528497f3` |
| 2026-07-13 | `agent1_v4_raw_review_policy.json`, `agent1_v4_terra_review_prompt.md` | 18 new-family sources × 20 raw documents, `gpt-5.6-terra` at low effort, 8 explicitly excluded source IDs, and the owner-approved Heinrich Böll exception (8 available unique documents, never padded to 20). | `bf81861a`, `372a837d` |
| 2026-07-15 | `agent1_v5_eiger_pipeline.json`, `agent1_v5_requirements.txt` | The lane that shipped: 18 sources, runtime pins (GlossAPI `a2aace04`, DataTrove 0.9.0 / `87f7bad5`, NanoChat `e1d54136`, Rust 1.85.1), the frozen dedup parameters, and a dated licence override scoped to the two private versioned releases only. | `c144116c` |

## Outcome

- `sources.json` + `source_license_adjudication.json` are the pair that every downstream stage rehashes; changing either invalidates existing cleaning and release receipts by design.
- `cleaning_policy.json` is the reason no ToC or bibliography text was removed in this corpus: it froze `audit_only` before Stage 10 and was never edited.
- **Known staleness:** `agent1_v5_eiger_pipeline.json` still records `dedup.max_bucket_documents: 5000`. The 2026-07-21 resolution raised it to 50,000 through a cluster-side `…resolved.json` and deliberately left the tracked config untouched — see [`../../../../docs/AGENT1_V5_LSH_OVERSIZED_DIAGNOSIS_2026-07-21.md`](../../../../docs/AGENT1_V5_LSH_OVERSIZED_DIAGNOSIS_2026-07-21.md).
- `release.private_only: true` in the same file also predates the 2026-07-28 publisher change (`e8fbec2c`) that allowed an explicitly scoped public release.
