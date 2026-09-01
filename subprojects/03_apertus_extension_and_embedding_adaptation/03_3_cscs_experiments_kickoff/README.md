# 03_3 — CSCS experiments kickoff

> **In one line:** the one-day planning bridge (2026-05-20) between a stale experiment plan and the first Clariden submission — it audited every open decision, rebuilt two unloadable tokenizer bundles into shippable ones, verified the polytonic vocab budget, wrote the CSCS auth workflow, and chose the replay languages; **no Slurm job ran here**.
> **Period:** 2026-05-20 (`ec5ee52b`, `01d7befa`) → 2026-05-21 (review packet, `3aa2cf71`). The `ship/` directory later gained the 2026-07-29 production tokenizer.
> **Status:** completed; its docs were explicitly superseded by cpt_plan v0.7 and then by [`../CPT_MASTER_20260526.md`](../CPT_MASTER_20260526.md), but the `ship/` bundles and the auth runbook stayed live for the rest of the programme.
> **Came from / led to:** [`../03_1_greek_embedding_diagnostic/`](../03_1_greek_embedding_diagnostic/README.md) + [`../03_2_apertus_c3_dedup_audit/`](../03_2_apertus_c3_dedup_audit/README.md) → this → [`../03_4_implementation_experiments/`](../03_4_implementation_experiments/README.md).

## Why this existed

The experiment plan (v0.12, 2026-05-12) had six decision nodes and ten open questions, and eight days of follow-on work had answered several of them without the plan being updated. Before spending GPU-hours, someone had to walk every node, mark it resolved or open, and — separately — verify that the artifacts the experiment would actually load (tokenizers, auth, corpus path) worked. The rule was that this sub-subproject *analyses* and does not edit the plan; anything needing sign-off was parked as a review checkpoint.

## History

| Date | What happened | Result / decision | Evidence |
|---|---|---|---|
| 2026-05-20 | State audit of all six decision nodes written | Node A (cutoff) already resolved at 17,408; D (framework) resolved to **Megatron-LM-Swiss-AI** over HF Trainer per the user's "closest to Apertus original process" directive; E (mix) → 70/30; F (polytonic) → in scope of CPT at 153,600; **C (decision-rule thresholds) left open** — it stayed open forever | [`ANALYSIS.md`](ANALYSIS.md) §7 |
| 2026-05-20 | **Ship-tokenizer reconstruction** | Both on-disk variants emitted `tokenizer_class: TokenizersBackend`, which `AutoTokenizer.from_pretrained()` cannot load. The underlying BPE was fine; a 3-file wrapper swap produced two loadable bundles: `apertus_greek_modern_only_148480` (256 × 580) and `apertus_greek_extended_153600` (256 × 600). Polytonic-NT fertility win 60 → 28 tokens (−53.3 %) and 60 → 20 (−66.7 %); base 131,072 IDs byte-identical, English/Russian unchanged | [`SHIP_TOKENIZER_RECONSTRUCTION.md`](SHIP_TOKENIZER_RECONSTRUCTION.md), [`scripts/build_and_verify_ship_tokenizer.py`](scripts/build_and_verify_ship_tokenizer.py) |
| 2026-05-20 | **Polytonic vocab budget checked against the sub-1B-language scaling pattern** | Corpus is ~222.7 M tokens pre-extension and ~162.7 M post-extension; the script-isolated power-law fit (`vocab_fired ≈ 0.1341 × tokens^0.5688`, R² 0.783, n=194) predicts **4,000–6,300** distinctive tokens at that scale. **+5,120 sits inside the band** — no change needed | [`POLYTONIC_VOCAB_BUDGET_CHECK.md`](POLYTONIC_VOCAB_BUDGET_CHECK.md) |
| 2026-05-20 | Replay-language selection | Proposed **34 languages** in 3 tiers from four user criteria (geographic / Western-Europe bridge / historical-Greek / global diversity). v0.7 then trimmed this to **24**; the doc survives as the convergence rationale, and it corrects v0.7's Tier-3 "near-zero exposure" framing (all five Tier-3 languages have ≥ 1 B sampled FineWeb-2 tokens) | [`REPLAY_LANGUAGE_SELECTION.md`](REPLAY_LANGUAGE_SELECTION.md) |
| 2026-05-20 | CSCS auth captured against the new `cscs-key` Rust tool (the old `sshservice-cli` is deprecated) | Account `a0140`; daily `cscs-key sign --headless --duration 1d`; verified end-to-end — `ssh ela` → `ela5`, `ssh clariden` → `clariden-ln001`; working uenv `pytorch/v2.6.0:v1` | [`CSCS_AUTH_WORKFLOW.md`](CSCS_AUTH_WORKFLOW.md) |
| 2026-05-20 | Curriculum + init-corpus decision | **Fresh-only for the init comparison** (Apertus-overlap overlay applied), mixed for main CPT. Its proposed 4-phase curriculum and 85/15 split were both overridden by v0.7's single shuffled bulk + anneal tail at 70/30 | [`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md) |
| 2026-05-20 (later) | **GCloud access lost** | The CPT corpus build moved from a GCP scratch VM to Clariden `xfer`; the held-out contamination re-run lost its only cheap path | constraint-update blocks in [`ANALYSIS.md`](ANALYSIS.md) and [`REVIEW_PACKET.md`](REVIEW_PACKET.md) |
| 2026-05-20 → 05-21 | v0.7 adopted as canonical and propagated (`01d7befa`); reviewer packet assembled and revised twice | Every doc in this directory gained a "v0.7 supersedes this" banner. The packet's own claim that the **composite 153,600** bundle is the active CPT base was reversed the same day by the bakeoff scope decision, which used the modern-only 148,480 | [`REVIEW_PACKET.md`](REVIEW_PACKET.md), [`../03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md`](../03_4_implementation_experiments/init_bakeoff/BAKEOFF_PLAN.md) |
| 2026-07-29 | A third bundle added to `ship/` long after this sub-subproject closed | [`ship/apertus_greek_modern_polytonic_148992/`](ship/apertus_greek_modern_polytonic_148992/) — vocab 148,992 = 256 × 582, modern 17,408 + polytonic **512**, `tokenizer.json` sha256 `bbb08e71…`, `release_audit.json` status `passed`. This is the tokenizer subprojects 07/08 used, and it retires the 153,600 bundle to historical status | [`../03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md`](../03_4_implementation_experiments/polytonic_cutoff_probe/PRODUCTION_DECISION_20260729.md) |

## Outcome

- **Three ship tokenizers**, all `AutoTokenizer`-loadable and byte-identical to Apertus on IDs 0–131,071: 148,480 (used by the bakeoff's extended arms), 153,600 (never trained — the polytonic specialization run was postponed and then superseded), 148,992 (the production tokenizer from 2026-07-29).
- **Working Clariden access**, which every subsequent subproject inherited.
- **A resolved framework question** — Megatron-LM-Swiss-AI — that decided the shape of all later runs.
- **One decision deliberately not made:** the §5.6 / Node-4 numerical thresholds. Review checkpoint C was left for later and never closed; that is the single largest methodological gap of the whole subproject ([`../CPT_MASTER_20260526.md`](../CPT_MASTER_20260526.md) §5.1 D1).
- **A planned deliverable that never landed:** `PLAN_DIFF.md`, the set of edits to fold back into the experiment plan after sign-off. v0.7 superseded the need.

## Where things are

| What | Where |
|---|---|
| Ship tokenizer bundles | [`ship/`](ship/) — three bundles, each with `manifest.json` + SHA-256s |
| Tokenizer rebuild + verification | [`scripts/build_and_verify_ship_tokenizer.py`](scripts/build_and_verify_ship_tokenizer.py) |
| NFC enforcement (V9) used by the corpus build | [`scripts/verify_and_normalize_nfc.py`](scripts/verify_and_normalize_nfc.py) |
| Polytonic budget verifier | [`scripts/verify_polytonic_budget.py`](scripts/verify_polytonic_budget.py) |
| Clariden auth runbook | [`CSCS_AUTH_WORKFLOW.md`](CSCS_AUTH_WORKFLOW.md) |

## Working documents

All superseded; kept for the path-of-arrival.

- **State audits / plans:** [`ANALYSIS.md`](ANALYSIS.md) (the main 2026-05-20 deliverable; its body still says "still open" in places the header banner marks resolved — trust the banner), [`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md), [`REPLAY_LANGUAGE_SELECTION.md`](REPLAY_LANGUAGE_SELECTION.md) (34 languages; v0.7 shipped 24).
- **Reviewer material:** [`REVIEW_PACKET.md`](REVIEW_PACKET.md) — assembled under v0.5/v0.6 framing, banner-corrected twice; its "composite 153,600 is the active CPT base" line is wrong for the bakeoff that actually ran.
- **Verification notes:** [`SHIP_TOKENIZER_RECONSTRUCTION.md`](SHIP_TOKENIZER_RECONSTRUCTION.md) (carries a 2026-07-29 production-update banner recovered from the working tree), [`POLYTONIC_VOCAB_BUDGET_CHECK.md`](POLYTONIC_VOCAB_BUDGET_CHECK.md).
- **Late arrival:** `ship/apertus_greek_modern_polytonic_148992/` was never committed during the work; it was recovered from the owner's working tree on 2026-09-01 (`2aec4a66`).
