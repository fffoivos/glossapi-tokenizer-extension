# Review Packet — 03_3 + 03_4

*2026-05-20. Single entry point for a reviewer who has read
[`Apertus_plan.md`](../Apertus_plan.md) and / or
[`experiments_plan.md`](../experiments_plan.md) and now needs to
evaluate the new sub-subprojects.*

> **Updated 2026-05-20 (round 2)** after the first reviewer pass. Five
> findings landed; all are now addressed. See [§"Round-1 review
> response"](#round-1-review-response) at the bottom for the per-finding
> changelog with quotes and links.
>
> **Constraint update 2026-05-20 (later)**: **GCloud access is gone.**
> The CPT corpus build moves from "GCP scratch VM" to "Clariden `xfer`
> partition" (same runbook steps, different host). The held-out
> contamination check via instance restart is no longer available;
> three substitute options at [ANALYSIS.md § Review checkpoint B](ANALYSIS.md#b-held-out-contamination-check-rerun).
> Other gcloud-dependent items (tokenizer merged-variant builds,
> further worker-side validation) are now complete or moved on-cluster.

## What's being submitted

Two new sub-subprojects under `03_apertus_extension_and_embedding_adaptation/`:

- **`03_3_cscs_experiments_kickoff/`** — bridge between planning and execution. Contains the state audit, the polytonic-budget verification, the rebuilt ship-tokenizer bundle, the CSCS auth workflow, and the curriculum + init-corpus decision.
- **`03_4_implementation_experiments/`** — the hands-on subproject. Currently has the auth-and-node probe + first calibration-run sizing. No sbatch submitted yet.

Pre-existing project files are **not edited** (no PR-style diffs to
`experiments_plan.md`, `Apertus_plan.md`, `GLOBAL_DECISIONS.md`,
or any source-of-truth doc). Suggested edits to those files are
parked as **review checkpoints** in
[ANALYSIS.md § 7](ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off);
landing them is gated on this review.

## Reading order (≈ 60 min total)

| # | Doc | Time | Purpose |
|---:|---|---:|---|
| 1 | [`ANALYSIS.md`](ANALYSIS.md) | 20 min | Δ since `experiments_plan.md` v0.12: resolved / partial / open / new directions. The map. |
| 2 | [`SHIP_TOKENIZER_RECONSTRUCTION.md`](SHIP_TOKENIZER_RECONSTRUCTION.md) | 5 min | What's in `ship/apertus_greek_extended_153600/`, why a rebuild was necessary (HF-loadability bug in the polytonic builder's wrapper config), and the structural integrity + fertility-on-real-text verification. |
| 3 | [`POLYTONIC_VOCAB_BUDGET_CHECK.md`](POLYTONIC_VOCAB_BUDGET_CHECK.md) | 10 min | Does the +5,120 polytonic budget make sense? Yes — sits in the middle of the 4,800-7,500 band predicted by the sub-1B-language scaling pattern derived from the May-13 vocab-attribution run. |
| 4 | [`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md) | 15 min | Reconciles `Apertus_plan.md` (colleague's PPL/quality/novelty ranking) with the dedup audit's per-source recommendations and a HPLT-broad foundation. Answers the two questions: (a) "fresh-only vs mixed for init pilots?" → **fresh-only** with explicit replay; (b) "what's the phase ordering?" → HPLT-broad → register diversity → academic+legal → dictionary gap restoration. |
| 5 | [`CSCS_AUTH_WORKFLOW.md`](CSCS_AUTH_WORKFLOW.md) | 5 min | Daily cert refresh using `cscs-key sign --headless --duration 1d`. Verified end-to-end 2026-05-20. |
| 6 | [`../03_4_implementation_experiments/AUTH_AND_NODE_FINDING.md`](../03_4_implementation_experiments/AUTH_AND_NODE_FINDING.md) | 5 min | Live Clariden probe: wait time is reservation-bound (next general window opens tomorrow ~03:33 UTC), QoS allows up to 4 nodes friction-free, recommended calibration shape is 1 node × 4× GH200 × 10 h × 1 B tokens. |

Skip the README.md files — they're indexes for browsing, not narratives.

## Verifications you can re-run

Everything in this packet that touches data or live systems is
reproducible. To re-run my checks:

```bash
# Polytonic vocab-budget pattern (~5 sec on home; no network)
/home/foivos/.venvs/glossapi-merge-docling/bin/python3 \
  scripts/verify_polytonic_budget.py

# Composite tokenizer rebuild + structural + fertility checks (~10 sec on home; uses cached Apertus tokenizer)
/home/foivos/.venvs/glossapi-merge-docling/bin/python3 \
  scripts/build_and_verify_ship_tokenizer.py

# CSCS auth health + cluster probe (requires fresh cert in ~/.ssh/cscs-key-cert.pub)
ssh-keygen -L -f ~/.ssh/cscs-key-cert.pub | grep -E 'Valid|Key ID|Principals'
cscs-key list
ssh ela 'hostname; whoami'
ssh clariden 'sinfo -o "%P %t %D"'
```

## Decisions I'd like your input on

These are the **review checkpoints** from
[ANALYSIS.md § 7](ANALYSIS.md#7-review-checkpoints--what-still-needs-your-explicit-sign-off).
You can mark each one inline in that doc, or in a reply. Each blocks
something concrete downstream:

| | Checkpoint | What blocks until you decide |
|---|---|---|
| A | C3 cutoff (was a checkpoint — **already resolved 2026-05-18: 17,408**; this checkpoint is now informational and just needs you to confirm the resolution before I fold the plan-diff) | Plan-diff into `experiments_plan.md` §2.4, §3 Node 1, §10 Q1, §11. |
| B | Held-out contamination check on C3 val/test (the dedup audit's run skipped this). **GCloud access lost 2026-05-20**, so the "restart the terminated gcloud instance" path is closed. Three remaining options: (1) document the gap and ship; (2) re-derive the val/test partition by re-running the splitter on a Clariden `xfer` allocation with the original seed (~1-2 h, most defensible); (3) substitute `virgin_hplt` slice as the held-out throughout (narrower but cheap). See [ANALYSIS.md § Review checkpoint B](ANALYSIS.md#b-held-out-contamination-check-rerun) for the full option list. | Eval-integrity claim on C3 val/test. Honest reporting requires either the rerun or a documented gap. |
| C | Lock literal values for the §10 Q8 decision rule (X / M_progress / M_ext / M_van / T) and the gate-language list (en / fr / de / ru / it?). Plan's suggested defaults: X=5 %, M_progress=3-5 %, M_ext=1-2 %, M_van=3-5 %, T=2-3 %. | Must be in writing **before any arm completes CPT** (hard temporal constraint per plan §3 Node 4). |
| D | CSCS training-harness pick: Swiss-AI Megatron-LM-Swiss-AI (closest to Apertus's recipe, highest setup) / nanotron (middle) / HF transformers + accelerate (lowest setup, biggest training-dynamics drift). My read: nanotron for the three-arm pilot (the *internal* delta dominates; absolute match to Apertus doesn't), Megatron-LM-Swiss-AI for the post-winner main CPT. | sbatch authoring; one node calibration of the first arm. |
| E | CPT mix ratios + non-Greek replay percentage. Default proposed in [`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md) §2.2: 85/15 Greek/English-replay, periodic Krikri-style mini-replay every 20 steps (every 10 in Phase 3). Greek mix weights from the colleague's ranking + dedup audit. | Final corpus build off the runbook (per [`../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md`](../03_2_apertus_c3_dedup_audit/CPT_DATASET_BUILD_RUNBOOK.md)). |
| F | Whether to formally close the polytonic/Katharevousa question (the colleague's plan + my polytonic-extension reconciliation say the +5,120 stacked layer addresses it; the parent `experiments_plan.md` still lists it as open). Cosmetic if you agree it's covered. | Plan tidy-up. |

## What I executed (so you know what's actually on disk vs proposed)

Executed (artifacts persist):

- **Generated and verified the composite ship tokenizer** at [`ship/apertus_greek_extended_153600/`](ship/apertus_greek_extended_153600/) — 153,600 vocab, loadable via `AutoTokenizer`, polytonic NT compression 60→20 tokens, English/Russian unchanged.
- **Derived the sub-1B-language slope** from the May-13 vocab attribution run; projected to the polytonic 223 M-token corpus; confirmed +5,120 is in-band.
- **Installed `cscs-key 1.1.0`** at `~/.local/bin/cscs-key` + **generated fresh Ed25519 keypair** at `~/.ssh/cscs-key{,.pub}` with explicit comment `cscs-key fffoivos@home 2026-05-20`. (The 19-byte placeholder files that were there got backed up to `.old.20260520`; they were "Permanent Redirect" HTML, not real keys.)
- **Live-probed Clariden**: cert validity / queue depth / partition state / expected start times / QoS limits. Numbers in [`03_4 AUTH_AND_NODE_FINDING.md`](../03_4_implementation_experiments/AUTH_AND_NODE_FINDING.md).
- **Updated parent [`03/README.md`](../README.md)** to index the new sub-subprojects, mark resolved decisions, and link to the open ones.

Proposed (not executed; awaiting your sign-off on the review checkpoints):

- The plan-diff against `experiments_plan.md` (sections to fold in 17,408 cutoff + polytonic layer + diagnostic-v2 sharpening + auth + decision-rule numbers).
- Any sbatch submission (no jobs have run since the March-28 / April-8 smoke tests).
- ~~Restart of the terminated gcloud instance for the held-out contamination rerun~~ — **closed 2026-05-20: GCloud access lost.** The held-out check is now an option set, not a single planned action; see [ANALYSIS.md § Review checkpoint B](ANALYSIS.md#b-held-out-contamination-check-rerun).
- ~~The GCP scratch-VM build of the actual CPT corpus per the runbook~~ — **substituted 2026-05-20: same build steps, Clariden `xfer` partition instead** (256 vCPU, 500 GB RAM, 24 h walltime — ample for the 14.4 M-doc pool).

## What you should send to your reviewer

The simplest send: a single instruction pointing at this file —

> Land at:
> `subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/REVIEW_PACKET.md`
> The reading order is in the doc; the decisions I'd like your input on are in §"Decisions I'd like your input on" at the end.

If the reviewer is off-system, tar the two new sub-subproject
directories — but the tokenizer JSONs are 19 MB each, so consider
excluding the `ship/` binary payloads (the manifest's `sha256` lets a
reviewer verify them against the artifacts in the repo without
needing the bytes):

```bash
cd /home/foivos/Projects/glossapi-tokenizer-extension
tar -czf /tmp/03_3_03_4_review_packet.tgz \
  --exclude='ship/*/tokenizer.json' \
  --exclude='ship/*/tokenizer_config.json' \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff \
  subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments
ls -lh /tmp/03_3_03_4_review_packet.tgz
```

(Resulting tarball ≈ 150 KB — all the markdown plus the verification scripts plus the manifests, no large binary payload.)

## Round-1 review response

Five findings from the first reviewer pass on 2026-05-20. All
addressed in the same session; reviewer cited line numbers shifted
slightly after the edits, but the content claims and verifications
they made are all preserved (the verification scripts still produce
the documented numbers; spot-check with `scripts/`).

### Blocker #1 — 148,480 modern-only tokenizer was not HF-loadable

*Reviewer's finding:* "The referenced `c3_added_17408_curated_padded` still has `tokenizer_class: TokenizersBackend`; `AutoTokenizer.from_pretrained(..., local_files_only=True)` fails with the same wrapper bug described in `SHIP_TOKENIZER_RECONSTRUCTION.md:23`. Only the 153,600 bundle was reconstructed to be loadable."

*Fix:* built a second Apertus-compatible ship bundle for the modern-only variant at [`ship/apertus_greek_modern_only_148480/`](ship/apertus_greek_modern_only_148480/). Same approach as the 153,600 rebuild: copy `tokenizer.json` from the C3 source, replace `tokenizer_config.json` with Apertus's canonical `PreTrainedTokenizerFast` config, copy `special_tokens_map.json`. `scripts/build_and_verify_ship_tokenizer.py` was extended to build and verify **both** bundles in one pass. All structural-integrity + AutoTokenizer-load + fertility checks pass for the new bundle. Cross-references in [`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md) and [`SHIP_TOKENIZER_RECONSTRUCTION.md`](SHIP_TOKENIZER_RECONSTRUCTION.md) now point at the new bundle path, not at the broken `c3_added_17408_curated_padded` directory. Spot-test:

```python
from transformers import AutoTokenizer
tk = AutoTokenizer.from_pretrained("ship/apertus_greek_modern_only_148480")
assert tk.vocab_size == 148480 and type(tk).__name__ == "PreTrainedTokenizerFast"
```

### Blocker #2 — calibration sizing inconsistent

*Reviewer's finding:* "The table says 1 node is ~24k tok/s and 1B tokens takes ~11.6h, but the proposed job is `--time=10:00:00` with target 1B tokens. That will likely walltime before the target. Same section also says 4-node 10B is '~3.6 days each', but the table's 4-node row says 10B is ~1.2 days and 30B is ~3.6 days."

*Fix:* changed [`AUTH_AND_NODE_FINDING.md`](../03_4_implementation_experiments/AUTH_AND_NODE_FINDING.md) §6.2 calibration to `--time=12:00:00` (full partition cap) with the 1 B target unchanged — at 24 k tok/s × 12 h = 1.04 B tokens, the target now fits with a small margin. Added an explicit note that if real throughput is lower than the estimate the job will walltime out at the latest checkpoint (the right failure mode for a calibration run). §6.3 rewritten to distinguish "**~1.2 days per arm** (if run in parallel)" from "**~3.6 days total for all three** (if run in series)", explicit on 12-hour chaining via `afterok`.

### Medium #3 — CSCS auth state internally inconsistent

*Reviewer's finding:* "`ANALYSIS.md:352` says the cert is expired and `ANALYSIS.md:530` says CSCS auth was not exercised. But `AUTH_AND_NODE_FINDING.md:6` says auth is verified working."

*Fix:* `ANALYSIS.md` §5 rewritten to point at [`CSCS_AUTH_WORKFLOW.md`](CSCS_AUTH_WORKFLOW.md) as the canonical workflow doc, with a short "current state (verified 2026-05-20)" callout showing the cert window + the `ssh ela / clariden` smoke results. §8 line about "auth not exercised" updated to note that read-only probes were exercised, no `sbatch` has been submitted.

### Medium #4 — pool-size argument mixed pre/post internal-dedup

*Reviewer's finding:* "`CURRICULUM_AND_INIT_CORPUS.md:20` argues from ~96 M fresh docs, while the actual build path immediately applies `drop_intra_and_inter`, and `ANALYSIS.md:210` says that drops to ~14.4 M docs. The recommendation may still be right, but the 10 B/arm and 20-40 B claims should be backed by post-dedup token counts, not pre-dedup HF-pool docs."

*Fix:* [`CURRICULUM_AND_INIT_CORPUS.md`](CURRICULUM_AND_INIT_CORPUS.md) §1 reworked. Added an explicit three-state framing (raw 98.2 M HF pool → 95.98 M after Apertus-overlap drop → ~14.4 M after internal `drop_intra_and_inter`). Token-budget claim now anchors on the post-internal-dedup pool: ~14.4 M docs × ~100 B chars (from `C3_TRAINING_DATASETS.md`) ÷ chars/token = **~26 B tokens at the extended 148,480 tokenizer, ~38 B at base Apertus**. Notes that 3 arms × 10 B = 30 B is ~80 % of one epoch at the extended tokenizer (tight but viable for pilots).

### Low #5 — polytonic byte-size typo

*Reviewer's finding:* "The TL;DR says 18,716 rows / 510,571,970 chars / '802 MB UTF-8', but the table and script show 1,001,289,826 bytes for all kept rows. The 802 MB number is the train split only."

*Fix:* [`POLYTONIC_VOCAB_BUDGET_CHECK.md`](POLYTONIC_VOCAB_BUDGET_CHECK.md) TL;DR row corrected to **1,001 MB UTF-8**, with a parenthetical noting that the 802 MB number is train-split-only. The supporting table at §1 and the verification script both already report the correct values (the bug was in the TL;DR row alone).

### What the reviewer checked that still holds

The reviewer's "what checked out" section reported clean reproductions
of: (a) 153,600 ship-bundle manifest sha256s, (b) `AutoTokenizer` load
returning `PreTrainedTokenizerFast` with vocab 153,600, (c) special-token
ids 0/1/2/3, (d) base/C3/poly id-range checks, (e) the polytonic-budget
fit `0.1341 × tokens^0.5688 / n=194 / 4,793-7,516 projection`. None of
those are touched by the round-1 fixes — re-running the two scripts now
will reproduce the same numbers plus the second (modern-only 148,480)
bundle's checks. No files were changed for the reviewer's run because
the reviewer did not need to write.
