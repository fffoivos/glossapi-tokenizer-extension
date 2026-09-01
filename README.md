# train-apertus-with-glossapi — Greek continued pretraining of Apertus-8B

This repository is the record of a five-month program (2026-04-10 → 2026-08-24)
that took `swiss-ai/Apertus-8B-2509`, gave it a Greek-extended tokenizer, and
continued its pretraining on a Greek corpus built from GlossAPI and HPLT on the
CSCS Clariden supercomputer. Almost all of the work was carried out by AI coding
agents under the owner's direction, in numbered subprojects under `subprojects/`.

**Where it ended.** A production tokenizer of 148,992 tokens (Apertus's 131,072
plus 17,408 modern-Greek and 512 polytonic BPE units); a 51.8 M-document /
63.8 B-token cleaned Greek corpus (`fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`);
one full 8B continued-pretraining run of 76.685 B active tokens (18,284 updates,
completed 2026-08-11) whose GreekMMLU went 35.78 % → **56.81 % at update 9,536**
→ 54.86 % at the end, with 19 scored checkpoints released to Hugging Face and
a checkpoint average (`avg_peak5`) as the default revision; a matched 8B + 1.5B
curriculum replication that did *not* reproduce an earlier 59.96 % headline; and a
post-training survey that stopped at the pilot stage. Every number here is
sourced in the subproject READMEs linked below.

## How to read this repository

Every subproject and sub-subproject has a `README.md` written as a **history**:
why it existed, a dated account of what happened (including reversals and dead
ends), the outcome with numbers and their sources, where the load-bearing
artifacts are, and a list of the remaining working documents marked as
historical. Nothing was deleted; the READMEs make the clutter navigable.

- Start with the table below, then the README of the subproject you care about.
- [`docs/PROJECT_INDEX.md`](docs/PROJECT_INDEX.md) says what each root-level document is and whether it still governs anything. Two rules still do: [`docs/LOSS_MEASUREMENT_POLICY.md`](docs/LOSS_MEASUREMENT_POLICY.md) and the frozen training config [`subprojects/CURRENT_HYPERPARAMETERS.md`](subprojects/CURRENT_HYPERPARAMETERS.md).
- Old status snapshots (`docs/CURRENT_STATUS.md`, `docs/ACTIVE_BACKLOG.md`, `subprojects/SUBPROJECTS_OVERVIEW.md`) carry a banner and are kept as records of what was planned at the time.
- Paths under `/capstor/…` and `/iopsstor/…` are CSCS Clariden storage, not in git. Run receipts, checkpoints and corpora live there or on Hugging Face.

## The program, in order

| # | Subproject | Period | Question it answered | How it ended |
|---|---|---|---|---|
| 01 | [`_archive/01_*`](subprojects/_archive/README.md) — corpus phase | 2026-04-10 → 05-14 | Can a clean, deduplicated Greek corpus be built for tokenizer training? | Yes: HPLT clean60 slice, full dedup, the 50/50 `mix.parquet` C3 trained on. Archived when C3 converged. |
| 02 | [`02_apertus_tokenizer_spec`](subprojects/02_apertus_tokenizer_spec/README.md) | 04-10 → 04-14 | What exactly must an extension preserve? | The Apertus tokenizer contract (131,072 ids, first-1,000 block, regex + ByteLevel, untied embeddings). |
| 02.1 | [`02_1_tokenizer_experiments`](subprojects/02_1_tokenizer_experiments/README.md) | 04-10 → 06-11 (+07-29) | How many and which Greek BPE units to append? | Four-arm exploration closed by fiat for C3; sweep fixed 17,408 curated units (vocab 148,480); polytonic arm later added +512 → 148,992. |
| 02.2 | [`02_2_tokenizer_implementation`](subprojects/02_2_tokenizer_implementation/README.md) | 04-10 → 05-18 | Which language does each Apertus token belong to? | Became the per-token language-attribution machinery (86.35 % of the vocab attributed); the merge-rule implementation it was named for happened elsewhere. |
| 03 | [`03_apertus_extension_and_embedding_adaptation`](subprojects/03_apertus_extension_and_embedding_adaptation/README.md) | 04-10 → 08-21 | How to initialise the new embedding rows, and does the extension help? | Four-arm init bakeoff to 5 B tokens; TD-layer11 vs Vanilla was a close call with no pre-committed thresholds; every arm lost native Greek vs base — which became 04's question. Polytonic cutoff decided 07-29. |
| 04 | [`04_cpt_training_regime_on_vanilla`](subprojects/04_cpt_training_regime_on_vanilla/README.md) | 05-28 → 06-11 | Was the bakeoff's training *regime* the cause of the Greek loss? | Yes: an Apertus-faithful regime on unmodified Apertus reached 0.4973 native-Greek MCQ (+6.69 pp over the bakeoff's Vanilla), and a geometry probe locked Path-A RoPE for Task 2. |
| 05 | [`05_token_distillation_cpt`](subprojects/05_token_distillation_cpt/README.md) | 06-10 → 08-12 | Does the extended tokenizer + TD init win at scale, and with what recipe? | 13.5 B two-arm pilot: 48.3 % base → 55.3 % vanilla → 58.7 % TD; five sweeps froze the recipe on 07-11; the v2 corpus was built (05/04) and the shard bridge written (05/05). |
| 06 | [`06_dataset_scheduling_experiments`](subprojects/06_dataset_scheduling_experiments/README.md) | 08-01 → 08-05 | Does the temporal order of HPLT vs GlossAPI matter? | Five 0.5B arms: order matters and trades off; stationary D0 accepted on point estimate for the 8B run. |
| 07 | [`07_full_8b_cpt`](subprojects/07_full_8b_cpt/README.md) | 08-05 → 08-11 | The production run. | First trajectory stopped at update 7,152 for a missing PII pass; sanitized rerun completed 08-11 (76.685 B tokens); GreekMMLU peak 56.81 % at update 9,536. |
| 08 | [`08_targeted_8b_cpt_experiments`](subprojects/08_targeted_8b_cpt_experiments/README.md) | 08-11 → 08-23 | Does a hard HPLT→OpenArchives curriculum replicate, and does 1.5B predict 8B? | Neither: the 59.96 % headline missed by 2.06 pp; 1.5B and 8B moved in opposite directions; an exploratory no-decay branch was stopped by its pre-registered gate. |
| 09 | [`09_full_8b_cpt_results_analysis`](subprojects/09_full_8b_cpt_results_analysis/README.md) | 08-12 → 08-23 | What did the 8B run actually learn and forget? | Peak-then-decline on GreekMMLU while BPB kept improving; the 0.5B proxy rejected; `avg_peak5` checkpoint average ties the peak and wins retention; 19 checkpoints released (09_2). |
| 10 | [`10_early_cooldown_causal_experiment`](subprojects/10_early_cooldown_causal_experiment/README.md) | 08-12 → 08-14 | Does cooldown timing *cause* the 9,536 peak? | Abandoned: five launch attempts, none trained; the restart-invariant lessons are the output. |
| 11 | [`11_greek_posttraining`](subprojects/11_greek_posttraining/README.md) | 08-23 → 08-24 | What would SFT of the Greek checkpoint take? | Survey + two 100-row pilots; Greek SFT is greenfield; no corpus built, no SFT trained. (Renumbered from `10_` on 2026-09-01 — it collided with the row above.) |
| — | [`papers`](subprojects/papers/README.md) | 06-11 | Do the cited papers say what the recipe cites them for? | 18 of 40 citations not fully confirmed, two contradicted; resolved by the sweeps rather than by citation. |

## Chronology of the program

**April 2026 — corpus and tokenizer groundwork.** The repo was created on
2026-04-10 (`f21eed85`, then named *GlossAPI Tokenizer Extension*) around a
corpus pipeline (`glossapi_corpus_cli/`): HPLT filtering, dedup repair, a
GlossAPI + HPLT mix builder, and four waves of cleaner iteration
([01](subprojects/_archive/README.md)). In parallel the Apertus tokenizer
contract was pinned ([02](subprojects/02_apertus_tokenizer_spec/README.md)) and
a four-arm tokenizer experiment matrix (F1/F2 fresh BPE, C1/C2 continuous BPE)
was launched ([02.1](subprojects/02_1_tokenizer_experiments/README.md)).

**May 2026 — convergence, attribution, and the bakeoff.** On 2026-05-11 the
tokenizer track converged on the C3 arm by decision rather than by finishing
the matrix ([docs/C3_CONVERGENCE.md](docs/C3_CONVERGENCE.md)); a 0→25,600
cutoff sweep on the swiss-ai TokEval suite fixed 17,408 curated units
(Greek fertility 2.41 → 1.345, vocab 148,480) on 05-18. The per-token
language-attribution run (1,933 languages, 113.4 B firings) gave the vocabulary
its language map ([02.2](subprojects/02_2_tokenizer_implementation/README.md)).
On CSCS Clariden, four embedding-initialisation arms (Vanilla / ReTok / Centroid /
TD-layer11) were trained to 2 B → 3.5 B → 5 B tokens; the 2 B headline
(Vanilla wins) flipped at 5 B on downstream aggregates and flipped back on a
vetted native-Greek suite, with base Apertus above every continued arm
([03](subprojects/03_apertus_extension_and_embedding_adaptation/README.md)).
Artifacts were published to `fffoivos/apertus-tokenizer-extension` on 05-25.
Because that loss hit even the unmodified Vanilla arm, a 5 B-token regime
diagnostic ran from 05-28: the Apertus-faithful regime (Goldfish loss, AdEMAMix,
1.2 B-token warmup) recovered native Greek, and a geometry probe showed the
bakeoff's RoPE override was itself costing ~1 B tokens of re-adaptation
([04](subprojects/04_cpt_training_regime_on_vanilla/README.md)).

**June–July 2026 — Task 2: recipe, corpus, and the bibliography detour.** The
13.5 B two-arm pilot (06-10/11) showed the extended-tokenizer + TD arm ahead of
vanilla on native Greek MCQ (58.7 % vs 55.3 %) at tied bits/byte and −31 %
tokens; five 13.5 B sweeps (replay, LR, α, β₃, β₂) froze the production recipe on
07-11 ([05](subprojects/05_token_distillation_cpt/README.md),
[`CURRENT_HYPERPARAMETERS.md`](subprojects/CURRENT_HYPERPARAMETERS.md)). Corpus
preparation ran alongside — an HPLT cleaning audit that promoted no new rule, a
decontamination redesign, anonymisation parity, and a six-week academic
bibliography-removal track that ended in a receipt-bound dry run stopped at the
apply boundary ([05/02](subprojects/05_token_distillation_cpt/02_corpus_preparation/README.md)).
The full v2 corpus was built and deduplicated on Clariden in the second half of
July (53.0 M → 51.8 M documents; [05/04](subprojects/05_token_distillation_cpt/04_full_corpus_preparation/README.md)),
and on 07-29 a pre-committed probe added +512 polytonic merges to the tokenizer
(vocab 148,992). Much of July's work ran on parallel `codex/*` branches that were
consolidated on 07-22.

**August 2026 — the 8B run and what it taught.** Five 0.5B data-order arms
(08-01 → 08-05) settled on stationary mixing ([06](subprojects/06_dataset_scheduling_experiments/README.md)).
The full 8B run launched on 08-05, was stopped at update 7,152 because the
corpus had never had Apertus's PII pass, and was rerun sanitized to completion
on 08-11 ([07](subprojects/07_full_8b_cpt/README.md)). Results analysis (08-12 →
08-23) found GreekMMLU peaking mid-run at update 9,536 while loss kept
improving, rejected the 0.5B proxy, and found a five-checkpoint average that
ties the peak and wins every retention task ([09](subprojects/09_full_8b_cpt_results_analysis/README.md)).
A causal early-cooldown branch from that peak never trained
([10](subprojects/10_early_cooldown_causal_experiment/README.md)); a matched
8B + 1.5B replication of the historical hard HPLT→OpenArchives curriculum
(08-11 → 08-23) failed to reproduce its headline and showed 1.5B does not mirror
8B ([08](subprojects/08_targeted_8b_cpt_experiments/README.md)). The program
closed with a post-training survey and two SFT pilots on 08-23/24
([11_greek_posttraining](subprojects/11_greek_posttraining/README.md)).

## Subproject summaries

Each block is condensed from the subproject's README, which carries the dated
history and the sources.

### 01 — Corpus phase (`subprojects/_archive/01_*`) — 2026-04-10 → 05-14 — completed
Four subprojects built the tokenizer-training corpus: the HPLT clean60 slice (48,728,774 rows), four waves of tokenizer-guided cleaner iteration, one full-corpus dedup (49,292,755 → 49,090,905 kept), and the 50/50 `mix.parquet` (14,453,413 rows / 104.94 B chars) that C3 trained on. Archived on 05-14 when C3 converged; its quality gates and its known splitter row-vs-doc leak both carried into 02.1's cutoff sweep. → [README](subprojects/_archive/README.md)

### 02 — Apertus tokenizer spec — 2026-04-10 → 04-14 — completed
Pinned the exact Apertus-8B-2509 tokenizer contract any extension had to reproduce. Two files, written once; the contract became the acceptance test downstream (0 mismatches across ids 0..999 in ship-bundle verification). → [README](subprojects/02_apertus_tokenizer_spec/README.md)

### 02.1 — Tokenizer experiments — 2026-04-10 → 06-11 (epilogue 07-29) — completed
The four-arm exploration was closed without completing when C3 (continuous BPE, GlossAPI+HPLT 50/50) was declared the arm on 05-11; a 0→25,600 sweep fixed the cutoff at 17,408 added units — Greek fertility 2.41 → 1.345 (−44.2 %), 82.4 % of the achievable gain, vocab 148,480 (`02_1_7/CHOSEN_CUTOFF.md`). 69 noise tokens are structurally skipped and backfilled. An earlier analytic answer (11,264) was superseded. The polytonic arm recommended +5,120, but the 07-29 production probe froze +512 → vocab 148,992, the tokenizer every CPT run from 05 onward uses. → [README](subprojects/02_1_tokenizer_experiments/README.md)

### 02.2 — Tokenizer implementation — 2026-04-10 → 05-18 — attribution track completed
Never held the merge-rule extension it was named for (variants were built in 02.1.2, ship bundles in 03.3). It became the language-attribution machinery: a 05-13 run tokenising ~1 B tokens for each of 1,933 languages (113.4 B firings, ~$100), joined to CLDR char-admissibility masks by PMI promotion. 113,184 of 131,072 tokens (86.35 %) attributed; its outputs anchored the cutoff recommendation, the eval-language selection, and the 03.1 embedding diagnostics. → [README](subprojects/02_2_tokenizer_implementation/README.md)

### 03 — Apertus extension and embedding adaptation — 2026-04-10 → 08-21 — completed
Four init arms on an identical 70/24/4/2 mix to 5 B tokens: Centroid broken (BPB 0.8994), ReTok dominated; at 5 B TD-layer11 led all three downstream aggregates while Vanilla kept tokenizer-fair BPB (0.4602 vs 0.4872) — but a vetted native-Greek suite reversed the Greek headline to Vanilla (0.4305 vs 0.4109) with base Apertus above every arm at 0.4817. No thresholds had been pre-committed, so the bakeoff produced data, not a winner; Vanilla stayed the safe default and the universal Greek loss became 04's question, while TD went to 05. A properly pre-committed probe on 07-29 chose +512 polytonic merges. → [README](subprojects/03_apertus_extension_and_embedding_adaptation/README.md)

### 04 — CPT training regime on Vanilla — 2026-05-28 → 06-11 — completed
A 5 B-token CPT of unmodified Apertus-8B under the Apertus-faithful regime (Goldfish k=h=50, LR 1.1e-5 with 1.2 B-token warmup, AdEMAMix β₃=0.99): iter 1192 native-Greek MCQ 0.4973 [0.4779, 0.5156], +6.69 pp over the bakeoff's Vanilla-5B and +1.56 pp over base, all CIs clear of zero. A 0.5 B probe then showed the bakeoff's RoPE geometry cost ~1 B tokens of re-adaptation (+5.51 pp at matched tokens), locking Path A. 217.2 GPU-h; regime, geometry, sidecar-eval pattern and adversarial-review runner inherited by 05. → [README](subprojects/04_cpt_training_regime_on_vanilla/README.md)

### 05 — Token-Distillation CPT (Task 2) — 2026-06-10 → 08-12 — completed / superseded by 07
The 13.5 B two-arm pilot: native Greek MCQ 48.3 % base → 55.3 % vanilla → 58.7 % TD over 18,489 questions, tied bits/byte, −31 % tokens per unit of Greek. Five 13.5 B sweeps froze the recipe on 07-11 (replay 79/20/1, peak LR 5.5e-5, AdEMAMix α=4 / β₃=0.999 / β₂=0.999, 400-iteration warmup). The 25 B probe built to gate the big run was frozen but never trained; its dataset stage and TD init fed the LR-floor reconstruction and the 8B run. Sub-tracks: [corpus preparation](subprojects/05_token_distillation_cpt/02_corpus_preparation/README.md) (cleaning audit, bibliography removal, decontamination, anonymisation — mostly measured no-ops for the CPT run), [full-corpus v2 build](subprojects/05_token_distillation_cpt/04_full_corpus_preparation/README.md) (53,046,533 → 51,839,746 documents / 63.78 B tokens, published to HF), [training-dataset bridge](subprojects/05_token_distillation_cpt/05_training_dataset_bridge/README.md) (the shard builder every later subproject imports). → [README](subprojects/05_token_distillation_cpt/README.md)

### 06 — Dataset scheduling experiments — 2026-08-01 → 08-05 — completed
Five 0.5B trajectories (D0 stationary; D1/D2 hard and D3/D4 gradual HPLT↔GlossAPI orders) over an identical 80.73 B-token stream, all to update 38,496. Order matters and trades off (D1 wins curated GlossAPI, +0.0473 BPB on HPLT; D2 the reverse, +0.0855 on polytonic). D0 led the predeclared hierarchy but the receipt records `winner_selected: false` (no document-cluster intervals); the owner accepted D0 on point estimate. → [README](subprojects/06_dataset_scheduling_experiments/README.md)

### 07 — Full Apertus-8B mixed CPT — 2026-08-05 → 08-11 — completed
D0 79/20/1 mix, 148,992-token tokenizer, untied layer-11 TD init, WSD-10 peak 5.5e-5. The first trajectory was stopped at update 7,152 because the corpus had never had the Apertus PII pass; masked, OCR-flagged rows excluded and globally deduplicated, the horizon reset from 19,248 updates / 80.73 B to 18,284 / 76.685 B. DP64 rejected on drift, DP32 selected; the rerun completed 08-11 with zero non-finite updates. GreekMMLU 35.782 % → 56.810 % (update 9,536) → 54.855 % on the 16,159-question clean subset while Greek BPB kept improving. → [README](subprojects/07_full_8b_cpt/README.md)

### 08 — Targeted 8B CPT experiments — 2026-08-11 → 08-23 — completed
Two planned continuations (A: academic + polytonic mixture; B: from update 9,536) were shelved and replaced by a matched 8B + 1.5B replication of the historical hard HPLT→OpenArchives curriculum, hardened by two 12-agent reviews. Both trained to update 3,694: the historical 59.9627 % GreekMMLU was not replicated (best 57.9004 %, a 2.06 pp miss outside a ±1.0 pp band ratified in advance); 1.5B does not mirror 8B (deltas −3.09 pp vs +1.27 pp); a constant-LR branch from iter 2,499 fell across every interval and was stopped by its pre-registered gate. → [README](subprojects/08_targeted_8b_cpt_experiments/README.md)

### 09 — Full 8B CPT results analysis — 2026-08-12 → 08-23 — completed
GreekMMLU peaks at update 9,536 = 56.81 % (init 35.78 %, terminal 54.85 %) across all 19 checkpoints while per-document BPB keeps improving; the best checkpoint differs per task; the 0.5B proxy was rejected (late-trajectory direction agreement 3/9). Decided: never select from GreekMMLU alone; preserve 9,536. Later, unreviewed work found `avg_peak5` (uniform mean of iters 7,152–11,920) ties the peak (56.78 %) and beats it on all nine retention tasks; it became the HF default revision. Sub-tracks: 09_1 downstream instability, 09_2 checkpoint-trajectory release, 09_3 added-token adaptation audit (adaptation is monotone through update 18,284, so the vocabulary extension does not explain the peak). → [README](subprojects/09_full_8b_cpt_results_analysis/README.md)

### 10 — Early-cooldown causal experiment — 2026-08-12 → 08-14 — abandoned
Would branch from the update-9,536 checkpoint and start the 3,657-update cooldown immediately with everything else fixed. It never trained: two restarts proved a historical gradient norm is not a cross-allocation invariant, the paired gate rejected a probe over one display quantum (2.011 vs 2.010), one launch died on a Slurm `--switches` relaxation, and the final sandwich run failed its control gate nine minutes in. 6.08 GPU-h, zero scientific updates. → [README](subprojects/10_early_cooldown_causal_experiment/README.md)

### 11 — Greek post-training — 2026-08-23 → 08-24 — ended at the pilot stage
Found Greek SFT to be greenfield (~3.5 k reusable quality rows anywhere), sized a minimal mix (~62 k rows) whose binding constraint is human review, and ran two 100-row classify → translate → generate pilots on no_robots that passed deterministic gates 99/100 while proving those gates blind to naturalness. Pilot 2 was completed but never reviewed; no corpus built, no SFT trained. → [README](subprojects/11_greek_posttraining/README.md)

### papers — reading library and citation audit — 2026-06-11 — completed
Twenty papers read in full and audited claim-by-claim against the frozen recipe: 18 of 40 citations not fully confirmed, two contradicted. The tensions were settled empirically by the sweeps. → [README](subprojects/papers/README.md)

## Where the artifacts are

| Artifact | Location | Documented in |
|---|---|---|
| Production tokenizer (148,992) | `subprojects/03_apertus_extension_and_embedding_adaptation/03_3_cscs_experiments_kickoff/ship/apertus_greek_modern_polytonic_148992/`; HF `fffoivos/apertus-tokenizer-extension` (`greek-modern-polytonic-tokenizer`) | [03.4 polytonic probe](subprojects/03_apertus_extension_and_embedding_adaptation/03_4_implementation_experiments/polytonic_cutoff_probe/README.md) |
| Bakeoff checkpoints, tokenizer (148,480), evals | HF `fffoivos/apertus-tokenizer-extension`; local mirror `release/` | [03](subprojects/03_apertus_extension_and_embedding_adaptation/README.md), [docs/PROJECT_INDEX.md](docs/PROJECT_INDEX.md) |
| Cleaned Greek corpus v2 | HF `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2` (revisions pinned per subproject) | [05/04](subprojects/05_token_distillation_cpt/04_full_corpus_preparation/README.md) |
| 8B CPT checkpoints (19 scored) + `avg_peak5` | Private HF branches per 09_2; CSCS `/capstor/scratch/cscs/fffoivos/runs/07_full_8b_cpt/…` | [09_2](subprojects/09_full_8b_cpt_results_analysis/09_2_checkpoint_trajectory_release/README.md) |
| Frozen training configuration | [`subprojects/CURRENT_HYPERPARAMETERS.md`](subprojects/CURRENT_HYPERPARAMETERS.md) | [05](subprojects/05_token_distillation_cpt/README.md) |
| Results presentations (self-contained HTML + JSON) | `subprojects/0{7,8,9}_*/presentations/` | each subproject README |

## Repository map

| Path | What it is |
|---|---|
| `subprojects/` | The numbered subprojects above; each with its own history README. |
| `docs/` | Root-level decision records, policies and (bannered) status snapshots — see [`docs/PROJECT_INDEX.md`](docs/PROJECT_INDEX.md). |
| `glossapi_corpus_cli/` | The April corpus pipeline (continuous BPE, text dedup, pipeline driver) used by 01/02.1. |
| `legacy/` | The pre-April `add_tokens(...)` baseline and exploratory HPLT/cleaning material (2026-04-10 → 05-14). |
| `ops/` | Worker orchestration, smoke and efficiency harnesses, HF upload handoff for the April–May phase. |
| `tests/` | Repo-level tests (pipeline contracts; bibliography-model tests from 05/02). |
| `release/` | Local mirror of the `fffoivos/apertus-tokenizer-extension` HF layout (May 2026). |
| `tokenizer_analysis/` | Wave-2/wave-4 tokenizer artifact analyses (April–May). |
| `config/apertus_greek_extension.yaml` | Machine-readable extension config from the tokenizer phase. |
| `outputs/` | One July bibliography header-mask audit output. |
| `*.py` at root | HF release assembly/publish scripts and a PDF re-evaluation script from the April–May phase. |

## Provenance of this history (consolidated 2026-09-01)

The history above is only visible because the branches were brought together.
Before 2026-09-01, GitHub's `main` stopped at 2026-06-11 (subprojects 02–05);
the local `main` continued to 2026-07-22; subprojects 06–10 existed only on
`agent/*` branches in two families that never merged (`full8b-production-launch`
→ `replay-reader-v1` / `early-cooldown-causal` / `full8-results-analysis`
carrying 06/07/09/10, and the `h2g-*` family carrying 06/08); and several units
of work — the 2026-07-29 polytonic tokenizer decision and ship bundle,
`05/07_8b_lr_floor_reconstruction`, `09_1`, `09_3`, `08/publication`, all of
`11_greek_posttraining` — existed only as uncommitted files in working trees.

The consolidation branch merges every leaf branch (ten branches; the four
add/add conflicts are documented in the merge commit) and adds one commit that
recovers the uncommitted files with their provenance (`2aec4a66`). Left out on
purpose: `agent/h2g-safe-open-verifier-20260817` (13 commits, 2026-08-17; conflicts
with the later `h2g` line — described in [08](subprojects/08_targeted_8b_cpt_experiments/README.md)),
119 working-tree copies of 06/07 files that differ from the committed versions,
and the root-level `frozen_*` deployment bundles in the `h2g` worktrees.

## Open items

- **Numbering:** `11_greek_posttraining` was `10_greek_posttraining` until 2026-09-01 (it collided with `10_early_cooldown_causal_experiment`); its own documents still say "10".
- **Unreviewed results:** the post-2026-08-19 findings in 09 (`avg_peak5`, ensembles, retention lm-eval) are explicitly not independently reviewed; 09's canonical 19-checkpoint presentation contains no averaged model, so "best model" differs between documents.
- **No floor decision from `05/07_8b_lr_floor_reconstruction`:** its three tails completed with receipts but no comparison or decision was committed, and 06/07 do not cite it although both use WSD-10.
- **Approval receipts outside git:** the 1.5B TD acceptance policy in 08 is still marked `proposal_pending_owner_approval` in-tree though the 1.5B run trained; the approval receipt lives on Clariden.
- **Stale contracts:** `07/configs/recipe_8b_full_mixed.json` still carries the pre-sanitization geometry (19,248 updates / 80.73 B); `06/FACTORIAL_EXPERIMENT_DESIGN.md` still says "not authorized" though it launched.
- **Docs referenced but absent:** e.g. `reports/cpt_curriculum_forgetting_learning.html` (cited by the 05 mix decision), `ERROR_CATALOG.md` (cited throughout `05/02/10_clean_hplt`), `reports/train_logs_cache_5b/` (04). Each is flagged in the relevant README.
- Each subproject README ends with a *Working documents* section listing the plans, snapshots and logs that could be archived; nothing has been moved.
