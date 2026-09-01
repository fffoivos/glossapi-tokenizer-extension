# 15 — Clean academic (reference / bibliography removal)

> **In one line:** six weeks of building a bibliography remover for the Greek academic CPT sources — from a 25-agent method study, through four generations of classifier and two sealed annotation cohorts, to a byte-verified Rust port and a receipt-bound 202,792-document dry run that removed 2.93 B characters (6.06%) and passed a 209-item human QA gate.
> **Period:** 2026-06-13 → 2026-07-28 (commits `056396fd` → `6f12e9d5`), plus a handover document recovered from an uncommitted worktree on 2026-09-01 (`2aec4a66`). **Status:** completed to **dry-run + QA**; the apply contract was authorized on 2026-07-28 but no apply, materialization, token count or publication receipt exists in this tree.
> **Came from / led to:** [`../10_clean_hplt`](../10_clean_hplt/README.md) → this → [`../20_dedup`](../20_dedup/README.md); production orchestration in [`../../04_full_corpus_preparation`](../../04_full_corpus_preparation/README.md)

## Why this existed

The ~60 B-token CPT mix draws academic Greek from eight sources (originally four:
openarchives.gr, greek_phd, Apothetirio_Kallipos, Apothetirio_Pergamos). The GlossAPI cleaner
had **no** reference or citation handling — verified by grep, not assumed — so bibliographies,
reference lists and citation apparatus were going into pretraining as-is. This stage was
net-new: decide whether the scholarly-corpus methods (peS2o/GROBID, MEDITRON, S2ORC) transfer
to Greek OCR'd theses, then build and validate whatever does.

## History

### 2026-06-13 — the method study, and the finding that reframed everything

A 25-agent investigation ([`REFERENCE_CLEANING_INVESTIGATION.md`](REFERENCE_CLEANING_INVESTIGATION.md),
raw output in [`investigation/`](investigation/README.md)) concluded **REMOVE-by-segmentation**;
STRUCTURE (S2ORC links, MEDITRON summaries) and MASK-to-`<ref>` were both rejected — no
downstream consumer, marker→entry links unrecoverable from OCR, and injecting a new mid-prose
token into the existing Apertus tokenizer is a fertility risk. The load-bearing finding, verified
on greek_phd doc 006: **the dominant reference sink in humanities theses is the body-interleaved
footnote stream, not the end list** — 60,837 chars of end bibliography against 586 footnotes of
which 429 are prose/hybrid carrying 201,290 chars of genuine Greek commentary. So a literal
MEDITRON end-matter cut misses ~5× the mass, and blanket footnote removal destroys real text.
The ano-teleia separator was counted at 983 occurrences of U+0387 with **zero** U+00B7 — the two
are never folded. A Rust `reference_detector` crate was built the same day (DETECT → SEGMENT →
EMIT SPANS → COUNT, never hard-delete, fail-closed) and run at corpus scale over all four raw
sources: bib header found in 80% of greek_phd / 66% of openarchives docs; β-gate keeps 29%
(Kallipos) / 36% (Pergamos) of `predicted_section==β` as *non*-bibliography, confirming a blind
β-drop would have removed that much real content. A production-text run (Clariden job 2527626)
then proved the central architectural split: on the doc-level `selected` parquet, whole-doc
detection finds a header in 0% of Kallipos and 1% of Pergamos, because their bibliographies are
per-chapter — they must use the section-β path.

### 2026-06-16 — the span pivot: one boundary is not enough

The end-boundary detector was measured at precision 0.986 / recall 0.619 and, on a
goal-oriented window eval, tail recall 0.80 against **body recall 0.46** — end-of-chapter
bibliographies were structurally invisible. A 3,186-span / 1,581-doc Opus dataset
(`9146721d`) showed `end_of_chapter` (1,153) ≈ `end_of_document` (1,122) and **41% of documents
carrying ≥2 spans**. A 22-feature line-level logistic regression plus a hysteresis span decoder
replaced the header→EOF rule: **prose amputation 51.9% → 5.6%**, paired-bootstrap ΔFβ0.5 +0.32
[+0.27, +0.37] (`49b698dd`). The operating point was then re-tuned away from strict
bibliography precision toward **prose protection**, recovering ~30 points of recall
(recall 0.59 → 0.87 at 99.8% prose protection, `66223fac`); an independent Opus audit of 353
removed lines put genuine prose removal at ~0.14% and found **67% of apparent false positives
were bibliography the annotation had missed** (`bec847fd`). Deployed to Rust with parity
verified at max per-line |Δp| 2.4e-5 and 60/60 identical decoded spans (`3de740fd`).

### 2026-06-19 — two heads, and a change of annotator

A 2,000-document structural gold set (`STRUCT_2K`) was annotated by **gpt-5.5 via codex**
rather than Opus — a bake-off gave κ 0.937, accuracy 0.984, zero prose eaten, and it recovered
7 bibliographies Opus had missed, while Opus weekly usage sat at ~82% (`2a148a9b`). Two binary
per-line heads (bib over 22 features; ToC over 27, front-gated) were trained and validated:
bib recall 0.876 at line-precision 0.97 (vs 0.857 deployed), ToC 0.972 precision / 0.602 recall
after smoothing (`15946f88`, `e3f24b9d`). Details and the full reversal list are in
[`eval/README.md`](eval/README.md).

### 2026-07-11 → 07-12 — hardening, and the honesty pass

`30f1fd71` hardened structural detection and `103ac2ec` added a leak-free ToC/bibliography
sequence evaluation. Then `e7236f48` — *"Reconstruct sequence supervision as explicit LLM
silver"* — retro-labelled the whole 2,000-item set: it is LLM silver, never human gold, despite
the `STRUCT_2K_gold.jsonl` filename. Eight further commits bound the classifier to the source
registry and receipts (`01cba0ee`, `09208335`, `a34f8af6`, `31a06901`, `9014a705`, `074aa621`,
`78be23a5`). The consequence, stated in the old README and in
[`../../04_full_corpus_preparation`](../../04_full_corpus_preparation/README.md): for the CPT
run the tracked cleaning policy stayed **`audit_only`**, both structural flags false, and
Stage58 a deterministic no-op.

### 2026-07-13 → 07-20 — the sequence-model program

~200 commits of ladder runs, positional-role pipelines, feature research, silver
reconstruction, sealed annotation lanes and a controlled evolution harness. That work has its
own history: [`eval/sequence_models/README.md`](eval/sequence_models/README.md).

### 2026-07-22 — worktree consolidation, and two gates that were measuring the wrong thing

Four `codex/toc-bib-*` branches were merged into `codex/worktree-consolidation-v2-20260722`
(`3b8fe685` sealed-annotation, `e2378060` evolution, `de10e413` sealed-inference, `8f07b616`
header-deploy). Two blockers were then removed. First, `151b9dc1`: **a single silver-label
disagreement** — `## ΒΙΒΛΙΟΓΡΑΦΙΑ` in one document out of a 939 k-line corpus — had
disqualified *all 256* gated-heading decoder candidates, so heading emission had never
deployed. Second, `7e00989b`: the zero-BIB spurious-block gate was a discrete count over 133
documents where 0.02 admits at most 2 blocks; improving the decoder moved it 2 → 3 and cost
1.1 points of line recall — *"the selection got worse because the model got better."* Both were
replaced by a continuous measured body-character-damage criterion.

### 2026-07-23 — the model decision

A fresh, disjoint 150-document cohort (`bibliography_150_20260723_v2`, 210,704 lines, dual-annotator
consensus silver, A/B agreement 0.99733) was sealed and opened once. **`heading_lexgate` at scope
threshold 0.98 strictly dominates the deployed incumbent: 86.0% of bibliography characters removed
against 53.9%, while destroying 0.258% of body characters against 0.505%** (`445f984a`,
[`eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json`](eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json)). The
candidate does **not** clear the historical 0.98 line-precision gate — the argument is that the
gate is denominated in the wrong unit: the incumbent scores 0.9788 line precision but 0.9127
character precision, because its false positives are long lines and corpus cleaning loses
characters. Adopting the recommendation meant adopting that change of measure, recorded as an
explicit owner decision (accepted 2026-07-27). Known limitation carried forward: Kallipos recall
0.677 at this operating point.

### 2026-07-25 → 07-26 — a decision reversed inside 24 hours

[`BIB_DETECTOR_BAKEOFF_20260725.md`](BIB_DETECTOR_BAKEOFF_20260725.md) scored three detectors on
the same cohort with the same metric — a comparison never previously made. The Rust
`reference_detector` came out with *higher* overall char recall than `heading_lexgate`
(0.8541 vs 0.8437) at **~1,250× the speed** (0.76 s wall for the cohort against 16 minutes;
~4 CPU-hours against ~9,300 extrapolated to the academic slice), winning outright on greek_phd
and losing on Kallipos. Its explicit decision: *"Adopt the Rust `reference_detector`; do not
port `heading_lexgate`."* **The next day the port began anyway** (`62e94175`, 2026-07-26), and
`heading_lexgate` is what production shipped. The bake-off document is not marked superseded;
the reversal is visible only in the commit sequence and in
[`production/policy.json`](production/policy.json), whose `model_policy.name` is `heading_lexgate`.

### 2026-07-26 — the port, at decision-equivalence

Sixteen commits in one day took the Python `heading_lexgate` stack to Rust against a stated bar
of **decision-equivalence**, not bit-exactness: `87f866e7` — *"END-TO-END MASK MATCHES —
210704/210704 (100.000000%)"*, 19,117 positives on both sides. The port also surfaced eleven
silent semantic traps (Python's `\w` vs Rust's; a `_SENTENCE_TERMINAL` set member that NFKC
makes unreachable; sklearn's `char_wb` off-by-one worth 932,469 wrong term weights; an exporter
bug inverting a scaler's `with_mean`). Full record: [`bib_line_model/README.md`](bib_line_model/README.md).

### 2026-07-24 → 07-27 — a failed production attempt, then a rebuilt one

[`BIB_CLEANING_HANDOVER_20260727.md`](BIB_CLEANING_HANDOVER_20260727.md) — recovered on
2026-09-01 from an uncommitted worktree — is the honest account of the first attempt. Two dry
runs (jobs 2908121 and 2910962) both stopped early and **both wrote receipts into the same
directory under a plan-relative naming scheme**, so 169 receipts from two plans over the same
documents could not be told apart; the contamination is provable arithmetically
(`glossAPI/libiep` reported 12,010 documents against a source of 6,005 — exactly 2×). Its §11
is a list of operational failures: a 6-hour walltime sized from the wrong benchmark
(20,346 lines/s for deterministic features against ~1,400 for the full chain), fire-and-forget
with no monitoring, units too large to pack, receipts not cleared before the re-run, and a job
script left in `/tmp` so an optimisation check never ran. It also carries two findings that
survived: Kallipos cleans at 61% with this model against 0.7% with the old regex detector, and
openarchives has documents removing >50%, at least one at 100%.

[`BIB_CLEANING_IMPLEMENTATION_20260727.md`](BIB_CLEANING_IMPLEMENTATION_20260727.md)
supersedes it. The mixed directory was kept as forensic evidence and not reused; the workflow
was rebuilt receipt-bound with immutable contracts and stable row-group unit IDs.

### 2026-07-27 — the complete dry run

Preflight job 2912077: 431 files, **51,839,746 rows**, 141,797,094,485 bytes, zero drift,
Hub commit `c368d37c…`. Sealed parity job 2912714: **210,704/210,704 line masks equal,
19,117/19,117 positives**. Dry run `20260727T193808Z-ddf94a84b8b7`, Slurm array 2912781,
157 units, all exit 0, 157/157 receipts, 202,792/202,792 documents, zero partial files, zero
`would_empty` documents:

- **159,142 / 202,792 documents had at least one bibliography cut (78.48%)**
- **2,933,770,472 of 48,373,473,465 characters removed (6.06%)**; 28,190,335 / 401,650,438 lines; 1,980,170 spans
- removal-fraction p50 0.052, p95 0.210; 1,540 documents over 30%, 192 over 50%
- per source (removed %): Greek PhD 7.96, elocus 7.46, Pergamos 6.82, libduth 6.16, Kallipos 5.60, OpenArchives 5.09, ELLAK articles 0.056, libiep 0.049

The regex hardening that made this possible is itself a story: the dry run **failed closed** on
previously unseen long lines, isolating `_PLACE_PUBLISHER_SHAPE`, then `_VOLUME_MARKER` on a
2,228-byte near-match, each fixed by semantics-preserving atomic rewrites. A broader proactive
author-pattern rewrite changed **9 of 210,704 sealed decisions and was completely reverted**
(`1cdb89b6` → `284c120a`). *"This sequence is why exact parity is a contract gate rather than
an informal test."*

QA: 30 deterministic median-sized Kallipos cuts, every OpenArchives document removing >50%,
and every would-be-empty document — **209/209 decisions complete and acceptable, zero
catastrophic / body-only / uncertain, 30/30 Kallipos primarily bibliography, 179/179
OpenArchives >50% removals acceptable.**

### 2026-07-28 — authorization, and the stop

The owner authorized an apply contract for the 175,242-document scope (greek_phd,
openarchives.gr, elocus, libduth), explicitly including libduth, and directed the v2 Hugging
Face target stay public. Kallipos is **not** promoted into apply scope. The libduth licence
directive is recorded *without* changing the existing source-rights warning and is explicitly
*"not represented as rightsholder permission."* Five commits (`b8cc3ea2` … `6f12e9d5`) added
and hardened the apply wrapper, the release publication workflow, dedup-ledger binding and
parallel fragment validation. **These are the last commits in this directory.** No apply
receipt, materialized release, token count or publication record exists here.

## Outcome

- **Method:** REMOVE by segmentation, never hard-delete, fail-closed, per-family counters that
  are never summed. Adopted and never reversed.
- **Model:** `heading_lexgate` at scope threshold 0.98, line threshold 0.9 — 86.0% of
  bibliography characters removed for 0.258% body damage on a sealed 150-document cohort
  ([`eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json`](eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json)).
- **Measure change:** line precision replaced by a character-denominated body-damage budget;
  owner-approved 2026-07-27 ([`production/policy.json`](production/policy.json)).
- **Corpus effect (dry run only):** 2.93 B characters, 6.06% of the 48.37 B-character academic
  slice, across 159,142 documents.
- **For the CPT run itself, nothing was removed.** The tracked structural policy stayed
  `audit_only` and Stage58 was a deterministic no-op — the bibliography cleaning targets the
  *published v2 dataset*, not the training corpus that had already been built.
- **Left open:** the apply run and everything after it; Kallipos promotion; the >50%
  OpenArchives documents; and the five sources (elocus, libduth, libiep, Pergamos,
  eellak-articles) that were never in any validation cohort — the fidelity guarantee is that
  Rust matches Python, not that the model is correct on unlabelled sources
  ([`BIB_CLEANING_HANDOVER_20260727.md`](BIB_CLEANING_HANDOVER_20260727.md) §10-D).

## Sub-subprojects

| Dir | Role | Period | Status | Result |
|---|---|---|---|---|
| [`investigation/`](investigation/README.md) | The 25-agent method study that chose REMOVE-by-segmentation | 2026-06-13 | completed | Footnote stream, not the end list, is the dominant sink |
| [`reference_detector/`](reference_detector/README.md) | Rust crate: end-boundary, footnote stream, β-gate, span/ToC heads | 2026-06-13 → 07-11 | superseded as the production model | Won the 07-25 bake-off on speed and greek_phd recall; not what shipped |
| [`eval/`](eval/README.md) | The measurement system: annotation protocols, evals A/B, iteration loop, the NEXTGEN bake-offs | 2026-06-13 → 07-23 | completed | `heading_lexgate` @0.98 recommended over the incumbent |
| [`eval/sequence_models/`](eval/sequence_models/README.md) | Ladders, positional-role pipeline, sealed annotation, evolution harness | 2026-07-13 → 07-23 | completed | Supplied the consensus-silver cohorts and the decoder families |
| [`bib_line_model/`](bib_line_model/README.md) | Rust port of the `heading_lexgate` line model | 2026-07-25 → 07-26 | completed | Decision-equivalent: 210,704/210,704 line mask, 142 s/cohort |
| [`production/`](production/README.md) | Receipt-bound dry-run → QA → apply → release workflow | 2026-07-27 → 07-28 | dry-run + QA complete; apply authorized, unrun here | 6.06% characters removed; 209/209 QA decisions passed |

## Where things are

| Path | What |
|---|---|
| [`REFERENCE_CLEANING_INVESTIGATION.md`](REFERENCE_CLEANING_INVESTIGATION.md) | The 2026-06-13 method verdict, per-source design, risks and the corpus-scale calibration tables. |
| [`BIB_DETECTOR_BAKEOFF_20260725.md`](BIB_DETECTOR_BAKEOFF_20260725.md) | Three detectors, one cohort, one metric — and a decision that was reversed the next day. |
| [`BIB_CLEANING_HANDOVER_20260727.md`](BIB_CLEANING_HANDOVER_20260727.md) | The failed first production attempt, its contaminated receipts, and ten open issues. Recovered 2026-09-01. |
| [`BIB_CLEANING_IMPLEMENTATION_20260727.md`](BIB_CLEANING_IMPLEMENTATION_20260727.md) | The record that supersedes it: commits, hashes, jobs, dry-run numbers, QA gate. |
| [`production/policy.json`](production/policy.json) | The frozen policy: 13 ranks, 202,792 analysed / 175,242 apply rows, model, QA gate, libduth override. |
| [`eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json`](eval/RECOMMENDED_BIBLIOGRAPHY_MODEL.json) | Machine-readable model decision with receipts, same-cohort comparison and known limitations. |
| [`driver/run_reference_detect.py`](driver/run_reference_detect.py) | Thin I/O driver: parquet→grouped-JSONL (sections) or `.jsonl.zst`→binary (whole-doc). |
| [`review/sample_refspans.py`](review/sample_refspans.py) | Full-doc, post-cleaner, inline `<match kind=…>` stratified review sampler. |
| [`clariden/`](clariden) | `run_academic_refs.sbatch`, `split_selected_by_source.py`, `sync_and_build.sh` — the corpus-scale CPU jobs. |
