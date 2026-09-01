# Bibliography cleaning — handover for review, 2026-07-27

Everything done, where it lives, and what is trustworthy. Written for someone picking
this up cold or reviewing it.

**Bottom line:** the detector is finished and verified. Nothing has been written to any
corpus, nothing published. The remaining work is operational, and the last dry run was
cancelled mid-flight with a contaminated results directory that must be cleaned before
its numbers mean anything.

---

## 1. What exists and how much to trust it

| thing | state | trust |
|---|---|---|
| Rust bibliography line detector | complete | **high** — decision-equivalent, 210,704/210,704 |
| Integration into glossAPI | complete, uncommitted | high — 58 Rust + 15 Python tests pass |
| v2 corpus preflight | complete | **high** — 431 shards re-hashed, match manifest and Hub |
| Throughput measurements | complete | high — measured, not extrapolated |
| Dry-run results | **partial and contaminated** | **low — see §6** |
| Code optimisation | written, **unverified** | **do not use** — see §7 |
| Apply run / token count / publication | not started | — |

---

## 2. The line model

**What it is.** A Rust port of the Python `heading_lexgate` stack (`line_hist_v3` at
threshold 0.9, v3 feature contract, no citation-grammar block). Fidelity bar was
**decision-equivalent**: the emitted line mask must match Python document-for-document.

**Verification — the headline result:**

```
LINE MASK @0.9: 210704/210704 agree (100.000000%)
  ref positives 19117   rust positives 19117
```

Checked for robustness rather than taken at face value: on **zero** lines was
`|p − 0.9| < |dp|`, i.e. no line's error was large enough to have flipped it. Errors
land at margins 0.44–0.78 (mid-range probabilities where the ensemble is uncertain);
lines near the threshold agree far better than their margin. Three lines sit within
1e-4 of 0.9 — none was at risk here, and that residual is inherent to decision
equivalence, not a defect.

Per-stage gates, all against the deployed pipeline's own output at scale:

| stage | result |
|---|---|
| 35 count features | bit-exact, 210,704 lines |
| 34 `line_shape` values | bit-exact |
| 7 gap summaries | bit-exact |
| 5 structure flags | bit-exact |
| `probability:entry` | bit-exact 210704/210704 |
| negative roles (8) + header kind | exact 210704/210704 |
| heading candidate mask | exact — 28,620 both sides, matches the run receipt |
| heading probabilities (3) | max 1.19e-7, 0 rows > 1e-6 |
| `probability:signal_tcn` | max 1.49e-7, 0 rows > 1e-6 |
| connector features (177) | 177/177 within 1e-6, 185,478 candidates, index aligned |
| TF-IDF char_wb + word | 0 support, 0 value mismatches vs *fitted* sklearn |

**Locations**

| what | where |
|---|---|
| Canonical source | `~/Projects/glossapi-development/rust/glossapi_rs_bib/` |
| Python wrapper | `~/Projects/glossapi-development/src/glossapi/bibliography.py` |
| Model artifacts (16 MB) | `~/Projects/glossapi-development/src/glossapi/models/bibliography/` |
| Full gate record + trap list | `rust/glossapi_rs_bib/PORT_STATUS.md` |
| Research original | `.../15_clean_academic/bib_line_model/` (superseded; glossAPI is canonical) |
| On CSCS | `/capstor/scratch/cscs/fffoivos/classifier_research/code_bundles/bib_port_b93d83bd/glossapi_rs_bib/` |

**Python API**

```python
from glossapi import BibliographyCleaner
cleaner = BibliographyCleaner()          # loads bundled artifacts once
spans   = cleaner.detect(text)           # pure — modifies nothing
cleaned = cleaner.clean(text, metrics=m) # m accumulates the ledger
```

Accounting closes exactly:
`len(original) - len(cleaned) == metrics["bibliography_chars_removed"] + len(spans)`
(the `+ spans` is the separating newline each removed run consumes).

**Building.** Two feature flags, and this matters:
- Python module: `maturin build --release` (default features) → `cdylib`, abi3-py38.
- CLI: `cargo build --release --no-default-features --features cli`. **Must** have
  `python` off — pyo3's `extension-module` deliberately doesn't link libpython, so a
  standalone binary built with it fails on `_PyExc_*`. `mimalloc` lives only here; a
  global allocator inside a Python extension would hijack the host interpreter.

Built artefacts on CSCS: extension wheel at
`bib_cleaning_20260724/port/target_ext/wheels/glossapi_rs_bib-0.1.0-cp38-abi3-manylinux_2_34_aarch64.whl`
(installed into `~/.venvs/hfupload`); CLI at `port/target_glossapi/release/bib_line_detect`.

---

## 3. glossAPI integration

Repo `github.com/eellak/glossAPI`, branch `development`, EUPL-1.2. **Uncommitted** —
the repo had unrelated in-flight deepseek work, so staging was left to the owner.

New: `rust/glossapi_rs_bib/`, `src/glossapi/bibliography.py`,
`src/glossapi/models/bibliography/`, `tests/test_bibliography.py`.
Modified: `src/glossapi/__init__.py` (2 lazy exports), both `dependency_setup/*.sh`,
`docs/code_map.md`, `.gitignore`.

**The `.gitignore` change is not cosmetic.** Line 67's `models/` rule (intended for
downloaded OCR weights) also covers `src/glossapi/models/`; the existing models are
tracked only because someone force-added them. Without the negations I added, a fresh
clone gets code with no model and a runtime error.

---

## 4. Corpus facts (verified, not from notes)

v2 = `fffoivos/glossapi-greek-nanochat-pretraining-dataset-v2`,
local at `/capstor/scratch/cscs/fffoivos/cpt_corpus_clariden/agent1-v5-clariden-debug-20260715T111552Z-30c72e9/release-deduplicated`.

```
waterfall      53,046,533 - 1,206,787 = 51,839,746
files          431 on disk == 431 in manifest
local re-hash  431 files, 51,839,746 rows, 141,797,094,485 bytes, drifted=0
vs Hub         commit c368d37c474bbef3d603d111f13997551c8cd2e0
               compared 431 files, missing=0, mismatched=0
```

**Only 13 of 431 files hold academic sources.** One source per file.

```
rank   2  base  Apothetirio_Kallipos      4,784    rank 416  new  glossAPI/elocus     7,699
rank   3  base  Apothetirio_Pergamos     11,071    rank 417  new  glossAPI/libduth    9,254
rank 263  base  greek_phd                17,419    rank 418  new  glossAPI/libiep     6,005
rank 264  base  greek_phd                14,273    rank 421  new  eellak-articles     5,690
rank 266-270 base openarchives.gr       126,597
```

**Policy point flagged and approved by the owner:** 158,289 of the 175,242 documents in
the apply set live in `nanochat_base` ranks. The "don't touch nanochat_base" rule
governs *deletion during dedup*; cleaning deletes no rows. Row count must stay
51,839,746 exactly and is asserted.

Measured size: **49,206,345,743 characters** across the 8 academic sources.
Note `chars`, `non_whitespace_chars`, `utf8_bytes`, `approx_word_count` are **entirely
null** for greek_phd, Kallipos and Pergamos (17,419/17,419 nulls in rank 263).

---

## 5. Throughput (measured)

```
Kallipos, 40 docs, 128,845 lines, 3,221 lines/doc
    1 threads:  837.91s   154 lines/s
    8 threads:  160.84s   801 lines/s
   72 threads:  199.43s   646 lines/s   <-- slower than 8
```

**Parallelism collapses past ~8 threads** — allocator contention inside one process.
The fix is more processes, not more threads. Per-process rate **197,860 chars/s**, so
the academic slice is ~70 process-hours ≈ 2.2 node-hours at 32 processes/node.

Calibration caveat for whoever sizes the next run: on the heaviest documents the
detector runs **~2× the char-based estimate**. Size walltimes accordingly.

---

## 6. Dry-run results — READ THIS BEFORE USING ANY NUMBER

Two dry runs happened. **The results directory contains both, and they overlap.**

- Run 1 (job 2908121, 69 units): stopped at 48/69 — units too large to pack, nodes ~80% idle.
- Run 2 (job 2910962, 157 units): cancelled at 121/157 on owner instruction.

`bibclean/dryrun/receipts/` holds **169 receipts from both runs over the same
documents**. Both name receipts `{rank}-u{unit}` and the unit numbering differs between
plans, so the aggregator cannot tell them apart. Proof of contamination:

```
glossAPI/libiep   12,010 docs reported, source has 6,005   <- exactly 2x
openarchives.gr  213,934 reported, source has 126,597
greek_phd         46,731 reported, source has 31,692
```

**Percentages are approximately usable** (ratios, same detector, same data).
**Every absolute count is inflated and must not be quoted.**

Indicative percentages only:

| source | cleaned% | chars% | p50 | p95 |
|---|---|---|---|---|
| greek_phd | 98.0% | 7.98% | .085 | .242 |
| elocus | 94.8% | 7.44% | .068 | .211 |
| libduth | 88.1% | 5.94% | .055 | .202 |
| openarchives.gr | 79.8% | 5.05% | .041 | .182 |
| Apothetirio_Kallipos | 61.2% | 5.02% | .037 | .149 |
| glossAPI/libiep | 11.7% | 0.05% | .000 | .011 |
| eellak-articles | 0.3% | 0.06% | .049 | .194 |
| **Apothetirio_Pergamos** | **no data** | — | — | — |

**To get a clean table:** move run 1's 48 receipts aside, re-aggregate run 2 alone
(`scripts/dryrun_table.py`), and note it is still ~77% coverage with Pergamos missing.

### Two findings worth a reviewer's attention

1. **Kallipos cleans at 61% with this model, against 0.7% with the old regex detector** —
   a source the previous study wrote off as unreachable. Profile looks safe (median
   3.7%, p95 15%, zero documents over 50%), and the largest cuts inspected are plainly
   real citations. **But** Kallipos bibliographies are per-chapter and header-less, which
   is exactly where many small mislocated cuts would hide. The headline examples are the
   *largest* removals and don't test the median case. Not yet sufficient to overturn the
   skip verdict; worth ~30 median-sized cuts read in context.

2. **openarchives has documents removing >50%, at least one at 100%.** It's the largest
   source in the apply set. The apply script refuses to empty a document, so nothing can
   be silently destroyed, but these need eyeballing before that source is finished.

### Projected removal (from contaminated data — indicative)

~2.6 billion characters, **~5.3% of the academic slice**, ≈620–630M tokens at the
pinned tokenizer's 4.17 chars/token. Under 1% of the whole corpus, since HPLT dominates
and is out of scope. No line counts — receipts record characters and spans only, which
is an omission worth fixing.

---

## 7. The unverified optimisation — DO NOT USE AS-IS

Two genuine inefficiencies found and fixed in code:
- `features::analyze` (the most expensive call — 51 backtracking regexes) ran **twice
  per line**: once building `lines`, again inside `deterministic_row`.
- `line_shape` was computed **up to 6× per line** (own row, heading numerics, connector
  row, twice per neighbour pair).

Both now computed once per document and passed by reference. Compiles, 58 tests pass.
**The 210,704-line mask-equivalence check never ran** — the job script was left in
`/tmp` on a login node and the resubmission failed with
`sbatch: error: Unable to open file /tmp/optcheck.sbatch`.

**This must pass before the optimisation goes near data.** Move the script into the
workspace first.

---

## 8. Where everything lives on CSCS

Workspace: `/capstor/scratch/cscs/fffoivos/bib_cleaning_20260724/`

| path | what |
|---|---|
| `scripts/preflight.py` + `.sbatch` | v2 integrity check (local↔manifest, local↔Hub) |
| `scripts/bibclean_shard.py` | the shard scorer/cleaner — `--dry-run` or apply, row-group units |
| `scripts/make_plan.py` | builds work plans sized from measured throughput |
| `scripts/run_units.sh` | per-node worker pool (processes, not threads) |
| `scripts/dryrun2.sbatch` / `dryrun3.sbatch` | the two array jobs |
| `scripts/dryrun_table.py` | aggregates receipts → per-source table |
| `scripts/bench.py` | throughput / thread-scaling benchmark |
| `scripts/buildext.sbatch` | builds the extension into `~/.venvs/hfupload` |
| `work_plan.json` / `work_plan_small.json` | 69-unit (bad) and 157-unit (good) plans |
| `bibclean/dryrun/receipts/` | **169 receipts, contaminated — see §6** |
| `preflight/v2_preflight.json` | the passing preflight record |
| `port/artifacts/` | exported model artifacts (source of the glossAPI copy) |
| `glossapi/src/` | glossAPI wrapper + models, as used by the runs |
| `out/` | all Slurm logs |

Environment: `~/.venvs/hfupload/bin/python` (3.11, pyarrow 25, huggingface_hub) — the
uenv has no pyarrow. Account `a0140`. Heavy reads must go through Slurm; the login node
reaps them. SSH cert valid to **2026-07-28 13:59**; refresh with `cscs-key sign`.

Reference artifacts used as gates:
`.../experiments/bib_nextgen_devfix_20260722/unseen_features_cohort2_v7/features.npy`
(210,704 × 126) and `port/ref_line_prob.npy` (the reference line probability).

---

## 9. What remains, in order

1. Separate the two runs' receipts; re-aggregate; get a clean table. Pergamos still needs data.
2. Verify the optimisation (§7) or drop it.
3. Apply run — greek_phd, openarchives.gr, elocus, libduth (9 files, 175,242 docs).
   Owner defaults taken: size columns updated **only where v2 has them non-null**;
   Kallipos excluded pending review.
4. Verify: untouched files checksum-identical to v2; rows exactly 51,839,746; 39 columns
   byte-identical; zero emptied documents; ledger waterfall per document.
5. Token count — both v2 and the variant (no v2 baseline exists anywhere).
   Tokenizer pinned: `fffoivos/apertus-tokenizer-extension` rev `a4826df7…`,
   `tokenizer.json` sha256 `358ae3f2…`, vocab 148,480.
6. Publish via `publish_private_agent1_v5.py` (dry run first, to a *different* output —
   receipts are immutable).

### Owner decisions still open

- **Licence scope.** The recorded approval covers *"these two private, versioned Hugging
  Face releases only"*. A third repo is outside it.
- **`libduth` is CC BY-NC-ND** and the newer adjudication marks it
  `local_training.eligible: false`. It is in the apply set.
- **The >50% openarchives documents** — reviewed before that source is cleaned?
- **Kallipos** — include on the strength of 61%, or hold pending the median-cut review?

---

## 10. Open issues — my best guess at what is actually wrong or unresolved

Ordered by how much they could change a decision. These are assessments, not findings;
where I am guessing I say so.

**A. Kallipos 61% may be many small false positives.** *Highest-stakes unknown.*
The old detector fired on 0.7%, this one on 61%, and the bake-off did predict this model
would do better here (0.759 vs 0.583). But Kallipos bibliographies are per-chapter and
header-less, and the prior investigation found that in **0%** of Kallipos documents does
a bibliography-word line sit above the citation-dense block — the anchor simply is not
there. So what is this model anchoring on? A model that fires 87× more often on a source
whose structural signal is documented absent deserves suspicion, not celebration. The
safe-looking distribution (median 3.7%, zero over 50%) is equally consistent with "many
small correct cuts" and "many small wrong cuts". **Resolve by reading ~30 median-sized
cuts in context, not the largest ones** — the largest are the most likely to be real.

**B. libiep fires on 11.7% of documents but removes 0.05% of characters.** That ratio is
strange: roughly one document in nine gets a cut averaging a few hundred characters.
Either it is finding small genuine reference lists, or it is making tiny spurious
detections. Low stakes (libiep is a skip source) but it is a cheap window onto the
detector's false-positive behaviour, and I would look at it precisely *because* it is
low-stakes.

**C. eellak-articles: 0.3% of documents cleaned, but p50 removal 4.9%.** Only ~15
documents, so this is probably just a small-sample artefact. Noted so the next person
does not spend time on it.

**D. Three sources are out of the validation distribution.** The 210,704/210,704 gate and
the cohort-2 bake-off cover greek_phd, openarchives and kallipos. **elocus, libduth,
libiep, Pergamos and eellak-articles were never in the validation cohort.** elocus and
libduth are in the apply set and clean at 94.8% and 88.1% — plausible, and consistent
with the older study, but *unvalidated against labels by this model*. The fidelity
guarantee is that Rust matches Python; it is not a guarantee that the model is correct on
sources nobody labelled.

**E. openarchives has at least one document removed at 100%.** The apply script refuses
to empty a document, so this cannot silently destroy content — but I do not know whether
that document *is* a pure reference list (in which case removal is right and the guard
would abort the shard wrongly) or a false positive. **This will halt the apply run** when
it hits that shard. Decide the policy before launching: abort, quarantine, or skip.

**F. I do not know why the detector runs ~2× the char-based estimate on heavy documents.**
Completed units matched estimates at ×0.96–1.15; the heaviest ran ×2–3. Candidate causes:
fancy-regex backtracking is superlinear in line length (Kallipos averages 247 chars/line),
or memory-bandwidth contention (workers held 4–6 GB RSS each, 23 per node). Untested. It
matters only for sizing, but it means walltimes should carry 2× headroom until understood.

**G. `chars`/`utf8_bytes`/`approx_word_count` are 100% null for greek_phd, Kallipos and
Pergamos.** I treated this as a fact to work around, but I never asked *why*. It may
indicate those sources bypassed a stats stage during v2 ingestion, which could imply other
untracked differences between base-corpus sources. Worth one look before assuming the rest
of their columns are sound.

**H. The receipt naming scheme allows silent collision.** Receipts are `{rank}-u{unit}`,
and unit numbering is plan-relative, so two plans over the same shard produce colliding or
interleaved names with no way to tell them apart — exactly what contaminated §6. The
scheme needs a plan identifier before any further run.

**I. No line counts anywhere.** Receipts record characters and spans only. The model
decides per *line*, so line-level accounting is the natural sanity check and it is missing.

**J. Unverified: the optimisation (§7), and whether `+len(spans)` accounting holds at
scale.** The newline-per-span term is unit-tested and held on every document processed in
the dry runs, but those runs were never completed or reconciled end-to-end.

---

## 11. Honest account of what went wrong

The detector performed correctly throughout. Every failure was operational and mine.

1. **6-hour walltime from the wrong benchmark.** Sized from the deterministic-features
   figure (20,346 lines/s) rather than the full chain (~1,400 lines/s), and assumed
   1,405 lines/doc when it is 3,221. Four tasks killed at the walltime, zero output.
2. **Fire-and-forget.** No monitoring, so six hours passed before the failure surfaced —
   and only because the owner asked. Now standing practice: a 15-minute monitor loop on
   any long job, and a measured throughput number before sizing a walltime.
3. **Units too large to pack.** 69 units into 108 slots meant nothing ever queued; peak
   utilisation 64%, falling to ~20% as units finished, with finish time set by the worst
   straggler. Fixed with 157 smaller units — 121 completed in one hour versus 48 in three
   and a half.
4. **Receipts not cleared before the re-run**, contaminating the results directory (§6).
5. **Job script in `/tmp`**, so the optimisation check never ran (§7).
6. **Stop conditions set on thin reasoning and then overridden.** Twice justified, but
   the underlying error was committing to thresholds before understanding the workload.

Two things did come out of it: real throughput numbers, and the Kallipos finding.
