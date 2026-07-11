<!-- Generated 2026-06-19. The historically named STRUCT_2K_gold.jsonl contains LLM-silver labels, not human gold. -->

> Historical research report: “gold” below is only the legacy artifact name.
> No full human annotation is required or planned; production safety uses the
> separate receipt-bound 100-case deletion audit (50 ToC + 50 BIB), Stage54 and
> a policy approved before Stage10. The current CPT policy is `audit_only`.

The skeptics' key code-level corrections are confirmed: `_GRK` at span_signals.py:47 omits the polytonic Greek Extended block (`Α-Ωα-ωΆ-ώϊϋΐΰ`, no `ἀ-῿`), while the Rust `greek_letter_fraction` exists separately — a real KEEP-IN-SYNC drift. The DL gate (`dl_probe.py:66`) is hardcoded `target=0.94`, not the contract's 0.97. The operating point's 0.999/0.997/0.995 frontier rests on 45/161/206 prose lines with no CI. These land, so I'll favor the cheaper/safer corrections in the synthesis.

Now writing the definitive document.

---

# Line classifier design — ToC + Bibliography (2,000-doc LLM silver)

This is the historical implementation-guiding design for the line-level structural tagger over the reported STRUCT_2K LLM-silver run (2,000 docs; labels `0=other` 78.35% / `1=bibliography` 15.31% / `2=table_of_contents` 6.34%; leak-free doc-grouped split, train 1392 / test 608). The raw joint corpus is now absent, so these numbers are historical results rather than currently reproducible promotion evidence. Every decision is anchored to the existing `reference_detector` crate and `eval/` harness, and to the precision-first / Rust-deployable contract.

The one rule that overrides everything below: **what ships is a synced Rust dot-product + hysteresis scalar pass.** A fired DL gate authorizes *distill-and-mine*, never *ship-the-net*. The bibliography precedent already followed this (the e5 probe gate fired, the net was not shipped), and that is the binding default — not an exception to be relitigated per facet.

---

## 1) Best way to implement it

Ship **two parallel binary per-line heads** that extend the already-deployed, parity-verified bibliography head — not a new architecture.

Concretely, per present (non-blank) line:

1. **Two scalar heads.** `p_bib` and `p_toc`, each computed as `standardize(features) → dot(weights) → sigmoid`. The bib head is the *existing* 22-feature span LR in `reference_detector/src/span_line_model.rs` (unchanged, no refit, no re-parity). The ToC head is **new**: the same standardize→dot→sigmoid shape over a small set of cheap, deterministic, Rust-portable signals (below).

2. **Position handling as score-gates/features, not constraints.**
   - ToC: a **hard front-gate applied to the score before decoding** — `p_toc *= 1[abs_idx < min(300, 0.30·N)]`, where `N` is the **true full-document line count** (verify the denominator at deploy is true `N`, not the present-line count — training used absolute indices). Gating the score, not post-filtering spans, means the decoder never opens a ToC run outside the front window.
   - Bib: **ungated** (multi-span survives), with the existing soft back-position feature (`pos`, weight +0.54) and the `under_bib_block` header-scoping feature that lets a chapter-bib header re-open a run mid-document. Front→back ordering stays **emergent**, never imposed.

3. **Per-class hysteresis decoder (the existing primitive, run twice).** For each class independently: open a run at `p ≥ θ_hi_c`, extend while `p ≥ θ_lo_c`, merge runs ≤ `G_c` **present-lines** apart, drop runs shorter than `Lmin_c`. Emit `Span{kind}` records (`bib_span` / `toc_span`) into the **same** decoder + output schema, which already carries a `kind` discriminator (`reference_module.rs`). The gap-merge `G` is in present-line units at both tune and deploy time — `span_line_model.rs` already filters to present lines first; preserve that for the ToC pass.

4. **Overlap resolution.** Near-zero by construction (measured: dot/underscore-leader fires on ~53–60% of ToC lines vs ~0.0001 of bib; the classes are orthogonal on existing signals). On the rare contested line, tie-break by `argmax(p_bib, p_toc)` **only on contested lines** — no global per-line argmax, no 3-way softmax. A line winning no class defaults to `other`, so the 172 all-other docs and 239 no-bib docs fall out for free.

**ToC head signals** (cheap, deterministic, and — critically — *not* dot-leader alone, which the skeptic measured at only ~53% recall on TEST):
- dot/underscore leader run (`\.{4,}` / `_{4,}`);
- **multi-level section-number prefix** `^\s*\|?\s*\d+(\.\d+){1,4}\.?\s` (covers bare-number and pipe-table ToC rows);
- **markdown-table-row-in-front-window** signal (silver-labelled ToCs were frequently rendered as `| … |` tables);
- trailing page-number signal, independent of leaders;
- front-position prior (monotone in `abs_idx/N`);
- low year-density (anti-bib context, reusing the bib `hasyear` signal — bib `hasyear≈0.87` vs ToC `≈0.013`, so this separates the two minority classes for free);
- **gated** ATX/short-line `Περιεχόμενα`-stem header — must be on a short line (<40 chars) AND ATX-headed or front-window-adjacent to leader/section-number rows, never fired inline (see the polytonic `περιεχόμενο` prose-collision fix in §the constraints note below).

**Two mandatory pre-work fixes the skeptics surfaced and I verified in code** (do these before any ToC fit or eval, or the precision target inherits a bug):
- **Polytonic `latin_fraction` bug / KEEP-IN-SYNC drift.** `eval/span_signals.py:47` `_GRK = [Α-Ωα-ωΆ-ώϊϋΐΰ]` omits the polytonic Greek Extended block; the Rust `greek_letter_fraction` (`reference_signals.rs:180`) already covers `0x1F00–0x1FFF`. The Python mirror drifted. Fix `_GRK` to include `ἀ-῿`. This is a **shared** feature both heads inherit; today it inflates `latin_fraction` on ~21 classical/Byzantine citation lines, pushing label-0 footnotes toward bib-positive — a direct precision-first violation. Re-run eval on TEST **after** the fix; do not inherit the buggy MODEL_TRADEOFF numbers.
- **Rust `fold_char` polytonic-capital hole.** `reference_signals.rs:29` folds polytonic *lowercase* vowels but **zero capital/adscript forms** (Ἀ Ἁ … ᾼ ῌ ῼ). Polytonic-cased headers (`ΒΙΒΛΙΟΓΡΑΦΙΑ`, `ΠΕΡΙΕΧΟΜΕΝΑ` in the U+1F00 block) miss the H_BIB / H_CV / contents anchor that the aux flags, the chapter-bib rule, and the ToC header signal all depend on. Either extend the fold table to capital/adscript ranges, or document the recall hole and log every doc where a header anchor is expected-but-missing. Add a polytonic-capital-header regression fixture to the parity tests so it is caught, not silently shipped.

This is a pure scalar Rust pass: no torch or ONNX on the hot path. Current corpus-scale work runs on Clariden CPU allocations, not `home` or the Mac.

---

## 2) One model or two?

**Two binary heads. Decisively.**

Reasoning:
- **The proven bib head is preserved untouched.** Two heads keep the deployed, parity-verified bib LR with no refit and no re-parity; a 3-way softmax would force a refit of the thing that already works.
- **Each class keeps its own everything.** Separate `cost_fp` class-weight, separate operating point, separate hysteresis `(θ_hi, θ_lo, G, Lmin)`, separate gate. ToC at 6.34% prevalence is fit cleanly as its own binary problem instead of being a minority arm under 78% "other". A shared softmax normalization *couples* the two thresholds, making per-class precision-first tuning harder — the opposite of what the asymmetric-cost contract needs.
- **The classes never touch.** ToC is front-located and single-block in 1260/1320 docs; bib is mid/back and multi-span. There is no `bib↔toc` adjacency for a joint model to exploit — a 3-state transition matrix would model noise.
- **It stays a Rust dot-product.** Two thresholded sigmoids + two run-length passes; no shared machinery, no new failure modes.

**Runner-up:** a single **3-way softmax line model** over `{other, bib, toc}` with per-class hysteresis on the softmax channels. It is one simpler artifact and an acceptable quick sanity baseline, but it couples the ToC front-gate to whole-doc bib logic, shares one standardization and one decoder, and dilutes the per-class precision-first operating point. Keep it as a baseline to measure against, not the ship path.

**Cost the research underweighted:** two heads = a **second** `*_line_model.json` + a per-class `{"bib":{…},"toc":{…}}` `span_smooth_params.json`, all byte-synced to Rust constants (`MU/SD/W/BIAS/THETA_*`) under KEEP-IN-SYNC. **Make a Rust per-line parity test for the ToC head a hard gate on the head landing** — mirror the existing verified bib parity test. A silent Python-fit-vs-Rust-constant drift breaks parity invisibly.

---

## 3) Other heads on top for the extra info?

Sub-flags (`is_chapter_bibliography`, `is_authors_own_works`, `has_header`, `n_entries`, `doc_type`) are **span-level attributes**, not line classes — the silver labels collapse them all to label 1/2 with no per-line supervision. They attach as a thin **post-decode per-span attribute pass** over the conservatively-decoded (under-capturing) spans the precision-first decoder already produces. **No multi-task neural heads** — there is no neural backbone in the shipping artifact to hang them on.

Per signal, measured on the 2000-doc `ann_*.json`:

| Signal | Verdict | How it attaches |
|---|---|---|
| **has_header** | **Derive (rule)** | "span start is ATX header OR short (<40 char) line with an H_BIB stem." Precision 0.996 / recall 0.976. Lexicon + ATX regex already in `span_line_model.rs`. Free. |
| **n_entries** | **Derive (count)** | Count present lines in the span (median 1.04 lines/entry; for numbered lists count `NUM_PREFIX`/`LEADNUM`). Never *learn* a count. Latent caveat: ~1.3× inflation at p90 for wrapped entries — add an author-head counter only if a consumer ever needs exact counts (none does today). |
| **doc_type {article\|book}** | **Derive (rule)** | `book ⟺ a ToC span exists` + source prior (greek_phd/openarchives → thesis-like, kallipos → textbook). 2-line rule. Weakest derived signal (recall 0.745 vs gpt-5.5, which itself left 347/2000 "unknown"), but safe because its only consumer (chapter-mode bib-keeping) is a soft gate, not a deletion. |
| **is_chapter_bibliography** | **Derive (structural rule), flag low-confidence** | **Do NOT** use naive position (naive "not-last-bib" scores only 0.69/0.59; 377 docs have a chapter-bib as their *only* bib). Correct rule: chapter-bib ⟺ the span is followed by main-text/chapter prose AND is NOT in the terminal back-matter region, off the already-derived front-matter/main-text/appendix segmentation. Mark low-confidence on single-bib docs. **Keep it confined to keep/segment decisions — never a deletion conditioned on it** (that would breach precision-first). |
| **is_authors_own_works** (CV / "Δημοσιεύσεις") | **Deterministic deny-lexicon FIRST; learned head GATED** | The only content-driven, precision-relevant flag (a CV list misread as the document bibliography is a real keep/derive error). Position is uninformative (CV median doc-fraction 0.45, full spread). |

**On `is_authors_own_works` I take the skeptic's cheaper correction over the research's "ship one learned head" default.** The research wanted a per-span learned scalar shipped by default. But there are only **244 positive spans / ~73 in the test split** — recall estimates carry a ±0.06–0.11 binomial CI, so a "precision ≥0.97" claim is not even *measurable* to that precision, and it adds a second Python-fit→Rust-constant sync burden for a flag whose only consumer tolerates approximation.

So:
- **Ship now:** the existing CV deny-lexicon (`CV_HEADER_FAMILY` / `H_CV` in `reference_signals.rs`) as a **post-decode span attribute** and **precision floor** — measured high-precision / ~50% recall, **zero new sync burden**.
- **Gate the learned head** behind an explicit "the CV-recall hole costs corpus" decision. When built: a **low-capacity scalar** over a handful of lexicon/aggregate features (fraction of entries sharing the doc-author surname, presence-window header cue, entry-style homogeneity) — never an embedding, never neural — trained and reported on the leak-free doc-grouped split, shipped **only if a doc-bootstrap CI lower bound clears the target**. Oversample CV docs in any future annotation round to lift N first.

Net shape: one interpretable per-line backbone (two heads) → decode spans → thin per-span pass that computes four flags by rule and runs (gated) one scalar classifier for CV lists.

---

## 4) Reference architectures — strong baseline vs SOTA ceiling

There are **two distinct notions of "best"** and they must not be conflated:

- **Reference architecture for ACCURACY (the ceiling probe):** a line-based **linear-chain CRF / BiLSTM-CRF** over a BIO tag set — the GROBID / Neural-ParsCit / Körner-2017 lineage. The closest published match to our exact task is **Körner 2017, "Reference String Extraction Using Line-Based CRFs"** (arXiv:1705.08154): a whole-document line-stream BIO CRF, text-only, that deliberately avoids a separate reference-area detector. **CERMINE** (SVM zone classifier into METADATA/REFERENCES/BODY/OTHER, ~91.7% P / 87.3% R) is the same anchor-then-derive framing and the strong classical baseline.

- **Reference architecture for a SHIPPABLE INTERPRETABLE system (what we adopt):** the **two binary hysteresis heads** of §1.

**I explicitly demote the `sota` facet's "ship a CRF" headline to ORACLE/CEILING.** Verified: there is **no CRF/Viterbi/transition matrix anywhere in `reference_detector/src/`** — the shipped decoder is `hysteresis()` (`span_line_model.rs:158`), a pure scalar run-length pass. A linear-chain CRF needs a learned transition matrix + Viterbi decode (not a synced Rust dot-product), and its MAP path optimizes **joint likelihood**, not our actual metric (max recall s.t. prose-protection ≥ floor). It belongs in the same bucket as XLM-R, not on the ship line.

Why hysteresis wins as the *shipped* system:
- **The metric is a per-class threshold property.** Hysteresis exposes `θ_hi/θ_lo/G/Lmin` as four scalar knobs that `operating_point.py` grids directly against the real objective (DEFAULT prose-protection floor 0.999). A CRF buries that trade in transition logits.
- **The DL upside is small.** GROBID's own benchmarks bound it: BiLSTM-CRF improves *reference-section segmentation* by only ~**+0.5 to +1 F1** over plain CRF at 2–3× runtime, and fine-tuned transformers (SciBERT/LinkBERT) sit *below* BiLSTM-CRF on structure tasks. So a transformer isn't even the ceiling here.
- **Low-data + precision-first.** 1392 train docs, 6.34% ToC; log-linear CRF can match neural CRF when data is scarce, and per-line marginals let us set precision≥0.97 and read recall-at-that-precision **without retraining**.
- **LayoutLM/LayoutLMv3/DiT rejected:** their entire edge is 2D layout + image patches; our input is a text-only line stream (no bbox/font/image), so they collapse to text-only RoBERTa. The text-vs-visual evidence (Najem-Meyer & Romanello, arXiv:2212.13924) shows text features are *more stable* and competitive on text-heavy pages, and visual models are *weak on exactly the heading/title classes* — i.e. on our input the layout family gives up its edge.

**Adopted:** two binary hysteresis heads ship; the BiLSTM-CRF is the Clariden-GPU accuracy-ceiling probe. **If it beats the two-pass decoder at matched prose-protection, mine the win into cheap Rust features — do not ship Viterbi.** Given near-zero `bib↔toc` adjacency, its expected ship-value here is ~0.

---

## 5) Embed the text? Which embedding model?

**No — do not embed for the shipped tagger. Embeddings are a feature-discovery ORACLE only.**

The embed-or-not question is already answered for the bib half and the answer is no-to-ship: `dl_probe.py` was run; `results_dl_probe.json` shows frozen e5-small lifts recall **+0.042** (at line-precision 0.94) / **+0.048** on Greek-script bib — and `MODEL_TRADEOFF.md` records the standing decision: ship the interpretable Rust line-LR (prose-amputation 52% → 5.6%), keep DL as an oracle. CPU embedding is infeasible at scale anyway (~80 lines/s → ~3 h for 840k silver-labelled lines; production corpus ~47M rows), and the deployment contract is a Rust dot-product. The **new** ToC class is *even more* hand-separable than bib (layout/positional cues), so embeddings are even less justified for it.

**Oracle model (when run):** `intfloat/multilingual-e5-small` — 384-d (cheapest on the 31 GB / 12 CPU box), frozen, via its **ONNX build on onnxruntime-CPU** (the one DL runtime home has), already wired into `embed_lines.py` / `dl_probe.py`. XLM-R SentencePiece byte-fallback → mixed polytonic + Latin reference strings never OOV, and no NFC assumed (matches the Apertus no-normalization rule).

**Fallback (corpus-scale, only if a gate fires and a frozen feature must actually be computed):** `Alibaba-NLP/gte-multilingual-base` (mGTE) via its onnx-community ONNX build, for its faster long-context CPU path.

**Ceiling / fine-tune fallback (Clariden GPU, burden-of-proof only):** `XLM-RoBERTa-base` 3-class token classifier (shares e5/mGTE's SentencePiece tokenizer).

**Rejected:** Greek-monolingual encoders (nlpaueb Greek-BERT is uncased/monotonic — strips the casing AND polytonic the task depends on; single-script) and heavyweight multilingual encoders (BGE-M3, e5-large, LaBSE — same Greek coverage as e5-small, far more CPU/RAM, no discrimination win the historical silver comparison could not already get from hand-features).

**Two corrections to the research's embedding claims that I take from the skeptics:**
1. **The mining loop is OPEN, not closed.** The bib oracle gap *still* shows 0.972 line-recall@prec0.94 vs the shipped model's 0.865 at the strict floor — one mining round happened (monotonic-Greek recall 0.33→0.47), not full closure. Do not assert "wins already mined." Run the **same loop fresh for ToC**: ship cheap Rust ToC signals first, run the e5 probe on ToC only to discover a residual recall hole, never ship the embedding.
2. **Verify polytonic tokenization before trusting any oracle number.** "Never OOV" ≠ "good tokenization." Apertus has no single-token merges for polytonic accented forms, so polytonic words may shatter into many subwords, degrading the frozen representation *precisely on the classical/Byzantine lines*. Before concluding "hand-features suffice," tokenize a fixed probe set (monotonic / polytonic-precomposed U+1F00 / NFD-decomposed / Latin-heavy) through e5's SentencePiece and report tokens-per-word; if polytonic fragments >2× its monotonic form, the +0.042 lift is polytonic-degraded and must be re-measured on a polytonic-stratified slice. Also confirm e5 does not silently NFC-normalize (`normalization_rule_name`) — if it does, it violates the no-NFC alignment.

---

## Comparison table

| Option | Accuracy | Precision control (≥0.97 / prose-protection) | Interpretability | Rust-deployable | Compute | Verdict |
|---|---|---|---|---|---|---|
| **Two binary hysteresis heads** (bib + ToC) | High (bib proven; ToC TBD) | **Best** — 4 scalar knobs/class, independent `cost_fp` | **Best** — auditable dot-product + thresholds | **Yes** (already shipped pattern) | Clariden CPU, scalar pass | **SHIP** |
| 3-way softmax + per-class hysteresis | ~equal | Worse — coupled thresholds via softmax norm | Good | Yes | Clariden CPU | Sanity baseline only |
| Linear-chain CRF / BiLSTM-CRF (BIO) | **Ceiling** (+0.5–1 F1 prior) | Poor for our metric — buried in transition logits | Lower (Viterbi) | **No** (transition matrix, not dot-product) | Deferred research probe | **Oracle/ceiling probe** |
| XLM-R / SciBERT token classifier | ≤ BiLSTM-CRF on structure tasks | Poor (softmax head) | Lowest | No | Clariden GPU | Probe only; never ship |
| Frozen e5-small embedding feature | small lift (+0.042 @0.94) | n/a (feature) | Opaque feature | No (infeasible at corpus scale) | ONNX-CPU oracle / ~3h for 840k lines | **Feature-discovery oracle only** |
| LayoutLMv3 / DiT | n/a (no layout input) | n/a | Low | No | GPU | **Rejected** (text-only input) |
| Header-regex anchor baseline | Low (misses chapter-bibs + CV lists) | n/a | High | Yes | Trivial | Floor baseline to beat, per Körner 2017 |

---

## Staged build order (tied to existing code)

1. **Pre-work fixes (gate everything else).** Fix `_GRK` polytonic range in `eval/span_signals.py:47`; extend/Document the Rust `fold_char` polytonic-capital hole in `reference_signals.rs`; add the polytonic-capital-header parity fixture. Re-baseline bib on TEST after the `_GRK` fix.
2. **Extend the bib line model to the second head.** Add the parallel ToC head in `eval/line_lr.py` (`pos`, `under_bib_block`, `dist_*` surface already there); add ToC signals (leader run, multi-level section number, md-table row, page-tail, front-pos, low-year, gated `Περιεχόμενα` header) to `eval/span_signals.py` and mirror in `reference_signals.rs`. Keep the bib head frozen.
3. **Decoder.** Run `eval/decode_spans.py` hysteresis per class; freeze `span_smooth_params.json` as `{"bib":{…},"toc":{…}}`. Tune **each** class on TRAIN with `eval/operating_point.py` (max recall s.t. prose-protection ≥ floor; ToC stricter, ≥0.999 — a wrongly-removed ToC line is front-matter prose loss). ToC front-gate on the score; bib ungated. Add the ToC Rust parity test as a landing gate.
4. **Three-view eval harness on the doc-grouped split.** Report **per-class** recall@prose-protection on the 608-doc TEST, **with doc-level (not line-level) bootstrap CIs** — extend the existing paired-bootstrap. At 45/161/206 prose lines, the 0.999/0.997/0.995 points likely overlap, so frame it as "one operating region within noise," not "separable frontier." Print **train-vs-test gap** per head/grid. Separate descriptive whole-corpus stats (labeled as such) from test-only performance claims. Budget the named ToC FP families (lists-of-figures/tables, index leaders, footnote leaders, English/Latin-titled thesis ToCs ≈8.7% of test ToC lines) as anti-features.
5. **DL ceiling probe (Clariden, only if step 4 plateaus below target at precision 0.97).** **Re-run the gate at the REAL operating point first** — `dl_probe.py:66` is hardcoded `target=0.94`; re-run `recall_at_prec` at **0.97** and report lift in **prose-protection** terms, per class, before citing "the gate fires." Report the BiLSTM-CRF / e5 gap **separately for the polytonic-heavy slice** (132 docs) vs monotonic, after confirming the tokenizer doesn't fragment/normalize polytonic.
6. **Distill wins to Rust.** Any oracle win that survives on the polytonic slice → mine into a cheap Rust signal (the bib playbook: +0.042 Greek-script recall → Greek-author/edition-marker regexes). Never ship the net or the embedding.
7. **(Gated) CV-list scalar head** only if the deny-lexicon recall hole is decided to cost corpus, with a doc-bootstrap CI lower bound clearing the target.

**Highest-value validity correction across all stages:** treat the reported 608-doc TEST as feature-design evidence, not an unbiased final estimate. If a small future source-balanced holdout is ever authorized, freeze it before feature/threshold selection and report one final per-class result. No new 2,000-item annotation run is required or planned.

---

## WHAT SHIPS vs WHAT'S A RESEARCH PROBE

- **SHIPS (Rust hot path, Clariden CPU, interpretable):** two binary per-line heads (bib unchanged + new ToC), each `standardize→dot→sigmoid`, each with its own front-gate/position feature and its own hysteresis run-length decoder; a thin post-decode per-span pass deriving `has_header` / `n_entries` / `doc_type` / `is_chapter_bibliography` by rule and applying the CV deny-lexicon as a precision floor. Pure scalar pass, parity-tested against the Python fit.
- **RESEARCH PROBE (CPU first; neural ceiling deferred, never the hot path):** frozen e5-small embedding feature-discovery; BiLSTM-CRF / XLM-R accuracy ceiling; the optional CV-list learned scalar head. A fired gate authorizes **distill-and-mine into cheap Rust features**, not shipping the model. The binding precedent: the bib e5 gate fired, the net was not shipped.
