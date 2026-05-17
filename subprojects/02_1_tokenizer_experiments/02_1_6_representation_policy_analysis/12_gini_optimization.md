# Gini-on-FLORES+ optimization for the C3 cutoff — experiment plan

**STATUS (2026-05-18): SUPERSEDED by 02_1_7.** The Gini-only
optimization plan described in this doc was never run as designed
here. The actual C3 cutoff decision was made on 2026-05-18 via the
TokEval multi-metric sweep in
[`../02_1_7_intrinsic_eval_sweep/`](../02_1_7_intrinsic_eval_sweep/);
chosen cutoff is **17,408 added units (curated + backfilled)**, see
[`../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md`](../02_1_7_intrinsic_eval_sweep/CHOSEN_CUTOFF.md).

**This doc's prediction was empirically wrong.** §3 below predicted
N* in the +3-5k range under Gini-only logic; the actual chosen
cutoff is +17,408, more than 3× the prediction. The Gini-only frame
overweighted aggregate-fairness vs. in-domain Greek fertility and
multi-metric quality. The 02_1_7 REPORT uses an in-domain Greek
fertility curve plus tokenizer Greek-fertility-on-Apertus-55 (TFG)
plus MorphScore-V2 plus added-vocab utilization — four metrics on
actual training-distribution held-outs — rather than the single
Gini-on-FLORES+ metric this doc proposed.

**Kept as historical record** because the contrast between the
policy-reasoning prediction (+3-5k) and the empirical measurement
(+17,408) is itself informative about the limits of single-metric
optimization. The §3 U-curve intuition remains mathematically
correct; what was wrong was assuming Gini-on-FLORES+ alone would
dominate the cutoff decision. In practice, multi-metric on
in-domain Greek dominated.

The text below is the original plan as written 2026-05-17, preserved
unchanged from that point. **Do not act on it.**

---

**One-line summary**: by Apertus's primary stated fairness criterion
for tokenizer selection (lowest Gini coefficient on FLORES+ 55
languages, alongside a secondary preference for smaller vocab and
non-degraded fertility/compression/utilization), an optimization
over the C3 cutoff candidates may produce a uniquely best `N*`.
Reasonable extrapolation from current Greek fertility numbers
suggests `N*` is well below the +11,264 pick — but this is a
prediction pending measurement.

**Source paper**: Apertus tech report, **arXiv:2509.14233v2** (1 Dec
2025), "Apertus: Democratizing Open and Compliant LLMs for Global
Language Environments". Local copy pulled into
[`sources/apertus_2509.14233v2.pdf`](sources/apertus_2509.14233v2.pdf)
(4.87 MB) so quotes are reproducible offline.

## 1. Why this experiment

Per [`11_tokenizer_provenance.md`](11_tokenizer_provenance.md),
Apertus inherited Mistral's tekken v3 BPE table verbatim. Apertus's
stated rationale for choosing Mistral-Nemo's tokenizer was
**multi-criteria** (paper §2.2, Appendix I), not single-criterion:

- **Gini coefficient on FLORES+ 55 languages** — the *binding
  differentiator*. Mistral-Nemo had the lowest Gini among Llama-3.1
  / Mistral-Nemo / Qwen-2.5 / Gemma-2.
- **Fertility rate / compression ratio / vocabulary utilization** —
  verification criteria. Apertus reports Mistral-Nemo "matches or
  outperforms the other tokenizers" on these.
- **Smaller vocabulary (128k vs 256k)** — explicitly preferred for
  pretraining efficiency.

The C3 extension is the first per-language vocab decision Apertus is
making inside its own stack. This experiment focuses on the Gini
criterion specifically because it's the named *differentiator* —
the dimension Apertus singled out as binding. The other criteria
still apply and should be tracked in the sweep:

- A C3 extension of +N tokens grows vocab from 131,072 to
  131,072+N. For N up to ~25,600, vocab stays well under the 256k
  alternative Apertus rejected, so the "smaller vocab" preference
  is approximately preserved.
- Fertility, compression, and vocab utilization shift as we extend
  Greek-side. These are reported alongside Gini in §5.

Apertus's primary stated fairness criterion gives a sharp test:

> If extending the tokenizer with N added Greek merges *lowers* the
> Gini below Mistral's value while keeping fertility / compression /
> vocab utilization non-degraded across the 55 FLORES+ languages,
> the extension produces a strictly better tokenizer along Apertus's
> stated dimensions.

The N that minimizes Gini, subject to non-degradation of the other
criteria, is the principled choice **along this axis**. That
replaces rhetorical "match-X" anchors in
[`02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md)
with a measured number — but measured against *one* of Apertus's
criteria (the differentiator), not the whole multi-objective bundle.

## 2. Primary-source citations — Apertus's FLORES+ Gini selection

### 2.1 §2.2 Tokenizer (p. 10) — the selection statement, verbatim

> "We based our choice on a comparison of the tokenizers of several
> large language models (e.g., Llama-3.1, Mistral-Nemo, Qwen-2.5, and
> Gemma-2) using four intrinsic evaluation metrics: **fertility rate,
> compression ratio, vocabulary utilization, and Gini coefficient**
> (Foroutan et al., 2025a). … **We conduct these evaluations using
> the FLORES+ development set covering 55 languages** (nll, 2024).
> Figure 1 presents the comparison results. **Mistral-Nemo achieves
> the lowest Gini coefficient, indicating more equitable tokenization
> costs across languages.** … we select Mistral-Nemo as the preferred
> tokenizer because it is **fairer across languages** and uses a
> smaller vocabulary (128k vs. 256k), making it more efficient for
> pretraining without sacrificing performance."

**That is the only paragraph in the Apertus paper that justifies the
choice of tokenizer.** It states a multi-criteria framework, but
Gini-on-FLORES+ is the **primary differentiator** Apertus reports as
binding (Mistral-Nemo had the lowest Gini); fertility / compression /
vocab utilization are reported as verification ("matches or
outperforms"), and smaller vocab (128k vs 256k) is reported as a
secondary preference for pretraining efficiency.

### 2.2 Appendix I (p. 92–93) — the precise definitions

The four metrics are defined verbatim in Appendix I. Quoted with
their equation numbers as numbered in the paper.

**Compression rate** (eq. 1, p. 92):

> "Using lines (documents) as units, it is defined as:
>
> CR(D; τ) = (1/|D|) · Σ_{b∈D} |b|_u / |τ(b)|"

Higher = better. Apertus's "u" subscript indicates the chosen
normalization unit (line, byte, word, or character).

**Fertility** (eq. 2, p. 93):

> "Using **words** as the normalization unit (as determined by the
> HuggingFace Whitespace Pretokenizer), fertility is defined as:
>
> Fertility(T) = Σ_{b∈D} |τ(b)| / Σ_{b∈D} |b|_u"

Lower = better. Words are detected by HF Whitespace Pretokenizer.

**Vocabulary utilization** (eq. 3, p. 93):

> "VocabUtil(T) = |{v : v ∈ τ(b), b ∈ D}| / |V|"

Fraction of vocab actively used on the corpus.

**Tokenizer Fairness Gini Coefficient** (eq. 4, p. 93):

> "We adapt the Gini coefficient (commonly used to measure inequality
> in economics) to quantify fairness across languages (Meister, 2025).
> Let L = l₁, l₂, …, lₙ be the set of languages, and let
> c₁ ≤ c₂ ≤ … ≤ cₙ denote their tokenization costs under T. **Here,
> cost is defined as the average number of tokens required to encode
> one normalization unit** (e.g., a byte, word, or line); **for
> parallel corpora, cost per line is often used to control for
> differences in character byte lengths across scripts**. The Gini
> coefficient is given by:"
>
> ```
>                  1   ⎡         Σᵢ₌₁ⁿ(n+1−i)·cᵢ ⎤
>    Gini(T) =   ─── ·⎢ (n+1) − 2·──────────────⎥          (eq. 4)
>                  n   ⎣            Σᵢ₌₁ⁿ cᵢ      ⎦
> ```
>
> "Values range from 0 (perfect equality) to 1 (maximum inequality)."

**Footnote 57** (p. 93, immediately after eq. 4): *"This is
equivalent to fertility, or the inverse of the compression rate."*

Three things to note:

1. **The cost unit is parameterizable** — byte, word, or line.
   Apertus says "cost per line is often used" for parallel corpora.
2. **Footnote 57 equates cost-per-X with fertility (for X=word) or
   inverse-compression-rate (for X=line)** — these are different
   numbers, and the paper isn't fully explicit about which Apertus
   actually used in their Figure 1 Gini. We'll report both.
3. **The 55 languages are enumerated explicitly in Appendix I**:
   Afrikaans, Albanian, Arabic, North Azerbaijani, Basque, Belarusian,
   Bengali, Bulgarian, Catalan, Chinese, Czech, Danish, Dutch,
   English, Estonian, Finnish, French, Galician, Georgian, German,
   **Greek**, Gujarati, Hebrew, Hindi, Hungarian, Indonesian, Italian,
   Japanese, Korean, Latvian, Malay, Malayalam, Marathi, Macedonian,
   Norwegian Bokmål, Persian (Farsi), Polish, Portuguese, Romanian,
   Russian, Slovak, Southern Sotho, Spanish, Swahili, Swedish, Tamil,
   Tajik, Telugu, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh,
   Yoruba.
   Greek is included; Latin and Ancient Greek are NOT.

### 2.3 The reference Foroutan et al. 2025a

Apertus cites this paper as the source of the four metrics. Full
reference (Apertus paper bibliography):

> Negar Foroutan, Clara Meister, Debjit Paul, Joel Niklaus, Sina
> Ahmadi, Antoine Bosselut, and Rico Sennrich. **Parity-aware
> byte-pair encoding: Improving cross-lingual fairness in
> tokenization.** arXiv:2508.04796, 2025.

That paper:
- Defines the same Gini formula adopted by Apertus.
- Proposes a "Parity-aware BPE" algorithm achieving ~6× lower Gini
  than classical BPE on a 30-language unbalanced set (Classical BPE
  Gini = 0.064 vs Parity-aware = 0.011 at 128k vocab).
- **Apertus did NOT use Parity-aware BPE** — they used Mistral-Nemo's
  classical BPE, picked from off-the-shelf tokenizers by Gini.

### 2.4 The dataset used to compute fertility / Gini

**FLORES+ development set** — `openlanguagedata/flores_plus` on HF.
Apertus cites it as "(nll, 2024)" — the FLORES+ release card from the
NLLB team's continuation. FLORES+ is a *parallel translation*
benchmark: each sentence has 200+ language renditions of the same
source, hand-translated. The dev split has ~1012 sentences per
language. The source content draws from Wikinews + Wikijunior +
Wikivoyage.

This is the dataset on which Apertus computed the per-language costs
that go into eq. (4).

### 2.5 The criterion as we'll use it

Apertus's stated criterion is "minimize Gini on FLORES+ 55 languages."
For our optimization:

```
N* = argmin Gini(T_c3(N))
     N
```

where:
- `T_c3(N)` = merged Apertus-compatible variant with N added Greek
  merges (N ∈ {0, 1024, …, 25600})
- Gini computed per eq. (4), `n=55`
- Costs `c_i` = per-line tokens on FLORES+ dev for each of the 55
  Apertus-enumerated languages (primary metric per Apertus §I);
  per-word fertility (eq. 2) also reported as secondary cross-check.

## 3. Why we expect a U-shaped curve

This is the key piece of intuition. **Gini measures inequality — it
doesn't care which direction the outliers are**. So as we add Greek
tokens (lowering Greek's cost), Gini doesn't just monotonically drop.

### 3.1 Intuition

Imagine a kindergarten with 6 kids and a fixed pile of cookies. Five
kids have {1, 4, 8, 12, 15} cookies; the 6th kid has some variable
number `x`. We want the most equal distribution.

- If `x = 25` (the 6th kid is rich), the distribution is
  unequal — the 6th kid is an outlier high.
- If `x = 8` (the 6th kid is at the median), the distribution is
  most equal — no outlier in either direction.
- If `x = 0` (the 6th kid is poor), the distribution is unequal
  again — the 6th kid is now an outlier low.

Inequality (Gini) has a **minimum somewhere around the median** of the
others. Adding to the rich kid's pile beyond the median doesn't make
things more equal — it just shifts which direction the outlier is.

Greek's situation in Apertus is the same. Greek under Apertus base
has **high** per-line cost on FLORES+ — it's an outlier high. Adding
Greek tokens lowers Greek's cost. As long as Greek is moving down
*toward* the median of the other 54 languages, Gini drops. Once Greek
crosses the median and becomes an outlier *low* (better-served than
the rest), Gini starts rising again.

### 3.2 Worked toy example — actual Gini computation

Let's compute. Say the other 5 languages have fertility costs
`{1.0, 1.5, 1.8, 2.1, 2.5}` — a roughly realistic FLORES+ spread.
Add Greek with variable cost `x` and compute Gini for each `x`:

| Greek's cost `x` | sorted (n=6) | Gini |
|---:|---|---:|
| 3.0 (well above median) | [1.0, 1.5, 1.8, 2.1, 2.5, **3.0**] | **0.186** |
| 2.4 (somewhat above) | [1.0, 1.5, 1.8, 2.1, **2.4**, 2.5] | 0.155 |
| **1.8 (at median)** | [1.0, 1.5, **1.8**, 1.8, 2.1, 2.5] | **0.145 ← min** |
| 1.4 (somewhat below) | [1.0, **1.4**, 1.5, 1.8, 2.1, 2.5] | 0.160 |
| 0.8 (well below) | [**0.8**, 1.0, 1.5, 1.8, 2.1, 2.5] | 0.208 |

The U is real and pronounced. Greek at the median (1.8) gives Gini
0.145; Greek at outlier-high (3.0) gives Gini 0.186 (28 % worse);
Greek at outlier-low (0.8) gives Gini 0.208 (43 % worse).

The Gini-optimum is approximately where Greek's cost equals the
median of the other 54 — not exactly (the optimum shifts slightly
based on the distribution's shape), but very close.

### 3.3 What this means for the C3 cutoff

Greek's cost under Apertus base on `modern_greek_eval` is 2.41. The
54 other FLORES+ language costs are unknown (haven't measured them)
but presumably range from English ~1.0 to long-tail Welsh / Yoruba /
Tajik ~2.5+. The median is probably 1.7-1.9.

Adding Greek tokens:

| Cutoff (added N) | Greek fertility (modern_greek_eval) | Greek position relative to predicted median ~1.8 |
|---:|---:|---|
| 0 (base) | 2.41 | ~ +0.6 above median (outlier high) |
| +1024 | 2.09 | ~ +0.3 above median |
| **+3072** | **1.83** | **~ at median** |
| +5120 | 1.69 | ~ -0.1 below median |
| +6144 | 1.63 | ~ -0.2 below median |
| +8192 | 1.55 | ~ -0.3 below median |
| **+11264** | **1.47** | **~ -0.4 below median (outlier low)** |
| +17408 | 1.37 | ~ -0.5 below median (outlier low) |

**Hypothesis (NOT yet measured on FLORES+)**: the Gini-minimum is
plausibly around +3,072 to +5,120 added tokens. The C3 REPORT's
current pick of +11,264 *may* put Greek far enough below the FLORES+
median that it becomes an outlier-low — increasing Gini compared to
a more modest cutoff.

This prediction extrapolates from `modern_greek_eval` fertility
numbers, which use noisy GlossAPI+HPLT text and likely measure
**higher** fertility than FLORES+ Greek would (FLORES+ is clean
Wikinews / Wikijunior / Wikivoyage translation). Two ways the
extrapolation could fail:

- **FLORES+ Greek fertility under Apertus base could be lower** than
  the 2.41 we measured on `modern_greek_eval` — closer to the
  FLORES+ median. In that case Greek is already near-median without
  extension, and `N*` shifts toward 0; under this metric, no
  extension is needed.
- **FLORES+ Greek fertility could be much higher** than expected
  (unlikely given FLORES+'s clean register, but possible if Mistral
  tokenized Wikinews-style Greek particularly poorly) — in which case
  `N*` shifts higher than +5,120, possibly past +11,264.

The U-curve shape is mathematically robust regardless. The
**location** of `N*` is empirical and requires the FLORES+ sweep
in §5 to pin down.

If the prediction's middle case holds, Apertus's primary stated
criterion would favour +3-5k added tokens, not +11k. **This is
currently the strongest principled argument the analysis can make,
but only as a prediction pending measurement.**

### 3.4 Caveat — fertility on FLORES+ ≠ fertility on `modern_greek_eval`

The numbers above are from `modern_greek_eval`, which uses GlossAPI
+ HPLT-clean text. FLORES+ uses Wikinews/Wikijunior/Wikivoyage human
translations — simpler register. FLORES+ Greek fertility under
Apertus base is probably **lower** than 2.41 (maybe 1.9-2.2).

Same direction, though: Greek under Apertus base is plausibly above
the FLORES+ median, and the cutoffs slide it through the median
somewhere between +1024 and +5120. **The exact cutoff depends on
measurement.**

## 4. Why FLORES+ is not the ideal corpus for our specific question

Apertus picked FLORES+ for a defensible reason: it's a parallel
benchmark where each sentence has 200+ language translations of the
same source content. That makes cross-language fertility comparable
in a controlled way — a good fit for selecting *which existing
tokenizer is fairest across many languages at once*.

Our question is different. We're choosing the *Greek-specific
extension size* for a tokenizer we're going to deploy on real Greek
text. FLORES+ Greek is a thin slice of what "Greek text" looks like
in deployment. The mismatches:

### 4.1 Register and source mismatch

FLORES+ source content is **Wikinews + Wikijunior + Wikivoyage** —
news articles, simplified-for-kids encyclopedia entries, and travel
guide writing. ~1012 sentences total.

Our C3 training corpus (per
[`../../../docs/C3_TRAINING_DATASETS.md`](../../../docs/C3_TRAINING_DATASETS.md)):

- ~14.4 M Greek documents
- ~100 B characters
- Sources: Kallipos/Pergamos academic textbooks, ancient/classical
  texts, ecclesiastical texts, dimodis/folk literature, Greek
  PhD theses (didaktorika.gr), EU Parliament Greek, Greek legislation
  (eur-lex), openarchives.gr academic repository, opengov.gr public
  consultations, school books, Wikisource Greek (literature,
  historical), Greek Wikipedia, Greek legal code, HPLT web crawl.

FLORES+ Greek captures essentially none of the registers our model
will actually see in deployment: academic prose, legal text,
ecclesiastical Greek, OCR'd book text, historical katharevousa, web
crawl with crawl artifacts. Optimizing fertility on FLORES+ optimizes
the wrong register.

### 4.2 Translation artifact

FLORES+ Greek is **human-translated from English source**. Translated
text has measurably different statistics from natively-produced text:

- **Lexical**: translations tend toward shorter, more common vocabulary
  (translators reach for unambiguous terms).
- **Syntactic**: translations carry source-language structure
  (translatese — Greek translations of English-source tend to have
  English-like word order, simpler clause structure).
- **Idiomatic**: translations replace native idioms with literal
  paraphrases.

A tokenizer optimized for FLORES+ Greek is optimized for translatese
Greek, not for native Greek. The Greek tokens our model will need to
encode well in deployment are native-prose Greek tokens.

### 4.3 Monotonic-modern-only

FLORES+ Greek is monotonic modern Greek. Our corpus includes:

- Polytonic Greek (Wikisource, ancient/liturgical, classical works)
  — handled by the parallel polytonic arm at
  [`../02_1_polytonic_greek_extension/`](../02_1_polytonic_greek_extension/)
- Katharevounsa (19th-century formal Greek, in academic and historical
  sources)
- Modern dialects represented in literature

None of these appear in FLORES+. Even for the modern-Greek C3 arm
specifically, FLORES+ doesn't sample the diachronic range our corpus
contains.

### 4.4 No specialized domains

FLORES+ has effectively zero:

- Code / programming Greek-language comments
- Math / LaTeX / scientific notation
- Religious / liturgical text
- Legal / contractual Greek
- Medical / technical terminology
- Names, URLs, codes, identifiers

Our corpus has substantial volume in each of these. A Greek
tokenizer's fertility on academic / legal / religious Greek is a
different metric from its fertility on Wikinews Greek.

### 4.5 Small sample and homogeneous

~1012 lines per language for FLORES+ dev. Internally homogeneous
(consistent source, consistent translation pipeline). Compare to C3's
14.4 M Greek docs spanning ~30 distinct source pools.

### 4.6 Apertus picked it for a different question

Apertus's question: "of the four off-the-shelf tokenizers we have,
which one is fairest across many languages?" FLORES+'s parallelism +
controlled register made it well-suited.

Our question: "given Apertus's chosen tokenizer, what Greek extension
size makes Greek tokens best-allocated for the Greek text we will
actually deploy on?" FLORES+'s narrow register and translation
artifacts make it a thin proxy for the answer.

### 4.7 Aggregate-vs-Greek tension

Even setting register aside: Gini-on-55 is an aggregate fairness
metric. It says "across these 55 languages, is no language an
outlier?" It does *not* say "is Greek well-served on the kinds of
Greek text our users will write?" These are different questions:

- Gini-on-FLORES+-55 minimization could leave Greek's fertility *on
  actual deployment text* still poor, as long as the spread across
  55 languages is balanced.
- Greek-specific-fertility minimization could over-serve Greek
  relative to its peer languages on FLORES+, but produce better
  user experience for Greek-language deployment.

### 4.8 What a better-suited corpus would look like

Two cross-checks worth running alongside the FLORES+ canonical:

1. **Native-text Gini on FineWeb-2**. FineWeb-2 v2.0.1 has Greek
   plus the other 54 Apertus-Appendix-I languages (verified — all 55
   are present in FW2). FW2 is native-language web crawl, not
   translation. Sample ~1000 docs per language; compute the same
   eq. 4 Gini.
2. **Greek-specific fertility on `modern_greek_eval` and HPLT-virgin
   held-out**. Already measured in
   [`../02_1_4_cutoff_analysis/REPORT.md`](../02_1_4_cutoff_analysis/REPORT.md)
   §2. The C3 fertility curve there is on Greek-only text from our
   actual training pool — directly relevant.

The **Apertus-criterion answer** is Gini-on-FLORES+-55 (so we can
make the argument "we beat Mistral on Apertus's own metric"). The
**deployment-relevant answer** is Greek-fertility on
`modern_greek_eval` / HPLT-virgin (already known from the cutoff
REPORT) or Gini-on-FineWeb-2-55.

If they agree on `N*`, we have a strong principled answer. If they
disagree, the user decides which to honour — but the disagreement
itself is informative (it would mean "FLORES+ undersells the gain
from extension" or "deployment text overstates it").

### 4.9 Pragmatic choice

Run the canonical FLORES+ Gini sweep as the *primary* experiment —
it's the answer to Apertus's stated criterion. Add a *secondary*
Gini sweep on FineWeb-2 same-55 native samples as cross-check.
Compare the curves. Both are achievable in the same ~30-minute
run-bundle.

The C3 REPORT's existing fertility table (already on
`modern_greek_eval`) serves as the *third* cross-check — the
"deployment-relevant" answer that's already been measured.

## 5. The experimental plan

### 5.1 Data needed

- **Apertus base tokenizer** — local cache:
  `~/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/.../tokenizer.json`
  (verified).
- **C3 merged-Apertus-compatible variants** at each cutoff
  N ∈ {0, 1024, 2048, …, 25600} — built by
  [`../02_1_2_cutoff_variant_builder/`](../02_1_2_cutoff_variant_builder/).
  Variants may already exist on the gcloud worker at
  `~/runs/c3_wave2_broad_latest_cleaner_20260506/50_50/tokenizers/`;
  if not, build them on `home`.
- **FLORES+ dev set** for the 55 languages in Apertus Appendix I,
  from `openlanguagedata/flores_plus` on Hugging Face. Public,
  small (~1012 parallel lines per language).

### 5.2 Steps

1. **Pull FLORES+ dev** for the 55 Apertus-Appendix-I languages.
2. **Tokenize 54 non-Greek FLORES+ sets** with Apertus base. Compute
   `c_i = total_tokens / total_lines` for each. These costs are
   **fixed** for the entire sweep — they don't change when we add
   Greek tokens.
3. **For each cutoff N**:
   - Pull or build merged Apertus-compatible variant `T_c3(N)`.
   - Tokenize FLORES+ Greek dev with `T_c3(N)`.
   - Compute `c_Greek(N)`.
   - Combine with the 54 fixed costs → 55 values.
   - Compute Gini per eq. (4) above.
4. **Plot** Gini vs N. Find `N* = argmin`.
5. **Sub-1024 refinement** near `N*` if needed (256-aligned grid).
6. **Report**: `N*`, the full Gini curve, the predicted-vs-measured
   FLORES+ Greek fertility table, and the position of Greek among
   the 55 sorted costs at `N*`.

### 5.3 Compute requirements

- 55 langs × ~1000 lines × ~30 μs/line per language ≈ 1.6 s of
  tokenization per language → ~90 s for the full 55-lang baseline pass.
- 25 cutoffs × 1000 Greek lines ≈ ~40 s of Greek tokenization in total.
- Building variants (if needed) — depends on the builder; estimated
  ~5 min per variant × 25 = ~2 hours if doing fresh builds. If
  pulling from gcloud, ~5 minutes total.

**Total wall-clock** on `home`: under 30 minutes if variants already
exist; under 2.5 hours if building from scratch.

No gcloud workload needed if variants are already built and pulled.

### 5.4 Deliverables

- **`scripts/run_gini_sweep.py`** — runs the sweep, writes a
  `results/gini_sweep.json` with per-cutoff Gini + Greek fertility.
- **`artifacts/gini_curve.png`** — plot of Gini vs N.
- **`artifacts/per_language_costs_apertus_base.json`** — the 54
  fixed FLORES+ costs we measured, for posterity and downstream reuse.
- **`REPORT_RESULTS.md`** — short post-run report with `N*`, the
  curve shape, and the recommended cutoff.

## 6. Open design choices

1. **Cost unit**: per-line (paper's choice for parallel corpora) is
   primary. Per-byte and per-char are secondary checks (controls for
   script encoding length and script density). Report all three.
2. **All 55 vs HQ-20 only**: paper specifies 55. HQ-20 would weight
   Greek's peer set more heavily — defensible variant. Run both;
   report the primary as 55-language.
3. **Granularity near `N*`**: if `N*` lands between two grid cutoffs,
   do a finer 256-aligned sweep around it. Cheap addition.
4. **Whether to also report the constrained metric `max(c_i)`**:
   alongside Gini, the max cost (worst-served language) is informative.
   Report both.

## 7. Risks and caveats

1. **FLORES+ fertility may not match `modern_greek_eval` fertility.**
   The shapes might differ. The U-curve still holds qualitatively,
   but the location of `N*` could shift.
2. **Apertus's Gini definition is sensitive to which 55 languages we
   pick.** Apertus paper Appendix I enumerates the 55, but if the
   FLORES+ release we pull has slightly different language codes
   (e.g., `cmn_Hans` vs `cmn_Hant`), we need to match the paper's
   list precisely. Worth double-checking.
3. **The 54-fixed-costs assumption can break**: if the C3 added
   tokens accidentally encode tokens used by non-Greek languages,
   their costs shift too. The C3 REPORT §4 says 99.5 % of added
   tokens are Greek-payload — the remaining 0.5 % might marginally
   affect other languages. Sanity-check by re-tokenizing one or two
   non-Greek languages with `T_c3(N)` at a few N values; if costs
   don't move, the 54-fixed assumption is safe.
4. **The Gini-optimum doesn't address fertility-elbow.** Two
   criteria, two answers. Gini-optimum gives the *fairest* cutoff
   under Apertus's stated metric. Fertility-elbow gives the cutoff
   beyond which marginal returns disappear. They might agree or they
   might not. If they disagree, the user picks which criterion to
   honour.
5. **Mistral's reference Gini value** (what we'd need to compare our
   `T_c3(N*)` against to confirm "we improved on Mistral's Gini")
   isn't directly reported in the Apertus paper — only the *fact*
   that Mistral-Nemo had the lowest. Our experiment computes Apertus
   base Gini directly (`N=0`) so we have a numeric reference even
   without Mistral's paper-reported value.

## 8. Expected outcome shape

If the prediction holds, the curve looks like:

```
Gini  ↑
      │  Greek as outlier-high                     Greek as outlier-low
      │       \                                          /
0.06  │        \                                        /
      │         \                                      /
0.05  │          \           min                      /
      │           \         /                        /
0.04  │            \_______/_______________________/
      │             N=3k   ↑                       N=25k
      │                  N*≈4k
      └──────────────────────────────────────────────────→
        0       1k     5k     10k     15k     20k     25k    N
                                ↑
                          C3 REPORT's current pick (+11,264)
```

The Gini value at `N*` is what tells us whether the extension is a
*real* improvement over Mistral-Nemo's base. If `Gini(T_c3(N*)) <
Gini(T_apertus)`, the extension is strictly better by Apertus's own
stated criterion.

## 9. What this experiment commits us to

A single number, `N*`, derived without rhetoric. The user can still
override `N*` with a different criterion (HQ-20-equal-share,
fertility-elbow, computational budget). But after this experiment,
the **default principled answer** has a number, not a menu.

This replaces the §1 anchor table in the C3 REPORT.
