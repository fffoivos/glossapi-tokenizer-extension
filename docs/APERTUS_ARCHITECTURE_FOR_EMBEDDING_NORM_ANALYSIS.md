# Apertus Architecture Decisions Relevant to the Cross-Language Embedding-Norm Convergence

Captured 2026-05-11. Context: the Phase-A diagnostic
([EXTENSION_DOC_FEEDBACK_20260511.md](EXTENSION_DOC_FEEDBACK_20260511.md))
found that Apertus-8B-2509's existing **1,506 Greek vocab entries have
L2 norms statistically indistinguishable from the English-baseline**
(Greek E p50 / English E p50 = 0.999, Greek U p50 / English U p50 =
0.989) — even though the data-share Path-A measurement
([APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md](APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md))
showed Greek is only **0.023 %** of Apertus-8B-2509's realised 13.5 T
pretraining tokens — a ~2,400× data gap to English and a ~41× gap in
per-vocab-entry training occurrences (85 M occurrences/English token vs
2.06 M occurrences/Greek token).

That apparent paradox is *not* an accident — Apertus's recipe contains
multiple deliberate choices that **force** per-token-norm convergence
across languages, regardless of their corpus share. This doc enumerates
those choices, anchors each to a paper section, and explains the
mechanism. The implication matters for the tokenizer-extension plan:
**norm parity confirms training reached the saturation plateau, not
that the merge content is well-targeted for the language.**

Anchors: arXiv:2509.14233v2 (Apertus v1 Technical Report, 1 Dec 2025),
sections §2.1 (architecture), §2.3 (optimizer & training recipe),
§2.4 (ablations), §2.6 (final-run retrospective).

---

## 1. The four convergence-forcing mechanisms

### 1.1 Aggressive global-norm gradient clipping at **0.1** (the biggest one)

Paper §2.6 "Gradient Clipping":

> "From our experience and ablations, the AdEMAMix optimizer is more
> sensitive to the value of gradient norm clipping since the added
> momentum keeps a much longer history of gradient values. Our
> experiments led to **set a clipping value of 0.1**. This means that
> when considering the gradient norms of Figure 3, in practice,
> **clipping is applied at almost every step**."

Standard LLM training uses gradient clipping at 1.0. Apertus uses
**0.1 — 10× tighter** — and confirms it fires basically every step.
Global-norm clipping at 0.1 caps the per-step movement of any single
parameter:

```
g_clipped = g × min(1, 0.1 / ||g||_2)
```

When ||g||_2 > 0.1 (almost every step per the paper), every parameter's
update is scaled down by the same factor. High-frequency tokens
(English, which would dominate the gradient norm) and low-frequency
tokens (Greek, which barely participates per batch) are throttled to
the *same* per-update budget. Over 2.6 M training steps (paper Table 2),
this drags every token's embedding to a similar plateau regardless of
how often it was sampled — exactly the cross-language norm parity we
observe.

### 1.2 Pre-Norm + RMSNorm (forward pass is scale-invariant to embedding norm)

Paper §2.1 "Model Architecture":

> "**Pre-Norm and RMSNorm.** We use pre-normalization before the residual
> in the transformer block, which has better training stability than
> post-normalization (Xiong et al., 2020). We replace LayerNorm (Ba
> et al., 2016) with RMSNorm (Zhang & Sennrich, 2019), which has
> equivalent performance while improving efficiency."

Pre-Norm puts RMSNorm *before* attention and MLP at every layer. The
residual stream gets RMS-normalized at every layer entry. Consequence:
the model's output is **invariant to scaling the input embedding by a
constant** — formally, `f(c · x) = f(x)` for the pre-attention residual
stream up to the very last LM head. The loss gradient therefore has no
direct preference for any particular *absolute* norm of an input
embedding `E_t`. The norm is a free parameter shaped only by the
equilibrium between weight decay (pulling toward 0) and gradient
updates (pushing away from init) — not by language-specific loss
signal.

For the output embedding `U_t`, the picture is slightly different —
RMSNorm sits before the LM head, so `logit_t = RMSNorm(x_last) · U_t`,
and `||U_t||` does directly scale the logit. But this is bounded by
the cross-entropy softmax saturation (§1.4 below), not by per-language
gradient signal.

### 1.3 QK-Norm (caps attention logits independent of embedding scale)

Paper §2.1:

> "**QK-Norm.** We incorporate QK-Norm (Henry et al., 2020; Dehghani
> et al., 2023), which normalizes the queries and keys in the attention
> layers. QK-Norm improves training stability by preventing excessively
> large attention logits."

QK-Norm applies LayerNorm/RMSNorm-equivalent rescaling to Q and K per
head before the dot product. Removes the second-order coupling between
embedding norm and attention-logit magnitude (which would otherwise
scale as `||E_q|| × ||E_k||`). Combined with Pre-Norm + RMSNorm, the
entire forward graph is largely scale-equivariant in the embeddings.

### 1.4 Logit saturation (cross-entropy plateau)

Not an Apertus-specific choice, but load-bearing for the convergence
story. For the output (unembedding) matrix `U`:

```
∂L/∂U_t  ∝  (p_t - 1{y=t}) · RMSNorm(x_last)
```

As `||U_t||` grows in the direction of `x_last` for the contexts where
`y = t` (its typical contexts), the predicted probability `p_t → 1`
and the gradient `(p_t - 1{y=t}) → 0`. The token stops growing once
it's confidently predicted in its typical contexts. Both Greek and
English tokens hit this plateau; English just gets there faster.

This is the classic "embedding-norm saturation" phenomenon observed
across modern transformer LMs. With Apertus's aggressive clipping
(§1.1) the plateau is reached *more uniformly* across tokens — the
clipping prevents the rare-but-real bursts of gradient that would
otherwise push high-frequency tokens past the plateau.

### 1.5 AdEMAMix long-tail momentum (smooths rare-token signal)

Paper §2.3 "AdEMAMix":

> "AdEMAMix improves upon existing gradient-based training algorithms
> that rely on Exponential Moving Averages (EMA) of gradients, such as
> Adam (Kingma & Ba, 2014; Loshchilov & Hutter, 2017), by adding a
> long-term EMA in the form of an additional momentum vector. This
> addition better leverages old gradients for faster convergence,
> especially for long training runs."

The long-term EMA means that when a rare Greek token *does* show up
in a batch, its gradient contribution influences future steps for much
longer than under standard Adam. This effectively amplifies the
training signal from low-frequency events relative to a memory-less
optimizer. Couple this with the §1.1 clipping (which throttles
high-frequency burst gradients) and you get a system that explicitly
flattens the per-token gradient-signal distribution across the vocab.

---

## 2. The combined steady-state model

Under the four mechanisms above, the steady-state norm for a token
with `n_occ` training occurrences is approximately:

```
||U_t||_∞   ≈   f(lr, wd, clip, n_steps)  ×  log_factor(n_occ)
```

where `f(...)` depends only on global hyperparameters (learning rate
schedule, weight decay, clip value, total steps) and `log_factor(n_occ)`
grows **logarithmically** with occurrence count rather than linearly.
This is why a 41× gap in per-token occurrences (English 85 M vs Greek
2.06 M) collapses to a ~25 % gap in norms (log(85 M)/log(2 M) ≈ 18.2/14.5
= 1.25) — and Apertus's 0.1 clipping further compresses that gap to the
observed ~1 % difference at the median.

### 2.1 What the architecture is *deliberately* engineering

This combination is a recognised design pattern for *multilingual*
training where the training corpus has wildly unequal per-language
shares. It explicitly addresses the failure mode where high-resource
languages take all the gradient signal and saturate, while low-resource
languages remain near their initialization. Apertus's recipe is making
each token's *embedding training* approximately frequency-independent —
the per-token convergence regime is the same for Greek as for English
once enough updates have accumulated to escape the initialization
basin.

Apertus mentions multilinguality as a primary motivation throughout:

- Paper §1: "we focus on expanding the multilingual representation of
  Apertus... For Apertus, we massively expand the number of languages
  represented in our pretraining data, to over 1800 languages, and set
  aside a much larger proportion of our pretraining text data mixture
  (~40%) for non-English languages."
- Paper §2.2 (tokenizer selection): chose Mistral-Nemo over Llama-3 /
  Qwen-2.5 / Gemma-2 specifically because Mistral-Nemo "achieves the
  lowest Gini coefficient, indicating more equitable tokenization costs
  across languages."

The architecture choices in §1.1–§1.5 above complete that multilingual
fairness story at the *training-dynamics* level: a fair tokenizer
allocates vocab equitably, and the training recipe ensures every
vocab slot gets driven to a similar trained state regardless of its
data share.

---

## 3. What this implies for the tokenizer-extension project

### 3.1 The norm-parity finding is *expected* given the architecture

Phase A's headline ("Greek tokens are well-trained, norm-parity with
English") is exactly what these four mechanisms predict for *any*
sufficiently-represented language. If we ran the same diagnostic on
Catalan (0.34 % of FineWeb-2 docs, ~7 M training occurrences/token in
HQ), Finnish, Hebrew, etc., we should expect similar norm parity. The
diagnostic is **sensitive enough to flag truly-undertrained tokens
(those near the empirical floor of 0.46) but not sensitive enough to
distinguish "well-trained on lots of Greek" from "well-trained on a
little Greek with aggressive clipping"**.

### 3.2 Norm parity does *not* imply merge-content quality

This is the critical separation the extension project hinges on:

- **Norm training**: an embedding has accumulated enough gradient
  signal to escape initialization and reach the saturation plateau.
  All 1,506 existing Apertus Greek tokens pass this bar.
- **Merge content**: the BPE merges that produced this token are
  *correct* for the target distribution's morphology. This is decided
  by the tokenizer's training corpus, *not* by the model's training
  corpus.

Apertus inherited Mistral-Nemo's BPE merges (paper §2.2) and *never
retrained the tokenizer* on its own pretrain corpus. So whatever Greek
merges were learned during Mistral's tokenizer-training step are
frozen — Apertus then trained those merges to convergence (per §1.1–
§1.5 above), but didn't get to choose them.

The fertility number (4.74 bytes/token = ~2.4 chars/token for Greek
under Apertus, measured in
[APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md](APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md)
§5.3 sanity check) reflects this content problem, not a training
problem. C3's 25,600 added merges target the morpheme-sized chunks
that Mistral's tokenizer missed.

### 3.3 What CPT replay-ratio decisions look like under this lens

For subproject 03 (embedding adaptation + CPT), the norm-parity finding
**does not** mean Greek doesn't need more data exposure — it means more
data exposure won't produce a measurable *norm-level* signal of
improvement. The benefit of more Greek data in CPT is in:

1. **Adapting the existing Greek vocab merges' embeddings to
   GlossAPI-style Greek** (vs the web-Greek that Mistral and Apertus
   trained them on).
2. **Training the new added merges from the C3 extension** to good
   embeddings.
3. **Not catastrophically forgetting Apertus's multilingual breadth**
   — which means we need to maintain the multilingual replay ratio
   while pushing the Greek-share much higher than the 0.023 %
   baseline.

The norm diagnostic on the *extended-and-CPT'd* model is the right
behavioral check — we should expect both the inherited and the added
Greek tokens to be norm-parity with English after CPT if the recipe
is preserving Apertus's training dynamics.

---

## 4. References inside the project

- Embedding-norm diagnostic numbers + method:
  [EXTENSION_DOC_FEEDBACK_20260511.md](EXTENSION_DOC_FEEDBACK_20260511.md)
  §1, plus raw artifacts at
  `runs/apertus_greek_diagnostic_20260511/`.
- Greek pretraining-data share + per-dataset measurements:
  [APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md](APERTUS_PRETRAINING_DATA_AND_GREEK_SHARE.md)
  §5, plus
  `ops/greek_share_run/outputs/{summary,fineweb2_hq_ell,clean_wikipedia_el,europarl_el,euroblocks_el,paradocs_el}.json`.
- C3 cutoff sweep (fertility / compression curve that *does* respond
  to merge content): [C3_CUTOFF_REPORT.md](C3_CUTOFF_REPORT.md).
- Project plan for embedding adaptation:
  `subprojects/03_apertus_extension_and_embedding_adaptation/README.md`
  + `TODO.md`.

## 5. References outside the project (verbatim paper quotes used above)

- §1 (intro), multilinguality motivation.
- §2.1 (Model Architecture): Pre-Norm + RMSNorm, QK-Norm, untied
  embeddings, xIELU.
- §2.2 (Tokenizer): inherited Mistral-Nemo, never retrained.
- §2.3 (Optimizer & Training Recipe): AdEMAMix long-term EMA, WSD
  schedule, batch-size doubling, 1-sqrt cooldown.
- §2.4 (Ablations): Table 3 shows the +xIELU +AdEMAMix +QK-Norm +Goldfish
  +WSD stack reaches baseline loss with 30–40 % fewer tokens.
- §2.6 (Final Run Retrospective): training stability, **gradient
  clipping at 0.1 applied at almost every step**, 8B's gradient norms
  growing visibly larger than 70B's over training.
