# MorphBPE for Greek — application notes

Parked idea captured 2026-04-21. Do not commit as part of the shipping
tokenizer extension; this is a research-track alternative that may inform
a future experiment.

## Why it fits our approach

The `unified_classification_20260418/` taxonomy work classified every
Greek token in both Apertus+hplt arms into Prefix / Stem / Compound /
Ending / Proper-noun / Function-word / Acronym / Loanword / Other, with
morphological decomposition into `parts[]` per token. The core finding
on the "Other" bucket was that it is dominated by BPE **cross-morpheme
cutoffs** (`stem_partial`, `prefix+stem_partial`, `fragment`) —
26.1 % of C2's Greek tokens, 21.4 % of F2's — tokens that carry no
reusable morphological signal because BPE's greedy-merge procedure sliced
through morpheme boundaries.

This is exactly the class of token that morphology-aware BPE variants
forbid by construction. Our taxonomy effectively measures the gap a
morphology-aware tokenizer would close on Greek.

## How MorphBPE works (recipe)

1. **Morphological pre-segmentation of the corpus.** Run a morphological
   analyzer on every word before any BPE training happens. Produces
   per-word segmentations like `γράφουμε → γραφ + ουμε` (stem + ending),
   `αντιπρόσωπος → αντι + προσωπ + ος` (prefix + stem + ending),
   `δημοκρατία → δημο + κρατ + ία` (stem + stem + ending).
2. **Boundary-constrained BPE.** Train BPE within each morpheme
   independently. Candidate pair-merges that would cross a morpheme
   boundary are rejected. The resulting vocab contains only tokens that
   live inside one morpheme, plus whole short morphemes as single
   tokens. No cross-morpheme cutoffs possible.
3. **Inference-time tokenization.** Run the same analyzer on input
   text first, then apply BPE within each segment. The model never sees
   a token that spans two morphemes.

Variants in the literature include **MorphPiece** (Shrestha et al.,
2024), **MorphBPE** (Salavati et al., 2025), **Morph-aware Subword
Tokenization** (various), and **FLOTA** (Hofmann et al., 2021, uses a
lighter hint-based scheme rather than hard boundaries).

## Alignment with our taxonomy

| our Gemini-derived category | MorphBPE analogue | match |
| --- | --- | --- |
| Prefix | analyzer's `prefix` segment | direct |
| Stem | analyzer's `root` / `stem` segment | direct for simple stems |
| Compound | two analyzer roots concatenated | analyzer-dependent; most tools flatten to one chunk |
| Ending (inflectional) | analyzer's inflectional suffix | direct |
| Derivational suffix (our adverbial -ως etc.) | analyzer's derivational suffix | direct with a good analyzer |
| Proper noun / Loanword / Function word | treated as unsegmented single units | direct |
| Other (fragment / cutoff) | would not exist | this is the point |

So the taxonomy we built is essentially the output a perfect MorphBPE
run would produce on the Greek vocab.

## Greek-specific challenges that MorphBPE does NOT solve for free

1. **Compound-linking vowel handling**. `δημο-κρατ-ία`: split as one
   compound chunk, as two roots + linking -ο-, or something else?
   Analyzer-dependent. Our Compound bucket implicitly picked the
   "two-root" analysis.
2. **Phonological sandhi at prefix-stem boundary**. ἀντί + aspirate →
   ἀνθ- ; ἀπό + aspirate → ἀφ- ; κατά + aspirate → καθ- ; συν + velar →
   συγ- ; συν + labial → συμ- ; etc. Naive analyzers either (a) restore
   the underlying prefix and lose surface fidelity, or (b) keep surface
   form and lose the link to the base prefix. Neither is ideal.
   See `CATEGORY_ISSUES.md` for examples we found in the current
   vocabulary.
3. **Diachronic coverage.** GlossAPI spans Ancient / Koine / Byzantine /
   Katharevousa / Demotic. Off-the-shelf Greek analyzers are mostly
   Demotic-only. A segmenter for the full diachronic range does not
   exist and would need to be built or stitched together.
4. **Polytonic vs monotonic.** Same word in polytonic (ἄνθρωπος) vs
   monotonic (άνθρωπος) form segments identically in principle but
   orthographically differs. Analyzers trained on monotonic underperform
   on polytonic. Independent problem being tackled in
   `greek-mono-poly-tokenizer-normalization/` (EOWAF).

## Available Greek morphology tools (incomplete survey, 2026)

| tool | register | strength | weakness |
| --- | --- | --- | --- |
| **Morfessor** (unsupervised) | any | trains on any corpus; unsupervised | segmentation quality on Greek is poor; doesn't know prefixes vs stems vs endings |
| **AUEB NLP / Greek-Morph** | Modern Greek (mostly demotic) | high-quality rules + lemmatization | no Ancient/Katharevousa support; not always publicly available |
| **UDPipe Greek model** | Modern Greek | POS + morph features | weak segmentation; no explicit morpheme split |
| **CLTK** (Classical Language Toolkit) | Ancient Greek | ancient morphology, lemmatization | Modern Greek is outside scope |
| **MorphyNet** (Batsuren et al.) | 15 languages inc. Greek | derivational morphology, large data | Modern Greek only; derivational focus |
| **spaCy el_core_news_*** | Modern Greek | lemmatization, POS | no morpheme-level segmentation |

None covers the Ancient→Modern span natively. A bespoke analyzer, or a
hybrid pipeline, would be needed for the GlossAPI corpus.

## Concrete experiment path (if pursued)

1. **Pick / build a segmenter** that handles the GlossAPI diachronic mix.
   Most tractable starting point: run **Morfessor** on the monotonic-
   normalised corpus as a baseline + augment with a rule pass for the
   top ~50 ancient preverb alternants (αντι / αντ / ανθ, απο / απ / αφ,
   etc.).
2. **Train boundary-constrained BPE** with the target vocab size
   matching the current arms (50 k fresh, 25 k extension on top of
   Apertus's 131 k).
3. **Measure fertility and morphological cleanliness**:
   - Bytes-per-token on held-out Greek (target: ≤ current tokenizer).
   - % of tokens in "BPE cutoff" shape (target: ≈ 0 by construction).
   - Classify the new vocab with the same Gemini pipeline; compare the
     Prefix / Stem / Compound / Ending / Other distribution.
4. **Train a small LM comparison run** on the two tokenizers and compare
   loss, nanochat-style. Infra exists in `nanochat_glossapi_en_vs_el/`.
5. **Validate against the existing v2 classifications**. Our 26 016
   Gemini-derived classifications are an oracle: a good MorphBPE vocab
   should reproduce their morpheme boundaries; tokens where they disagree
   are the audit set.

## Gotchas to record before committing to this path

- **Compound-boundary analyzer choice materially changes the vocab.**
  Pick explicitly; don't inherit silently from whatever default the
  segmenter tool uses.
- **Phonological-alternation policy.** Before training, decide: do we
  surface-form the alternants (αντι → ανθ kept as ανθ) or normalize to
  the base prefix (ανθ → αντι)? The choice affects BPE-vocab quality,
  decoding fidelity, and cross-register generalisation.
- **Out-of-analyzer words.** Loanwords, proper nouns, typos, and
  morphology-lacking tokens (numbers, URLs) need a fallback path. Most
  MorphBPE implementations just BPE these as whole units; that's fine.
- **Inference cost of the analyzer.** If Greek morphological
  segmentation adds non-trivial latency per inference call, it becomes a
  production concern. Benchmark early.

## Relationship to EOWAF (sibling project)

EOWAF in `greek-mono-poly-tokenizer-normalization/` attacks a different
axis of the same problem: it preserves morpheme identity across
polytonic / monotonic form variants by separating letter skeletons from
accents. MorphBPE attacks the BPE-merge-cutoff axis by disallowing cross-
morpheme tokens. The two are **complementary**: an EOWAF-encoded corpus
fed into a MorphBPE tokenizer would address both problems at once.
Worth recording as a possible v4-or-later combined approach.

## Why this is "parked" and not "planned"

- Shipping critical path is the merge-rule extension of Apertus's
  existing tokenizer, per `docs/GLOBAL_DECISIONS.md`.
- MorphBPE requires a bespoke Greek segmenter that does not currently
  exist in the repo.
- Value is uncertain until the segmenter is built and measured.
- Needs to wait until the current classification-taxonomy work
  (`unified_classification_20260418/`) is fully exploited as an
  evaluation oracle.

Cross-reference: `CLAUDE.md` lists this file under Parked — pending review.
