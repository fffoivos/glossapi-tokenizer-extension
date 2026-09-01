# Root causes and fixes — v1

Status: **for discussion.** Nothing here has been applied.
Fifteen review findings reduced to eleven causes. Each has **what went wrong** (with the real
examples, not codes), **why**, and **the fix** — what we build, what it costs, and how we will know
it worked.

Companion: `TRANSLATION_SPEC_v1.md` (the rules themselves), `pilot_no_robots_100/FEEDBACK.md` (raw
findings, in the order they were found).

---

## The findings, in plain language

Codes are kept only so the other documents stay cross-referenceable.

| # | what it was |
|---|---|
| F1 | **The Piggy name.** The pig-latin game became κορακίστικα, but the bot stayed *Το Γουρουνάκι* — a name that no longer referred to anything |
| F2 | **How much English is in the Greek text** — measured at 4.94% of all tokens |
| F3 | **Users type Greeklish or English** because switching keyboard is a hassle; we have zero such rows |
| F4 | **`chatbot`** — is that actually what colloquial Greek uses, or did we just never think about it |
| F5 | **Unnatural word order** — *Γύρω στους 21 βαθμούς έχει σήμερα* instead of *Έχει γύρω στους 21 βαθμούς σήμερα* |
| F6 | **"Pamela Pleasantly"** lost its PP alliteration; and *Πάμελα* / *Ντάνιελ* are not Greek names |
| F7 | **The sample wasn't representative** — you were seeing nothing but named chatbots |
| O1 | The two pipeline stages **disagreed about which rows were defective** |
| O2 | The pipeline **silently fixed mistakes in the English source** — 8+ rows |
| O3 | **Greek forced a gender** that the English left open — 7 rows |
| O4 | **38 transliteration decisions** made with no standard behind them |
| O5 | **"Neutral" register wasn't always reachable** — 3 rows were forced to commit |
| O6 | The unit-conversion hazard was **over-flagged** — 10 rows flagged, 1 needed it |
| O7 | Gates fired on **correct** Greek: jargon rows, and the Three Little Pigs "mixed register" |
| D1 | **Your decision:** go hard on transposing into Greek reality |

---

## RC1 · The prompt has no memory from one row to the next

**What went wrong.** The same kind of thing was handled in opposite ways depending on which batch it
landed in. Invented characters were given **Greek names in some categories** (Jenna→Ελένη, Henry
Watson→Ανδρέας Βασιλείου) and **transliterated in others** (Pamela→Πάμελα, Daniel→Ντάνιελ) — 6 versus
28, with nothing in the instructions asking for that split. The word **`chatbot` was chosen 30 times
across 14 rows without a single alternative ever being considered**. Thirty-eight rows made a
transliteration call against no standard. Gender was invented afresh seven times. *(F2, F4, F6, O3,
O4, O5, O7.)*

**Why.** Every row is an independent model call. A decision made in row 3 is invisible in row 40. From
inside the process, "consistent" and "arbitrary" are indistinguishable — there is nowhere for a
decision to live.

### The fix — a style guide the pipeline actually reads, and that grows itself

1. **Write the decisions down once.** `TRANSLATION_SPEC_v1.md` already holds them: the frame/content
   test, register, gender, word order, naming, and a **glossary** seeded with the vocabulary we
   measured (`chatbot` 30 uses, `email` 16, `bot` 12, the piercing jargon, the transliteration
   standard). Right now several entries say *pending* — those are the decisions this discussion needs
   to close.
2. **Inject it into every call**, so it is not a document someone remembers to consult but part of
   the instruction itself.
3. **Make the model account for it.** Two new fields: `guide_applied` (which entries it used) and
   `guide_gaps` (decisions it had to invent because the guide was silent, as `{term, decision, why}`).
4. **Close the loop.** After each run, triage `guide_gaps` into the next version of the guide. Drift
   stops being invisible and becomes a worklist.

The prompt text:

> ## The style guide
> The project style guide and glossary are below. **They are decisions already taken. Apply them; do
> not re-litigate them per row.**
> {{STYLE_GUIDE}}
>
> When you apply a glossary entry, name it in `guide_applied`. When the guide does not cover a
> decision you had to make, make the decision and record it in `guide_gaps`. An empty `guide_gaps`
> is a claim that the guide covered everything in this row.

**Cost:** ~1,500 tokens added per call; negligible in money, and it *reduces* per-row thinking.
**How we'll know it worked:** person names get one treatment across all categories instead of two;
`guide_gaps` shrinks run over run.

---

## RC2 · "Translate" is the default verb, so preserving the surface is the default behaviour

**What went wrong.** The unnaturalness you flagged: *Γύρω στους 21 βαθμούς έχει σήμερα*, *Αλλά
ΚΑΘΟΛΟΥ εύκολο δεν ήταν*, and *Όλα καλά είναι* — the last from a completely neutral English
*Everything is okay*. Separately, *Pamela Pleasantly* kept its spelling and lost its alliteration, and
English words survived in the Greek by inertia rather than by decision. *(F5, F6, F2.)*

**Why.** The prompt's framing verb is *translate*, so the model preserves surface and adapts only
where explicitly licensed. There is also a specifically Greek mechanism at work: **free word order is
a knob English does not have**, and the model reached for it to compensate whenever an English
emphasis device would not carry — marking the same thing twice and overshooting into the theatrical.
*It wasn't easy AT ALL* already marks emphasis with capitals; the Greek kept the capitals **and**
added fronting.

### The fix — change what the task *is*, then name the two behaviours that follow

Your transposition decision (D1) already makes this official; the prompt just has not caught up. The
job is reframed from *translate this row* to *write the Greek row a Greek writer would have produced*,
with fidelity demoted from goal to constraint list. Then two specific rules, because the general
instruction alone will not reliably produce them:

> You are **writing the Greek row that a Greek writer would have produced for this task**. You are not
> translating an English row. The English row is your source of *content and intent*, not of surface
> form. Fidelity is a list of things you must not change — it is not the goal.
>
> **Word order.** Write canonical, unmarked Greek. Greek lets you front a complement for emphasis;
> English cannot, so English marks emphasis by other means. Use marked order **only** when the source
> is itself marked **and** that markedness is not already carried by something you preserved
> (capitals, an intensifier, punctuation). **Never add markedness to compensate for a device that did
> not transfer** — if it does not transfer, let it go.
>
> **Names and catchy phrases.** A name may work by *sound* (alliteration, rhyme), by *transparent
> meaning*, or by *reference*. Transliteration preserves spelling and destroys the first two.
> Identify what the name does and reproduce **that**. A real person or brand is reference — leave it
> alone. An invented character gets a **common Greek name**, never a transliteration.

**How we'll know it worked:** the three sentences above, re-run, come back canonical; *Pamela
Pleasantly* comes back as something like *Γαλήνη η Γλυκιά*; and the naturalness judge (RC9) scores the
re-run above the baseline.

---

## RC3 · A substitution is applied to one thing and not swept across the row

**What went wrong.** The Piggy name. The game changed, the name did not, and the pun's referent
vanished. In the same row, *Pamela Pleasantly*'s alliteration was lost the same way. At a larger
scale, your own coherence point: a half-transposed scene is worse than an untouched one. *(F1, F6, D1.)*

**Why.** The prompt permits substitution but never requires a sweep for whatever depends on the thing
substituted.

### The fix — a propagation step that runs *before* translating, plus a field that forces the check

Placement matters: if the sweep comes after the translation step, it is a review; before, it is a
precondition. And the field is what makes it real — a required list means "I didn't think about it"
becomes a claim the model has to make.

> ## Propagate every substitution
> **Nothing may be replaced in isolation.** Whenever you substitute or localize anything, scan the
> *whole row* — system prompt, every user turn, the reference answer, titles, greetings, sign-offs —
> for elements whose meaning **derives from what you replaced**, and re-derive each from the
> replacement.
>
> A chatbot called **Piggy** is named after **pig latin**. Once the game becomes **κορακίστικα**, the
> name must be re-derived — **Κοράκι** — or the row keeps a name that refers to nothing. The
> substitution succeeded and the row still broke.
>
> Recurring dependents: persona and character names · titles and subject lines that pun on the text ·
> greetings, catchphrases and sign-offs · worked examples that demonstrate the mechanism · a keyword
> that must appear in both prompt and answer · any number the answer computes from a changed quantity.
>
> Record each in `derived_elements`. An empty list is a **claim**, not a skipped field.
>
> **At scene scale:** if you relocate a scenario, relocate it *completely*. A Greek family with Greek
> names eating Thanksgiving turkey is worse than leaving the scene American.

**How we'll know it worked:** the Piggy row comes back with a re-derived name and a populated
`derived_elements`; transposed scenes contain no leftover American props.

---

## RC4 · The classification is made once, by the stage that knows least

**What went wrong.** Stage 1 marked **2** rows as "deliberately defective — do not repair". Stage 2
then behaved as though **5 more** were, naming an ambiguity instead of silently answering. Those five
rows now carry a label that contradicts what they contain. *(O1.)*

**Why.** Stage 1 classifies from the prompt alone. Stage 2 sees the English reference answer and
discovers the defect only when it tries to answer — by which point the class is frozen.

### The fix — give stage 1 the missing information (preferred), or let stage 2 object

**(a) Give stage 1 the reference answer.** Withholding it was an accident of design rather than a
decision. It is the cheaper fix and keeps one source of truth:

> You are also given the English reference answer. Use it **only** to detect properties of the
> *prompt* you could not otherwise see — that it is ambiguous, underspecified, built on a false
> presupposition, or that the reference itself fails the prompt's own constraint. Do not let it
> influence your translation of the prompt.

**(b) Let stage 2 disagree**, as a backstop:

> If, while writing the answer, you conclude stage 1's class is wrong, honour it anyway and record
> `class_disagreement: {assigned, should_be, why}`.

I would do both: (a) prevents most of it, (b) catches the rest and tells us whether (a) is working.

**How we'll know it worked:** the count of `class_disagreement` records is near zero, and the rows
that behave as defective are the rows labelled defective.

---

## RC5 · "Be faithful to the English" was never defined — faithful to *what*?

**What went wrong.** The pipeline silently corrected the human-written English in **8+ rows**:
misspellings (*Alberdeen*, *Nort-East*, *Amepere*), a wrong name (*Sir John Dalton*), a claim that
"only three franchises" did something followed by a list of four, a queen-consort passage that
contradicted itself, and two rows where the English answer violated its own prompt's paragraph count.
*(O2.)*

**Why.** The prompt says to use the English answer as a content anchor, but never says whether its
*errors* count as content. Both readings are defensible, so the model chose row by row.

### The fix — allow repair, forbid *silent* repair, and say so on the dataset card

The corrections are genuinely improvements; the problem is that they were invisible. So the policy is
repair-and-record, with one carved-out exception where the defect is the point of the exercise:

> **The English reference may be wrong.** It was written by human annotators and contains
> misspellings, factual errors, internal contradictions, and rows that violate their own prompt.
>
> **Silent repair is forbidden; repair-and-record is required.** Write the correct Greek and record
> every correction in `reference_corrections` as `{what, english, greek, why}`.
>
> **Exception:** where the defect is the *point* — the row is marked as deliberately defective, or
> the task is to fix the text — reproduce it faithfully and do not correct it.

There is a downstream consequence worth accepting deliberately: combined with your transposition
decision, the result is a **corrected, transposed variant** of no_robots, not a translation of it.
That belongs in the dataset card, not in a footnote.

**How we'll know it worked:** every correction appears in `reference_corrections`; spot-checking
finds no unrecorded ones.

---

## RC6 · The prompt treats ten very different tasks as one task

**What went wrong.** Nothing visible yet — which is the problem. One generic instruction covers all
ten categories, and **five of them were never tested**. The analysis found that the differences are
severe: in Closed QA the answer must stay recoverable from the passage; in Coding, **code is unfenced
in ~206 of 334 rows**, so our fence-based protection would mangle most of it; in Extract the answer
must be a literal span of the source; in Chat the persona *is* the content.

**Why.** The class taxonomy describes *how to translate*. Nothing describes *what kind of object this
row is*.

### The fix — inject the family and category profile from the spec

The spec's key structural finding does the heavy lifting: the ten categories are **two families**, and
branching on family first settles most questions before category-specific rules are even consulted.

> ## What kind of row this is
> Family: **{{FAMILY}}** — {{generative: the answer is composed, the prompt is a short instruction,
> and you have wide latitude to transpose the scenario into Greek reality | source-bearing: the prompt
> contains a source block that is ground truth, the answer derives from it, and your latitude is near
> zero}}
>
> Category: **{{CATEGORY}}**
> Essential features — a Greek version that loses any of these is destroyed: {{CATEGORY_INVARIANTS}}
> Safe to adapt: {{CATEGORY_SAFE}}
> {{CATEGORY_RULES}}

Alongside it, a **structural code detector** for Coding that finds code by indentation and line
shape rather than by fences — otherwise that category fails silently and catastrophically.

**How we'll know it worked:** the proportional re-run covers all ten categories; Coding rows survive
a parse check and an identifier-multiset diff.

---

## RC7 · One label is being asked to describe two separate decisions

**What went wrong.** Every Rewrite row needed a *secondary* class to say what it actually was, because
"literal" there describes only how the source block moves into Greek and says nothing about the
transformation — which is the task. And two Closed QA rows deliberately preserved a misspelling in the
prompt while being labelled "literal", so anyone reading the label alone could not tell the prompt was
intentionally wrong.

**Why.** For rows that contain a source text, *how you move the source* and *what the answer does with
it* are independent choices. One field cannot hold both, so one of them silently loses.

### The fix — two fields for source-bearing rows, one for the rest

> `source_handling` — how the source block moves into Greek:
> `LITERAL` · `PRESERVE-DEFECT` · `VERBATIM-FREEZE` · `LOCALIZE`
>
> `output_handling` — what the answer must do with it:
> `TRANSFORM` (state the target: Q&A, bedtime story, news script…) · `EXTRACT` · `SUMMARIZE` ·
> `CLASSIFY` · `ANSWER`
>
> Generative rows keep the single `translation_class`.

**How we'll know it worked:** no row needs a secondary class to be intelligible; rows that preserve a
defect are labelled as preserving a defect.

---

## RC8 · The note about a defective source is written inside the answer

**What went wrong.** Every one of the five rows that blew the length band was this: the Greek answer
names the source's ambiguity *before* answering, while the English answer just answers. The worst was
**5.26× the reference length** — in Closed QA, where the median answer is **110 characters** and
terseness is the whole point. The Greek row stopped being the same *type* of example as its English
counterpart.

**Why.** Flagging an ambiguity rather than guessing is good behaviour and the prompt encourages it —
but it was allowed to change the shape of the deliverable.

### The fix — your decision: split it by family

Keep the behaviour where it teaches something, move it out of the way where it destroys the form:

> **If the source is defective or ambiguous:**
> - **generative rows** — say so *in the answer*, as a good assistant would: name the ambiguity, then
>   answer what you can. This is part of what we are teaching.
> - **source-bearing rows** — the answer keeps the reference's shape and length. Put the observation
>   in `defect_note` instead, and answer as directly as the reference does.

**How we'll know it worked:** the length band tightens; no source-bearing row runs multiples of its
reference.

---

## RC9 · The checks test correctness; nothing tests whether it reads well

**What went wrong.** The unnatural word order you found is **invisible to automated checking**. I
tried: a regex scan over the whole corpus returned **11 hits, 1 of them real**, and **missed 2 of the
3 known cases**. The register check flagged the Three Little Pigs row as "mixed register" because the
wolf addresses one pig and then two — that is grammatical **number**, not formality. Meanwhile
**99 of 100 rows pass every hard gate**, and the Chat batch passed with **zero flags and zero
advisories** while containing both defects you found. *(F5, O7.)*

**Why.** The gates test what a regex can see — script, encoding, ratios, lengths. Naturalness is not
that kind of property, and no amount of tuning will make it one.

### The fix — prevent it in the prompt with worked examples; detect only to check that it worked

**Corrected 2026-08-24 (owner).** I framed this as a detection problem and reached for a judge. Wrong
order. **Fix it at generation time with before/after examples in the prompt** — the same mechanism as
RC2 — so the marked word order never gets written. Detection then only has to answer *did the
examples work*, which is a **sample-scale** question a spot-check can answer, not a row-scale gate.

Consequences: no corpus-wide naturalness detector is needed; the 9-row re-run tests the fix on the
exact rows where it failed; a judge, if used at all, ranks a sample rather than gating the corpus.

Superseded below (kept for the record):

This is the one cause with no cheap fix, so the honest response is to stop pretending and re-allocate:

1. **A stage-3 naturalness pass** with an explicit rubric, scoring each row on: *word order* —
   canonical, or marked, and is the markedness in the source? · *register consistency* — one address
   form throughout, matching the prompt and every other turn? · *idiom* — any phrase reading as a
   calque of English syntax? · *naturalness* — would a Greek writer produce this unprompted?
2. **Its job is to rank, not to reject.** It sorts your review queue so the worst rows arrive first.
3. **Re-point the human budget.** The deterministic gates already have correctness at 99/100 and are
   structurally incapable of the rest — so your reading time is worth far more spent on naturalness
   than on correctness.
4. **Calibrate before trusting it**: ~100 hand-checked native Greek pairs, because every published
   Greek reward-model result is on translated benchmark data.

**How we'll know it worked:** the rows the judge ranks worst are the rows you would have flagged.

---

## RC10 · Three of the checks measure the wrong thing

**What went wrong.** The pilot's **only** hard failure was a row keeping English piercing terminology
(*septum*, *bridge*, *anti-eyebrow*) — arguably correct Greek usage. The lowest-scoring Chat row kept
Tarkov game vocabulary, also correct. A single formal verb form in turn 3 of a Chat row was invisible
because the check ran on all turns joined together. And all three Rewrite register warnings were rows
where the check measured *the rewritten advert addressing its reader* while the target described *the
assistant addressing the user* — two different relationships reported as one. *(O7.)*

### The fix — three targeted corrections, and a rule about new gates

- **Greek-ratio floor → per-row allowlist.** Build the allowlist from that row's own recorded
  localization decisions and glossary entries, and flag only Latin tokens *not* on it. A global floor
  cannot distinguish correct jargon from leakage; a per-row list can.
- **Register check → per turn.** Report a within-row inconsistency separately from a mismatch against
  the prompt.
- **Register in source-bearing rows → two axes.** Assistant-to-user and deliverable-to-its-reader are
  measured separately and never compared.

And a standing rule earned the hard way: **`gates.py` was wrong three times, and every single sigma
flag across all three versions was a false positive** — elisions (*σ' αγαπώ*), an abbreviation
(*κ.σ.*), and one row where the Greek faithfully mirrored the English source's own run-together
*Singersongwriteractress*. So: **new gates start as advisories** and earn hard-flag status only after a
clean pass on real data.

**How we'll know it worked:** the piercing and Tarkov rows stop being flagged; the single-formal-turn
row starts being flagged.

---

## RC11 · The corpus represents one slice of reality

Two findings, one cause: what we sampled is not what exists.

**(a) The sample wasn't representative — the thing you noticed.** It was stratified for
stress-testing, not proportion. Closed QA was over-represented **7.75×**, Chat **2.39×**, and
Generation *under*-represented at **0.44×** despite being 45.8% of the dataset. **Five categories —
24.3% of the data — were absent entirely**, including Coding. Re-weighting moved real numbers:
transposable scope 20% → **28.6%**, and constraint-bearing rows turn out to be the **largest class
corpus-wide at ~37%**, not literal ones. *(F7.)*

**(b) Users type Greeklish or English — your point, and we have none of it.** All 100 prompts are
clean, correctly-accented Greek *because they were produced by translation*. We have one input mode of
at least four: clean Greek (100 rows), Greeklish (**0**), English-typed-expecting-Greek (**0**),
code-switched (**0** as input). *(F3.)*

### The fix — two samples now, one new stream later

- **Sample A — the 9 rows you have reviewed**, re-run for a clean before/after on the Piggy name, the
  word order, and the Pamela alliteration. Small, fast, and it verifies the fixes exactly where the
  problems were found.
- **Sample B — a proportional 100**: Generation 46 · Open QA 12 · Brainstorm 11 · Chat 8 · Rewrite 7 ·
  Summarize 4 · Coding 4 · Classify 4 · Closed QA 3 · Extract 2. Covers all ten categories and finally
  tests Coding.
- **A Greeklish / code-switched stream**, built by transliterating Greek prompts we already have and
  pairing them with the same Greek answers — so one row becomes a robustness pair. The model always
  answers in well-formed Greek regardless of input script; we do not want it *producing* Greeklish.
  One practical catch: **no language-ID tool detects Greeklish** — it reads as English or Welsh with
  high confidence — so this stream must be exempted from that gate or it will all be rejected.

**How we'll know it worked:** the re-run's numbers are corpus-level rather than sample-level, and
Coding either passes its parse check or fails loudly.

---

## What I would do first

1. **RC1 — the style guide.** Most findings, cheapest fix, and it compounds: every run improves it.
   Blocked only on closing the *pending* glossary entries.
2. **RC6 — category profiles**, with the structural code detector. Unblocks the five untested
   categories; Coding is a silent catastrophe today.
3. **RC2 — reframe from translating to writing.** This targets the unnaturalness, which is the thing
   a reader actually feels.
4. **RC9 + RC10 — fix what the checks measure, then add the judge.** Without this we cannot tell
   whether RC2 worked.
5. **RC3, RC5, RC7, RC8** — precise, bounded, one class of defect each.
6. **RC4** — smallest effect, but stage 1 is being edited anyway.
7. **RC11** — changes what we can *claim*, not what we produce.
