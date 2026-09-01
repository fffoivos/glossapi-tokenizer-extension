# Pilot 1 — review working document

Living document for the 100-row no_robots pilot. Owner feedback, my analysis, and the
cross-cutting themes we should resolve **together at the end** rather than patching one at a time.

Artifact: https://claude.ai/code/artifact/927a2ebe-8ad9-4519-aafd-64759554cc30

> ⚠️ **Read every rate in this document with F7 in mind.** The 100 rows are a **stress sample**
> (20 each from 5 of 10 categories), not a proportional one. Chat is over-represented 2.39×,
> Closed QA 7.75×, Generation *under*-represented at 0.44×, and five categories (24.3% of the
> dataset) are absent entirely. Where a corpus-level estimate exists, F7 gives the re-weighted figure.

**Working rules for this review**
- The artifact is **frozen while the owner reads**. No republishing — a republish reloads the page
  and loses the reader's place. Changes are staged locally and shipped in one batch on request.
- Findings get logged here, not fixed immediately. Point fixes during a review create a moving
  target and hide the pattern; we look for the general rule once the whole sample has been seen.
- Owner words are quoted verbatim and kept separate from my analysis.

Status key: **open** · **prompt-staged** (change written, not run) · **fixed** (verified in output) ·
**wontfix** (deliberate, with reason)

---

# Part 0 — Decisions taken

## D1 · Hard transposition into Greek reality (owner decision, 2026-08-24)

> Maybe we should go hard on not just translating but translating entities, places, events in Greek
> reality, in effect change it enough so that it is coherent, with the same style and tone, in Greek
> reality

**Decided:** the target is not a translation of `no_robots`. It is **Greek-native instruction data
seeded by `no_robots`** — invented entities, places and events are relocated into Greek reality, and
relocated *far enough to be internally coherent*, preserving the original's style, tone, register and
task shape.

### The operational test

The word doing the work in the decision is **coherent**. Half-transposition is worse than none: a
Greek family with Greek names sitting down to Thanksgiving turkey is more jarring than leaving the
whole scene American. So transposition has to be **complete within a scenario** — which is F1's
propagation rule applied at the scale of a scene rather than a name.

The boundary that makes this safe, stated as a single test:

> **Would swapping this entity change whether the answer is true?**
> **Yes → it is content. Freeze it.  No → it is frame. Transpose it.**

Applied: *Aberdeen* in "why is Aberdeen called the Granite City" is content — the answer is about
Aberdeen's granite. *Jenna*, in "write an email to my daughter Jenna", is frame — nothing about the
email is true or false because of her name. The clean illustration is row `27ad5dbc`, the only row in
the sample flagged both invented *and* culture-bound: *"a casual email to a friend named Bowen about
the Legend of Faust"* — **Bowen transposes, Faust does not.** Transpose the frame, keep the content.

### Scope — measured over the 100, so the cost is known

| bucket | rows |
|---|---|
| **TRANSPOSABLE** — invented frame | **20** |
| LOCKED — a real entity is the subject | 52 |
| LOCKED — a source text is the object of the task | 12 |
| NEUTRAL — nothing culture-bound either way | 16 |

And it is highly concentrated by category:

| category | transposable | locked | neutral |
|---|---|---|---|
| **Chat** | **12** | 8 | 0 |
| **Generation** | **7** | 4 | 9 |
| Open QA | 1 | 12 | 7 |
| Rewrite | 0 | 20 | 0 |
| Closed QA | 0 | 20 | 0 |

**19 of the 20 transposable rows are Chat or Generation.** Rewrite and Closed QA are structurally
immune, because both carry a source text that *is* the object of the task. So the policy is
aggressive in principle but bounded in practice: it touches about a fifth of rows, in two categories.

### What this commits us to

- The corpus is deliberately **mixed**: Greek-native chat and creative writing, faithful translation
  for factual and source-bearing tasks. That is coherent — it mirrors what a Greek assistant actually
  does: it chats in a Greek social world and answers factual questions about the whole world.
- It **compounds O2**. Combined with the 8 rows that corrected the source's errors, the result cannot
  honestly be described as "no_robots translated into Greek". It should be named and documented as a
  Greek-native derivative, with `no_robots` credited as the seed and the CC-BY-NC terms still carried.
- Every transposition must be **recorded**, not just made — a `transposition` field alongside
  `localization_decisions`, so a reviewer can audit the frame/content calls rather than re-derive them.
- The **failure mode to gate for is a mixed frame**, not an over-eager one: a row that is half-Greek
  and half-American. Worth an explicit check in stage 3.

**Status:** decided in principle; the prompt changes and the frame/content field are staged work for
the end of the review.

---

# Part 1 — Owner feedback

## F1 · `6d2fe44a` — substitution not propagated to a dependent name

> Piggy is related to pig-latin, so it could be κοράκι for κορακίστικα.

**What happened.** Stage 1 correctly reclassified the row REGENERATE-NATIVE and replaced the game
(pig latin → κορακίστικα) but kept the persona name as a literal translation of the English:
`Το Γουρουνάκι`. In English the name **puns on the mechanism** — Piggy ← pig latin. Once the
mechanism changes, the pun's referent is gone and the Greek carries a name that no longer means
anything. Target: `Το Κοράκι` / `Κορακάκι`, re-derived from κορακίστικα.

**Why it generalises.** The substitution was applied to one element in isolation while another
element still derived its meaning from the old value. Same shape as: a character named for the
language being taught; an acrostic whose letters spell a word used elsewhere; a keyword that must
appear in both prompt and answer; a number the answer computes from a quantity that was localized.

**Staged change** (stage 1, new Step 4 "Propagate every substitution"): nothing may be replaced in
isolation; scan the whole row for elements whose meaning derives from what was replaced and
re-derive each from the replacement. New required output field `derived_elements`, which must be an
explicit empty list when there are none — a claim, not a skipped field. Stage 2 must honour it
rather than re-translating the English names.

**Status:** prompt-staged. Needs a re-run to verify, plus native ratification of the κορακίστικα
encoding rule (the model self-flagged that regional variants differ).

---

## F2 · How many English words are in the Greek text

> We have also to consider about the number of words that are English in the Greek text.

**Measured over all 100 rows** (prompts + responses, code/URLs stripped): **1,687 of 34,125 tokens
are Latin-script = 4.94%**, across **539 distinct** Latin tokens.

The important finding is that this single number conflates **three different things**, which is
exactly why the ratio gate misfired on the piercing row (O7):

| kind | example | verdict |
|---|---|---|
| **Proper nouns and titles** | `Kill Bill`, `Lady Gaga`, `Tarantino`, `Charles Stewart III`, `The Blue Shift` | Legitimate. Includes the English function words `the/of/in/and/a` — ~100 occurrences, almost all *inside* preserved English titles. |
| **Loanwords in Latin script** | `chatbot` (30×), `email` (16×), `bot` (12×), `online`, `check` | **The real policy question.** These are choices, made silently and consistently. |
| **Untranslated leakage** | `piercing`, `football`, `loot`, `quarter`, `flats` | Some are jargon with no settled Greek term; some are probably just leakage. |

Only **97 distinct lowercase Latin tokens (393 occurrences)** exist — i.e. the non-proper-noun
vocabulary is small enough to enumerate and legislate **term by term**, which is far better than any
ratio threshold. A ratio can only ever tell us that *something* is off; a glossary tells the writer
what to do.

**My proposal for the end:** stop treating this as one number. Report the three kinds separately,
and govern kind 2 and 3 with a **glossary** in the style guide rather than a threshold. A ratio gate
stays only as a coarse tripwire for the leakage kind.

**Status:** open — needs a glossary decision, see F4.

## F3 · The user might write in Greeklish or in English

> a greek user has to change keyboard language while writing and for convenience they might not do it

**This is the most consequential item in the review so far, and it is not a translation bug — it is
a corpus coverage gap.** Every one of our 100 prompts is clean, well-formed, correctly accented
Greek, because it was *produced by translation*. Real Greek users frequently do not type that way,
and a model trained only on clean Greek input will be weakest exactly where real usage is messiest.

At least four input modes exist in the wild, and we currently have data for **one**:

1. **Clean Greek** — 100/100 of our rows.
2. **Greeklish** — Greek typed in Latin characters (`ti kaneis`, `pos mporo na...`). **0 rows.**
3. **English input** — the user writes the request in English but wants a Greek answer, or is
   indifferent. **0 rows.**
4. **Code-switched** — Greek sentence with English technical nouns, or a Latin-script word dropped
   mid-sentence. Present incidentally in our *outputs*, never as *input*.

**The behaviour we need to decide and then teach:** what should the model answer in? My
recommendation is **always well-formed Greek, whatever the input script** — answering Greeklish with
Greeklish would train the model to *produce* Greeklish, which we do not want in a model meant to
write good Greek. But that is a decision, and it needs to be represented in the data to be learned.

**Why this is cheap to fix and worth doing properly:**
- Greeklish inputs can be **generated systematically** from the Greek prompts we already have
  (transliterate the prompt, keep the Greek answer) — turning one row into a robustness pair.
- The transliteration is not arbitrary: `AUEB-NLP/ByT5_g2g` via `gr-nlp-toolkit` (Apache-2.0) is a
  real Greeklish↔Greek model we can run in reverse to produce plausible Greeklish variants.
- Real evidence for what Greek users actually type exists: the **619 Greek conversations in
  WildChat-1M** (ODC-BY) are genuine user prompts, not translated ones. Those should be inspected
  before we design the stream, rather than us guessing the distribution.
- Note the detection asymmetry we already established: **no language-ID tool can recognise
  Greeklish** — it reads as English, Welsh, or Tsonga with high confidence. So this stream also has
  to be exempted from the LID gate, or the gate will reject it all.

**Status:** open — this is a new stream in the corpus plan, not a prompt fix.

## F4 · Is `chatbot` really what Greek uses?

> there seem to be very common words like "chatbot" that we should seriously think about if that is
> used verbatim in colloquial greek, or if there are better terms for it

**Measured:** `chatbot` is the **single most frequent Latin-script token in the whole pilot** — 30
occurrences across 14 rows. And the alternatives appear **zero times**: `τσατ` 0, `μποτ` 0, `ρομπότ`
0, `βοηθός` once. So the pipeline made the same lexical choice 30 times without ever considering an
alternative, which is precisely the kind of silent consistency that should be a deliberate decision.

Candidates: `chatbot` (Latin, as-is), `τσατμπότ` (transliterated), `ρομπότ συνομιλίας` /
`συνομιλιακό ρομπότ`, `ψηφιακός βοηθός`, `βοηθός`. The honest argument for the status quo is that
colloquial Greek tech register does borrow English terms in Latin script very freely. The argument
against is that these strings sit in **system prompts defining the assistant's own identity**, where
a model that is supposed to write native Greek describing itself in English reads oddly.

`email` (16×, 8 rows) and `bot` (12×, 6 rows) are the same question and should be decided together,
not one at a time — which is the argument for a glossary rather than per-row judgement.

**Status:** open — first entries in the glossary.

## F5 · Unnatural word order — marked constituent fronting

> I find sometimes that there are unnatural expressions
> eg in `86786cab` "Why, it's 70 °F today" → "Γύρω στους 21 βαθμούς έχει σήμερα" … I would write here
> "Έχει γύρω στους 21 βαθμούς σήμερα"
> and in `89c788bc` "But, It wasn't easy AT ALL" → "Αλλά ΚΑΘΟΛΟΥ εύκολο δεν ήταν" and here
> "Αλλά δεν ήταν ΚΑΘΟΛΟΥ εύκολο"

**Verified in the output**, plus a third instance in the same row that was not flagged:

| row | stage | produced | canonical |
|---|---|---|---|
| `86786cab` | 2 (assistant) | Γύρω στους 21 βαθμούς **έχει** σήμερα | Έχει γύρω στους 21 βαθμούς σήμερα |
| `89c788bc` | **1 (user turn)** | Αλλά ΚΑΘΟΛΟΥ εύκολο **δεν ήταν** | Αλλά δεν ήταν ΚΑΘΟΛΟΥ εύκολο |
| `89c788bc` | 2 (assistant) | **Όλα καλά είναι**, Ντάνιελ! | Όλα είναι καλά / Όλα καλά! |

**It is in both stages.** One instance is stage 2 generating, one is stage 1 *translating a user
turn* — so a fix has to go into both prompts, not just the generator.

**Mechanism — over-marking by compensation.** Greek has relatively free word order, so fronting a
complement is always *grammatically available*. The model appears to reach for it when the English
carried an emphasis or tone device it cannot transfer directly, and then marks the same thing twice.
`It wasn't easy AT ALL` marks emphasis with capitals **and** the post-verbal `at all`; the Greek kept
the capitals (correct — that transfers) **and** added syntactic fronting on top. Two markers where
English had one, so the Greek overshoots into something theatrical. `Everything is okay` is entirely
neutral in English and came back marked in Greek for no source reason at all.

This is the classic translationese signature, but with a specifically Greek mechanism: free word
order gives the model a knob English does not have, and it turns it when it feels it has lost
something elsewhere.

**Answering your question — yes, fronting does serve a purpose, but not here.** It is correct and
natural under **contrastive focus** (*Γύρω στους 21 έχει, όχι 30*), when the fronted element is the
narrow answer to a *how much / what exactly* question, and in proverbial or literary registers where
the markedness is the point. The test is whether the **source** is itself marked. In all three cases
above it was not: the English was neutral or marked by a device that was already carried over
separately.

**Detection is hard, and I want to be honest about how hard.** I tried to find these automatically
with regex over the whole corpus: it returned **11 hits, of which 1 was a true positive**, and it
**missed two of the three known instances** — one because it sat in a user turn rather than a
response, one because the pattern shape did not match. Marked word order is not regex-detectable;
`Η επαγγελματική εξουθένωση δεν είναι μόδα` is perfectly canonical and looks identical to a fronted
predicate from a pattern's point of view. This needs either a parser or, more practically, an
**LLM-judge rubric** ("is the word order canonical? if marked, is the markedness present in the
source?") — or a native reader. Same lesson as the register gate in O7: the checks that matter most
here are the ones a cheap heuristic cannot do.

**Proposed rule for the style guide:** default to canonical, unmarked Greek word order. Use marked
order **only** when the source is itself marked *and* the markedness is not already carried by
another device you have preserved (capitals, an intensifier, punctuation). Never add markedness to
compensate for something lost elsewhere — if a tone device does not transfer, let it go.

**Status:** open — needs a rule in both prompts, and a decision on whether stage 3 gets an
LLM-judge naturalness pass.

## F6 · Names that work by sound, and how far to transpose culturally

> "Η Πάμελα η Ευχάριστη" from "Pamela Pleasantly" — these names are meant to have a soundbite quality
> to them, in this case it is the PP effect … maybe a more general formulation of the instruction
> than the "Piggy" thing … when we have rhymes, catchy names or phrases, even places and so on,
> events and so on; maybe we should transfer the whole thing to the Greek cultural sphere …
> Maybe we shouldnt even be considering rare Greek names like Pamela and Daniel

### How common is this? Measured over the whole `no_robots` train split

- **795 Chat rows**; **773 (97%)** open with a named persona — so naming is essentially universal in
  this category.
- Only **22 names are multi-word**, but **10 of those 22 (45%) are alliterative**: *Dynamite Dan,
  Tilly Tolerance, Movie Max, Helpful Harry, **Pamela Pleasantly**, Annie Ambitious, Hippie Howard,
  Adj Add, Matt Murdock*, plus *Meow Meow* (reduplication).

**Read:** soundbite names are **rare in absolute terms — ~10 rows of 795, 1.3%** — but the pattern is
deliberate: when the author used two words, they made it alliterative about half the time. So this
does not justify a subsystem, but it does justify a rule — and the same rule covers rhymes, catchy
phrases, and puns elsewhere (Generation carries the poems and wordplay).

### A finding your comment exposed: we are already inconsistent, and it splits by category

Classifying every person-name decision in the 100 rows: **6 localized to Greek names, 28
transliterated, 9 kept in Latin.** The split is not random —

| category | treatment | examples |
|---|---|---|
| Generation / Rewrite | **localized** | Jenna → Ελένη · Henry Watson → Ανδρέας Βασιλείου · George → Γιώργος · Ann → Άννα · Amy → Άννα · Shailee → Σοφία |
| Chat | **transliterated** | Pamela → Πάμελα · Daniel → Ντάνιελ · Kurt → Κερτ · Felicia → Φελίσια · Hannah → Χάνα · Richard → Ρίτσαρντ · Marmee → Μαρμί |

**The same kind of entity — an invented character carrying no factual content — was treated in
opposite ways depending on which category it appeared in.** Nothing in the prompt asked for that; it
is exactly the ad-hoc drift predicted by Theme A, now demonstrated rather than asserted. And your
point lands precisely: *Πάμελα* and *Ντάνιελ* are not Greek names, they are transliterations that
read as foreign, so the Chat personas ended up in an uncanny middle — Greek script, foreign people.

### The general formulation (better than the "Piggy" rule)

F1 was about **dependency**: B derives from A, so replacing A forces B. This is a different
property — **function vs form**. A name can do work through:
- **sound** — alliteration (*Pamela Pleasantly*), rhyme, reduplication (*Meow Meow*);
- **transparent meaning** — *Pleasantly*, *Helpful*, *Ambitious* describe the persona;
- **reference** — a real person or brand (*Charles Dickens*, *Top Gun*), which must NOT be touched.

Transliteration preserves *form* and destroys *function*. The rule should be: **identify what the
name does, and reproduce that, not its spelling.** *Pamela Pleasantly* does two jobs at once —
alliteration plus a transparent adjective — so the Greek should ideally do both. Candidates:
«Γαλήνη η Γλυκιά» (Γ-Γ, and *Γαλήνη* genuinely means calm, matching the persona's brief),
«Χαρά η Χαρωπή», «Καλλιόπη η Καλοσυνάτη». Your call — you have the ear.

### The bigger fork: how far do we transpose?

Your proposal — *transfer the whole thing to the Greek cultural sphere* — is a real design decision,
not a detail, and it interacts with everything above. The boundary we already drew (LITERAL for real
entities, LOCALIZE for invented ones) survives it: **Aberdeen must stay Aberdeen.** What changes is
how *aggressive* localisation is on the invented side:

| stance | invented character | invented setting | catchy name | cultural reference |
|---|---|---|---|---|
| **current** (implicit, inconsistent) | sometimes Greek, sometimes transliterated | usually kept | transliterated | usually kept |
| **full transposition** (your proposal) | common Greek name | Greek setting | re-created for effect in Greek | swapped for a Greek equivalent |

Arguments for full transposition: the corpus reads native; the model gets Greek cultural grounding
it will otherwise lack; translationese drops. Arguments against: it drifts further from `no_robots`
(compounding O2 — we would no longer be publishing a translation at all), and it costs more per row.

My read: **take it**, but bound it explicitly — transpose freely inside invented scenarios, never
across a factual claim, and record every transposition. The risk is not transposing too much, it is
transposing something that was actually a fact.

**Status:** open. Three separate decisions: (a) the function-over-form naming rule, (b) one
consistent person-name policy across all categories, (c) how far cultural transposition goes.

## F7 · The sample is not representative — and I under-stated that

> if there are only 700 named chatbots, then the sample you used was not representative because I am
> only seeing named chatbots so far … are there actually metadata that you can use to make the 100
> representative?

**You are right.** The 100 was **stratified, not proportional** — 20 rows from each of 5 categories,
chosen so each would stress a *different* failure mode. That is a defensible design for finding bugs
fast, but I described it as "chosen because each stresses a different way translation can fail"
without ever saying plainly that **the resulting frequencies do not describe the dataset**. The
reading experience you are having is the direct consequence: Chat is 20% of what you are reading and
**8.4%** of what actually exists.

### How skewed

| category | in split | % of split | in pilot | over/under |
|---|---|---|---|---|
| Generation | 4,346 | **45.8%** | 20 | **0.44× — under** |
| Open QA | 1,182 | 12.4% | 20 | 1.61× |
| Brainstorm | 1,060 | 11.2% | **0** | **absent** |
| Chat | 795 | 8.4% | 20 | **2.39×** |
| Rewrite | 625 | 6.6% | 20 | 3.04× |
| Summarize | 395 | 4.2% | **0** | **absent** |
| Coding | 334 | 3.5% | **0** | **absent** |
| Classify | 334 | 3.5% | **0** | **absent** |
| Closed QA | 245 | 2.6% | 20 | **7.75×** |
| Extract | 183 | 1.9% | **0** | **absent** |

**Five categories — 24.3% of the dataset — are entirely missing.** The most consequential absence is
**Coding**: it is the natural home of `VERBATIM-FREEZE`, so we currently have **one** row of evidence
about how the pipeline handles code, despite code being a known hazard.

### Re-weighting the findings to corpus level

Per-category rates from the pilot, re-weighted by true category shares (covering the 75.7% of the
dataset the pilot touched):

| finding | pilot figure | corpus estimate |
|---|---|---|
| transposable (D1 scope) | 20% | **28.6%** |
| locked | 64% | **38.5%** |
| culturally neutral | 16% | 32.9% |
| class = CONSTRAINT-PRESERVING | 18% | **37.3%** |
| class = LITERAL | 50% | 28.8% |
| class = LOCALIZE | 5% | 9.9% |
| register = neutral | 63% | 63.6% |
| register = ενικός | 31% | 29.2% |

Two of these matter:

- **D1's scope is larger than I told you** — ~29% of the corpus is transposable, not 20%, because
  Generation is 46% of the data and is 35% transposable, and I under-sampled it.
- **`CONSTRAINT-PRESERVING` is the biggest class in the corpus at ~37%, not `LITERAL`.** The pilot
  inverted this. That re-prioritises everything: the Greek constraint checker — which we already know
  is broken in four ways and needs rebuilding from EuroEval's — is the single highest-leverage piece
  of the pipeline, not a side issue.
- Register is the one headline that **does** generalise: 63% neutral either way.

### Answering the metadata question directly

The dataset carries exactly four fields: `prompt`, `prompt_id`, `messages`, `category`. **`category`
is the only native stratification key there is.** Everything else has to be derived, and the obvious
derived features turn out to be redundant with it: multi-turn rows (794), rows with a system prompt
(794) and Chat rows (795) are **the same rows**, so those axes add no information beyond category.

The derived feature that *does* add information is **length**, especially inside Generation, which is
nearly half the corpus and by far the most heterogeneous (prompt chars p25/p50/p75/p95 =
86 / 157 / 303 / 1,685; answers 282 / 601 / 1,201 / 2,110).

### What I propose

Keep both samples, and label them honestly for what they are:

1. **Pilot 1 (this one) = the stress sample.** Its job is finding failure modes, which it did well —
   every finding F1–F6 came out of it. Its frequencies are not evidence about the corpus, and the
   document should say so wherever a rate is quoted.
2. **Pilot 2 = a proportional sample of 100**, drawn to the real distribution: Generation 46,
   Open QA 12, Brainstorm 11, Chat 8, Rewrite 7, Summarize 4, Coding 4, Classify 4, Closed QA 3,
   Extract 2 — with Generation's 46 stratified across length bands. That sample answers "how often",
   covers the five blind categories, and is the one to measure accept/edit/reject rates on.

**Status:** open — needs your go-ahead on running pilot 2 before the corpus-level rates in this
document can be trusted.

---

# Part 2 — My observations, awaiting your verdict

Grounded in counts over the actual output, not impressions. These are *candidates* for the tally —
none is acted on.

## O1 · The two stages disagree about what counts as a defect

Stage 1 labelled **2** rows `PRESERVE-DEFECT`. Stage 2 then behaved as if **5 more** were defective
— naming an ambiguity or refusing to silently repair — on `ca55b8ed`, `cd643d95`, `89a32a4e`,
`99bc89eb`, `4b9a2990`.

Stage 2 is better placed to notice, because it sees the reference answer and discovers the flaw when
it tries to answer. But the class is assigned in stage 1 and never revised, so those rows carry a
label that contradicts their content. **Question for the end:** should stage 2 be allowed to
*propose* a class change (a `class_disagreement` field), or should defect detection move into a
cheap pre-pass that sees both prompt and reference?

## O2 · The pilot silently improved the human data — 8 rows

`d72eb4b7`, `dca7c1d3`, `1f4757a0`, `02e03bb7`, `9f90f44f`, `316a8caf`, `96c2d5a6`, `7210810f`
corrected errors in the *human-written* English reference: misspellings (“Alberdeen”, “Nort-East”),
a factual contradiction (“only three franchises” followed by four), a wrong name (“Sir John
Dalton”), an internal contradiction about a queen consort's powers.

This is good for the model but has a consequence worth deciding deliberately: **the Greek set is
then not a translation of no_robots, it is a corrected variant of it.** That is defensible and
arguably better, but it should be a stated policy with a `reference_corrections` field, not an
emergent behaviour. It also affects how we describe the dataset if it is ever released.

## O3 · Greek forces grammatical gender that English leaves open — 7 rows

`425a5425`, `8f90d1c6`, `f23bcac0`, `e9dd9385`, `31c6605a`, `89c788bc`, `e3cfb165`. English "thank
you, you've been wonderful" carries no gender; the Greek adjective or participle must choose one.
The model picked, and flagged that it had.

This is a **systematic bias-injection channel** — every such row teaches the model an assumption the
source never made. Options: (a) infer from context and accept it, (b) prefer constructions that
avoid gendered agreement, (c) treat it as a localisation decision that must be recorded, (d) split
the row. Needs one policy, applied everywhere.

## O4 · 38 rows required a transliteration decision, with no standard behind it

Σρέντινγκερ, Μπαρτολντί, Αμπερντίν, Latin-script NBA team names, `Ford Pinto` kept as-is,
`Sykkuno` kept as-is. Each call was reasonable and individually justified, but **there is no rule**,
so the corpus will be internally inconsistent at scale. Candidates: ΕΛΟΤ 743, "commonest Greek
usage", or "keep Latin script for brands/handles, transliterate persons and places".

## O5 · "neutral" register is not always reachable — 3 rows

`ba5caa49` (→ ενικός), `e9dd9385` (→ πληθυντικός), `dc1b3cb2` (→ πληθυντικός). All three were
targeted `neutral` by stage 1 and then committed to an address form by stage 2, because numbered
instructions and direct-address advertising copy force one.

That suggests `neutral` should be defined as **"no preference — choose and report"** rather than
"must avoid committing". Worth fixing in the prompt wording, because it currently reads as an
instruction the writer sometimes cannot obey.

## O6 · Unit hazard over-flagged

10 rows carried `us_specific_unit_or_currency` but only **1** actually needed a conversion. Harmless,
but if hazards are ever used to route review effort, this one is noisy.

## O7 · Carried over from the build (documented in README)

- **The gate suite was wrong three times**; every sigma hard-flag across v1–v3 was a false positive.
- **The register gate cannot separate formality from grammatical number** — the Three Little Pigs
  row reads as "mixed register" because the wolf addresses one pig, then two. No marker-counting
  heuristic fixes this; it is the standing argument for human review of register-critical rows.
- **One genuine open call:** `9602a769` keeps English piercing jargon (*septum*, *bridge*,
  *anti-eyebrow*) at a 0.76 Greek ratio. Correct usage or leakage? Native-speaker decision.

---

# Part 3 — Cross-cutting themes forming

The findings are not eight independent patches. They cluster into **three problems**, and F3 turns
out to belong to a different category from everything else.

### Theme A — We keep making the same decision ad hoc, per row

F2 (which English words survive), F4 (`chatbot`/`email`/`bot`), F6 (person names), O3 (gender),
O4 (transliteration — 38 rows), O5 (what `neutral` register means), O7 (foreign jargon). Every one
was decided sensibly in isolation and **guaranteed to drift at scale**, because nothing records the
decision for the next row.

F6 turned this from a prediction into a measurement: the *same* kind of entity — an invented
character — was **localized to a Greek name in Generation and Rewrite rows, but transliterated in
Chat rows** (6 localized vs 28 transliterated overall). No instruction asked for that split. This is
the clearest evidence that the style guide is the load-bearing artefact, not a nicety.

→ **One artefact: a project style guide with a glossary**, cited by both stage prompts. The glossary
is the part that F2's measurement makes tractable — only 97 distinct lowercase Latin tokens exist in
the whole pilot, so the vocabulary needing legislation is small and enumerable. Governing this with
a *ratio threshold* was always going to fail; governing it with a *word list* will not.

### Theme B — Consistency within a single row

F1 (a substitution not propagated to a dependent name) and O1 (the two stages disagreeing about
whether a row is defective). Both are failures of *internal* coherence: one part of a row was
changed and another part was left pointing at the old state.

→ Two decisions: the **propagation rule** (staged), and **whether stage 2 may revise stage 1's
class**, i.e. whether the pipeline is one-directional or has a feedback edge.

### Theme C — What the corpus is *of*

O2 (we silently corrected 8 human-written references) and, much more importantly, **F3 (we have zero
Greeklish, zero English-typed, and zero code-switched input)**.

These are not quality bugs — the output is fine. They are questions about **what the dataset
represents**. O2 means the Greek set is a *corrected variant* of no_robots rather than a translation
of it. F3 means the set represents **one input mode out of at least four**, and specifically not the
messy ones real Greek users produce when they cannot be bothered to switch keyboard.

→ F3 is the one finding so far that **changes the corpus plan rather than the prompts**. It adds a
stream, and cheaply: existing Greek prompts can be transliterated to Greeklish and paired with the
same Greek answers.

### Theme D — Naturalness, which no cheap gate can see

F5 (marked word order) sits on its own, and it is the theme with the worst tooling story. The output
is grammatical, the terminology is right, every deterministic gate passes it — and it still reads
wrong to a native speaker. Two independent attempts at cheap detection have now failed for the same
reason: the register gate could not separate formality from grammatical number (O7), and the
word-order scan returned 1 true positive in 11 hits while missing two of three known cases (F5).

→ This is the strongest argument yet that **the human review budget should be spent on naturalness,
not on correctness** — the deterministic gates already handle correctness well (99/100), and they
are structurally incapable of handling naturalness. It also raises the question of whether stage 3
should include an **LLM-judge naturalness pass** with an explicit rubric, to pre-sort rows for the
human rather than replace them.

**The through-line:** Theme A is consistency across the corpus, Theme B is consistency within a row,
Theme C is coverage of the real world, and Theme D is naturalness. Everything found so far is one of
those four — and A, B and C are all fixable by writing something down, while D is not.

---

# Part 4 — Open questions for the end of the review

**Sampling (F7)**
- Run pilot 2 as a **proportional** 100 (Generation 46 · Open QA 12 · Brainstorm 11 · Chat 8 ·
  Rewrite 7 · Summarize 4 · Coding 4 · Classify 4 · Closed QA 3 · Extract 2)?
- Stratify Generation's 46 by prompt-length band, since it is 46% of the corpus and the most varied?
- Coding is the untested blind spot for `VERBATIM-FREEZE` — prioritise it even beyond its 3.5% share?

**Corpus design**
- Do we add a Greeklish / English-input / code-switched stream, and how large (F3)?
- What must the model answer in, given a Greeklish or English prompt (F3)? My recommendation:
  always well-formed Greek, never Greeklish out.
- Do we inspect the 619 real Greek WildChat conversations first to ground the input distribution (F3)?
- Do we accept the corrected-variant framing (O2), or force fidelity to the source's errors?

**Cultural transposition — settled by D1, remaining sub-questions**
- ~~How far do we transpose?~~ **Decided (D1): hard transposition of the frame, freeze the content.**
- One person-name policy for **all** categories: common Greek names for invented characters
  (D1 implies yes). Current output does both, split by category, for no stated reason — needs
  making explicit so it stops happening.
- Names that work by sound (F6): adopt "reproduce the function, not the spelling"? And what should
  *Pamela Pleasantly* become — «Γαλήνη η Γλυκιά», «Χαρά η Χαρωπή», something else?
- How do we name and describe the dataset now that it is not a translation (D1 + O2)?
- Does stage 3 gain a **mixed-frame check** — a row that is half-Greek, half-American (D1)?

**Style guide + glossary**
- `chatbot`, `email`, `bot` — Latin as-is, transliterated, or Greek terms (F4)?
- Which English words are allowed to survive at all, and on what principle (F2)?
- Transliteration standard — ΕΛΟΤ 743, commonest usage, or keep-Latin-for-brands (O4)?
- Gender where Greek forces a choice English never made — infer, avoid, or record (O3)?
- εσύ/εσείς: fixed policy, mirror-the-prompt (current provisional), or labelled both ways?
- Does `neutral` mean "avoid committing" or "no preference, choose and report" (O5)?

**Naturalness**
- Adopt the canonical-word-order rule in both prompts (F5)?
- Does stage 3 get an LLM-judge naturalness pass to pre-sort rows for human review (F5, Theme D)?
- Should the human review budget be re-pointed at naturalness rather than correctness, given the
  deterministic gates already pass 99/100 and cannot see this class of problem?

**Pipeline**
- Does stage 2 get to overrule stage 1's class (O1)?
- Is the κορακίστικα encoding rule right, and is `Κοράκι` the name you want (F1)?
