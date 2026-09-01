#!/usr/bin/env python3
"""Build the pilot artifact. Reads pipeline outputs, emits a self-contained HTML page."""
import json, os, html, collections

D = os.path.dirname(os.path.abspath(__file__))
s1 = {r['row_id']: r for i in range(5) for r in json.load(open(f'{D}/out/stage1_b{i}.json'))}
s2 = {r['row_id']: r for i in range(5) for r in json.load(open(f'{D}/out/stage2_b{i}.json'))}
src = {json.loads(l)['row_id']: json.loads(l) for l in open(f'{D}/sample_100.jsonl')}
gated = {r['row_id']: r for r in json.load(open(f'{D}/out/gated.json'))}

rows = []
for rid, s in src.items():
    a, b, g = s1[rid], s2[rid], gated[rid]
    en_asst = [m['content'] for m in s['messages_en'] if m['role'] == 'assistant']
    en_user = [m['content'] for m in s['messages_en'] if m['role'] == 'user']
    rows.append({
        'id': rid[:8], 'cat': s['category'], 'cls': a.get('translation_class'),
        'sec': a.get('secondary_classes') or [], 'sub': a.get('subtype') or '',
        'haz': a.get('hazards') or [],
        'reg_en': a.get('en_register'), 'reg_ev': a.get('register_evidence') or '',
        'reg_t': a.get('target_register'), 'reg_u': b.get('register_used'),
        'reg_n': b.get('register_note') or '',
        'sys_en': s.get('system_en'), 'sys_el': a.get('system_el'),
        'p_en': s['prompt_en'], 'p_el': a.get('prompt_el') or '',
        'u_en': en_user, 'u_el': a.get('user_turns_el') or [],
        'r_en': en_asst, 'r_el': ([b.get('response_el')] if b.get('response_el') else []) + (b.get('assistant_turns_el') or []),
        'loc': a.get('localization_decisions') or [], 'cn': a.get('constraint_notes') or '',
        'sn': a.get('substitution_note') or '', 'tn': a.get('translator_notes') or '',
        'dev': b.get('deviations') or [], 'sf': b.get('self_flags') or [],
        'cc': b.get('constraint_check') or '',
        'flags': g.get('flags') or [], 'adv': g.get('advisories') or [],
        'gr': g['gates'].get('greek_ratio_response_nonproper'),
        'lr': g['gates'].get('len_ratio_el_en'),
        'rd': g['gates'].get('register_detected'),
    })
rows.sort(key=lambda r: (r['cat'], r['id']))

stats = {
    'n': len(rows),
    'cls': collections.Counter(r['cls'] for r in rows),
    'cat': collections.Counter(r['cat'] for r in rows),
    'haz': collections.Counter(h for r in rows for h in r['haz']),
    'regt': collections.Counter(r['reg_t'] for r in rows),
    'matrix': collections.Counter((r['cat'], r['cls']) for r in rows),
    'hard': collections.Counter(f for r in rows for f in r['flags']),
    'adv': collections.Counter(f for r in rows for f in r['adv']),
    'loc_rows': sum(1 for r in rows if r['loc']), 'loc_n': sum(len(r['loc']) for r in rows),
    'dev_n': sum(len(r['dev']) for r in rows), 'sf_n': sum(len(r['sf']) for r in rows),
    'clean': sum(1 for r in rows if not r['flags']),
    'clean_adv': sum(1 for r in rows if not r['flags'] and not r['adv']),
}
CATS = ['Generation', 'Open QA', 'Chat', 'Rewrite', 'Closed QA']
CLSS = ['LITERAL', 'LOCALIZE', 'REGISTER-CRITICAL', 'CONSTRAINT-PRESERVING',
        'PRESERVE-DEFECT', 'VERBATIM-FREEZE', 'REGENERATE-NATIVE', 'RE-EXECUTE']

payload = {
    'rows': rows,
    'stats': {k: (dict((('|'.join(kk) if isinstance(kk, tuple) else kk), vv) for kk, vv in v.items())
                  if isinstance(v, collections.Counter) else v) for k, v in stats.items()},
    'cats': CATS, 'clss': CLSS,
}
prompts = {
    's1': open(f'{D}/prompts/stage1_classify_translate.md').read(),
    's2': open(f'{D}/prompts/stage2_generate.md').read(),
    'gates': open(f'{D}/gates.py').read(),
}

# ---- per-category spec: what the translation task must preserve -------------
FAMILY = {
    'generative': dict(
        cats=['Generation', 'Open QA', 'Chat', 'Brainstorm', 'Coding'],
        recall='0.00 – 0.10',
        desc='Short instruction, long answer. The answer is <b>composed</b>. Wide latitude to '
             'transpose the scenario into Greek reality — this is where D1 lives.',
        defect='the note goes <b>inline in the answer</b>, as a good assistant would'),
    'source-bearing': dict(
        cats=['Rewrite', 'Closed QA', 'Summarize', 'Classify', 'Extract'],
        recall='0.53 – 0.88',
        desc='Long prompt containing a source block, short answer. The answer is <b>derived</b>. '
             'The source is ground truth; latitude is near zero.',
        defect='the note goes in a <b>separate field</b>, so the answer keeps the reference’s shape'),
}
SPEC = {
 'Generation': dict(n=4346, share='45.8%', fam='generative', piloted=True,
   shape='Bare instruction, single turn. Prompt median 155 chars, answer 872 — the answer is ~5× the prompt. Essentially never a source text.',
   key='<b>The output spec is inside the instruction.</b> 29.7% name a count of words/paragraphs/items/lines/stanzas and 21% give an explicit numeric size limit — five to fifty times the rate of any other category.',
   ess=['The countable constraint, restated so it stays countable in Greek.',
        'Formal devices when the row is about form — ABAB, sonnet, alliteration. These <b>transfer</b>: rhyme is fully available in Greek, so they are not regeneration cases.',
        'Required literal strings (a proverb the story must contain) — the Greek wording becomes the checkable string and must be agreed in advance.',
        'Real-world referents, and the requested slant or persona.'],
   safe='Everything that is scenery: invented names and settings, US props inside invented scenes, units (with a gloss), sentence rhythm.',
   cls='CONSTRAINT-PRESERVING where a count is stated; REGISTER-CRITICAL where the deliverable is a message to a named human; LOCALIZE where the whole cast is invented.',
   chk='Strong. Parse the count from the Greek prompt and verify the Greek answer — words, paragraphs, lines, list items. Rhyme is checkable from the stressed vowel to line end. Length ratio was tight (median 0.99), so outliers mean something.',
   pri='<b>High and rising</b> — 46% of the corpus, and the pilot under-sampled it at 0.44×. Watch the reference-improvement reflex.'),
 'Open QA': dict(n=1182, share='12.4%', fam='generative', piloted=True,
   shape='The smallest prompts in the dataset — median 52 chars, p99 223. A bare factual question, no source. Answer median 364 chars.',
   key='<b>This is where the transposition instinct must be actively suppressed.</b> 12 of 20 pilot rows are about a real entity that <i>is</i> the question.',
   ess=['The referent. Aberdeen stays Aberdeen; “the Americas” must not narrow to «ΗΠΑ», which would change the correct answer.',
        'The correctness of the answer — translating the prompt must not change what is true.',
        'Rows <i>about English</i> freeze the English: the etymology of “soccer” hinges on “Association Football” → “soccer”.',
        'Numbers are not rescaled.'],
   safe='Established Greek names of real entities; number formatting; unit relabelling for the reader (151 ft → ~46 m); loanwords where Greek genuinely uses English.',
   cls='LITERAL (17/20) — meaning <i>translate the words, keep the world</i>.',
   chk='Weakest — no source to check against. Checkable: entity survival (every proper noun and number, mapped through the row’s own decisions, must appear in the Greek pair); a length band, since this category out-grows its reference (median 1.38, the highest).',
   pri='Medium-high, focused on factual drift and out-growth.'),
 'Chat': dict(n=795, share='8.4%', fam='generative', piloted=True,
   shape='The only multi-turn family: 99.9% multi-turn, median 7 messages, 82% with exactly 3 user turns. The <code>prompt</code> field <i>is</i> the system prompt in 794/795 rows. Tiny — a whole row is ~650 chars.',
   key='<b>The persona is the content</b> — 18 of 20 rows. If the bot is deliberately rude, that is the point, not a defect to soften.',
   ess=['The persona voice, sustained across every turn.',
        '<b>Turn-to-turn coherence</b> — the invariant unique to this category. If an assistant turn offers cookies and the next user turn accepts them, the cookies cannot be localized away.',
        'Real facts the persona rests on — the 1986 Red Sox–Mets Series does not become ΠΑΟ–ΟΣΦΠ.',
        'One consistent surface form for every name across all turns; emphatic caps preserved.'],
   safe='The joke vehicle, aggressively — an untranslatable pun is rebuilt natively. Endearments, exclamations, colloquial register. Institutions inside invented scenarios (vocational school → ΕΠΑΛ/ΙΕΚ), keeping year counts exact.',
   cls='REGISTER-CRITICAL (16/20). Override to CONSTRAINT-PRESERVING when the system prompt states a mechanical output rule (every answer a haiku, a numbered list). ενικός in 16/20 — the only category where it dominates.',
   chk='Best structural checkability: turn-count and role parity; <b>per-turn</b> register, never concatenated; one surface form per entity across turns; cross-turn referent linkage; per-turn length ratio.',
   pri='<b>Highest per row.</b> This is where naturalness and register live. Note the pilot’s Chat batch had zero flags and zero advisories while containing both defects you found — the gates say nothing here.'),
 'Brainstorm': dict(n=1060, share='11.2%', fam='generative', piloted=False,
   shape='Bare instruction; only 20 of 1,060 prompts contain a newline at all. <b>Longest answers of any category</b> (median 188 words) and 94% list-formatted, median 5 items. 60% open with a conversational preamble.',
   key='The freest category — and the one where “substitute” and “do not translate” apply to things that look identical.',
   ess=['<b>The item count</b> — 303 of 1,060 prompts name a number.',
        'The item↔rationale pairing — every item keeps its gloss.',
        'List-marker style (numbered vs bulleted) is stable within a row.',
        'Factual claims attached to named entities — keep the name, keep its numbers.'],
   safe='More than anywhere else. US-only recommendation lists should be <b>substituted</b> with Greek equivalents — while official product titles (game names) must <b>not</b> be translated. Same category, opposite treatment; the frame/content test decides.',
   cls='LOCALIZE or REGENERATE-NATIVE for recommendation and naming rows; CONSTRAINT-PRESERVING where a count is stated.',
   chk='Item count vs the number named in the prompt; list-marker type preserved. Not checkable: whether the ideas are any good.',
   pri='Medium per row, <b>high in aggregate</b> — 11% of the corpus, never tested. Needs one consistent decision for the ~636 conversational preambles.'),
 'Coding': dict(n=334, share='3.5%', fam='generative', piloted=False,
   shape='Bare instruction in ~80% of rows. Python 151, JavaScript 52, HTML/CSS 20, Java 17, Bash 17.',
   key='<b>Code is almost never fenced.</b> 9 of 334 prompts and 48 of 334 answers use fences — yet <b>254 answers contain code</b>. In ~206 rows the code is delimited <i>only by indentation</i>. Our masking assumes fences, so it would protect 14% and mangle the rest, silently.',
   ess=['Every code token — keywords, identifiers, library and method names, operators, literals.',
        'Indentation — load-bearing in Python, and in 206 rows the only marker of where code begins and ends.',
        'Identifiers, which the surrounding prose references by name. Translate one without the other and the explanation desynchronises from the code.',
        'User-supplied code round-trips byte-identically in modify/debug rows.'],
   safe='Prose scaffolding, the preamble, comments, and user-facing string literals — the last two are judgement calls, not defaults. Culture markers are near zero here.',
   cls='VERBATIM-FREEZE on the code span; prose handled per family.',
   chk='<b>Strongest of all.</b> Extract code spans and <i>parse</i> them — anything that parsed in English and fails in Greek is a hard fail. Diff the identifier/keyword multiset, allowing string literals to differ.',
   pri='<b>High until proven.</b> Entirely untested, and the failure is catastrophic and silent. ~5 rows are English-semantics-bound (palindromes, vowel sets) and need different Greek test data.'),
 'Rewrite': dict(n=625, share='6.6%', fam='source-bearing', piloted=True,
   shape='Short imperative line, blank line, source block. 100% contain a blank line, 86% contain a transform verb, 94% have a first line under 200 chars. Prompt median 1,084 chars.',
   key='<b>Deliberate defects, when the task is to fix them.</b> “Make it grammatically correct where needed” requires the Greek to <i>contain</i> roughness to fix.',
   ess=['The translated source must itself be the thing transformed.',
        'The transformation target (Q&amp;A, bedtime story, news script, rap lines).',
        '<b>Source defects, when fixing them is the task</b> — keep the <code>????!</code>, the shouting caps, the stray bracket, the run-on. Cleaning them at translation time is the most destructive error available here.',
        'Real facts and figures inside the source; stated format constraints.'],
   safe='Invented senders and recipients; the whole currency frame when the scenario is invented — relabel dollars to euros <i>without rescaling</i> so internal comparisons hold, and convert genuinely only where a figure would otherwise be absurd. Puns rebuilt natively.',
   cls='<b>Two-axis</b>: source-handling × output-handling. A single label hides the preserve-defect requirement, which is the one that most reliably destroys the row.',
   chk='Richest. Split at the first blank line; assert the answer’s recall against the Greek source is within ±0.15 of the English pair’s own value; assert the answer is not a substring of the source; defect parity; numeric multiset equality modulo declared relabels.',
   pri='High — the highest density of judgement calls per row.'),
 'Closed QA': dict(n=245, share='2.6%', fam='source-bearing', piloted=True,
   shape='The most rigid shape in the dataset. Question first then passage: 87% have “?” in the first line, 98% a blank line. Prompt median 956 chars, <b>answer median 110</b>; prompt &gt; answer in 100% of rows.',
   key='<b>Answer-recoverability from the translated passage</b> is the entire job. And terseness is the type — this is where the inline defect note did the most damage (one row ran 5.26× its reference).',
   ess=['The answer must stay recoverable from the Greek passage.',
        'Question–passage lexical match — a term glossed one way in the question and another in the passage silently breaks the row.',
        'Figures in their original units and currency — <b>no conversions in this category</b>.',
        'Source defects including in the question; scrape artefacts (footnote markers, run-together tokens).'],
   safe='Number formatting; infobox field labels but not values; quotation marks → «»; transliteration with the original at first mention; one-off gloss coinages, then reused identically.',
   cls='LITERAL (18/20), with VERBATIM-FREEZE and PRESERVE-DEFECT as the two escape hatches. The pilot <b>under-used</b> PRESERVE-DEFECT — two rows preserved a misspelling while labelled literal.',
   chk='Nearly fully machine-gradable. Assert every content token of the Greek answer appears in the Greek passage or question (English baseline 0.88); key-fact anchoring; numeric multiset equality; footnote-marker parity; answer sentence-count parity.',
   pri='<b>Low per row</b> once the checks are in — this is where automation carries the most weight.'),
 'Summarize': dict(n=395, share='4.2%', fam='source-bearing', piloted=False,
   shape='100% embedded source, 386/395 with a blank-line separator. Answer is free prose, 85% a single paragraph, median 54 words. Only 3% have a preamble.',
   key='<b>No new facts.</b> The summary must be entailed by the source — and regenerating it from a truncated Greek source is where hallucination enters.',
   ess=['No new facts — the defining constraint.',
        'Named entities, numbers, dates and money — 330/395 answers reuse a proper noun from the prompt; stated arithmetic must stay consistent.',
        'Length constraints stated in the prompt. Note the English answers honour explicit sentence counts in only 48/73 rows, so compliance cannot be a pass criterion — only “must not get worse”.',
        'The instruction’s focus scope — “what does it say about X” is a scoped summary.'],
   safe='Very little. The instruction wrapper phrasing and register. You may translate source and summary together; you may not swap the source’s subject and keep the summary.',
   cls='LITERAL on the source; the summary regenerated from the <b>Greek</b> source.',
   chk='Number preservation (every numeral in the answer must appear in the source) catches hallucination cheaply; entity overlap; compression-ratio band (English median 0.29).',
   pri='Medium — concentrated on hallucination, and on the <b>12 rows that put the instruction last</b>, which any position-assuming pipeline mangles.'),
 'Classify': dict(n=334, share='3.5%', fam='source-bearing', piloted=False,
   shape='Bimodal: 181 of 334 embed a source, the rest are short instructions or batch-labelling tasks. Answer median 20 words, 74% single-line, 34 rows ≤15 chars.',
   key='<b>~89 rows have a closed label set</b> where the answer must be lexically identical to an option in the prompt. Translate the option list <b>once and reuse the exact string</b>.',
   ess=['Closed-set membership — two independent translations of “Not Kid-Friendly” leave prompt and answer disagreeing and the row stops being a classification.',
        'Label↔item alignment in batch rows (52 prompts have ≥3 numbered items). Reordering breaks the row <i>invisibly</i> — it still looks well-formed.',
        'Evidence cited in the justification must survive into the translated source.',
        'World knowledge the prompt does not supply — boxers→weight classes cannot be re-derived after substitution.'],
   safe='The classified content in the ~150 rows where the label is genuinely inferable from the text. Names and brands may be localized <b>as long as the label does not change</b>.',
   cls='CONSTRAINT-PRESERVING for closed-set rows; LITERAL otherwise.',
   chk='Closed-set membership (32/40 batch rows already pass in English, so a Greek failure where English passed is a regression); item/label count parity; label-spelling consistency across the corpus.',
   pri='Medium. Tone and toxicity are culturally calibrated — a faithful translation may read more or less aggressive than the label asserts.'),
 'Extract': dict(n=183, share='1.9%', fam='source-bearing', piloted=False,
   shape='96% embedded source. Longest prompts of the untested five (median 1,321 chars) and the shortest relative answer — median answer/prompt ratio 0.10. <b>Zero preamble.</b> Format explicitly specified in 90/183, ordering in 34.',
   key='<b>Literal substring-hood</b> — 52% of rows have every extracted item verbatim in the source. Never translate source and answer independently; that is the fatal mode.',
   ess=['Extracted spans must be literal substrings of the translated source.',
        'Proper nouns and diacritics exactly; capitalisation where demanded; inline artefacts (one row extracts a section including <code>[citation needed]</code>).',
        '<b>The requested order</b> — “alphabetised” is a <i>different order</i> in Greek.',
        'The requested format, literally; and completeness — “all the locations” means all.'],
   safe='The least of any category. The instruction wrapper only. You may translate source and spans together, provided each span is translated identically in both places.',
   cls='LITERAL + VERBATIM-FREEZE on the spans.',
   chk='Most checkable, and the gate is <b>mandatory</b>: split into items, assert each is a substring of the normalised prompt, and gate <b>relatively</b> (Greek ≥ English) — never absolutely, since extract-then-synthesise hybrids are legitimately near zero.',
   pri='Low per row with the gate in place; high for the ~4 untranslatable rows. <b>Greek swaps the decimal and thousands separators</b> — 200,000 → 200.000 and 1.35 → 1,35, in source and answer together.'),
}

CAUSES = [
 dict(t='The prompt has no memory from one row to the next', lead=True,
  wrong='The same kind of thing was handled in opposite ways depending on which batch it landed in. Invented characters got <b>Greek names in some categories</b> (Jenna→Ελένη, Henry Watson→Ανδρέας Βασιλείου) and were <b>transliterated in others</b> (Pamela→Πάμελα, Daniel→Ντάνιελ) — 6 versus 28, with nothing asking for that split. <b>“chatbot” was chosen 30 times across 14 rows without a single alternative ever considered.</b> 38 rows made a transliteration call against no standard. Gender was invented afresh 7 times.',
  why='Every row is an independent model call. A decision made in row 3 is invisible in row 40, so from inside the process “consistent” and “arbitrary” are indistinguishable — there is nowhere for a decision to live.',
  fix=['Write the decisions down once — the frame/content test, register, gender, word order, naming, and a <b>glossary</b> seeded with the vocabulary we measured.',
       'Inject it into <b>every call</b>, so it is part of the instruction rather than a document someone remembers to consult.',
       'Make the model account for it: <code>guide_applied</code> (entries used) and <code>guide_gaps</code> (decisions it had to invent because the guide was silent).',
       'Triage <code>guide_gaps</code> into the next version after each run. Invisible drift becomes a worklist that shrinks.'],
  know='Person names get one treatment across all categories instead of two; <code>guide_gaps</code> shrinks run over run.',
  block='Blocked on closing the pending glossary entries: <b>chatbot, email, bot</b>, the jargon policy, the transliteration standard.'),
 dict(t='“Translate” is the default verb, so preserving the surface is the default behaviour', lead=True,
  wrong='The unnaturalness you flagged: <i>Γύρω στους 21 βαθμούς έχει σήμερα</i>, <i>Αλλά ΚΑΘΟΛΟΥ εύκολο δεν ήταν</i>, and <i>Όλα καλά είναι</i> — the last from a completely neutral English <i>Everything is okay</i>. Separately, <i>Pamela Pleasantly</i> kept its spelling and lost its alliteration.',
  why='The framing verb is <i>translate</i>, so the model preserves surface and adapts only where licensed. There is a specifically Greek mechanism too: <b>free word order is a knob English does not have</b>, and the model reached for it whenever an English emphasis device would not carry. <i>It wasn’t easy AT ALL</i> already marks emphasis with capitals — the Greek kept the capitals <b>and</b> added fronting, marking it twice.',
  fix=['Reframe the job: <b>write the Greek row a Greek writer would have produced</b>, with fidelity demoted from goal to constraint list. Your transposition decision already made this official; the prompt has not caught up.',
       'Add the word-order rule: canonical Greek unless the source is itself marked <b>and</b> that markedness is not already carried by something preserved. <b>Never add markedness to compensate for a device that did not transfer.</b>',
       'Add the naming rule: a name works by sound, by transparent meaning, or by reference. Transliteration preserves spelling and destroys the first two. Reproduce what the name <i>does</i>.'],
  know='The three sentences above come back canonical; <i>Pamela Pleasantly</i> comes back as something like <i>Γαλήνη η Γλυκιά</i>.'),
 dict(t='A substitution is applied to one thing and not swept across the row',
  wrong='The Piggy name. The pig-latin game became κορακίστικα, but the bot stayed <i>Το Γουρουνάκι</i> — a name that no longer referred to anything. The same row lost <i>Pamela Pleasantly</i>’s alliteration the same way.',
  why='The prompt permits substitution but never requires a sweep for whatever depends on the thing substituted.',
  fix=['A propagation step placed <b>before</b> the translation step — so it is a precondition, not a review.',
       'A required <code>derived_elements</code> list. An empty list becomes a <b>claim</b> rather than an oversight — the same trick that made localisation decisions surface properly.',
       'Extended to scene scale for transposition: relocate completely or not at all. A Greek family eating Thanksgiving turkey is worse than an untouched American scene.'],
  know='The Piggy row comes back with a re-derived name and a populated dependency list; transposed scenes contain no leftover American props.'),
 dict(t='The classification is made once, by the stage that knows least',
  wrong='Stage 1 marked <b>2</b> rows as deliberately-defective. Stage 2 then behaved as though <b>5 more</b> were, naming an ambiguity instead of silently answering. Those rows carry a label that contradicts their content.',
  why='Stage 1 classifies from the prompt alone. Stage 2 sees the English reference answer and discovers the defect only when it tries to answer — by which point the class is frozen.',
  fix=['<b>Give stage 1 the reference answer</b> — withholding it was an accident of design, not a decision. Use it only to detect properties of the <i>prompt</i>, not to influence its translation.',
       'As a backstop, let stage 2 record a <code>class_disagreement</code> rather than silently diverging.'],
  know='Disagreement records fall to near zero, and the rows that behave as defective are the rows labelled defective.'),
 dict(t='“Be faithful to the English” was never defined — faithful to <i>what</i>?',
  wrong='The pipeline silently corrected the human-written English in <b>8+ rows</b>: misspellings (<i>Alberdeen</i>, <i>Nort-East</i>, <i>Amepere</i>), a wrong name (<i>Sir John Dalton</i>), a claim that “only three franchises” did something followed by a list of four, and two rows where the English answer violated its own prompt’s paragraph count.',
  why='The prompt says to use the English answer as a content anchor but never says whether its <i>errors</i> count as content. Both readings are defensible, so the model chose row by row.',
  fix=['Repair is allowed. <b>Silent repair is forbidden.</b> Every correction recorded in <code>reference_corrections</code>.',
       'One exception: where the defect is the point of the exercise, reproduce it faithfully.',
       'Say so on the dataset card — combined with transposition, the output is a <b>corrected, transposed variant</b> of no_robots, not a translation of it.'],
  know='Every correction appears in the record; spot-checking finds no unrecorded ones.'),
 dict(t='The prompt treats ten very different tasks as one task', lead=True,
  wrong='Nothing visible yet — which is the problem. One generic instruction covers all ten categories, and <b>five were never tested</b>. The differences are severe: Closed QA answers must stay recoverable from the passage; in Coding <b>code is unfenced in ~206 of 334 rows</b>; Extract answers must be literal spans; in Chat the persona is the content.',
  why='The class taxonomy describes <i>how to translate</i>. Nothing describes <i>what kind of object this row is</i>.',
  fix=['Inject the family and category profile — the “Spec” tab is exactly this content, and branching on family first settles most questions before category rules are consulted.',
       'Add a <b>structural code detector</b> for Coding that finds code by indentation and line shape rather than by fences — otherwise that category fails silently and catastrophically.'],
  know='The proportional re-run covers all ten categories; Coding rows survive a parse check and an identifier diff.'),
 dict(t='One label is being asked to describe two separate decisions',
  wrong='Every Rewrite row needed a <i>secondary</i> class, because “literal” there describes only how the source block moves into Greek and says nothing about the transformation — which is the actual task. Two Closed QA rows deliberately preserved a misspelling while labelled literal, so the label could not tell you the prompt was intentionally wrong.',
  why='For rows containing a source text, <i>how you move the source</i> and <i>what the answer does with it</i> are independent choices. One field cannot hold both, so one silently loses.',
  fix=['Two fields for source-bearing rows: <code>source_handling</code> (literal / preserve-defect / verbatim-freeze / localize) × <code>output_handling</code> (transform / extract / summarize / classify / answer).',
       'Generative rows keep the single class.'],
  know='No row needs a secondary class to be intelligible; rows that preserve a defect are labelled as doing so.'),
 dict(t='The note about a defective source is written inside the answer',
  wrong='Every one of the five rows that blew the length band was this: the Greek answer names the ambiguity <i>before</i> answering while the English just answers. Worst was <b>5.26×</b> the reference — in Closed QA, where the median answer is 110 characters and terseness is the whole point.',
  why='Flagging an ambiguity rather than guessing is good behaviour and the prompt encourages it — but it was allowed to change the shape of the deliverable.',
  fix=['<b>Your call, split by family:</b> inline in the answer for generative rows, where it teaches something worth having.',
       'A separate <code>defect_note</code> field for source-bearing rows, so the answer keeps the reference’s shape and length.'],
  know='The length band tightens; no source-bearing row runs multiples of its reference.'),
 dict(t='The checks test correctness; nothing tests whether it reads well', lead=True,
  wrong='The unnatural word order is <b>invisible to automated checking</b>. I tried: a regex scan over the whole corpus returned <b>11 hits, 1 of them real</b>, and missed 2 of the 3 known cases. The register check flagged the Three Little Pigs row as “mixed register” because the wolf addresses one pig and then two — grammatical <b>number</b>, not formality. Meanwhile <b>99 of 100 rows pass every hard gate</b>, and the Chat batch passed with zero flags while containing both defects you found.',
  why='The gates test what a regex can see — script, encoding, ratios, lengths. Naturalness is not that kind of property, and no amount of tuning will make it one.',
  fix=['A stage-3 <b>naturalness judge</b> with an explicit rubric: word order (canonical or marked, and is the markedness in the source?), register consistency, calqued syntax, and “would a Greek writer produce this unprompted?”',
       'Its job is to <b>rank, not reject</b> — it sorts your review queue so the worst rows arrive first.',
       '<b>Re-point the human budget at naturalness.</b> The deterministic checks already have correctness at 99/100 and are structurally incapable of the rest.',
       'Calibrate on ~100 hand-checked native Greek pairs first — every published Greek reward-model result is on translated benchmark data.'],
  know='The rows the judge ranks worst are the rows you would have flagged.'),
 dict(t='Three of the checks measure the wrong thing',
  wrong='The pilot’s <b>only</b> hard failure was a row correctly keeping English piercing jargon (<i>septum</i>, <i>bridge</i>). A single formal verb form in turn 3 of a Chat row was invisible because the check ran on all turns joined together. All three Rewrite register warnings measured <i>the rewritten advert addressing its reader</i> while the target described <i>the assistant addressing the user</i>.',
  why='Each check was written against a plausible idea of the failure rather than against the data.',
  fix=['Greek-ratio floor → <b>per-row allowlist</b> built from that row’s own recorded decisions. A global floor cannot tell correct jargon from leakage; a per-row list can.',
       'Register → <b>per turn</b>, reporting within-row inconsistency separately from a mismatch against the prompt.',
       'Register in source-bearing rows → <b>two axes</b>, measured separately and never compared.',
       '<b>Standing rule:</b> new checks start as advisories and earn hard-flag status only after a clean pass. The gate suite was wrong three times and every sigma flag it ever raised was a false positive.'],
  know='The piercing and Tarkov rows stop being flagged; the single-formal-turn row starts being flagged.'),
 dict(t='The corpus represents one slice of reality',
  wrong='<b>The sample was not representative</b> — the thing you noticed. Stratified for stress-testing, not proportion: Closed QA over-represented <b>7.75×</b>, Chat <b>2.39×</b>, Generation <i>under</i>-represented at <b>0.44×</b> despite being 46% of the data, and five categories (24.3%) absent entirely. Separately, <b>every prompt is clean Greek because it was produced by translation</b> — we have zero Greeklish, zero English-typed, zero code-switched input.',
  why='Two different sampling decisions, both of which narrowed what the data can represent.',
  fix=['<b>Sample A</b> — the 9 rows you have reviewed, re-run for a clean before/after on the Piggy name, the word order and the Pamela alliteration.',
       '<b>Sample B</b> — a proportional 100 across all ten categories, which finally tests Coding.',
       '<b>A Greeklish stream</b> built by transliterating Greek prompts we already have and pairing them with the same Greek answers — one row becomes a robustness pair. The model always answers in well-formed Greek; we do not want it <i>producing</i> Greeklish.',
       'One catch: <b>no language-ID tool detects Greeklish</b> — it reads as English or Welsh with high confidence — so that stream must be exempted from the check or it will all be rejected.'],
  know='Re-weighting already moved real numbers: transposable scope 20% → 28.6%, and constraint-bearing rows turn out to be the largest class corpus-wide at ~37%, not literal ones.'),
]
payload['spec'] = SPEC
payload['family'] = FAMILY
payload['causes'] = CAUSES

TPL = open(f'{D}/template.html').read()
out = TPL.replace('/*__DATA__*/', json.dumps(payload, ensure_ascii=False)) \
         .replace('/*__PROMPTS__*/', json.dumps(prompts, ensure_ascii=False))
open(f'{D}/artifact.html', 'w').write(out)

# The whole page is rendered by JS, so a syntax error yields a blank page. Check before shipping.
import re as _re, subprocess, tempfile, shutil
blocks = _re.findall(r'<script>(.*?)</script>', out, _re.S)
if shutil.which('node') and blocks:
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(blocks[-1]); tmp = f.name
    p = subprocess.run(['node', '--check', tmp], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit('JS SYNTAX ERROR in template:\n' + p.stderr[:1500])
    print('js syntax: OK')
# JSON payloads must survive a real parse too
for tag in ('data', 'prompts'):
    m = _re.search(rf'<script id="{tag}" type="application/json">(.*?)</script>', out, _re.S)
    json.loads(m.group(1))
print('json payloads: OK')
print('wrote artifact.html', len(out), 'bytes; rows:', len(rows))
print('classes:', dict(stats['cls']))
print('clean:', stats['clean'], 'clean+noadv:', stats['clean_adv'])
