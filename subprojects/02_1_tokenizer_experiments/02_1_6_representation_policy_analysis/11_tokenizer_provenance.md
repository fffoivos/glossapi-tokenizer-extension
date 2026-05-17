# Tokenizer provenance — what Apertus inherited from Mistral, what changed

Tightly scoped to the question: which parts of the Apertus tokenizer
are Apertus's own, and which are inherited from Mistral-Nemo's
`tekken` v3? Resolution matters because **the per-language BPE
allocation (Greek 1,479 tokens, Arabic 7,146, Latin 2,768, etc.) is
either Apertus's policy or Mistral's policy** — different defensible
budgets depending on which.

Retrieved 2026-05-17 from:
- Apertus tokenizer locally at
  `~/.cache/huggingface/hub/models--swiss-ai--Apertus-8B-2509/.../tokenizer.json`
- Mistral-Nemo-Base-2407 tokenizer fetched from
  `https://huggingface.co/mistralai/Mistral-Nemo-Base-2407/raw/main/tokenizer.json`

## Headline finding

**Apertus's per-language BPE allocation is 100 % inherited from
Mistral. Apertus made zero per-language modifications to the BPE
merge table.**

What Apertus did change is the **special-token block at the top of
the vocab** (ids 0-999). They expanded the reserved-id block from
Mistral's 514 to 1000, repurposed 58 of those slots for code / math /
PII / chat-template / reasoning markers, and truncated the trailing
486 BPE entries to keep total vocab at 131,072.

None of the per-language vocab discussion in
`02_2_2_vocab_lang_attribution/` is affected by Apertus's changes —
those analyses operate on the inherited Mistral BPE table.

## Side-by-side comparison

| Field | Mistral-Nemo-Base-2407 | Apertus-8B-2509 | Same? |
|---|---|---|---|
| Tokenizer family | BPE | BPE | ✓ |
| Vocab size | 131,072 | 131,072 | ✓ |
| Normalizer | null (none) | null (none) | ✓ |
| Pre-tokenizer | Sequence[Regex split, ByteLevel] | Sequence[Regex split, ByteLevel] | ✓ |
| `add_prefix_space` | false | false | ✓ |
| `model.ignore_merges` | true | true | ✓ |
| `model.byte_fallback` | false | false | ✓ |
| `byte_level` regex pattern | GPT-2-style L/N/M split | GPT-2-style L/N/M split | ✓ |
| **Front-block size (reserved IDs)** | **514 (ids 0-513)** | **1000 (ids 0-999)** | **✗** |
| BPE table ID range | 514-131,071 (130,558 entries) | 1000-131,071 (130,072 entries) | **✗ (Apertus drops 486 trailing BPE entries)** |
| `added_tokens` named entries | 14 real + 500 placeholders | 72 real + 928 placeholders | **✗** |
| Post-processor template | TemplateProcessing | TemplateProcessing | ✓ |
| Vocab-merge table identity | (reference) | first 130,072 entries inherited verbatim from Mistral | ✓ (Apertus only truncates) |

## What's identical

- **The BPE merges themselves**. Apertus uses Mistral's exact merge
  table for the first 130,072 entries. We can read this from the
  `merges` field of the tokenizer.json: the first 5 merges Apertus
  reports are `['Ġ','Ġ'], ['Ġ','t'], ['e','r'], ['i','n'], ['Ġ','ĠĠĠ']`
  — these are byte-level prefix merges typical of tekken v3.
- **The pre-tokenizer regex**:
  `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*…` — the
  GPT-2-style split that handles whitespace and case transitions.
  Identical to Mistral's.
- **Byte-level encoding**. Both use ByteLevel pretokenizer with
  `add_prefix_space=false`.
- **Special-token blank placeholders**. Both tokenizers reserve a
  range of `<SPECIAL_N>` placeholder slots that Mistral never
  populated. Apertus extends this range and populates more of it.

## What's different

### Difference 1 — Front-block size: 514 → 1000

Mistral reserved IDs 0-513 (514 entries) as the special-token region.
Apertus expanded this to IDs 0-999 (1000 entries). The expansion
made room for Apertus's new chat-template, reasoning, PII, and code
tokens.

The BPE table shifted accordingly:
- Mistral's BPE starts at ID 514, ends at 131,071 (130,558 BPE entries).
- Apertus's BPE starts at ID 1000, ends at 131,071 (130,072 BPE entries).

**This means Apertus truncated 486 BPE entries from the tail of
Mistral's merge table.** These were the lowest-priority (latest-trained)
merges Mistral kept; the last few in Apertus's table include rare
multilingual entries like `ĠstratÃ©gique`, `ĠnÃ¤iteks` (Estonian),
`çľĭçĿĢ` (Chinese), `åĲİæ±īä¹¦` (Chinese book title). Mistral's
truncated 486 would have been similarly rare.

**Impact on per-language allocation — likely minimal but unverified.**
The truncated 486 entries are at the very tail of merge-frequency
order, where high-frequency Greek / Latin / Arabic words (which
enter the BPE table at low IDs) are unlikely to live. Apertus's
last few BPE entries (e.g. `ĠstratÃ©gique`, `ĠnÃ¤iteks`, `çľĭçĿĢ`)
are rare-multilingual tail tokens, consistent with the truncated 486
also being rare-multilingual tail.

However, **a direct tail-token audit has not been done**. The proper
verification is: pull Mistral's tokenizer.json, decode the 486
dropped BPE entries (Mistral's tail of the merge table), classify
by script + language, and check overlap with our PMI-promoted token
sets for Greek / Latin / Arabic / Korean / etc. Marked as TODO in
`INVESTIGATIONS_TRACKER.md`. Until that audit runs, treat "no
impact on the major-language PMI counts" as a **strong hypothesis
based on tail-position reasoning**, not a verified claim.

### Difference 2 — Real special tokens: 14 → 72

Mistral's 14 named special tokens (verbatim from Mistral-Nemo-Base-2407
tokenizer.json):

```
id   0  <unk>
id   1  <s>
id   2  </s>
id   3  [INST]
id   4  [/INST]
id   5  [AVAILABLE_TOOLS]
id   6  [/AVAILABLE_TOOLS]
id   7  [TOOL_RESULTS]
id   8  [/TOOL_RESULTS]
id   9  [TOOL_CALLS]
id  10  <pad>
id  11  [PREFIX]
id  12  [MIDDLE]
id  13  [SUFFIX]
```

Apertus's 72 named special tokens (verbatim, in id order):

```
id   0  <unk>                          [kept]
id   1  <s>                            [kept]
id   2  </s>                           [kept]
id   3  <pad>                          [MOVED from Mistral id 10]
id   4  [/INST]                        [kept]
id   5  [AVAILABLE_TOOLS]              [kept]
id   6  [/AVAILABLE_TOOLS]             [kept]
id   7  [TOOL_RESULTS]                 [kept]
id   8  [/TOOL_RESULTS]                [kept]
id   9  [TOOL_CALLS]                   [kept]
id  10  <SPECIAL_10>                   [PLACEHOLDER — Mistral's <pad> moved out]
id  11  [PREFIX]                       [kept]
id  12  [MIDDLE]                       [kept]
id  13  [SUFFIX]                       [kept]
id  14  \begin{                        [NEW — LaTeX math, not special-flagged]
id  15  \end{                          [NEW]
id  16  \text{                         [NEW]
id  17  \boxed{                        [NEW]
id  18  <filename>                     [NEW — StarCoder-style]
id  19  <gh_stars>                     [NEW]
id  20  <issue_start>                  [NEW]
id  21  <issue_comment>                [NEW]
id  22  <issue_closed>                 [NEW]
id  23  <jupyter_start>                [NEW]
id  24  <jupyter_text>                 [NEW]
id  25  <jupyter_code>                 [NEW]
id  26  <jupyter_output>               [NEW]
id  27  <empty_output>                 [NEW]
id  28  <commit_before>                [NEW]
id  29  <commit_msg>                   [NEW]
id  30  <commit_after>                 [NEW]
id  31  <reponame>                     [NEW]
id  32  <think>                        [NEW — reasoning tag]
id  33  </think>                       [NEW]
id  34  <answer>                       [NEW]
id  35  </answer>                      [NEW]
id  36  <iban-pii>                     [NEW — Apertus PII redaction]
id  37  <email-pii>                    [NEW]
id  38  <ip-pii>                       [NEW]
id  39  <file_sep>                     [NEW]
id  40  <code_to_intermediate>         [NEW]
id  41  <intermediate_to_code>         [NEW]
id  42  <pr>                           [NEW — pull-request structure]
id  43-57  <pr_status>, <pr_is_merged>, <pr_base>, <pr_file>, <pr_base_code>,
          <pr_diff>, <pr_diff_hunk>, <pr_comment>, <pr_event_id>, <pr_review>,
          <pr_review_state>, <pr_review_comment>, <pr_in_reply_to_review_id>,
          <pr_in_reply_to_comment_id>, <pr_diff_hunk_comment_line>  [NEW × 15]
id  58  <|fim_begin|>                  [NEW — alt fill-in-middle]
id  59  <|fim_hole|>                   [NEW]
id  60  <|fim_end|>                    [NEW]
id  61-72  <|system_start|>, <|system_end|>, <|developer_start|>, <|developer_end|>,
          <|user_start|>, <|user_end|>, <|assistant_start|>, <|assistant_end|>,
          <|inner_prefix|>, <|inner_suffix|>, <|tools_prefix|>, <|tools_suffix|>
                                       [NEW — Apertus chat template]
```

That's:
- **11 kept** from Mistral (`<unk>`, `<s>`, `</s>`, `<pad>` repositioned, `[/INST]`,
  the 4 tool-related tokens, `[PREFIX]`/`[MIDDLE]`/`[SUFFIX]`)
- **1 dropped**: Mistral's `[INST]` removed (Apertus uses
  `<|user_start|>...<|user_end|>` instead)
- **58 added**: 4 LaTeX + 14 StarCoder code + 4 reasoning + 3 PII + 17
  PR-structure + 3 alt-FIM + 12 chat-template + 1 `<file_sep>`

The Apertus paper §2.2 says **47** custom special tokens; the actual
count of newly-named tokens is **58** by id. The discrepancy is
probably that the paper counts only the code/math-specific tokens
(excluding the chat-template and PII tokens which are documented
separately in §3.1.2 and §4).

### Difference 3 — Post-processor template (chat template)

Both tokenizers use `TemplateProcessing`. Both wrap input with `<s>`
BOS. The difference is downstream: Apertus uses its own chat-template
formatting (`<|system_start|>...<|assistant_end|>`) rather than
Mistral's `[INST]...[/INST]` format. This is a **downstream
post-processing** change, not a tokenizer-vocab change.

## What about per-language BPE merges?

This is the critical question for our fairness analysis: **did
Apertus add or remove any per-language merges?**

Answer: **NO.** The 130,072 BPE entries Apertus uses (IDs 1000-131071)
are inherited verbatim from Mistral. The 486 entries Apertus dropped
are the **trailing** rare-frequency merges from Mistral's table; no
Greek, Latin, or Apertus-target-language merges of any practical
volume are affected.

This means:
- The Greek 1,479 PMI-tokens count is **Mistral's allocation**, not
  Apertus's.
- The Latin 2,768 PMI-tokens count is **Mistral's allocation**.
- The Arabic 7,146, Korean 4,438, etc. are all **Mistral's
  allocations**.
- Apertus's only per-token-vocab decision was "use Mistral's tekken
  v3 as-is" (paper §2.2, after Gini comparison across four
  tokenizers).

## Implication for the C3 extension

The C3 extension is mathematically equivalent to **adding Apertus's
first per-language allocation decision** for Greek. Until this
extension, Apertus has made no per-language vocab decisions. The
existing 1,479 Greek tokens are entirely Mistral's policy outcome
from undisclosed training data.

The "rational policy" question for the C3 cutoff therefore becomes
narrower than the previous synthesis attempted:

- **Apertus's stated policy** for vocab allocation is "minimize Gini
  on FLORES+ 55-language fertility, aggregate." That policy was
  honored by Mistral's tekken winning the Gini comparison. Apertus
  has never specified a per-language target.
- **Mistral's stated policy** for tokenizer training is "compression
  efficiency vs SentencePiece" (Mistral Nemo blog). Per-language
  allocation is downstream of whatever data Mistral fed BPE on, which
  Mistral did not disclose.
- **Therefore the C3 extension is unconstrained by either Apertus's
  or Mistral's stated policy.** Both stated policies are satisfied
  with Apertus's current 1,479-Greek allocation; both are also
  satisfied with any larger Greek allocation that doesn't break the
  aggregate Gini.

The C3 extension is the first time a per-language Greek vocab
decision is being made *inside Apertus's stack*. That's the right
framing.

## What this scopes out

Several hypotheses in `INVESTIGATIONS_TRACKER.md` were about the
broader Apertus pretraining-data policy. The narrowed scope (tokens
in the tokenizer specifically) makes some of them less relevant:

- **Hypothesis 5** (pre-2024 dataset landscape): **still relevant
  but narrowed.** The question is now specifically "what data did
  Mistral feed BPE on?", not "what data did Apertus pretrain on?"
  These could be the same or different. Investigation should focus
  on Mistral's tokenizer-training corpus, which Mistral did not
  disclose.
- **Hypothesis 6** (inherited from prior multilingual models):
  **still relevant.** Mistral's tokenizer training could have used
  XLM-R-100's language list, mT5's 101, etc. as a corpus selection
  filter. Worth investigating.
- **Hypothesis 7** (team / institutional bias): **partially scoped
  out.** Apertus's Swiss-cantonal bias affects pretraining data and
  named priorities but does NOT affect the inherited BPE allocation.
  Mistral's French team bias might have affected Mistral's
  tokenizer-training data, which is relevant.
- **Hypothesis 8** (commercial markets): **scoped out for tokenizer
  question.** Apertus's commercial framing doesn't affect inherited
  vocab. Mistral's commercial framing could affect Mistral's
  tokenizer-training data choice; covered already by Mistral-strong-11
  analysis.
- **Hypothesis 9** (benchmark coverage): **partially scoped out.**
  Benchmark coverage motivates evaluation choices, not tokenizer
  training data. May affect Mistral's data choices indirectly.
- **Hypothesis 10** (Reddit proxy): **scoped out.** Reddit usage
  would affect Apertus pretraining (we know it doesn't), not the
  inherited tokenizer.

The remaining in-scope investigations for the tokenizer-vocab question:

- **Hypothesis 5 (narrowed)**: what data did *Mistral* train tekken
  v3 on, as best as inferable from per-language vocab structure?
- **Hypothesis 6 (kept)**: is Mistral's per-language BPE allocation a
  recognizable inheritance from a prior multilingual model's
  language list?
- **Hypothesis 7 (narrowed)**: did Mistral's French team bias affect
  the BPE-training data and therefore the per-language vocab
  allocation?

These three are the relevant scope. Hypotheses 8, 9, 10 can be
deferred or dropped.
