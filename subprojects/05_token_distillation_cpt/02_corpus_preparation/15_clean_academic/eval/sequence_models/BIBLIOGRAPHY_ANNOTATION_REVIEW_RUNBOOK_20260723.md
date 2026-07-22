# Bibliography annotation and review runbook

Date: 2026-07-23

Status: consolidated guidance for future bibliography-review runs. This records
what the completed source-matched review taught us. It does not retroactively
change the frozen consensus-silver cohort.

## 1. Recommended reviewer models

### Production annotation

- Use `gpt-5.6-terra` with `model_reasoning_effort="high"` for a single careful
  pass or as the primary annotator.
- Use two prediction-blind, mutually blind passes for evaluation labels. For
  maximum model diversity, use `gpt-5.6-terra` high for one pass and
  `gpt-5.6-sol` high for the other. Two independent Terra-high calls are also
  acceptable when consistency is more important than model diversity, but
  their errors will be more correlated.
- Record the actual model and reasoning effort for every batch. Never infer the
  model from a run or reviewer name. The historical reviewer IDs
  `sealed-role-sol-terra-high-{a,b}-v1` were misleading: the final passes mixed
  imported Sol-high batches with newly completed Terra-high batches.
- Use a human adjudicator if the intended result is human gold. A third Codex
  pass can produce adjudicated silver, but it does not turn model annotation
  into human gold.

### What the completed run actually used

The final repaired passes contained:

| pass | imported `gpt-5.6-sol`, high | new `gpt-5.6-terra`, high |
|---|---:|---:|
| A | 174 batches | 184 batches |
| B | 176 batches | 212 batches |

After the two deterministic annotation repairs and removal of seven documents
with systematic footnote-versus-bibliography disagreement, the retained
143-document cohort had 99.9133% A/B agreement on the binary BIB/NON_BIB task.
This demonstrates that both high-reasoning models were useful annotators. It
does not prove that one is more accurate, because there is no human-gold
benchmark for these lines.

### Models not to treat as final truth

- Historical `gpt-5.5` Struct2K labels are useful bootstrap silver and
  development material, not human gold or a fresh evaluation reference.
- Luna 5.6 was proposed as a cheap document-quality reviewer but was not
  executed or validated for the bibliography-role task. Benchmark it on an
  adjudicated pilot before using it here.
- Medium-reasoning Terra runs were attempted, but the terminal annotation was
  completed with high reasoning. Use high reasoning for the role taxonomy.

The existing runner invokes Codex in an empty, ephemeral, read-only workspace,
binds the response to a JSON schema, and records the runtime contract. Preserve
those properties.

## 2. Define the task before defining the labels

The annotation question is not “does this line contain something that looks
like a citation?” It is:

> What structural role does this physical line play in reconstructing a
> bibliography region that the cleaner may remove?

The labels therefore encode block behaviour:

- which lines can independently start a bibliography block;
- which lines can join a confirmed block but cannot start one;
- which headings are included boundaries or internal connectors;
- which lines stop growth and remain in the document; and
- which lines lack enough evidence to supervise a model.

Do not change the scope silently between annotation runs. In particular,
decide whether consolidated scholarly endnotes/footnotes are removal targets
before sampling. The completed run exposed unresolved policy disagreement on
footnote-heavy documents; seven such documents were excluded instead of being
silently forced into consensus.

## 3. Canonical labels

Only `ENTRY` can seed a bibliography block.

| Label | Operational definition | Block action | Included in confirmed bibliography? |
|---|---|---|---:|
| `ENTRY` | A physical line that independently looks like a bibliographic record or a recognizable start of one. | Seed or extend. | yes |
| `CONTINUATION` | Citation content split from an adjacent entry and too incomplete to establish an entry by itself. | Attach only to a supported region. | yes |
| `FILLER` | Non-citation material that belongs inside an entry-anchored bibliography region: extraction debris, a rule, a page artifact, an annotation, or another internal bridge. | Bridge supported material; never seed. | yes |
| `BIB_HEADER` | An ATX Markdown heading that introduces the overall bibliography/reference region. | Include; stop outward/upward growth at the heading. | yes |
| `BIB_SUBHEADER` | An ATX Markdown heading subdividing a bibliography by language, source, period, medium, or category. | Include and connect supported bibliography material on either side. | yes |
| `NON_BIB_HEADER` | An ATX Markdown heading for material outside the bibliography, normally the next or previous document section. | Exclude and stop growth in the outward direction. | no |
| `OTHER` | Any usable line that does not have one of the roles above. | Exclude. | no |
| `UNKNOWN` | Visible evidence is genuinely insufficient, usually because extraction or truncation prevents a sound judgment. | Mask from supervised targets. | no |

`ENTRY_ANCHOR` in early planning documents means the same operational concept
as the final `ENTRY` label. Use `ENTRY` in new annotations.

### 3.1 ENTRY

Use `ENTRY` when the line itself supplies enough independent evidence to act
as a bibliography anchor. Common forms include:

- author, title, year, journal, publisher, DOI, URL, volume, or page fields in
  a recognizable citation arrangement;
- a numbered or bulleted reference item;
- the recognizable beginning of a wrapped reference, even if later physical
  lines continue it;
- an item in a declared webography or source list, including a standalone URL
  when it is clearly one list item; and
- citation-formatted CV publication lists or primary-source lists, under the
  current corpus convention.

Do not use `ENTRY` for a bibliography heading, an inline citation inside prose,
a citation example embedded in instructions, a number alone, or a fragment
that needs an adjacent line to look bibliographic.

### 3.2 CONTINUATION

Use `CONTINUATION` for citation content that belongs to a neighbouring entry
but is not independently a recognizable entry. Examples include:

- the title, journal, publisher, DOI, or pages wrapped onto a second physical
  line after an author/year line;
- the remainder of a citation broken by PDF extraction; and
- a short fragment whose meaning as citation content depends on the adjacent
  entry.

It is not a generic “weak bibliography-like line.” It must belong to an actual
entry. If there is no `ENTRY` anchor in the contiguous bibliography-role
component, label it `OTHER`, not `CONTINUATION`.

### 3.3 FILLER

Use `FILLER` for non-citation material that must be crossed or included to keep
an already-supported bibliography region coherent. Examples include:

- Markdown table separators or repeated rules;
- page numbers, image markers, OCR fragments, and layout debris inside a
  bibliography;
- a blank-like or symbol-only extracted line inside the region;
- descriptive annotations attached to bibliography entries; and
- small internal bridge material that is neither an entry nor citation
  continuation.

“Filler” is relational: it fills a bibliography region. A separator, page
artifact, short prose line, or citation-looking fragment elsewhere in the
document is `OTHER`. If its contiguous bibliography-role component has no
`ENTRY`, it cannot be `FILLER`.

### 3.4 Heading roles

For this corpus, assign a heading role only when the source line is an ATX
Markdown heading matching:

```text
^\s{0,3}#{1,6}\s+\S
```

This restriction must be stated in the prompt. The original annotation prompt
allowed semantic-looking plain-text headings, producing hundreds of false
header labels that later had to be repaired to `OTHER`.

- `BIB_HEADER` is the overall entry point: `## ΒΙΒΛΙΟΓΡΑΦΙΑ`, `# References`,
  `### Αναφορές`. It is included with the block and is a hard outward/upward
  boundary.
- `BIB_SUBHEADER` is internal: `### Ελληνική βιβλιογραφία`, `### Foreign
  sources`, `#### Διαδικτυακές πηγές`. It connects independently supported
  bibliography material rather than terminating it.
- `NON_BIB_HEADER` denotes a Markdown heading outside the bibliography, such
  as the next chapter heading after the references. It remains in the corpus
  and is a hard outward boundary.
- A plain line that merely looks like a title is `OTHER` under this annotation
  contract. A separate deterministic lexicon may still use its text as model
  evidence, but it is not a trusted header label.

### 3.5 OTHER

Use `OTHER` for ordinary prose and all out-of-scope material, including:

- body prose containing authors, years, quotations, links, or inline
  citations;
- table-of-contents material;
- captions, ordinary tables, rosters, glossaries, and non-citation lists;
- plain-text section-like lines that are not ATX Markdown headings;
- isolated context/filler-like material with no entry anchor; and
- ordinary footnotes or endnotes unless the run's declared corpus policy
  explicitly includes a consolidated scholarly-note list.

### 3.6 UNKNOWN

Use `UNKNOWN` sparingly. It means the evidence is unusable, not merely that the
choice is difficult. Use a lower confidence score for a difficult but
answerable choice. Use `UNKNOWN` when missing or truncated text genuinely
prevents a defensible classification. Any `UNKNOWN` vote masks that line from
all derived supervised targets.

## 4. How to write the annotation instructions

Every prompt should contain these elements in this order:

1. **Purpose.** State that roles reconstruct removable bibliography regions;
   token presence alone is not the target.
2. **Independence.** State that no model predictions or other reviewer labels
   are present and the reviewer must judge the text from scratch.
3. **Unit and context.** Explain physical line indices, document position,
   chunk overlap, and any display truncation.
4. **Complete label table.** Give the definition and block action for every
   label. Do not provide one-sentence circular definitions such as “an entry is
   a bibliographic entry.”
5. **Hard invariants.** Only `ENTRY` seeds; context roles require an anchored
   component; header roles require ATX Markdown; `UNKNOWN` is masked.
6. **Positive and negative edge cases.** Explicitly contrast bibliography with
   inline citations, footnotes, ToCs, list-like prose, plain-text headings,
   webographies, CV publication lists, and wrapped citations.
7. **Output contract.** Require schema-bound JSON, every chunk exactly once,
   complete line coverage, valid inclusive offsets, and no prose outside the
   response schema.

Do not ask the annotator to reproduce the document with inline tags. Long
documents drift or truncate. Supply stable line identities and request labels
or run-length-encoded offset spans.

### Minimum prompt wording for the two learned invariants

The following wording should appear verbatim or equivalently in every future
role prompt:

> `CONTINUATION` and `FILLER` are context-only roles. They are valid only inside
> a contiguous bibliography-role component containing at least one `ENTRY`.
> Outside such a component, label the line `OTHER`.

> Assign `BIB_HEADER`, `BIB_SUBHEADER`, or `NON_BIB_HEADER` only to an ATX
> Markdown heading beginning with one to six `#` characters followed by a
> space and visible text. Otherwise use the appropriate non-header role.

## 5. Packet construction

- Select full documents before inspecting model predictions when building a
  test cohort. Balance sources and exclude train/development works by canonical
  work identity and near-duplicate checks.
- Present the full document where practical. For large documents, use bounded
  overlapping chunks with stable absolute indices and enough context to see
  entries on both sides of potential continuation, filler, or subheaders.
- A review call should label a coherent chunk or block, not one isolated line.
  Line roles are contextual even when `ENTRY` must be independently strong.
- Hide detector highlights, probabilities, candidate strata, previous labels,
  and the other pass during first judgment.
- Preserve source text and line identities. If a physical line is exceptionally
  long, show a bounded prefix and suffix around an explicit truncation marker;
  do not silently replace its underlying identity or text.
- The tested runner uses at most two documents per batch, two concurrent
  workers, schema validation, a 30-minute call timeout, and split-batch fallback.
  Keep runs resumable and retry only missing or invalid batches.

## 6. Independent passes, consensus, and adjudication

Preserve each raw pass unchanged. Derive corrected or consensus artifacts as
new, hash-bound outputs.

For silver labels, two-pass agreement should be calculated separately for each
downstream task:

| Task | Role mapping |
|---|---|
| bibliography membership | `ENTRY`, `CONTINUATION`, `FILLER`, `BIB_HEADER`, `BIB_SUBHEADER` -> `BIB`; all other non-UNKNOWN roles -> `NON_BIB` |
| entry seed | `ENTRY` -> `ENTRY`; every other non-UNKNOWN role -> `NOT_ENTRY` |
| heading type | the three heading roles remain distinct; everything else -> `NOT_HEADER` |
| context role | `CONTINUATION` and `FILLER` remain distinct; everything else -> `OTHER` |
| fine role | exact eight-label agreement |

Trust a task label only when the two passes agree after that task's recoding.
Thus `CONTINUATION` versus `FILLER` is still trusted as `BIB` and `NOT_ENTRY`,
but is unresolved for context subtype and exact role. If either pass uses
`UNKNOWN`, mask every task for that line.

Do not force all fine-role disagreements through a third model. The completed
run showed that keeping only A/B agreements and masking the rest is simpler
and defensible for consensus silver. Use human adjudication when disagreement
coverage is too large or when true human gold is required.

## 7. Quality gates and diagnostics

Before scaling an annotation run:

- manually inspect a small source-balanced pilot;
- report agreement separately for binary membership, entry seed, header
  detection, header subtype, context detection, context subtype, and exact
  role;
- report agreement per source and per document, not only globally;
- inspect documents that contribute the most disagreements instead of assuming
  every disagreement is random;
- distinguish annotation-policy disagreement from low-quality extraction; and
- verify overlap consistency within each pass.

Recommended terminal binary-membership gates from the completed run are:

- overall A/B agreement at least 98%;
- each source at least 95%; and
- unresolved fraction at most 0.5%.

These are binary-membership gates, not proof that the exact eight-way taxonomy
has the same reliability. The completed run's repaired and filtered cohort had
99.9133% binary agreement, while auxiliary role detection remained less
complete. Report both agreement and trusted-label coverage; they are different
quantities.

## 8. Lessons from the failed first specification

1. **Definitions must express use, not appearance.** “Looks like a reference”
   was insufficient for filler, continuation, and boundaries. Block action made
   the categories coherent.
2. **Context roles cannot exist globally.** Calling debris outside any
   bibliography `FILLER` made the label meaningless. Requiring an entry anchor
   fixed a large class of errors.
3. **Header must have a detectable surface contract.** Allowing inferred
   semantic headings caused high header-detection disagreement. Restricting the
   role to ATX Markdown headings removed that ambiguity.
4. **Binary agreement can survive subtype disagreement.** Task-specific
   recoding preserves useful BIB supervision without inventing exact-role
   consensus.
5. **Model names are provenance, not reviewer nicknames.** Mixed continuation
   runs must store runtime provenance per batch.
6. **Do not re-annotate completed work when changing models.** Import verified
   completed batches, continue only pending packets, and retain which model
   produced each batch.
7. **Keep test labels sealed until candidates are frozen.** Annotation can run
   in parallel with development, but labels and predictions must not meet until
   the evaluation boundary.
8. **Consensus silver is not human gold.** High agreement does not remove
   correlated model error or settle an undefined corpus-policy question.

## 9. Existing implementation entrypoints

- Prompt used by the historical run:
  `SEALED_BIBLIOGRAPHY_ROLE_PROMPT.md` (superseded for future runs by the two
  learned invariants in this document).
- Output schema: `sealed_bibliography_role.schema.json`.
- Model runner/coordinator: `sealed_bibliography_sol_coordinator.py`.
- Role contract: `bibliography_role_contract_v2.json`.
- Context-only repair: `repair_contextual_bibliography_roles.py`.
- Markdown-header repair: `repair_non_markdown_header_roles.py`.
- Task-specific consensus: `materialize_consensus_silver.py`.
- Completed-run report: `CONSENSUS_SILVER_20260719.md`.

Before the next annotation run, create a versioned prompt/contract that embeds
the context-anchor and Markdown-heading rules directly. The repair scripts
should remain available as validation assertions, but a correctly specified
new run should produce zero changes under both repairs.
