# Fresh bibliography edge and component review — 2026-07-14

## Outcome

The frozen asymmetric edge policy now has an independent, blinded review set,
and a separate source-balanced component-labeling set is ready for review.
Neither job fitted a model, selected a threshold, or changed corpus text.

Final code commit: `d0c22a9d103c5afe2be171911b147a81ea210172`

Clariden jobs:

- focused tests: `2759693`, `COMPLETED`, 12 passed;
- packet build: `2759694`, `COMPLETED` in 2 minutes 11 seconds.

Final Clariden artifact root:

`/capstor/scratch/cscs/fffoivos/runs/05_token_distillation_cpt/full_corpus_v2/classifier_research/unseen_block_reviews/fresh_edge_component_20260714_d0c22a9`

Local review copy:

`outputs/bibliography-fresh-edge-component-review-20260714`

## Freshness contract

The input is the canonical 500-document source-matched holdout. That holdout
already excludes all 2,000 STRUCT-2K identities and performs a same-source
near-duplicate check against STRUCT-2K. Before the new 90-document candidate
pool was selected, this run additionally excluded both the document IDs and
work IDs from the earlier 30-document review packet.

The final candidate pool contains 30 Greek PhD, 30 Kallipos, and 30
OpenArchives documents. It was selected before model access. Its overlap with
the earlier review packet is zero.

## Edge review

The reviewed policy is exactly the train-OOF frozen policy:

- left edge: all explicit deterministic negative roles;
- right edge: structural roles only;
- the independently anchored bibliography core is never removed.

The packet contains 30 documents, balanced 10 per source, and 170 line cases:

- all 85 lines the frozen edge policy removes;
- 85 deterministically sampled retained boundary controls.

The interface hides KEEP/REMOVE until the reviewer labels the line. After the
decision, it reveals the frozen action and whether it agrees. The controls are:

- `← BIB`;
- `→ NOT BIB`;
- `↑ UNSURE`;
- `↓ WEIRD`.

This is an independent precision/safety audit of the edge change. The removed
cases are exhaustive for the 30 selected documents; retained controls are a
sample, so they do not form a full recall estimate for the complete base
classifier.

## Component-labeling review

The frozen base classifier proposed 182 components in the 90-document pool:

- Greek PhD: 77;
- Kallipos: 68;
- OpenArchives: 37.

The largest exact source-balanced two-stratum set is 108 components: 36 per
source, split into 54 bibliography-like selections and 54 citation-dense
narrative-risk selections. Those strata are sampling metadata, not labels or
ground truth. Review choices are:

- `← BIB`;
- `→ CITATION PROSE`;
- `↑ OTHER`;
- `↓ WEIRD`.

Long components show at most 60 highlighted component lines, sampled from the
first, middle, and last parts, with explicit omission markers and three outer
context lines on either side. The original component line boundaries and total
line counts remain in the packet.

## Review isolation and export

Both readers save progress separately by reviewer name and resume at the first
undecided case. Use reviewer `foivos` for the user pass and `codex` for the
independent Codex pass. Each reader exports a JSON file bound to its packet
hash.

Packet hashes:

- edge: `8fc1a4f227ae488c277ee90eace7730bb01ea66aae0daf43624fa0640850cb4c`;
- components: `388d0310e2594f110b341a691724be0dfd460bed4e67830a10083950673c7121`.

## Next decision

After both reviewer exports exist:

1. summarize edge false removals, retained-control errors, agreement, and
   source breakdown without changing the frozen policy;
2. adjudicate reviewer disagreements;
3. materialize reviewed components as a document-grouped training/evaluation
   dataset;
4. train and cross-fit the component classifier, keeping this fresh edge packet
   sealed as evaluation evidence.

The complete machine receipt is `receipt.json` in the artifact root. The local
copy intentionally omits the 53.7 MB candidate-pool JSONL; that file remains on
Clariden and is hash-bound in the receipt.
