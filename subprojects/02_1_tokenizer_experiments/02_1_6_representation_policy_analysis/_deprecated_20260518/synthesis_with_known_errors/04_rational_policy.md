# Phase 4 — Rational policy + necessary/accidental split

For each of the 15 effective-policy rules in `03_effective_policy.md`,
classify *necessary* (forced or defensible-on-principle) vs
*accidental* (a contingent gap or unstated artifact).

The **rational core** is the union of all necessary rules. The
remaining accidents are policy that a defensible reproduction of
Apertus's stated goals would not regenerate the same way.

## Classification of the 15 rules

| # | Rule | Class | Reason |
|---:|---|:---:|---|
| 1 | English unconditional primary | **necessary (qualified)** | Forced by the choice to use English-only datasets. The choice TO use them is necessary because the field's quality pipelines exist only in English. |
| 2 | Multilingual content allocated by natural frequency | **necessary** | Stated policy by Penedo/Apertus, defensible on simplicity + scaling grounds. Equal-share alternatives have known failure modes (over-fitting low-resource langs). |
| 3 | Quality filtering only where classifier exists | **necessary (mechanism), accidental (specific list)** | The mechanism is correct. The HQ-20 specific list has 3 unexplained skips (Korean / Romanian / Ukrainian) and 1 unexplained inclusion (Vietnamese). |
| 4 | 10 % / 33 % retention per curriculum stage | **necessary** | Stated curriculum-design rationale. Any reasonable HQ filter requires a retention rate. |
| 5 | 0.95 sampler haircut on secondary HQ-20 ring | **accidental** | No paper or card documents the primary/secondary ring split or the 0.95 value. Project-internal artifact. |
| 6 | Toxicity haircut for 9-language cover set | **accidental** | Tooling-availability gap, not a stated principle. A principled rebuild would either cover more languages or apply a language-agnostic filter. |
| 7 | OCR post-processing for 5 EU languages (Institutional Books) | **accidental** | Same as Rule 6 — tooling-availability gap. Greek volumes exist; OCR pipeline didn't scale. |
| 8 | Parallel pairs follow ParaDocs's 6 en-X | **accidental** | Tooling-availability gap. Greek has EuroParl coverage; ParaDocs's narrower scope is a contingent dataset choice, not a principled exclusion of Greek. |
| 9 | Mistral-strong-11 sets inherited downstream priority | **accidental for Apertus** | Mistral's stated policy ≠ Apertus's stated policy. Apertus did not adopt Mistral's strong-language list as its own; it inherits the consequence by inheriting the tokenizer. |
| 10 | Tokenizer chosen for Gini-fairness on FLORES+ 55 | **necessary** | Stated fairness metric. FLORES+ includes Greek so Greek's fairness contributed to the Gini optimization. |
| 11 | Script-isolated languages need dedicated vocab slots | **necessary** | Mathematical, non-negotiable. |
| 12 | BPE merge order frequency-driven on Mistral's training data | **necessary (mechanism), accidental (Mistral's specific mix)** | The mechanism is BPE — necessary. The specific Mistral mix that produced the 1,479-Greek-token starting point is a contingent input Apertus inherited. |
| 13 | Public framing: Swiss-multilingual-global triad | **necessary (the multilingual commitment)** | The "Swiss" and "global" parts are positioning. The "multilingual" / "1,000+ languages" claim is a substantive commitment Apertus is publicly held to, including by SWI / Wikipedia. |
| 14 | Romansh / Swiss-German get named affordances | **necessary (for Apertus's specific mission)** | This is the Swiss-mission-justified policy. A defensible reproduction of Apertus's mission would keep these affordances. |
| 15 | Aggregate language count = primary marketing claim | **accidental** | Marketing rhetoric. The substantive commitment is multilingual quality, not the headline number. |

## The rational core

The rational-core policy, derived from the necessary rules only:

> **R1** (from Rule 1): Apertus uses the highest-quality available
> pretraining data per language. English-only datasets exist;
> multilingual analogues mostly don't. English will dominate consumed
> pretraining tokens.
>
> **R2** (from Rules 2, 3, 4): Multilingual content enters at natural
> web frequency. Where a per-language quality classifier exists, the
> top 10–33 % of documents are retained; otherwise random sampling at
> matching share.
>
> **R3** (from Rule 10): The tokenizer is chosen to minimise per-
> language fertility variance (Gini) over a held-out parallel
> multilingual benchmark (FLORES+).
>
> **R4** (from Rules 11, 12): Script-isolated languages receive
> dedicated vocab slots proportional to the tokenizer's discovery on
> in-distribution training data. Latin-script languages share merges.
>
> **R5** (from Rule 13): Public commitment to "1,000+ supported
> languages" obligates non-trivial coverage for every claimed
> language, where "non-trivial" means at least sufficient vocab to
> tokenize the language with fertility comparable to a fair-share
> baseline (FLORES+ Gini target).
>
> **R6** (from Rule 14): Apertus's Swiss mission justifies named
> per-language investment for Switzerland's official languages
> (German, French, Italian, Romansh) plus English. Other languages
> receive default treatment.

These 6 statements are the rational policy. Everything in Phase 3's
15 rules that didn't make it here is accidental: the specific HQ-20
list (Rule 3-specific), the 0.95 ring (5), the toxicity cover set
(6), the OCR-5 (7), ParaDocs-6 (8), Mistral-strong-11 (9), the
specific Mistral training mix (12-specific), the aggregate count
marketing (15).

## What R1-R6 imply for Greek

### From R1

Greek receives no English-only-dataset boost. Greek's pretraining
tokens come from multilingual sources only: FW2 + FW2-HQ +
Clean-Wikipedia + EuroParl + EuroBlocks-SFT. Total measured: ~3.11 B
Greek tokens. Necessary; nothing to change.

### From R2

Greek is in FW2-HQ HQ-20 → quality filter applies → 10 % or 33 %
retention. Stated policy. The 0.95 haircut (Rule 5) is accidental; a
defensible reproduction would not apply it. Greek's data share rises
slightly under the rational core (~0.024 % → ~0.025 % of pretraining
tokens).

### From R3

The Gini-fairness criterion **explicitly counts Greek** in the
55-language FLORES+ benchmark. Apertus's chosen tokenizer optimizes
for *aggregate* low-Gini, which can leave specific languages with
poor fertility while the average is fair. The rational core *does
not* fix Greek-specific fertility — it accepts Greek's specific
operating point as a consequence of the Gini-aggregate optimization.

But: if Apertus were redoing the tokenizer choice with the explicit
goal of per-language acceptable fertility (a stricter reading of R3),
Mistral-Nemo's 1,479-Greek-token starting point would not satisfy
"acceptable fertility for every FLORES+ language." Specifically,
Greek under Apertus base has fertility 2.41 on `modern_greek_eval`
vs ~1.5 for English on equivalent slices — a 60 % per-token-cost
premium. **This is the rational-core gap that motivates extension at
all.**

### From R4

Greek is script-isolated → needs dedicated vocab slots.

What fraction does R4 imply for Greek? The mathematical answer is
"whatever the BPE training on a Greek corpus produces, up to the
point of diminishing returns." Empirically (from `02_1_4_cutoff_analysis/
REPORT.md` §2 fertility table), Greek's natural elbow on C3's
in-distribution data is ~16-17 k added tokens (gains drop below
0.015 / +1k step beyond that). The mathematical optimum is the elbow,
not a peer-anchor.

### From R5

The public commitment to "1,000+ languages" obligates Greek to receive
"non-trivial coverage." The operative definition: Greek's fertility
should be **comparable to the fair-share Gini baseline** on FLORES+
languages. Empirically (per the cutoff REPORT), this lands somewhere
between Greek-tier (current) and Arabic-tier (the closest script-
isolated HQ-20 peer with adequate fertility).

Arabic's PMI footprint in Apertus is 7,146 = 5.45 % of vocab. Arabic's
fertility on FLORES+ is competitive. **R5 implies Greek's fertility
target should be at least Arabic-comparable.**

### From R6

Apertus's Swiss mission gives **no Greek-specific entitlement**.
Greek is not Swiss-mission language. Greek's allocation is
**default treatment**, where "default" means the rational core's
multilingual policy with no per-language boost.

## What R1-R6 collectively suggest for Greek's vocab budget

Three converging arguments:

### Argument A — from R3 + R5 (fairness on FLORES+)

Greek's current fertility on `modern_greek_eval` is 2.41 (vs ~1.5 for
fair-Gini benchmark languages). Bringing Greek to fair-Gini parity
requires fertility ~1.5, achieved at the +11k extension level per
`02_1_4_cutoff_analysis/REPORT.md` §2.

Result: **+11,264 added Greek tokens.**

### Argument B — from R4 + R11 (script-isolated peer parity)

Greek should match the script-isolated HQ-20 peer cluster. The cluster
mean (Arabic + Korean + Japanese + Chinese + Persian) is ~4,300 PMI
tokens; the highest member (Arabic) is 7,146.

Target Greek total: 5,000-7,000 tokens. Greek current base: 1,479.
Implied added: **+3,521 to +5,521 added** → closest grid cutoffs:
+3,072 (Korean-tier) or +5,120 / +6,144 (German-tier / Arabic-tier).

### Argument C — from R5 alone (the multilingual commitment as a floor)

The "1,000+ languages" public commitment plus the fact that Greek is
a HQ-20 quality-filtered language obligates Greek to be in the
HQ-20-typical vocab range, not the long-tail range. The HQ-20-typical
range (excluding outliers English and French): ~2,000-7,000 PMI
tokens (Polish 2,570, Dutch 3,045, Italian 4,712, Arabic 7,146).

Greek's current 1,479 is **below this range**. The minimum increase to
enter the range is to ~2,500 (added +1,021). The mid-range target is
~4,500 (added +3,021). The upper end is ~7,000 (added +5,521).

Result: **+1,024 to +5,632 added**, depending on whether Greek is
treated as "entering HQ-20 mainstream" or "matching HQ-20 high end."

## Convergence point

The three arguments give a range:

| Argument | Implied added |
|---|---:|
| A — FLORES+ fair-Gini fertility parity | +11,264 |
| B — script-isolated peer-cluster parity | +3,072 to +6,144 |
| C — HQ-20 mainstream membership | +1,024 to +5,632 |

The **rational core does not produce a unique answer**. Three
defensible budgets exist:

1. **Minimal-rational** (~+3,072 — Korean-tier): satisfies R4 + R5 + R6
   at the floor. Greek matches the script-isolated HQ-20 mean. Most
   conservative defensible budget.
2. **Mid-rational** (~+6,144 — Arabic-tier / German-tier): satisfies
   R4 + R5 strongly. Greek matches the highest script-isolated HQ-20
   peer (Arabic 7,146). The principled middle position.
3. **Maximal-rational** (~+11,264 — current C3 pick): satisfies R3 +
   R5 strictly (Greek reaches fair-Gini-on-FLORES+ fertility parity).
   The C3 cutoff REPORT's current recommendation.

**The C3 REPORT's current pick (+11,264) IS rational-core-defensible
under Argument A**, but it goes further than Arguments B and C
require. The "match English-unique 13 k" framing in the C3 REPORT is
rhetorical — the *substantive* defense of +11,264 is Argument A
(fair-Gini-on-FLORES+ fertility parity), which the C3 REPORT does
not currently articulate in those terms.

## What this leaves unresolved

The single biggest open question is which version of R5 — the
multilingual commitment — Apertus actually means:

- **Weak R5**: "1,000+ languages must be supported in some sense"
  — i.e. the tokenizer must produce non-byte-fallback output for
  these languages. Greek's existing 1,479 tokens already satisfy this.
- **Mid R5**: "1,000+ HQ-20 languages must have HQ-20-typical
  fertility" — Greek requires ~+5,000 added.
- **Strong R5**: "1,000+ FLORES+ languages must have fair-Gini
  fertility" — Greek requires ~+11,000 added.

Apertus's primary sources do not disambiguate. The model card's
"1,811 natively supported" claim leans weak; the apps page's
"Multilingual competence" leans mid; the §2.2 Gini-fairness
optimization leans strong. The three claims coexist without resolution.

Greek's vocab budget therefore depends on which R5 interpretation
Apertus *itself* would accept on principle, and that question is not
answerable from primary sources alone — it requires asking Apertus's
authors directly, or making a normative interpretation. The next
file (`FAIRNESS_DEFINITION.md`) makes that normative interpretation
explicit.
