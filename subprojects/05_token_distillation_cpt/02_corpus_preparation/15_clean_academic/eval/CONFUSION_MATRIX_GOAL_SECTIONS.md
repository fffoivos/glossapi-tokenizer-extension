# Goal-eval — section sources (Kallipos/Pergamos): TOTAL bibliography recall

sampled 300 sections across all classes (β + high/low entry-density non-β).

## Stacked-pipeline confusion (detector = section-classifier→β→gate; weighted to corpus)

| | truth: bibliographic list | truth: not |
|---|---:|---:|
| detector flags bib | 28954 | 1206 |
| detector keeps | 6753 | 861759 |

**precision = 0.960  recall = 0.811  F1 = 0.879**

## Where the misses are (the blind spot the β-only eval could not see)

- FN total (weighted) = 6753.  **Classifier miss** (true bib in a NON-β section the gate never saw) = 1927 (29%).  Gate miss (β but kept) = 4826.
- raw: 1 of the sampled true-bibliography sections were classified NON-β (predicted_section ∈ {'κ': 1}) — invisible to the β-gate.
- FN by predicted_section: {'β': 8, 'κ': 1}