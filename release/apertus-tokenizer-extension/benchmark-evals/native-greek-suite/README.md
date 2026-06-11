# Native Greek Suite

This is the 2026-05-26 vetted native-Greek evaluation pass across Apertus-Base
and every available experiment checkpoint.

Headline Greek score is native-first:

- includes GreekMMLU, ILSP Medical MCQA Greek, and ILSP ASEP MCQA;
- reports Plutus QA as a domain add-on;
- reports `greek-nlp/benchmark` as supporting evidence;
- excludes explicit machine-translated Greek diagnostics from the headline.

Main result: Vanilla is ahead of TokenDistil on the native MCQ headline, while
Apertus-Base remains above all continued checkpoints. TokenDistil still leads
Vanilla on the older fallback downstream/retention bundle; use
`supporting-material/provenance/evals/NATIVE_GREEK_SUITE_RESULTS_20260526.md`
for the full interpretation.

Files:

- `NATIVE_GREEK_SUITE_SUMMARY.md` - generated aggregate table.
- `native_mcq_aggregate.csv` - native MCQ aggregate by checkpoint.
- `native_mcq_per_task.csv` - native MCQ per-task scores.
- `greek_nlp_supporting_aggregate.csv` - non-translation GreekNLP support mean.
- `greek_nlp_per_task.csv` - GreekNLP per-task rows.
