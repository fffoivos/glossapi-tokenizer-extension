# Native Greek suite summary

Native headline uses vetted native Greek tasks only. MT diagnostics are excluded.

| Model | Native MCQ general | MCQ tasks | MCQ + Plutus | greek-nlp supporting mean | greek-nlp metrics |
|---|---:|---:|---:|---:|---:|
| Apertus-Base | 0.4817 | 3 | 0.4902 | 0.2150 | 6 |
| Centroid-2B | 0.2824 | 3 | 0.2796 | 0.1388 | 6 |
| ReTok-2B | 0.3685 | 3 | 0.3731 | 0.1577 | 6 |
| ReTok-3.5B | 0.3770 | 3 | 0.3772 | 0.1537 | 6 |
| TokenDistil-2B | 0.3961 | 3 | 0.4049 | 0.1750 | 6 |
| TokenDistil-3.5B | 0.4028 | 3 | 0.4121 | 0.1838 | 6 |
| TokenDistil-5B | 0.4109 | 3 | 0.4160 | 0.1733 | 6 |
| TokenDistil-Init | 0.2939 | 3 | 0.2915 | 0.1664 | 6 |
| Vanilla-2B | 0.4327 | 3 | 0.4256 | 0.1978 | 6 |
| Vanilla-3.5B | 0.4370 | 3 | 0.4333 | 0.1952 | 6 |
| Vanilla-5B | 0.4305 | 3 | 0.4329 | 0.1679 | 6 |

Notes:

- `native_mcq_general` averages GreekMMLU, ILSP Medical MCQA, and ILSP ASEP MCQA.
- `MCQ + Plutus` adds the domain-specific Plutus QA finance task.
- `greek-nlp supporting mean` excludes the upstream `machine_translation` task.
- Per-task CSVs remain authoritative for domain-specific interpretation.
