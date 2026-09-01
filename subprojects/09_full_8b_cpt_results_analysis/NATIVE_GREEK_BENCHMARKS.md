# Native-Greek checkpoint evaluation expansion

Date: 2026-08-12

The immediate goal is not a large benchmark zoo. It is to add independent
signals that can tell whether the update-9,536 GreekMMLU peak is a narrow exam
effect or a broader capability peak. “Native” here means the underlying prompts
or source material were authored in Greek, rather than an English benchmark
being machine-translated into Greek.

## Tier 1: score every saved base-model checkpoint

These tasks can be expressed as candidate continuation likelihood or a small
fixed label set, so they are appropriate for the base CPT checkpoints and do
not require an instruction-tuned model.

| Benchmark | Native evidence | Size and task | Recommended metrics | Priority |
| --- | --- | --- | --- | --- |
| [GreekMMLU](https://aclanthology.org/2026.findings-acl.448/) | Greek academic, professional and governmental examinations | 21,805 MCQs; 16,159-item decontaminated public subset already frozen here | accuracy, choice NLL, correct-answer BPB | Existing anchor |
| [DemosQA](https://huggingface.co/datasets/IMISLab/DemosQA) | Authentic `r/greece` questions and community-ranked Greek answers | 600 questions, four candidate answers | length-normalized choice NLL, accuracy, answer BPB | **Add first** |
| [Greek Medical MCQA](https://huggingface.co/datasets/ilsp/medical_mcqa_greek) | Past Greek DOATAP medical examinations | 2,034 MCQs across seven subjects; 432-item validation split | score the validation split with choice NLL, accuracy and subject macro-average | **Add first** |
| [Greek ASEP MCQA](https://huggingface.co/datasets/ilsp/mcqa_greek_asep) | Greek civil-service examination `1Γ/2025` | 1,200 MCQs across law, economics, administration, computing and history | choice NLL, accuracy, subject macro-average | **Add first** |
| [GPCR](https://huggingface.co/datasets/ilsp/greek_pcr) | 208 manually annotated Greek physical-commonsense items; about 40% are culturally specific | Two-choice PIQA-style completion | paired choice NLL and accuracy | Add after gated-access approval |
| [OYXOY](https://aclanthology.org/2024.findings-eacl.21/) | Expert-verified Modern-Greek linguistic judgments, partly derived from the Dictionary of Standard Modern Greek | NLI, two WSD formulations and metaphor detection | label NLL, exact/multilabel accuracy, macro F1 | Add after harness adaptation |
| [Greek Protipa Exams](https://huggingface.co/datasets/ilsp/greek-protipa-exams) | Greek Model and Experimental School admission exams from 2013–2025 | 1,702 test items plus 34 dev; mixed text/image | score the text-only MCQ subset first; report by year and subject | Add after gated-access approval |

DemosQA is especially valuable because it changes both register and source:
GreekMMLU is formal examination Greek, while DemosQA is naturally occurring
community discourse. Its community vote is a preference signal, not an
objective truth label, so report both accuracy and continuous choice loss.

## Tier 2: generation or critic-based evaluation

These benchmarks are useful, but a base model's prompt-following behavior and
the evaluator itself become part of the measurement. Run them on seven
decision checkpoints first: initialization, post-warmup, approximately 5B,
10B, the 39.997B GreekMMLU peak, cooldown start and terminal.

| Benchmark | What it adds | Evaluation design |
| --- | --- | --- |
| [GreekBarBench](https://arxiv.org/abs/2505.17267) | 310 native Greek Bar-exam problems spanning five legal areas, with facts, statutory citations and analysis | Use its published span-based rubric and a fixed multi-judge panel. The full task averages roughly 62k context tokens, so the current 4,096-context checkpoint cannot claim the official score; either evaluate a predeclared retrieval condition or defer the official run to a longer-context model. |
| [Plutus-ben](https://aclanthology.org/2025.emnlp-main.1535/) | Expert-annotated Greek financial NER, QA, topic classification and extractive/abstractive summarization | Use deterministic metrics where possible and a blinded pairwise judge for abstractive outputs. Report each of the six tasks separately. |
| [GreekSum / GreekT5](https://arxiv.org/abs/2311.07767) | Native Greek news summarization rather than exam answering | ROUGE/BERTScore are secondary; primary comparison should be blinded pairwise judging for factuality, coverage and Greek fluency against the same articles. |
| [Greek Civics QA](https://huggingface.co/datasets/ilsp/greek_civics_qa) | 407 Greek civics questions with long sourced reference answers | Score reference-answer BPB and generated answers with source-grounded critic rubrics. |
| [GEAR](https://huggingface.co/datasets/ilsp/GEAR) | Native Greek student-support prompts with human scores for empathy, understanding, reasoning and harm | Generate answers under one frozen completion prompt and use a calibrated judge panel; validate judge ordering against the released human-scored responses before scoring our checkpoints. |
| [Greek LLM Arena](https://llmarena.gr/) | 90 difficult Greek cases across public services, law, health, tax, language, culture and safety | Seek access to the fixed prompts/private rubrics or submit the selected checkpoints. Preserve its three-judge design and report judge agreement. |

For critic-based tasks, use at least two independent judges, blind model and
checkpoint identity, randomize answer order, repeat reversed pair order, and
calibrate on human-scored examples where the benchmark provides them. A judge
score without this calibration is diagnostic, not a selection gate.

## Relevant but not native-source primary evidence

The current ILSP suite also contains Greek translations of IFEval, MT-Bench,
MMLU, ARC, HellaSwag, TruthfulQA, MGSM and Humanity's Last Exam. They may be
useful for cross-language comparability or instruction following, but they do
not answer the native-Greek question and should not be averaged into the
primary native score.

The [Ancient-to-Modern Greek translation benchmark](https://huggingface.co/datasets/ilsp/ancient-modern_greek_translations)
is relevant to the polytonic extension, but it measures a specialized
translation capability. Keep it as a separately reported diagnostic rather
than mixing it into Modern-Greek checkpoint selection.

The newly released [Greek Culture Bench](https://huggingface.co/datasets/ilsp/greek_culture_bench)
contains 1,951 Greek questions with verification text and source URLs, and the
ILSP collection also lists small native-looking Modern-Greek and chronological
history QA sets. Their currently visible cards do not establish enough about
question authorship, answer construction or contamination to promote them yet.
Keep them on the watchlist and audit their gated cards and rows before use.

## Mandatory preparation before scoring

Every candidate benchmark must pass the same gate:

1. freeze the exact dataset revision and license/access terms;
2. normalize without destroying Greek diacritics;
3. search the complete executed training corpus for exact question, option and
   answer matches plus long n-gram and MinHash near-duplicates;
4. also measure collisions across benchmarks so overlapping exam questions are
   not silently counted multiple times in any aggregate;
5. freeze both full and decontaminated subsets, reporting removals by source;
6. freeze prompt, option ordering, length normalization and tokenizer revision;
7. score the same examples for every checkpoint;
8. report per-item outputs and paired bootstrap intervals, not only an average.

Official-exam datasets are particularly likely to overlap GlossAPI PDFs or HPLT
web pages. Their provenance is an advantage only if contamination is measured,
not assumed absent.

## Proposed execution order

1. DemosQA, Medical MCQA and ASEP MCQA over all 19 checkpoints.
2. GPCR, OYXOY and the text-only Protipa subset after access/schema review.
3. Plot all continuous metrics against GreekMMLU and source-conditioned BPB;
   test whether their best checkpoint lies before, near or after update 9,536.
4. Only then run the seven-checkpoint critic panel for GreekBarBench, GreekSum,
   Greek Civics QA and GEAR.

This ordering gives the cheapest independent evidence first and reserves judge
cost for checkpoints that remain scientifically ambiguous.
