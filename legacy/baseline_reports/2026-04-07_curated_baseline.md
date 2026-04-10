# Curated Baseline

## Sample

- artifact: `/home/foivos/data/glossapi_work/analysis/tokenizer_fertility_20260407_curated/evaluation_chunks.jsonl`
- chunk count: `18`
- doc count: `15`
- source dataset count: `8`

## Documents Used

- `train` `1000_prwta_xronia_ellhnikhs` `first1k_000812`
- `train` `1000_prwta_xronia_ellhnikhs` `first1k_000927`
- `train` `AI-team-UoA/greek_legal_code` `test_001633`
- `train` `AI-team-UoA/greek_legal_code` `train_014302`
- `train` `Apothetirio_Kallipos` `paper_A_0502`
- `train` `Apothetirio_Kallipos` `paper_B_1017`
- `train` `Ekklisiastika_Keimena` `ekkl_000054`
- `train` `Ekklisiastika_Keimena` `ekkl_000651`
- `train` `ellinika_dedomena_europaikou_koinovouliou` `GBY_106`
- `train` `ellinika_dedomena_europaikou_koinovouliou` `LTL_517`
- `train` `eurlex-greek-legislation` `doc_12040`
- `train` `eurlex-greek-legislation` `doc_4012`
- `train` `openbook_gr` `FKB_641`
- `train` `opengov.gr-diaboyleuseis` `opengov_000908`
- `test` `opengov.gr-diaboyleuseis` `opengov_001254`

## Headline Metrics

| Model | Tokens / 100 chars | Chars / token | Tokens / word | Avg Greek word tokens | Single-token Greek words |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ilsp/Llama-Krikri-8B-Base` | `29.47` | `3.39` | `2.01` | `2.42` | `33.5%` |
| `google/gemma-4-E2B` | `47.40` | `2.11` | `3.23` | `3.11` | `21.2%` |
| `swiss-ai/Apertus-8B-2509` | `52.94` | `1.89` | `3.61` | `3.57` | `15.1%` |

## Interpretation

- Krikri is the current Greek-efficiency reference.
- Gemma 4 is substantially worse than Krikri, but still materially better than Apertus on this sample.
- Apertus is the natural tokenizer-extension target because it loses the most on Greek segmentation while still being a model family we may want for other reasons.
