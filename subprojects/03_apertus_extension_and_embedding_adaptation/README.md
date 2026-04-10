# 03 Apertus Extension And Embedding Adaptation

## Scope

Plan and later implement model-side adaptation after the tokenizer extension is frozen.

## Already Decided

- this comes after tokenizer and corpus work, not before
- embeddings and `lm_head` both matter because `tie_word_embeddings = false`
- only the new rows need explicit initialization
- the intended schedule is:
  - frozen-base warmup
  - then full continued pretraining

## Still Open

- exact initialization method
- exact warmup schedule
- exact multilingual replay ratio
- exact acceptance criteria for model-side success

