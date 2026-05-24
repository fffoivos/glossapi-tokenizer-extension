# Production Base NFC Preprocess

Purpose: create the production-safe Vanilla/base-tokenizer Megatron binary from
the NFC-normalized CPT bulk stream.

Final job:

- Job: `2367579`
- Partition: `xfer`
- State: `COMPLETED`, exit `0:0`
- Elapsed: `00:16:07`
- Input JSONL:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix.nfc.jsonl`
- Output prefix:
  `/iopsstor/scratch/cscs/fffoivos/cpt_corpus/bulk_mix_base_nfc_megatron/bulk_mix_text_document`
- Output files:
  - `bulk_mix_text_document.bin`, `39,326,819,096` bytes
  - `bulk_mix_text_document.idx`, `115,083,482` bytes
  - `bulk_mix_text_document.manifest.json`

Manifest summary:

- rows: `5,754,172`
- sequences: `5,754,172`
- documents: `5,754,173`
- tokens: `9,831,704,774`
- workers: `64`
- wall seconds: `960.68`

Validation:

- Validation job: `2367575`
- Compared custom HF-tokenizer fallback output against the canonical
  Megatron-preprocessed `bulk_mix_base_megatron` output on the first `1000`
  original JSONL rows.
- First `1000` sequence lengths matched.
- First `1001` document indices matched.
- First `7,661,264` bytes of `.bin` matched exactly.
- SHA-256 for both compared byte ranges:
  `58fd30082e8ff9d20f75a81d260194b07abf248a67324dd1840960e0e156121a`

Why this exists:

- `bulk_mix.nfc.jsonl` is the production-safe corpus stream after NFC cleanup.
- The older bakeoff binary `bulk_mix_base_megatron` was built before NFC cleanup.
- Megatron's stock `preprocess_data.py` could not run on current `xfer` nodes
  because those nodes expose neither `uenv` nor a torch/transformers runtime.
- `preprocess_hf_jsonl_to_megatron.py` writes the same Megatron indexed-dataset
  format without importing torch, and was validated byte-for-byte on a prefix.
