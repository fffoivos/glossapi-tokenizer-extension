# TODO

- design the first compatible Greek `BPE` training config
- implement tokenizer diffing against Apertus vocab and merges
- implement merge-rule extension assembly
- add a regression checklist and executable tests

## In progress

- **`vocab_lang_attribution/`** sub-subproject — per-token language
  attribution for the 131k base-vocab entries. Run started 2026-05-13 on
  8 × c4-highcpu-192 workers; 87 % complete as of report capture, ETA
  ~30-45 min remaining. Status + open issues + scripts in
  [`vocab_lang_attribution/RUN_REPORT.md`](vocab_lang_attribution/RUN_REPORT.md).

