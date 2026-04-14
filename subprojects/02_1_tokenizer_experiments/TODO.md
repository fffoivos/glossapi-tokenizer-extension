# TODO

- define the exact fertility metric bundle for model selection
- define the exact decision rule for the elbow
- define the exact comparison rule for the four tokenizer arms:
  - fresh discovery `BPE` on `GlossAPI-only`
  - fresh discovery `BPE` on `GlossAPI + HPLT`
  - continuous `BPE` from Apertus on `GlossAPI-only`
  - continuous `BPE` from Apertus on `GlossAPI + HPLT`
- review whether additional modern-Greek-only control slices are needed
- define the exact continuous-`BPE` training procedure starting from the Apertus tokenizer and merge table
