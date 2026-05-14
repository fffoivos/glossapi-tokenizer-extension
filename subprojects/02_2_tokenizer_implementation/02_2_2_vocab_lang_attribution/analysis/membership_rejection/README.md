# membership_rejection

Applies `02_2_1_char_language_membership` to the full 1,933-row
`02_2_2_vocab_lang_attribution` histogram matrix as a rejection signal.

The rule is deliberately asymmetric:

- For rows with direct CLM bits, `rejected_decoded_mass` is a hard
  language-membership rejection: the decoded token fired in that dataset,
  but `bitmask_and` has none of the language group's bits.
- `partial_utf8`, `byte_unmapped`, and `special` tokens are reported as
  `unknown_standalone_mass`, not as rejections. Their zero masks mean the
  char-level tool cannot evaluate the standalone token.
- Rows without direct CLM bits use `script_proxy_bits` only as a coarse
  script-family signal. That is not a language-level proof.

Run:

```bash
PROJECT=/home/foivos/Projects/glossapi-tokenizer-extension
$PROJECT/.venv-hplt-review/bin/python build.py
```

Outputs:

- `language_membership_rejections.tsv`
- `direct_rejection_top_tokens.tsv`
- `summary.json`
