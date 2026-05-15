# English/German Main Token Set Prototype

This directory is the earlier two-language prototype for deriving
English and German token sets from the same firing-count matrix used by
the later PMI multi-language pass.

It is kept because it documents the comparative reasoning that led to
the full `main_token_sets_pmi/` method: English and German share many
Latin-script tokens, so language attribution cannot be decided by char
membership alone. The prototype compares empirical activation rates and
emits one main token list per target language.

For the current canonical multi-language outputs, use
`../main_token_sets_pmi/`.

## Files

- `build_main_sets.py` — builder for the two-language prototype.
- `main_tokens_en.txt` — English prototype token list.
- `main_tokens_de.txt` — German prototype token list.
- `summary.json` — counts, mass coverage, and English/German activation
  distance summary.

