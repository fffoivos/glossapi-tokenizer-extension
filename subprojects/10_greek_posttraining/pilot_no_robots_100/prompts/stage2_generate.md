# Stage 2 — Write the Greek response natively

You are writing the **assistant response** for a Greek instruction-tuning dataset. You write in
Greek, for a Greek reader, as though the Greek prompt were the original — never as a translation of
the English answer.

You receive: the Greek prompt (and system prompt / user turns for multi-turn rows), the translation
class and hazards assigned in stage 1, the target register, and the **English reference answer**.

## What the English reference is for

The source rows were written by human annotators, so the reference answer carries real human quality
— its content coverage, structure, and length are worth keeping. **Use it as an anchor, not as a
source text.** Convey the same substance at a similar level of detail and a similar length, but
compose the Greek freshly. Do not translate its sentences. If a sentence of it only makes sense in
English, drop or replace that part and record the deviation.

Where the reference is weak, thin, or wrong, you may write a better answer — say so in `deviations`.

## Register

Write in the `target_register` you are given: `ενικός` = singular/informal address,
`πληθυντικός` = plural/formal address, `neutral` = phrase so that no choice is forced (impersonal
constructions, avoiding direct address).

**The register must be internally consistent across the whole response, and consistent with the
prompt.** For multi-turn rows it must also stay consistent across every assistant turn and match the
persona in the system prompt. Report the register you actually used in `register_used`; if you had
to depart from the instruction, say why in `register_note`.

## Class-specific requirements

- `LITERAL` — the answer must remain about the same real entities as the English. Do not swap in
  Greek examples for facts.
- `LOCALIZE` — stay consistent with the localization decisions already made in the prompt. Do not
  introduce new foreign references that clash with them.
- `VERBATIM-FREEZE` — reproduce protected material (code, quoted text) unchanged; write the
  surrounding explanation in Greek.
- `PRESERVE-DEFECT` — the prompt is deliberately flawed. **Do not silently fix it.** Respond as a
  good assistant would to a flawed request: point out the ambiguity, false presupposition, or missing
  information, and ask for what is needed or answer conditionally.
- `CONSTRAINT-PRESERVING` — satisfy the constraint exactly as restated in Greek, and state in
  `constraint_check` how your answer satisfies it (counts, keyword present, format).
- `REGISTER-CRITICAL` — the persona is the point. Sustain the voice fully; a flattened, neutral
  assistant voice is a failure even if the content is right.
- `RE-EXECUTE` — perform the task on the Greek text using Greek's own machinery. If the task is
  about formality, use the address axis and register vocabulary, not a literal rendering of the
  English lexical changes.
- `REGENERATE-NATIVE` — answer the substituted Greek task on its own terms, and **honour
  `derived_elements`**: use the re-derived names and catchphrases exactly as stage 1 set them, never
  the literal translation of the English ones.

## Greek quality bar

Natural modern Greek, monotonic, NFC. No calques of English syntax. No Latin-script words unless
they are genuinely the Greek convention (technical terms, proper names, code). Correct final sigma.
Do not use Greeklish. Never mention that anything was translated, and never refer to yourself as an
AI model, a language model, or by any vendor name.

## Output

Return **one JSON object per input row**, in the same order, as a JSON array:

```json
{
  "row_id": "<copied verbatim>",
  "response_el": "the Greek assistant response (single-turn rows)",
  "assistant_turns_el": ["... one per assistant turn, in order (multi-turn rows only, else [])"],
  "register_used": "ενικός|πληθυντικός|neutral",
  "register_note": "... or null",
  "constraint_check": "... or null",
  "deviations": ["ways this departs from the English reference, and why"],
  "self_flags": ["anything a human reviewer should check: uncertain terminology, a cultural call, a weak reference you improved on"]
}
```

Output the JSON array and nothing else.
