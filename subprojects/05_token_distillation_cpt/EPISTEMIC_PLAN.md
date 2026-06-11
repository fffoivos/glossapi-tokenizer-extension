# Epistemic plan — experiments before the 60 B run

Living agenda. The pilot answered "does the TD/Greek-tokenizer recipe work?" (yes,
clean GreekMMLU win). What remains is a small set of experiments whose answers set
the 60 B design. Each is framed as: *question → decision it gates → method →
decision rule*. We read results against the rule — no post-hoc rationalizing.
Companion: `ROADMAP_20260611.md`, `reports/`.

TD is the production choice; Vanilla is kept only as a one-run tokenizer control.

---

## Evaluation policy (decided)

- **Adaptation = GreekMMLU only.** The one Greek benchmark we trust — decontaminated,
  broad (16.6 k Qs, 31 subjects), carried the headline. **Drop** ILSP-medical,
  ILSP-ASEP, Plutus-QA, and the `greek_nlp_s100` generative suite.
- **Forgetting = held-out LM loss** on 0.5 B slices of the *old* (Apertus-original)
  data — contamination-free by construction. We do **not** decide on English/
  multilingual *benchmark* accuracy (drop the retention suite, or keep it as cheap
  informational only).
- **Decontamination = GreekMMLU only** (forward, on the new training data, reusing
  the pilot pipeline). Because we don't decide on the other benchmarks, the
  "extended decontam" workstream is unnecessary, and the retrospective
  "were the pilot's multilingual gains contamination?" study is **off the critical
  path** (do only if we want to caveat the v1 report).

Both adaptation signals (GreekMMLU + new-Greek held-out loss) and the forgetting
signals (old-data held-out loss) are read on every run.

---

## (a) Prerequisites — three builds, in parallel

1. **Decontaminate + build the curriculum dataset.** New training data, screened
   against **GreekMMLU**. Built as the **two-phase** curriculum — **70 % HPLT then
   30 % GlossAPI** (GlossAPI = OpenArchives, pilot-comparable; phd/tail are 60 B
   additions). Two phases via the data-path/segment mechanism (HPLT binary →
   GlossAPI binary), **not** an ordered binary (the global shuffle defeats that —
   the pilot bug). Within-phase shuffle stays on; the cross-phase order is the
   curriculum.
2. **Forgetting held-out sets.** 0.5 B each (matching the 3 new-Greek held-outs),
   from the replay/Apertus-original pools, excluded from training, wired into
   extra-valid: `old_english`, a few replay languages (e.g. `old_de/ru/zh`),
   `old_code`, and **`old_greek`** (greek_replay) to separate forgetting-of-old
   from learning-of-new. Built once; used by (b)–(d).
3. (Implicit) the new-Greek held-outs already exist from the pilot.

---

## (b) Vanilla control — one run

Vanilla at the **new settings** (curriculum + baseline replay), one replay value.
Purpose: re-anchor the tokenizer reference (TD vs Vanilla) at the new config. Not a
sweep — the pilot already settled TD ≫ Vanilla rigorously.

---

## (c) Replay sweep — TD + curriculum

**Question.** How low can replay go before the old distribution degrades?
**Gates.** The replay % carried into (d) and the 60 B build.
**Method.** TD + curriculum, baseline settings; vary **only** replay % (grid e.g.
{35 = pilot control, 25, 15}; budget proxy vs full = open). Read GreekMMLU +
new-Greek held-out (adaptation) and old-data held-out (forgetting).
**Decision rule.** **User picks** the replay % at the best forgetting↔adaptation
balance (forgetting curves vs GreekMMLU). Carry it forward.
**Ride-along.** Instrument the HPLT→GlossAPI boundary (loss/grad). This is *only a
sanity watch* here — the real shift test is (e); see the scale caveat there. If the
baseline transition is unstable, pause and look before trusting the sweep.

---

## (d) Peak-LR sweep — TD + curriculum

**Question.** Where is the adaptation↔forgetting knee for peak LR?
**Gates.** The production peak LR.
**Method.** TD + curriculum, at the replay % chosen in (c). Sweep the **other 3**
around the 5.5e-5 control: **{2.75e-5, 8.25e-5, 1.1e-4}**. All same ext tokenizer →
held-out losses directly comparable. Read GreekMMLU + new-Greek (adaptation) and
old-data held-out (forgetting).
**Decision rule.** **User picks** the LR at the knee — max GreekMMLU / min new-Greek
loss while the forgetting curves are still flat.
**Caveat.** On saturated pilot data the LR effect is muted (Greek is data-limited);
expect compressed differences — read with that in mind, or confirm on the probe.

---

## (e) DEFERRED — distribution-shift / curriculum-optimizer study (scale-dependent)

**Do not run at pilot scale — it's the wrong scale.** At 13.5 B (3,218 iters) the
70/30 curriculum gives ~1,670 iters HPLT then **~715 iters GlossAPI**, which is
≈ AdEMAMix's slow-momentum memory (1/(1−β₃) ≈ 1,000 steps) — and β₃ is still
*warming* at the boundary. So you'd see the disturbance begin but the run ends
before the optimizer recovers: **a pilot-scale result would mislead.**

**The shift is a 60 B-scale phenomenon.** At ~14,300 iters the GlossAPI phase is
~4,300 iters ≈ 4× the memory → build-up, disturbance, and recovery all resolve,
and the optimizer has room to handle it.

**When we get there, design it as its own scaling-tier experiment:**
- Needs post-boundary GlossAPI ≥ ~2,000 iters → ~25–30 B+ at the 70/30 ratio, or a
  50/50 ratio at ~17 B for the test only; or a multi-shift run if we only want the
  immediate spike, not full recovery.
- Levers (gated on the boundary diagnostic — only if the transition misbehaves):
  **β₂** {0.99, 0.995, 0.999} with **warmup decoupled** (the `2/(1-β₂)` rule is
  about the *start*, not the boundary — don't confound); **α** {0, 4, 8} (α = 0 ≡
  AdamW = "does the slow momentum hurt across the shift / is AdEMAMix needed");
  and the surgical **boundary re-warm** (reset the slow EMA at the GlossAPI start).
- Home: the scaling probe (~20–30 B) or the 60 B run itself, boundary instrumented.

---

## Sequence

1. **(a)** prerequisites (parallel).
2. **(b)** Vanilla control.
3. **(c)** replay sweep (TD+curriculum) → **user picks replay %.**
4. **(d)** peak-LR sweep (TD) at that replay → **user picks LR.**
5. Carry replay % + LR into the 60 B recipe (`ROADMAP_20260611.md`).
6. **(e)** distribution-shift study at the scaling/60 B tier — **designed later.**

Pilot-scale sweeps (replay, LR — effects show early) stay at pilot scale; the
distribution-shift study lives at the scaling tier. Open parameters: the exact
forgetting-set list, the replay grid, and the sweep budget (proxy vs full).

## Later (stub)
- **Data scaling.** Does GreekMMLU break past the pilot's ~59 % plateau with fresh
  data from the full ~55 B pool? → is 60 B worth the compute. (Pairs naturally with
  the probe in (e).)
