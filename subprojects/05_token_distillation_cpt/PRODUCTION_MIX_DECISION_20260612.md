# Production (60 B) mix decision — replay fraction + old-Greek replay

**Date:** 2026-06-12 · **Status:** decided, ready to implement · **Scope:** the data
*mixture* for the ~60 B-new-token TD production run. Peak LR is now settled
separately in `PRODUCTION_LR_DECISION_20260613.md`; AdEMAMix and the
keep-vs-drop curriculum question remain separate.
**Evidence:** `reports/cpt_curriculum_forgetting_learning.html` (the `curriculum_sweeps_v2`
replay sweep) + `papers/notes/{ibrahim-cpt,stability-gap-cpt,gupta-rewarm-cpt}.md`.

---

## Decision

The 60 B production mix is, **as fractions of the total token budget**:

| Component | Share of total | Pilot was | Change |
|---|---:|---:|---|
| **New Greek** (HPLT → OpenArchives + phd + tail) | **~79 %** | ~74 % | +5 |
| **Foreign replay** (multilingual + code + math, Apertus-matched) | **~20 %** | ~21 % | ≈ |
| **Old-Greek replay** (`greek_replay` = nanochat ∩ Apertus-overlap) | **~1 %** | ~5 % | **−4** |
| Total replay | **~21 %** | ~26 % | −5 |

Equivalently in the curriculum-blend knob: **replay blend weight R = 0.25** (→ 20 %
foreign replay), with the old-Greek slot cut from ~5 % to ~1 % and the freed ~4 %
rolled into new Greek.

---

## Why (from the sweep, not from priors)

1. **Replay is a flat plateau over [0.25, 0.35]; do not go below it.** The TD replay
   sweep R ∈ {0.35, 0.25, 0.15}:
   - GreekMMLU peak: **R=0.25 → 55.1 %, R=0.35 → 54.9 %** (a tie — 0.2 pp, < the ~0.4 pp
     SE on 16,632 Qs), **R=0.15 → 52.7 %** (clearly worse).
   - Forgetting (mean old multilingual+code held-out Δ): **flat across all three**
     (+0.045 … +0.052 nats); old Greek **improves** at every R (≈ −0.19).
   - ⇒ Take the **low end of the tie (R=0.25 = 20 %)** for budget efficiency (less
     replay = more new Greek at equal forgetting). **Do NOT cut to ~13 % (R=0.15)** —
     that is where adaptation fell off; the ROADMAP's pre-sweep "10–15 %" guess
     (DA3) is **superseded** by this result.

2. **Drop the dedicated old-Greek replay (5 % → 1 %).** Old Greek was *never forgotten*
   — it improved ~−0.19 at every R, driven by positive transfer from the 79 % new
   Greek, not by its own replay slot. So a 5 % slot to *prevent* old-Greek forgetting
   is largely redundant. Keep **1 %** as (a) a cushion and (b) the source distribution
   for the old-Greek held-out probe. Ibrahim: *"even 1 % replay significantly reduces
   forgetting."* **One acknowledged bet:** part of the −0.19 is the model directly
   training on the 5 % `greek_replay` (in-distribution); the pilot can't tell us how
   much survives at 1 %. The 1 % slot + the old-Greek probe is the tripwire — if old
   Greek starts rising in the 60 B run, raise it back.

3. **Replay must stay distribution-matched to Apertus's real pretraining families.**
   Per stability-gap (2406.14833), *random* replay "barely moves the needle"; matching
   the pretraining mixture is what works. Keep the pilot's Apertus-family sources —
   English **FineWeb-Edu**, multilingual **FineWeb-2-HQ**, code **StarCoderData**
   (NOT CodeParrot), math **FineMath**. No new foreign families.

---

## Implement (for the execution agent)

1. **Start from** `03_training_experiments/dataset_build/bulk_13b.json` (the realized
   pilot recipe; `make_recipe_13b.py` is its builder).
2. **Re-weight to 79 / 20 / 1** of the ~60 B budget:
   - New-Greek bucket → ~79 % (it absorbs the freed ~4 % from old-Greek).
   - Foreign-replay bucket (multilingual + code + math) → renormalize to **~20 % total**.
     Keep the major + Balkan language tiers; the long-tail languages may be trimmed
     (ROADMAP DA3) but stay distribution-matched.
   - `greek_replay_apertus_original` → **~1 % total** (down from ~5 %).
3. **Keep building the held-out probes** unchanged: the new-Greek (hplt/openarchives/
   greek_phd) **and** the old-data forgetting sets (english/de/ru/zh/code/old_greek),
   carved at ≤25 % of each pool with id-drop, exactly as in `curriculum_sweeps_v2`.
   These are the monitoring instrument below.
4. **Extended decontamination** (ROADMAP §5) — screen *all* data including replay, not
   just GreekMMLU, since this is the shippable-design run.
5. Scale, schedule, curriculum keep/drop, and the LR/α sweeps are **out of scope here**
   — see ROADMAP §5–§6.

## Monitoring gate (the stability gap)

Forgetting in CPT is **front-loaded**: re-warming (and, for TD, the embedding swap)
causes a pronounced *early* old-data loss spike that then recovers — the stability
gap (2406.14833; Gupta 2308.04014). The pilot's end-state forgetting was mild, but the
**worst forgetting is the early transient trough, not the end** — and the gap
**deepens with both shift strength and token budget**, so 60 B may show a deeper gap
than 13.5 B did.

- **Watch the early multilingual held-out loss (en/de/ru/zh) over the first ~5–10 B
  tokens** — specifically the **depth and recovery speed of the early trough**, not the
  end-state value.
- **Trigger:** if that early gap is markedly deeper or recovers more slowly than the
  pilot (pilot trough ≈ warmup-end, recovery by ~1.4 B), **nudge foreign replay up
  toward 25 %** (R=0.35) — Ibrahim's strong-shift dose. English→Greek is a strong
  cross-lingual shift (their En→De used 25 %), so 20 % is deliberately *just under* the
  heuristic, justified by the pilot's flat empirical forgetting but not guaranteed at
  scale.

## Verify on review (acceptance criteria)

- Realized per-source token counts (from the built binaries) give **new Greek ≈ 79 %,
  foreign replay ≈ 20 %, old-Greek ≈ 1 %** of total.
- Foreign replay is **Apertus-family + distribution-matched**: FineWeb-Edu / FineWeb-2-HQ
  / **StarCoderData** / FineMath — **no CodeParrot**, no off-distribution sources.
- `greek_replay` reduced to ~1 % but **still present** (probe source intact).
- New-Greek + 6 forgetting held-out probes built and tokenized (×2 if a control arm runs).
- Decontamination covers replay too.
- Recipe diff vs `bulk_13b.json` is exactly the re-weighting above (no silent source
  swaps).

---

*Rationale trail:* the replay sweep (`curriculum_sweeps_v2`, 4 arms × 13.5 B, completed
2026-06-12) settles ROADMAP **DA3** at **R=0.25 / 20 %** (not the pre-sweep 10–15 %), and
adds the old-Greek 5 %→1 % cut. Grounded in Ibrahim (replay dose by shift; 1 % helps;
compute-equivalent) and the stability-gap paper (early-transient forgetting; matched
replay). Full numbers + figures: `reports/cpt_curriculum_forgetting_learning.html`.
