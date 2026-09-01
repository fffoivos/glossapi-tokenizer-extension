# Greek post-training data survey — SFT (and outlook to preference/thinking stages)

Date: 2026-08-23. Status: SURVEY / RECOMMENDATION (no decisions locked, no builds started).
Scope: instruction-tuning data for post-training `fffoivos/apertus-8b-greek-cpt`. Produced from three
parallel research agents (Apertus recipe / Greek landscape / translation candidates), each verifying
dataset ids, row counts, and licenses against the HF hub on 2026-08-23. Numbering note: 09_* lives on
the results worktree branch; this subproject takes 10 to avoid collision.

---

## 0. Headline findings

1. **Greek SFT is greenfield.** No large, quality, public Greek SFT corpus exists. ILSP (Meltemi,
   Llama-Krikri) released **zero** of their SFT/preference data. The only reusable quality Greek
   material anywhere: **623** human-written Aya rows, **~619** real Greek WildChat conversations,
   **501** human-translated MPEP prompts, **1,745** native GPT-4 self-instruct rows
   (CausalLM/GPT-4-Self-Instruct-Greek, CC-BY-4.0), and ILSP's small native QA sets (mostly NC).
   Everything else is MT-Alpaca or mass-templated (Aya collection 4.16M Greek rows = NLLB MT;
   xP3x 7.2M = templated).
2. **The Krikri paper (arXiv:2505.13772) is the proven blueprint** for exactly this program:
   translate a modern English backbone (Tulu-3, SmolTalk, Magpie, UltraFeedback) with post-editing,
   **regenerate responses natively** (Gemma-2-27B-IT) instead of shipping translated responses,
   Magpie-synthesize natively in Greek, and ground synthetic QA in Greek corpora. Stage-1 SFT
   856,946 pairs (43% Greek), Stage-2 638,408; DPO 92,394 triplets scored with
   Skywork-Reward-Gemma-2-27B-v0.2 (works on Greek). Their data is unreleased; the method is fully described.
3. **Apertus's own SFT mixture is public and reusable as the non-Greek backbone**:
   `swiss-ai/apertus-sft-mixture`, **3,942,208 rows, ODC-BY, ungated** — but it contains only
   **~3,215 Greek rows (0.082%)**, mostly incidental. Nothing to inherit for Greek; everything to
   inherit for task scaffolding, identity, safety, function-calling, and the chat template.
4. **Field consensus on translation** (Krikri, SmolTalk2-multilingual, Qwen2.5, Bactrian-X,
   naturalness studies): **translate prompts (with locale adaptation), generate responses natively**.
   Literally-translated responses measurably imprint translationese.
5. **No public Greek preference data exists at all** (largest find: 30 DPO pairs). RLHF stage will be
   built on-policy + reward model, not translated.

---

## 1. What Apertus itself did (swiss-ai, arXiv:2509.14233 §4)

Two stages: SFT on a curated mixture, then **QRPO** (quantile reward policy optimization,
arXiv:2507.08068, length-normalized variant, β_KL=5) — QRPO ≈ DPO at 8B scale.

**Published**: `swiss-ai/apertus-sft-mixture` (3,942,208 rows, ODC-BY; treat the shipped artifact as
ground truth over the paper's Table 12, which over-counts by ~242k). Top sources: smoltalk2
1,273,661 (only `no_think` subsets); multilingual WikiQA 881,422; The-Tome 544,951; Llama-Nemotron
446,536; tulu-3-sft-olmo-2-mixture-0225 397,114; EuroBlocks-Synthetic 162,336; function-calling
(glaive 112,537 + xlam 59,448 + apigen 5,000); Romansh 46,170; identity hardcoded 114 + charter-qa 88.
**The alignment/preference mixture was never released**; its prompt source is public
(`allenai/olmo-2-0325-32b-preference-mix`, minus Flan v2 / No Robots / robots.txt) and the rebuild
pipeline is at github.com/swiss-ai/posttraining.

**8B SFT hyperparameters** (verified in repro scripts, `reproducibility-scripts/sft/final/ademamix.py`):
1 epoch, GBS 512, LR 5e-6 linear decay, warmup ratio 0.03, seq 4096, **no packing**,
prompt_loss_weight 0.0 (loss on completions only), AdEMAMix β1 .9 / β2 .999 / β3 .99 / α 8,
weight_decay 0, bf16, TRL + DeepSpeed ZeRO-2, 256×GH200 on Clariden (we need far less).
SFT ran on the long-context (64k) checkpoint at seq 4096. Instruct EOS = `<|assistant_end|>` (id 68),
special tokens ids 61–72, template in `chat_template.jinja` of Apertus-8B-Instruct-2509; the
developer-role format spec is github.com/swiss-ai/apertus-format. **Deliberation: enabled/disabled**
in the developer role is their thinking hook — relevant to our "implement thinking later".

Compliance choices worth copying: license filter (they dropped NC/SA data — cost them 0.443→0.417 on
their eval avg), 8-gram + Ratcliff-Obershelp decontamination vs benchmark prompts (catches
translated contamination — directly relevant since our evals are Greek), branding scrub.

## 2. Greek landscape (what exists, verified)

### 2.1 ILSP recipes (data unreleased)
- **Meltemi-7B-Instruct-v1**: ~100k MT'd instructions (Open-Platypus permissive subsets,
  Evol-Instruct, Capybara) + hand-crafted Greek safety multi-turn.
- **Meltemi-v1.5** (arXiv:2407.20743): ORPO on 97,072 triplets (89,730 Greek) from "12 preference
  datasets" (unnamed), MT system undisclosed.
- **Llama-Krikri-8B-Instruct** (arXiv:2505.13772): see headline finding 2. Also: synthetic QA/dialogues
  grounded in Greek Wikipedia, EUR-LEX, school books, Kallipos; ELRC-SHARE parallel data for
  translation skill; ~43% Greek per SFT stage (the rest English — retention matters).
- ILSP publishes only evals/resources (49 datasets: mmlu_greek, ifeval_greek 541, mt-bench-greek 80,
  greek_civics_qa 407, medical_mcqa_greek 2,034, mcqa_greek_asep 1,200 CC-BY-4.0, …). Seeds/evals,
  not corpus; mostly CC-BY-NC-SA.

### 2.2 Reusable Greek material (the "free gold" pile, ~3.5k rows + prompts)
| Source | Rows | Provenance | License |
|---|---|---|---|
| CohereLabs/aya_dataset (`ell`) | 623 | human-written | Apache-2.0 |
| allenai/WildChat-1M Greek convs | 619 | real users × GPT-3.5/4 | ODC-BY |
| data-is-better-together/MPEP_GREEK | 501 prompts | human-translated curated prompts | none stated |
| CausalLM/GPT-4-Self-Instruct-Greek | 1,745 | native GPT-4 synthetic | CC-BY-4.0 |
| ilsp native QA sets | ~4k total | native | mostly NC(-SA) |
| Petrouil/gsm8k-greek | 5,480 | MT (undocumented) | MIT |

MT-Alpaca clones (iamshnoo, saillab, gsar78, gsoloupis; 45–62k each) and Aya-collection/xP3x mass
data: low quality, use only as filtered bulk if ever. oasst1/oasst2 contain **zero** Greek.
EuroBlocks Greek = 572–582 rows (0.05%). Teuken/Tower/Salamandra: no Greek instruct data.

### 2.3 Greek preference data
None public (30 pairs is the record). Krikri's bootstrap: contrast translated vs natively-regenerated
responses to form pairs; score with Skywork RM; ~92k triplets.

## 3. Translation candidates (ranked, licenses verified)

| # | Dataset | Rows | Provenance | License | Verdict |
|---|---|---|---|---|---|
| 1 | HuggingFaceH4/no_robots | 10,000 | human | **CC-BY-NC-4.0** | Quality anchor; fully one-person-reviewable; derivative must stay NC |
| 2 | OpenAssistant/oasst_top1_2023-08-25 | 13,637 | human | Apache-2.0 | Clean-license complement; triage ~20–30% first |
| 3 | allenai/coconot | 10,983 | GPT-4 synth | ODC-BY | Refusal behavior; translates near-losslessly; high value/row |
| 4 | databricks-dolly-15k | 15,011 | human | CC-BY-SA-3.0 | Only commercial-clean human set; plain register; US-trivia tail ~5–10% |
| 5 | LDJnr/Capybara | 16,006 | GPT-4-era synth | Apache-2.0 | Deep multi-turn reasoning prose; translate whole conversations |
| 6 | GAIR/lima | ~1,300 | human | CC-BY-NC-SA, gated | Long-form style; NC-SA taint; optional |
| 7 | smoltalk targeted subsets (rewrite/constraints/summarize/everyday/systemchat) | 30–60k sample | Llama-405B synth | Apache-2.0 | Adapt rather than translate rewrites; systemchats valuable |
| 8 | tulu-3-sft-personas-instruction-following | 29,980 | GPT-4o synth | ODC-BY | Verifiable constraints — machine-checkable post-translation |

Skip for translation: FLAN v2 (English-linguistics-bound — the canonical breaks-under-translation
set), ultrachat_200k / OpenHermes-2.5 (dated distilled slop; OpenHermes has **no license**), Magpie
dumps (don't translate — **run Magpie natively in Greek**, as Krikri did), WildJailbreak (adversarial
phrasing is language-specific — regenerate natively).

**no_robots detail**: 10 categories — Generation 4,560, Open QA 1,240, Brainstorm 1,120, Chat 850,
Rewrite 660, Summarize 420, Coding 350, Classify 350, Closed QA 260, Extract 190. ~10–15% needs
adaptation not translation (wordplay/poems, English-grammar asks, US culture; Rewrite tasks should be
re-executed on Greek's own formality axis). No Greek version exists on the hub. Precedents: German
(DeepL, deliberate informal-register policy — we must pick an ενικός/πληθυντικός policy up front and
encode it in the translation prompt), Dutch (regenerated with GPT-4 instead of translated; lost ~9%
to content filters — budget for that).

## 4. Recommended shape of the corpus (proposal, not locked)

Four streams, mirroring Krikri but sized to one reviewer:

- **A. Greek human-reviewed core (~25–40k)**: no_robots-el (translated prompts + natively generated
  responses, human-reviewed end-to-end) + oasst_top1-el (triaged) + coconot-el + free-gold pile
  (§2.2) + dolly-el if commercial-clean release matters. This is where all deep human review goes.
- **B. Greek native-synthetic bulk (scalable)**: Magpie natively in Greek (argilla/magpie-ultra
  pipeline as template) + **grounded QA/dialogues over our own GlossAPI corpus** (51.8M rows —
  our unique asset; Krikri validated the grounded-QA move with Wikipedia/EUR-LEX/school books) +
  personas-IF-el with machine-checked constraints. RM + langid + format filters; spot-review only.
- **C. English retention backbone (~40–60% of mix)**: subsample `swiss-ai/apertus-sft-mixture`
  as-is (ODC-BY, matched to base model, includes identity/safety/function-calling/systemchats).
  Krikri kept ~57% English; Apertus's compliance work comes for free.
- **D. Identity + register**: rewrite the 114 apertus-hardcoded identity rows for our model, in
  Greek + English; fixed register policy; Greek system prompts (Meltemi-v1.5 precedent: RAG/CoT/
  math/code system-prompt roles).

Decontamination gate (non-negotiable given our history): 8-gram + overlap scrub of ALL streams vs
GreekMMLU + every ilsp eval + our greek-reality-bench, before any training.

**Later stages (scoped, not planned)**: preference = translated/collected Greek prompts → on-policy
generations from our SFT checkpoint + strong models → RM scoring (Skywork line verified workable on
Greek) → QRPO (Apertus repro pipeline) or LN-DPO (Krikri). Thinking = translate problems only,
generate traces natively with an open reasoning teacher; s1K-style ~1k reviewed Greek trace set is
the feasible reviewed option; Apertus format already reserves the Deliberation/inner-sections hooks.

## 5. Checkpoint choice for SFT init

Candidates named by owner: lowest-loss checkpoint vs averaged-middle (best GreekMMLU; peak single
cp iter9536 = 56.81%). Recommendation: default to the **benchmark-best averaged checkpoint** (loss is
the weaker proxy for downstream ability), and treat it as an A/B: SFT at this scale is cheap enough
(~1 epoch over a few hundred k rows on 1–4 H100s) to run both inits on the frozen SFT mix and pick by
Greek instruct evals (ifeval_greek, mt-bench-greek, GreekMMLU-after-SFT). Note Apertus SFT'd their
long-context checkpoint at seq 4096; our CPT base descends from Apertus-8B-2509 (post-extension), so
the same seq-4096 recipe applies unchanged.

## 6. Open decisions for the owner

1. License posture: is CC-BY-NC acceptable for the released Greek SFT set (no_robots anchor), or do
   we want a clean commercial release (drop no_robots → dolly+oasst anchor)? Apertus chose clean.
2. Register policy for Greek responses (πληθυντικός ευγενείας vs ενικός; one policy, encoded in
   generation prompts).
3. Translator/generator model choice (frontier API vs Krikri/Gemma-class open model) — cost + the
   Dutch precedent's ~9% content-filter loss applies to API routes.
4. Stream B scale and whether grounded-QA over GlossAPI is in scope for v1.
5. Checkpoint A/B (§5) — run both or commit to averaged.

## 7. Caveats / not verified

LIMA card gated (license via secondary sources); OpenHermes/Skywork-80K/magpie-ultra/EuroBlocks have
no license tags; aya_dataset Greek test-split count unverified (train=623 solid); lmsys-chat-1m Greek
count gated-unverified; Apertus Greek row count uses a ≥4-consecutive-Greek-chars heuristic (upper
bound); Meltemi's "12 preference datasets" and both ILSP MT systems undisclosed; Apertus alignment
mixture composition known only from the paper.

Full agent reports (with per-claim source URLs) are preserved in the session transcripts; key
sources: arXiv 2509.14233, 2507.08068, 2505.13772, 2407.20743; hub cards linked by id throughout.
