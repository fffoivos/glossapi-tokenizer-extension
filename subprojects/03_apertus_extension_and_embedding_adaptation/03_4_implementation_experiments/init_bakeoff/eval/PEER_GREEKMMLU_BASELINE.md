# GreekMMLU comparative baseline — public base models (execution-agent handoff)

**Date:** 2026-06-16 · **Prepared by:** planning agent · **Runs on:** Clariden (execution agent) · **Status:** ready to submit · **Models: LOCKED (4)**

## 0 · Goal
Benchmark **4 public base models** on **GreekMMLU** as a comparative baseline for our Apertus-8B Greek CPT
(Vanilla + TD). Deliverable: one GreekMMLU **overall accuracy per model**, directly comparable to our
existing Vanilla/TD GreekMMLU points → a single comparison table + bar chart (planning builds the
presentation once the numbers are in).

## 1 · Models — LOCKED (all BASE, not instruct → matches our base CPT)
| label | hf_id | size | family | gated | role |
|---|---|---:|---|---|---|
| **Krikri-8B-Base** | `ilsp/Llama-Krikri-8B-Base` | 8B | Llama-3.1 + 56.7B Greek CPT | **no** | Greek SoTA peer |
| **Gemma-2-9B** | `google/gemma-2-9b` | 9B | Gemma-2 | **yes** | frontier multilingual base |
| **Qwen3.5-9B-Base** | `Qwen/Qwen3.5-9B-Base` | 9B | Qwen3.5 (latest gen, vision enc.) | **no** (Apache-2.0) | latest frontier base, size-matched |
| **Gemma-3-12B** | `google/gemma-3-12b-pt` | 12B | Gemma-3 (newer, multimodal) | **yes** | newer-gen ceiling — **larger, not size-matched** |

Optional (commented in the manifest): `meta-llama/Llama-3.1-8B` (gated; Krikri's own base → would show the
Greek-CPT lift). Home arms already evaluated: **Apertus-8B Vanilla-CPT**, **TD-CPT** (reuse their points).

## 2 · Method (established tooling — do not reinvent)
- **Runner:** `run_native_greek_mcq_eval.sbatch` → `run_native_greek_mcq_eval.py`, benchmark `greekmmlu`
  (= `dascim/GreekMMLU`, registry `native_greek_benchmark_registry.json`), **full split**, causal-LM
  **log-likelihood MCQ** scoring. **Tokenizer-agnostic** (accuracy, not loss) → 128k Llama/Krikri-extended,
  256k Gemma, Qwen vocab all directly comparable. 1 GPU / model, `normal`, ≤10 h.
- **Driver:** `submit_peer_models_greekmmlu.sh` reads `peer_models_greekmmlu.tsv`, submits one job per
  active (uncommented) model — 4 jobs.

## 3 · Prerequisites
1. **Sync** to the Clariden repo (the runner + registry are already there):
   `eval/peer_models_greekmmlu.tsv`, `eval/submit_peer_models_greekmmlu.sh`, this doc.
2. **`export HF_TOKEN=<token>`** with the **Gemma licenses accepted** for `gemma-2-9b` and `gemma-3-12b-pt`
   (gated). Krikri + Qwen3.5 are open (Apache-2.0). The token propagates via the sbatch `--export=ALL`.
3. The runner pip-installs the latest `transformers` into its venv (`PY_ENV`) — required for Gemma-3 / Qwen3.5.

## 4 · Run
```bash
cd .../init_bakeoff/eval
DRY_RUN=1 bash submit_peer_models_greekmmlu.sh        # preview the 4 sbatch lines, submit nothing
bash submit_peer_models_greekmmlu.sh                  # submit 1 job per model
# (override OUT_ROOT=... to pin the output dir; default is runs/eval/peer_greekmmlu_<UTC stamp>)
```
**Smoke the two vision-encoder models first** (Gemma-3-12B, Qwen3.5-9B-Base) — see §6.

## 5 · Outputs → what to hand back
- Per model: `OUT_ROOT/<label>/*_native_mcq_summary.csv` with rows `benchmark=greekmmlu, subject, n, accuracy, correct`.
- **Overall GreekMMLU acc** = `sum(correct) / sum(n)` over the per-subject rows, **skipping the `__all__` row**
  (identical recipe to `analysis/collect_greekmmlu.py`).
- Report back: the `OUT_ROOT` path + a **4-row** table `model | overall greekmmlu acc | N` (N confirms full split).

## 6 · Gotchas / caveats — READ before the full run
- **Gemma-3-12b-pt AND Qwen3.5-9B-Base both ship with a vision encoder** → `AutoModelForCausalLM.from_pretrained`
  may not load them as a plain causal LM (may need the text tower / `Gemma3ForCausalLM` / `Qwen3` text class /
  very recent transformers). **Run a `--sample-size 50` smoke on each first**; if it errors on load, load the
  text model or drop that arm and report it. (Krikri + Gemma-2-9b are plain text models — no issue.)
- **Contamination caveat (matters for how we frame the baseline).** GreekMMLU is *native Greek exams*; the
  public models may have trained on web copies of those exams → possible **train–test contamination we cannot
  control**. Our Vanilla/TD CPT were **decontaminated against GreekMMLU**. So a high public score may be partly
  memorization — this is a *context* baseline, not a training-controlled like-for-like. Note it with the numbers.
- **Size:** Gemma-3-12B is 12B — a newer-gen ceiling, not size-matched to the 8–9B peers / our 8B. Qwen3.5-9B
  and Gemma-2-9B are size-matched (9B).
- **Base, not instruct** — all base, so no chat-format MCQ advantage; matches our base CPT.

## 7 · Acceptance
- Each of the 4 models produces a `greekmmlu` summary with **N ≈ full split** (~16–22k) and an overall acc,
  no load errors (or an explicit "X failed to load as causal LM" note for a vision-encoder arm).
- Hand back the 4 overall accuracies + `OUT_ROOT`; planning aggregates with Vanilla/TD and builds the
  comparison presentation.

## 8 · Provenance / IDs
- Krikri-8B-Base: https://huggingface.co/ilsp/Llama-Krikri-8B-Base
- Gemma-2-9B: https://huggingface.co/google/gemma-2-9b · Gemma-3-12B: https://huggingface.co/google/gemma-3-12b-pt
- Qwen3.5-9B-Base: https://huggingface.co/Qwen/Qwen3.5-9B-Base
- GreekMMLU: dascim/GreekMMLU (arXiv 2602.05150). Eval code: `init_bakeoff/eval/run_native_greek_mcq_eval.{py,sbatch}`.
