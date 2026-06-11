# Paper library — Greek CPT of Apertus-8B

PDFs in `pdf/`, extracted text in `txt/`, per-paper notes + citation audit in `notes/`.
Consolidated citation audit: [`CITATION_AUDIT.md`](CITATION_AUDIT.md). Refresh with `bash fetch.sh`.

**20 papers.** Each note verifies the specific claim CURRENT_HYPERPARAMETERS.md cites it for.


## Base model / all
- **[apertus](notes/apertus.md)** (arXiv:2509.14233) — Fully-open 8B and 70B multilingual LLMs (1811 languages, 15T tokens) trained on EU-AI-Act-compliant data with a novel recipe — AdEMAMix optimizer, WSD schedule 

## Optimizer
- **[ademamix](notes/ademamix.md)** (arXiv:2409.03137) — AdEMAMix replaces Adam's single first-moment EMA with a sum of two EMAs — a fast one (beta1≈0.9) plus an alpha-scaled slow one (beta3≈0.9999) — letting the opti

## LR schedule
- **[hagele-wsd-scaling](notes/hagele-wsd-scaling.md)** (arXiv:2405.18392) — EPFL/HuggingFace study showing that a constant-LR-plus-cooldown schedule (the trapezoidal / WSD recipe) matches a well-tuned cosine schedule, scales predictably

## LR schedule (CPT)
- **[gupta-rewarm-cpt](notes/gupta-rewarm-cpt.md)** (arXiv:2308.04014) — A Pythia-410M study (Pile to SlimPajama, ~300B to ~297B tokens, val perplexity only) showing that re-warming then decaying the learning rate is necessary to lea
- **[ibrahim-cpt](notes/ibrahim-cpt.md)** (arXiv:2403.08763) — Empirical study (405M and 10B decoder-only LLMs, 200B+ new tokens) showing that LR re-warming + re-decaying + a small amount of replay of prior data matches ful
- **[practitioner-multimodal-cpt](notes/practitioner-multimodal-cpt.md)** (arXiv:2408.14471) — A large-scale empirical "practitioner's guide" to continual multimodal (CLIP/OpenCLIP) pretraining, introducing the FoMo-in-Flux benchmark (63 datasets) and swe
- **[stability-gap-cpt](notes/stability-gap-cpt.md)** (arXiv:2406.14833) — During continual pre-training, an LLM's target-domain AND general task performance first drops then recovers (a V-shape "stability gap"); training on a small hi

## Warmup
- **[ma-yarats-warmup](notes/ma-yarats-warmup.md)** (arXiv:1910.04209) — Refutes RAdam's variance-based motivation and shows that simple untuned linear warmup of Adam over 2/(1-beta2) iterations matches RAdam across settings, motivat

## Loss
- **[goldfish-loss](notes/goldfish-loss.md)** (arXiv:2406.10209) — A drop-in modification to next-token training that pseudo-randomly excludes a 1/k subset of tokens (via a localized h-gram hash) from the loss, sharply reducing

## New-token init
- **[artetxe-crosslingual-transfer](notes/artetxe-crosslingual-transfer.md)** (arXiv:1910.11856) — Shows that a monolingual transformer (BERT) can be transferred to a new language by freezing the entire transformer body and learning only a fresh token-embeddi
- **[eeve-vocab-expansion](notes/eeve-vocab-expansion.md)** (arXiv:2402.14714) — EEVE expands an English-centric LLM's vocabulary for Korean using subword-based embedding initialization plus a 7-stage parameter-freezing schedule that trains 
- **[jiang-tokenizer-aware-adaptation](notes/jiang-tokenizer-aware-adaptation.md)** (ACL 2026.eacl-long.357) — Extends Artetxe-style embedding relearning (swap tokenizer, retrain only the embedding layer on a frozen transformer body) from old encoder-only models to moder
- **[token-distillation](notes/token-distillation.md)** (arXiv:2505.20133) — A cheap, training-free-at-the-body method that initializes INPUT embeddings for newly added tokens by distilling (MSE-matching) the frozen model's hidden states

## Embedding LR
- **[allam-arabic-cpt](notes/allam-arabic-cpt.md)** (arXiv:2407.15390) — ALLaM adapts Llama-2 (7B/13B/70B) into a strong Arabic+English model via tokenizer/vocabulary expansion plus continued pretraining on 1.2T tokens, achieving SOT
- **[llrd-bert-finetuning](notes/llrd-bert-finetuning.md)** (arXiv:2006.05987) — An empirical study of BERT-Large few-sample fine-tuning instability that pins the main cause on the Adam debiasing omission, shows the top BERT layers are a poo
- **[optimal-embedding-lr](notes/optimal-embedding-lr.md)** (arXiv:2506.15025) — A theory+experiments paper arguing that under Adam, large vocabulary breaks muP's prediction of equal-constant embedding LR: the optimal embedding-to-hidden LR 
- **[tokenization-bottleneck](notes/tokenization-bottleneck.md)** (arXiv:2511.14365) — A Thoughtworks NeurIPS-2025 workshop paper showing that extending Llama3-8B's vocabulary with ~17.8k chemistry (SMILES) and text tokens before continued pretrai
- **[ulmfit](notes/ulmfit.md)** (arXiv:1801.06146) — Howard & Ruder (ACL 2018, arXiv:1801.06146) introduce ULMFiT, a transfer-learning recipe for NLP that pretrains an AWD-LSTM language model and fine-tunes it for

## Vocabulary cutoff
- **[magikarp-undertrained-tokens](notes/magikarp-undertrained-tokens.md)** (arXiv:2405.05417) — Embedding-weight indicators (output-embedding cosine distance to a mean unused-token reference, or input-embedding norm for untied models) plus repetitive-promp
- **[tao-vocab-scaling](notes/tao-vocab-scaling.md)** (arXiv:2407.13623) — A compute-optimal scaling-law study showing that the optimal vocabulary size grows with model/compute budget (so most LLMs under-size their vocab) — derived for