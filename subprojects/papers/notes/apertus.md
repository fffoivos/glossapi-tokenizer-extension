# Apertus: Democratizing Open and Compliant LLMs for Global Language Environments (Swiss AI / arXiv:2509.14233)

**Source:** arXiv:2509.14233  |  **Topic:** Base model / all  |  `txt/apertus.txt`, `pdf/apertus.pdf`

**TL;DR.** Fully-open 8B and 70B multilingual LLMs (1811 languages, 15T tokens) trained on EU-AI-Act-compliant data with a novel recipe — AdEMAMix optimizer, WSD schedule with 1-sqrt cooldown, Goldfish loss, xIELU, QK-Norm, cross-document attention masking — documenting every pretraining and long-context hyperparameter.

## Key load-bearing facts
- 8B main-pretraining recipe (Table 2 + Appendix Table C.4): AdEMAMix optimizer, seq len 4096, peak/max LR 1.1e-4, 15T tokens, 2.6M steps. 70B uses 1.0e-5 peak LR.
- WSD (Warmup-Stable-Decay) schedule: warmup starts from 0.1x peak LR, linearly ramped over 16.78B tokens (Table C.4 'LR Warmup Duration 16.78BT'); long stable plateau at constant peak LR; cooldown with negative-square-root (1-sqrt) shape. Final LR = 0.1x peak (1.1e-5 for 8B).
- WSD is chosen explicitly to enable continual training WITHOUT re-warming: '...allows us to continue pretraining without rewarming the learning rate' and they extended planned 9T->15T tokens 'thanks to no schedule change being required' (lines 942-944). The 5 data-mixture stage transitions reset only the dataloader state/seed (lines 2042-2044); no LR re-warm. Figure 3 caption notes mixture changes cause discontinuous loss jumps but training stayed stable.
- Goldfish loss (memorization mitigation): k=50 (2% masking rate), h=50 (50-token hash context window) for both prose (lines 923-924) and Table C.4 (Goldfish k 50, Goldfish h 50).
- AdEMAMix pretraining betas/alpha (Table C.4): Adam (beta1,beta2)=(0.9,0.999), AdEMAMix alpha=8, AdEMAMix beta3=0.9999. alpha and beta3 are warmed over the first 100,000 steps then held constant. CAUTION: post-training (SFT/QRPO) uses LOWER beta3=0.99 (lines 2818-2821, 2954-2956) — do not confuse with pretraining.
- RoPE geometry: main pretraining theta=500,000 with max position embeddings 4096; after the 64k long-context expansion theta=12,000,000 (Table C.4). NTK-aware RoPE scaling, factor 8, LLaMA-3 style.
- Batch-size doubling mid-run: 8B starts at 1024 seqs (4.2M tokens/batch), doubled to 8.4M after 8T tokens (nodes also doubled), LR kept unchanged (lines 947-950). 70B: 8.4M->16.8M after 4.4T.
- Other knobs (Table C.4): weight decay 0.1, gradient clipping 0.1 (AdEMAMix is clip-sensitive; clipping fires nearly every step), init std 0.008944, no bias terms, pre-norm + RMSNorm, QK-Norm, xIELU activation, cross-document attention masking, tokenizer = Mistral-Nemo-style 131k vocab.
- Cooldown coincides with a data-mixture switch to highest-quality sources at 13.5T tokens (Stage 5); cooldown decays to the 0.1x-peak final LR. On 1.5B ablations, cooldowns decayed to zero over 100B tokens with the 1-sqrt schedule.
- Long-context extension is a SEPARATE staged phase (8k->16k->32k->64k), each stage re-warming LR over 1.2B tokens, starting LR at the 0.1x-peak final cooldown value, RoPE theta increased per stage up to 12M, GBS held at 8M (8B)/16M (70B). This is the only place fresh warmups recur.
- Retrospective caveat the authors flag: 70B showed almost no benchmark/loss jump at cooldown onset, unlike 8B; they hypothesize the 70B peak LR (1.0e-5) was set too low and the model had not converged on the Phase-4 mixture. They did not derive proper LR scaling rules (tight schedule).
- Architecture (8B): 32 layers, dim 4096, MLP dim 21504, 32 Q / 8 KV heads (GQA), xIELU, context length 65536. Megatron-LM based; code + WandB logs are public.

## Citation audit — what CURRENT_HYPERPARAMETERS.md cites this for

**Claim:** 8B peak LR = 1.1e-4; WSD schedule with 1-sqrt cooldown; warmup from 0.1x peak over 16.8B tokens; final LR = 0.1x peak.
**Verdict:** `CONFIRMED`
**Evidence:** Table 2 (line 891) lists 8B Max LR = 1.1e-4. Lines 938-945: 'We employ the Warmup-Stable-Decay (WSD) learning rate (LR) schedule... Our LR warmup for both models starts from 0.1 the peak LR and is linearly increased for 16.8B tokens.' Lines 955-956: 'For the final learning rate annealing, we opt for a negative square root shape (also denoted 1-sqrt).' Lines 959-960: 'The final learning rate is set to a factor of 0.1 of the respective maximum.' Appendix Table C.4: 'LR Decay Style WSD', 'LR WSD Decay Style 1-sqrt', 'LR Warmup Duration 16.78BT' (lines 8835-8843).

**Claim:** Pretraining used ONE warmup only (no re-warm at data-mixture/curriculum stage transitions); constant peak LR between warmup and cooldown.
**Verdict:** `CONFIRMED`
**Evidence:** Lines 942-944: WSD '...allows us to continue pretraining without rewarming the learning rate in the future. In fact, we extended the initial planned training phase of 9T tokens thanks to no schedule change being required.' Lines 2037-2044: across the 5 data stages, transitions reset only the dataloader sampler state and modified the dataset seed (Stages 3,4,5) for reshuffling — no LR action. Figure 3 caption (lines 1360-1364): batch-size doubling and 'changes in data mixtures... result in discontinuous loss jumps' yet training stayed stable; 'Phase 5 coincides with the learning rate cooldown' (i.e., the only schedule change is the single cooldown). The single warmup is the 16.78B-token ramp at the very start. The structure is warmup -> long constant-peak stable plateau across Stages 1-4 -> 1-sqrt cooldown in Stage 5.

**Claim:** Goldfish loss k=h=50 was used.
**Verdict:** `CONFIRMED`
**Evidence:** Lines 922-925: 'we identify an optimal configuration of a 2% token masking rate (k = 50) and a 50-token context window for hashing (h = 50).' Appendix Table C.4: 'Goldfish k 50', 'Goldfish h 50' (lines 8845-8851).

**Claim:** Pretraining AdEMAMix used beta1=0.9, beta2=0.999, beta3=0.9999, alpha=8.
**Verdict:** `CONFIRMED`
**Evidence:** Appendix Table C.4 (lines 8817-8827): 'Adam beta (0.9, 0.999)', 'AdEMAMix alpha 8', 'AdEMAMix beta3 0.9999'. Prose lines 8784-8786: 'we increase beta2 to 0.999 and beta3 to 0.9999 during pretraining.' alpha and beta3 are warmed over 100,000 steps then fixed (lines 8782, 8829-8831).

**Claim:** Main-pretraining geometry was rope_theta=500000, seq=4096 (the 12M/65536 is the later long-context extension only).
**Verdict:** `CONFIRMED`
**Evidence:** Lines 797-798: 'We use RoPE embeddings... with a base Theta = 500,000 during pretraining, which we extend in the long-context phase.' Appendix Table C.4 (lines 8799-8807): 'RoPE theta during main pretraining 500,000', 'Max Position Embeddings during main pretraining 4096', 'RoPE theta after 64k context expansion 12,000,000', 'Rope Scaling Factor (NTK) 8'. The 65536 context length (Table line 790) and 12M theta are the long-context expansion targets, set per-stage in Table 5 (RoPE theta column rising 1M->2M->4M->12M as context goes 8k->16k->32k->64k).

**Claim:** Batch doubling 4.2M -> 8.4M tokens at 8T, LR unchanged.
**Verdict:** `CONFIRMED`
**Evidence:** Lines 946-950: 'an initial batch size of 1024 (4.2M tokens)... for the 8B... After 8T tokens for the 8B model and 4.4T for the 70B, we intentionally doubled both the number of nodes and the batch size at this stage, while keeping the learning rate unchanged.' Table 2 lists 8B 'Batch Size (Tokens) 4.2M -> 8.4M'.

## How we use it
Authoritative source recipe for our Apertus-8B Greek CPT. Use it as the default-inherited base config and a re-warm decision anchor. (1) Inheriting the base run: when continued-pretraining the 8B-2509 checkpoint, the model already left main pretraining at the cooldown floor LR (~1.1e-5 = 0.1x the 1.1e-4 peak). For CPT we are effectively restarting a schedule, so we must decide our own peak LR and warmup — do not assume 1.1e-4 carries over; the checkpoint sits at the post-cooldown state. (2) Re-warm question for our mixture change: the paper is direct evidence that the Apertus team did NOT re-warm at data-mixture transitions within main pretraining (only the dataloader seed/state was reset; lines 2042-2044, Figure 3). But our Greek CPT introduces a genuinely new mixture AND new tokens — closer to their long-context phase, where they DID re-warm (1.2B tokens/stage from the 0.1x-peak floor). The honest read: their precedent supports a short re-warm from the cooldown-floor LR for a new phase, not a from-scratch full 16.78B ramp. (3) Optimizer: if we mirror AdEMAMix, use pretraining betas (0.9/0.999/0.9999, alpha=8) only for a long run; for a SHORT CPT the paper's own finding favors the post-training setting beta3=0.99 — cite lines 2818-2821/8788-8789 in CURRENT_HYPERPARAMETERS.md rather than blindly copying 0.9999. (4) Geometry: keep rope_theta=500000 / seq=4096 for standard CPT; only adopt theta=12M/seq=65536 if we explicitly run long-context extension (separate phase). (5) Carry over weight decay 0.1, grad clip 0.1 (note AdEMAMix clip-sensitivity), 1-sqrt cooldown, and Goldfish k=h=50 if we want memorization parity. Flag the 70B 'peak LR maybe too low / no scaling rules derived' caveat (lines 1392-1398) as a known open risk we should not inherit uncritically when picking our 8B CPT peak LR.
