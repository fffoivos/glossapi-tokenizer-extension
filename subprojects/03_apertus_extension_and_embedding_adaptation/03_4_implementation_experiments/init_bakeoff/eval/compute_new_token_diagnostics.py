"""New-token integration diagnostic suite (cpt_plan v0.7 §5.3).

Seven diagnostics over the 17,408 new modern-Greek IDs in [131072, 148480),
read at every bakeoff checkpoint to detect failure modes the bakeoff exists
to compare arms on:

  D1. Rank of correct new token in next-token logits   → "new token invisible"
  D2. Aggregate prob mass on new Greek tokens          → under/over-emitted
  D3. New-token entropy by register                    → register-specific collapse
  D4. Top-k mix (new vs base) at new-token positions   → model still prefers old subpieces
  D5. Greedy-gen new-token utilization rate            → new rows exist but behaviorally dead
  D6. Embedding L2-norm distribution: new vs existing  → degenerate-subspace collapse
  D7. Cosine-similarity / effective-rank of new rows   → same-direction collapse

[Cite: cpt_plan.md v0.7 §5.3]

D6/D7 are embedding-only (no forward pass). D1-D4 need teacher-forcing
forward. D5 needs greedy autoregressive generation. Heaviest cost is D1-D4
forward on the eval set; ~5-15 minutes per checkpoint on one GH200.

Usage:
    python3 compute_new_token_diagnostics.py \\
        --model-path /path/to/checkpoint \\
        --eval-jsonl /path/to/held_out_greek.jsonl \\
        --output-json /path/to/diagnostics.json \\
        --new-id-range 131072 148480 \\
        --base-vocab-size 131072

JSONL format: one doc per line, fields {text, source?, register?, doc_id?}.

--embedding-only skips the forward-pass + greedy-gen diagnostics (runs only
D6, D7). Useful for local testing without GPU.
--skip-greedy skips just D5 (greedy gen is the most expensive of the
forward-pass diagnostics).
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path


def _doc_iter(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _embedding_diagnostics(E, U, new_id_range, base_vocab_size, sample_cos=500, seed=20_260_520):
    """D6 + D7: embedding-space diagnostics (no forward pass).

    E, U: torch tensors [vocab, hidden].
    Returns dict with norm stats + cosine-similarity / effective-rank stats.
    """
    import torch

    new_start, new_end = new_id_range
    vocab_size = int(E.shape[0])
    clipped_new_end = min(new_end, vocab_size)
    N_requested = new_end - new_start
    N_new = max(0, clipped_new_end - new_start)
    out = {
        "new_id_range": [new_start, new_end],
        "available_new_id_range": [new_start, clipped_new_end],
        "n_new_requested": N_requested,
        "n_new": N_new,
        "base_vocab_size": base_vocab_size,
        "vocab_size": vocab_size,
        "applicable": N_new > 0,
    }

    # D6: L2-norm distribution
    for name, M in [("E", E), ("U", U)]:
        M = M.float()
        existing = M[:base_vocab_size]
        new = M[new_start:clipped_new_end]
        existing_norms = torch.linalg.norm(existing, dim=1)
        new_norms = torch.linalg.norm(new, dim=1)
        norm_report = {
            "existing_mean": float(existing_norms.mean()),
            "existing_std":  float(existing_norms.std()),
            "existing_p50":  float(existing_norms.median()),
            "existing_p95":  float(existing_norms.quantile(0.95)),
            "new_mean":      None,
            "new_std":       None,
            "new_p50":       None,
            "new_p5":        None,
            "new_p95":       None,
            "new_to_existing_mean_ratio": None,
        }
        if new_norms.numel():
            norm_report.update({
                "new_mean":      float(new_norms.mean()),
                "new_std":       float(new_norms.std()),
                "new_p50":       float(new_norms.median()),
                "new_p5":        float(new_norms.quantile(0.05)),
                "new_p95":       float(new_norms.quantile(0.95)),
                # ratio test: if new << existing, new tokens "starve"; if >>, dominate softmax
                "new_to_existing_mean_ratio": float(new_norms.mean() / existing_norms.mean()),
            })
        out[f"{name}_norm"] = norm_report

    if N_new == 0:
        out["new_E_cos"] = {
            "n_sampled": 0,
            "mean_off_diag": None,
            "std_off_diag": None,
            "p95_off_diag": None,
            "p99_off_diag": None,
        }
        out["new_E_effective_rank"] = {
            "participation_ratio": None,
            "rank_at_99pct_var": None,
            "n_singular_values": 0,
            "sigma_max": None,
            "sigma_min": None,
        }
        return out

    # D7: cosine similarity + effective rank of new rows (E only; U is symmetric)
    # Subsample to keep O(sample_cos^2) tractable.
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(N_new, generator=g)[:min(sample_cos, N_new)]
    new_E = E[new_start + idx].float()
    new_E_norm = new_E / (torch.linalg.norm(new_E, dim=1, keepdim=True) + 1e-12)
    cos = new_E_norm @ new_E_norm.T  # [s, s]
    # off-diagonal stats
    n = cos.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    cos_off = cos[mask]
    out["new_E_cos"] = {
        "n_sampled":       n,
        "mean_off_diag":   float(cos_off.mean()),
        "std_off_diag":    float(cos_off.std()),
        "p95_off_diag":    float(cos_off.quantile(0.95)),
        "p99_off_diag":    float(cos_off.quantile(0.99)),
    }
    # Effective rank: 1) participation ratio of singular values:
    # eff_rank = (Σσ)² / Σσ²  (Roy & Vetterli participation ratio)
    s = torch.linalg.svdvals(new_E)
    out["new_E_effective_rank"] = {
        "participation_ratio": float((s.sum() ** 2) / (s.pow(2).sum() + 1e-12)),
        "rank_at_99pct_var":   int((s.pow(2).cumsum(dim=0) / s.pow(2).sum() < 0.99).sum().item()) + 1,
        "n_singular_values":   int(s.numel()),
        "sigma_max":           float(s.max()),
        "sigma_min":           float(s.min()),
    }
    return out


def _fmt_optional_float(value, digits=3):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _forward_diagnostics(model, tokenizer, eval_jsonl, new_id_range, max_context, max_docs, device):
    """D1 + D2 + D3 + D4: forward-pass diagnostics over the eval set."""
    import torch

    new_start, new_end = new_id_range
    n_pos_total = 0
    n_pos_new_target = 0
    # D1: rank of correct new token at new-target positions
    rank_buckets = {"top1": 0, "top5": 0, "top10": 0, "top50": 0}
    rank_sum = 0
    # D4: at new-target positions, what's the top-1 prediction's vocab range?
    top1_new = 0
    top1_base = 0
    # D2: aggregate prob mass on new IDs (average over all positions)
    prob_mass_new_total = 0.0
    # D3: per-register aggregators
    register_stats = defaultdict(lambda: {"n_pos": 0, "n_new_target": 0, "prob_mass_new": 0.0,
                                          "entropy_sum": 0.0, "rank_sum": 0})

    t0 = time.time()
    n_seen = 0
    for row in _doc_iter(eval_jsonl):
        if max_docs and n_seen >= max_docs:
            break
        n_seen += 1
        register = row.get("register", "unknown")
        ids = tokenizer.encode(row["text"], add_special_tokens=False)
        if len(ids) < 2:
            continue
        if max_context and len(ids) > max_context:
            ids = ids[:max_context]
        n_pos = len(ids) - 1
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits[0]  # [seq, vocab]
        # Predictions for positions 0..n-2 about targets 1..n-1
        pred = logits[:-1]                                   # [n_pos, vocab]
        target = torch.tensor(ids[1:], dtype=torch.long, device=device)  # [n_pos]

        # D2: probability mass on new ids (avg over all positions)
        probs = torch.softmax(pred.float(), dim=-1)
        prob_mass_new = probs[:, new_start:new_end].sum(dim=-1)  # [n_pos]
        prob_mass_new_total += float(prob_mass_new.sum())

        # D3: entropy at each position (averaged into register stats)
        with torch.no_grad():
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)  # [n_pos]
        register_stats[register]["entropy_sum"] += float(entropy.sum())
        register_stats[register]["n_pos"] += n_pos
        register_stats[register]["prob_mass_new"] += float(prob_mass_new.sum())

        # D1 + D4: only at positions where target is a NEW id
        is_new_target = (target >= new_start) & (target < new_end)
        new_pos = is_new_target.nonzero(as_tuple=False).squeeze(-1)
        if new_pos.numel():
            sub_pred = pred[new_pos]
            sub_target = target[new_pos]
            # rank of target among all logits at each position
            target_logits = sub_pred.gather(1, sub_target.unsqueeze(1)).squeeze(1)  # [k]
            ranks = (sub_pred > target_logits.unsqueeze(1)).sum(dim=1) + 1  # 1-indexed
            n_pos_new_target += sub_target.numel()
            rank_sum += int(ranks.sum())
            rank_buckets["top1"]  += int((ranks <= 1).sum())
            rank_buckets["top5"]  += int((ranks <= 5).sum())
            rank_buckets["top10"] += int((ranks <= 10).sum())
            rank_buckets["top50"] += int((ranks <= 50).sum())
            register_stats[register]["n_new_target"] += sub_target.numel()
            register_stats[register]["rank_sum"] += int(ranks.sum())
            # D4: top-1 prediction at these positions
            top1 = sub_pred.argmax(dim=-1)  # [k]
            top1_new  += int(((top1 >= new_start) & (top1 < new_end)).sum())
            top1_base += int((top1 < new_start).sum())

        n_pos_total += n_pos

        if n_seen % 25 == 0:
            elapsed = time.time() - t0
            rate = n_seen / max(elapsed, 1e-6)
            print(f"  doc {n_seen}: {rate:.2f}/s  n_new_targets={n_pos_new_target}", file=sys.stderr)

    out = {
        "n_docs_processed":   n_seen,
        "n_positions_total":  n_pos_total,
        "n_positions_new_target": n_pos_new_target,
    }
    if n_pos_total:
        out["d2_avg_prob_mass_new_per_pos"] = prob_mass_new_total / n_pos_total
    if n_pos_new_target:
        out["d1_rank_of_new_target"] = {
            "mean_rank": rank_sum / n_pos_new_target,
            "top1_rate":  rank_buckets["top1"]  / n_pos_new_target,
            "top5_rate":  rank_buckets["top5"]  / n_pos_new_target,
            "top10_rate": rank_buckets["top10"] / n_pos_new_target,
            "top50_rate": rank_buckets["top50"] / n_pos_new_target,
        }
        out["d4_top1_at_new_target"] = {
            "new":        top1_new,
            "base":       top1_base,
            "new_rate":   top1_new / n_pos_new_target,
        }
    out["d3_per_register"] = {}
    for reg, stats in register_stats.items():
        if not stats["n_pos"]:
            continue
        out["d3_per_register"][reg] = {
            "n_positions":          stats["n_pos"],
            "n_new_target":         stats["n_new_target"],
            "avg_entropy_nats":     stats["entropy_sum"] / stats["n_pos"],
            "avg_prob_mass_new":    stats["prob_mass_new"] / stats["n_pos"],
            "avg_rank_new_target":  (stats["rank_sum"] / stats["n_new_target"]) if stats["n_new_target"] else None,
        }
    return out


def _greedy_diagnostics(model, tokenizer, prompts, new_id_range, max_new_tokens, device):
    """D5: greedy-gen new-token utilization rate."""
    import torch
    new_start, new_end = new_id_range
    n_total_gen = 0
    n_new_gen = 0
    per_prompt = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        gen = out[0, input_ids.shape[1]:].tolist()  # only the generated suffix
        n_new = sum(1 for t in gen if new_start <= t < new_end)
        per_prompt.append({
            "prompt": prompt[:80] + ("…" if len(prompt) > 80 else ""),
            "n_generated": len(gen),
            "n_new": n_new,
            "utilization": n_new / max(len(gen), 1),
        })
        n_total_gen += len(gen)
        n_new_gen   += n_new
    return {
        "n_prompts":        len(prompts),
        "n_total_gen":      n_total_gen,
        "n_new_gen":        n_new_gen,
        "utilization_rate": n_new_gen / max(n_total_gen, 1),
        "per_prompt":       per_prompt,
    }


_DEFAULT_GREEDY_PROMPTS = [
    "Η ελληνική γλώσσα έχει μακρά ιστορία. ",
    "Στην ομιλία του Πλάτωνα, ο Σωκράτης λέει ότι ",
    "Στην Αθήνα, οι αρχαίοι ναοί έχουν ιδιαίτερη σημασία γιατί ",
    "Η σύγχρονη ελληνική κουζίνα ",
    "Στη φιλοσοφία του Αριστοτέλη, η έννοια της αρετής ",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--eval-jsonl", type=Path, default=None,
                    help="held-out JSONL; required unless --embedding-only")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--new-id-range", type=int, nargs=2, default=[131072, 148480],
                    metavar=("LO", "HI"), help="[LO, HI) IDs of new vocab")
    ap.add_argument("--base-vocab-size", type=int, default=131072)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--embedding-only", action="store_true",
                    help="skip forward-pass + greedy-gen diagnostics (D1-D5); compute only D6+D7")
    ap.add_argument("--skip-greedy", action="store_true",
                    help="skip just D5 (greedy-gen utilization rate)")
    ap.add_argument("--greedy-max-new-tokens", type=int, default=100)
    ap.add_argument("--greedy-prompts-file", type=Path, default=None,
                    help="optional: one prompt per line; defaults to _DEFAULT_GREEDY_PROMPTS")
    args = ap.parse_args()

    if not args.embedding_only and args.eval_jsonl is None:
        ap.error("--eval-jsonl is required unless --embedding-only")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    t0 = time.time()
    print(f"loading model + tokenizer: {args.model_path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    model = model.to(args.device)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    # D6 + D7
    print("computing embedding-space diagnostics (D6 + D7) ...", file=sys.stderr)
    E = model.get_input_embeddings().weight.detach().cpu()
    lm_head = model.get_output_embeddings()
    U = (lm_head.weight if lm_head is not None else E).detach().cpu()
    embedding = _embedding_diagnostics(E, U, tuple(args.new_id_range), args.base_vocab_size)

    report = {
        "model_path":      args.model_path,
        "new_id_range":    args.new_id_range,
        "tokenizer_vocab": tok.vocab_size,
        "embedding":       embedding,
    }

    # D1-D4
    if not args.embedding_only:
        print("computing forward-pass diagnostics (D1-D4) ...", file=sys.stderr)
        report["forward"] = _forward_diagnostics(
            model, tok, args.eval_jsonl,
            tuple(args.new_id_range), args.max_context, args.max_docs, args.device,
        )

        # D5
        if not args.skip_greedy:
            print("computing greedy-gen utilization (D5) ...", file=sys.stderr)
            if args.greedy_prompts_file:
                prompts = [l.strip() for l in args.greedy_prompts_file.read_text().splitlines() if l.strip()]
            else:
                prompts = list(_DEFAULT_GREEDY_PROMPTS)
            report["greedy"] = _greedy_diagnostics(
                model, tok, prompts, tuple(args.new_id_range),
                args.greedy_max_new_tokens, args.device,
            )

    report["wall_seconds"] = time.time() - t0
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Compact summary
    print("\n=== new-token diagnostic suite ===")
    for ax in ("E", "U"):
        if f"{ax}_norm" in embedding:
            n = embedding[f"{ax}_norm"]
            print(
                f"  D6.{ax}: existing μ={n['existing_mean']:.3f} (p50={n['existing_p50']:.3f}); "
                f"new μ={_fmt_optional_float(n['new_mean'])} "
                f"(p50={_fmt_optional_float(n['new_p50'])}) — "
                f"ratio {_fmt_optional_float(n['new_to_existing_mean_ratio'])}"
            )
    cos = embedding["new_E_cos"]
    print(
        f"  D7. new_E cos off-diag: μ={_fmt_optional_float(cos['mean_off_diag'], 4)}  "
        f"p95={_fmt_optional_float(cos['p95_off_diag'], 4)}"
    )
    er = embedding["new_E_effective_rank"]
    print(
        f"  D7. new_E participation ratio: {_fmt_optional_float(er['participation_ratio'], 1)} / "
        f"{er['n_singular_values']} (rank@99% variance: {er['rank_at_99pct_var']})"
    )

    if "forward" in report:
        f = report["forward"]
        if "d1_rank_of_new_target" in f:
            d1 = f["d1_rank_of_new_target"]
            print(f"  D1. rank of new target: mean={d1['mean_rank']:.1f}  top1={d1['top1_rate']:.3f}  top5={d1['top5_rate']:.3f}")
        if "d2_avg_prob_mass_new_per_pos" in f:
            print(f"  D2. avg prob mass on new ids: {f['d2_avg_prob_mass_new_per_pos']:.4f}")
        if "d4_top1_at_new_target" in f:
            print(f"  D4. top-1 at new-target positions: new_rate={f['d4_top1_at_new_target']['new_rate']:.3f}")
    if "greedy" in report:
        g = report["greedy"]
        print(f"  D5. greedy-gen utilization: {g['utilization_rate']:.4f} ({g['n_new_gen']}/{g['n_total_gen']})")

    print(f"\nwrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
