"""Per-added-token behavioural audit for the Apertus-8B Greek CPT checkpoints.

Paired document-level design: every document is scored twice by the SAME model,
once under the extended tokenisation and once under the base (131,072)
tokenisation, then aligned on character offsets.  Because both arms see the
identical text, training-set contamination cancels and no held-out corpus is
needed.

  T1  merged-vs-split log-likelihood parity
        delta_t = logP(t | ctx)  -  sum_j logP(b_j | ctx, b_<j)
      over every occurrence of added token t, where b_1..b_m is the base
      tokenisation of the same character span.
      delta > 0  => the added row predicts its own string better than the base
                    pieces it replaced: the merge pays for itself.
      delta <= 0 => alive but not yet worth its vocabulary slot.
      [tokenisation-invariance framing: Cao & Rimell 2021; Chirkova et al. 2023]

  T2  hidden-state agreement at the Token-Distillation layer
        cos( h_L[pos of t] , h_L[pos of b_m] )
      i.e. does the added token occupy the same representational slot as the
      phrase it replaced.  Default layer 11 = the layer TD was fitted at.
      [Token Distillation, Dobler 2025; inner lexicon, Kaplan et al. 2024]

  T3  Magikarp-style echo probe: P(t | t t t t) and the rank of t.
      [Land & Bartolo, EMNLP 2024]
"""
from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--base-tokenizer", required=True)
    p.add_argument("--corpus", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--new-id-range", nargs=2, type=int, default=[131072, 148992])
    p.add_argument("--layer", type=int, default=11)
    p.add_argument("--late-layer", type=int, default=30,
                   help="second probe layer: do merged and split paths reconverge downstream?")
    p.add_argument("--max-docs", type=int, default=20000)
    p.add_argument("--max-context", type=int, default=1024)
    p.add_argument("--max-split-tokens", type=int, default=3072)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--bucket-pool", type=int, default=1,
                   help="batches per length-sorted pool; 1 disables bucketing")
    p.add_argument("--echo-repeats", type=int, default=4)
    p.add_argument("--echo-batch", type=int, default=128)
    p.add_argument("--skip-echo", action="store_true")
    p.add_argument("--skip-hidden", action="store_true")
    p.add_argument("--progress-every", type=int, default=200)
    return p.parse_args()


def read_docs(path, limit):
    n = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("text"):
                yield row
                n += 1
                if limit and n >= limit:
                    return


def score_batch(model, seqs, pad_id, layers, want_hidden, device):
    """Teacher-forced log-probs of each realised token + optional hidden states."""
    import torch
    L = max(len(s) for s in seqs)
    x = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, device=device)
        mask[i, :len(s)] = 1
    with torch.no_grad():
        out = model(input_ids=x, attention_mask=mask,
                    output_hidden_states=want_hidden)
        lp = torch.log_softmax(out.logits.float(), dim=-1)
        # token_lp[i, k] = logP(seq[i][k] | seq[i][:k]) for k >= 1
        tgt = x[:, 1:].unsqueeze(-1)
        token_lp = lp[:, :-1].gather(-1, tgt).squeeze(-1)
    h = {L: out.hidden_states[L] for L in layers} if want_hidden else None
    return token_lp, h


def main():
    a = parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model)
    base_tok = AutoTokenizer.from_pretrained(a.base_tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16).to(device).eval()
    lo, hi = a.new_id_range
    bos = tok.bos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    want_hidden = not a.skip_hidden

    stat = defaultdict(lambda: {"n": 0, "d_sum": 0.0, "d_sq": 0.0, "d_pos": 0,
                                "merged_nll": 0.0, "split_nll": 0.0,
                                "pieces": 0, "cos_sum": 0.0, "cos_n": 0,
                                "late_cos_sum": 0.0,
                                "by_reg": defaultdict(lambda: [0, 0.0, 0])})
    n_docs = n_occ = n_unaligned = 0
    t0 = time.time()
    buf = []

    def flush(buf):
        nonlocal n_occ, n_unaligned
        m_seqs = [bo["m_ids"] for bo in buf]
        s_seqs = [bo["s_ids"] for bo in buf]
        LAYERS = (a.layer, a.late_layer)
        m_lp, m_h = score_batch(model, m_seqs, pad, LAYERS, want_hidden, device)
        s_lp, s_h = score_batch(model, s_seqs, pad, LAYERS, want_hidden, device)
        for i, bo in enumerate(buf):
            reg = bo["register"]
            s_start = {c: k for k, (c, _) in enumerate(bo["s_offs"])}
            s_end = {c: k for k, (_, c) in enumerate(bo["s_offs"])}
            for pos, t, (cs, ce) in bo["hits"]:
                j0, j1 = s_start.get(cs), s_end.get(ce)
                if j0 is None or j1 is None or j1 < j0:
                    n_unaligned += 1
                    continue
                npieces = j1 - j0 + 1
                if npieces < 2:
                    continue                    # no contrast: same segmentation
                # +1 for the BOS we prepended; lp index k-1 scores token k
                lm = m_lp[i, pos].item()        # token at m_ids[pos+1]
                ls = float(s_lp[i, j0:j1 + 1].sum().item())
                st = stat[t]
                d = lm - ls
                st["n"] += 1; st["d_sum"] += d; st["d_sq"] += d * d
                st["d_pos"] += int(d > 0)
                st["merged_nll"] += -lm; st["split_nll"] += -ls
                st["pieces"] = npieces
                r = st["by_reg"][reg]; r[0] += 1; r[1] += d; r[2] += int(d > 0)
                if want_hidden:
                    for L, key in ((a.layer, "cos_sum"), (a.late_layer, "late_cos_sum")):
                        hm = m_h[L][i, pos + 1].float()
                        hs = s_h[L][i, j1 + 1].float()
                        st[key] += torch.nn.functional.cosine_similarity(
                            hm, hs, dim=0).item()
                    st["cos_n"] += 1
                n_occ += 1

    for row in read_docs(a.corpus, a.max_docs):
        text = row["text"]
        me = tok(text, return_offsets_mapping=True, add_special_tokens=False)
        m_ids = me["input_ids"][: a.max_context - 1]
        m_offs = me["offset_mapping"][: len(m_ids)]
        hits = [(k, t, m_offs[k]) for k, t in enumerate(m_ids) if lo <= t < hi]
        if not hits:
            continue
        se = base_tok(text, return_offsets_mapping=True, add_special_tokens=False)
        # keep the base arm covering the same character prefix as the merged arm
        char_end = m_offs[-1][1]
        s_ids, s_offs = [], []
        for t, off in zip(se["input_ids"], se["offset_mapping"]):
            if off[1] > char_end:
                break
            s_ids.append(t); s_offs.append(off)
        # Greek base tokenisation is ~2x less efficient, so the split arm is the
        # binding length constraint.  Trim it (and the merged arm + hits with it)
        # instead of dropping the document, which would bias towards short docs.
        if len(s_ids) > a.max_split_tokens:
            s_ids = s_ids[: a.max_split_tokens]
            s_offs = s_offs[: a.max_split_tokens]
            char_end = s_offs[-1][1]
            keep = [k for k, off in enumerate(m_offs) if off[1] <= char_end]
            if len(keep) < 2:
                continue
            m_ids = m_ids[: keep[-1] + 1]
            m_offs = m_offs[: len(m_ids)]
            hits = [h for h in hits if h[0] <= keep[-1]]
        if len(s_ids) < 2 or not hits:
            continue
        buf.append({"m_ids": [bos] + m_ids, "s_ids": [bos] + s_ids,
                    "s_offs": s_offs, "hits": hits,
                    "register": row.get("register", "unknown")})
        n_docs += 1
        # Length-bucket within a pool before batching. Every sequence is scored
        # independently under its own attention mask, so grouping by length
        # changes only padding waste, never a returned value.
        if len(buf) >= a.batch * a.bucket_pool:
            if a.bucket_pool > 1:
                buf.sort(key=lambda b: max(len(b["m_ids"]), len(b["s_ids"])))
            for k in range(0, len(buf), a.batch):
                flush(buf[k:k + a.batch])
            buf = []
        if n_docs % a.progress_every == 0:
            print(f"[T1/T2] docs={n_docs} occ={n_occ} tokens={len(stat)} "
                  f"unaligned={n_unaligned} {time.time()-t0:.0f}s", flush=True)
    if buf:
        if a.bucket_pool > 1:
            buf.sort(key=lambda b: max(len(b["m_ids"]), len(b["s_ids"])))
        for k in range(0, len(buf), a.batch):
            flush(buf[k:k + a.batch])

    echo = {}
    if not a.skip_echo:
        ids_all = list(range(lo, hi))
        for i in range(0, len(ids_all), a.echo_batch):
            chunk = ids_all[i: i + a.echo_batch]
            x = torch.tensor([[bos] + [t] * a.echo_repeats for t in chunk],
                             device=device)
            with torch.no_grad():
                lg = torch.log_softmax(model(input_ids=x).logits[:, -1].float(), -1)
            for k, t in enumerate(chunk):
                echo[t] = {"logp": lg[k, t].item(),
                           "rank": int((lg[k] > lg[k, t]).sum().item()) + 1}
            if i % (a.echo_batch * 20) == 0:
                print(f"[T3] {i}/{len(ids_all)} {time.time()-t0:.0f}s", flush=True)

    out = {"model": a.model, "new_id_range": [lo, hi], "layer": a.layer,
           "late_layer": a.late_layer,
           "n_docs": n_docs, "n_occurrences": n_occ, "n_unaligned": n_unaligned,
           "elapsed_s": time.time() - t0, "per_token": {}}
    for t, st in stat.items():
        n = st["n"]
        mean = st["d_sum"] / n
        out["per_token"][str(t)] = {
            "surface": tok.convert_ids_to_tokens(t),
            "n_occ": n,
            "n_base_pieces": st["pieces"],
            "mean_delta_logp": mean,
            "sd_delta_logp": max(0.0, st["d_sq"] / n - mean * mean) ** 0.5,
            "frac_delta_positive": st["d_pos"] / n,
            "mean_merged_nll": st["merged_nll"] / n,
            "mean_split_nll": st["split_nll"] / n,
            "mean_hidden_cos": (st["cos_sum"] / st["cos_n"]) if st["cos_n"] else None,
            "mean_hidden_cos_late": (st["late_cos_sum"] / st["cos_n"]) if st["cos_n"] else None,
            "by_register": {k: {"n": v[0], "mean_delta": v[1] / v[0],
                                "frac_pos": v[2] / v[0]}
                            for k, v in st["by_reg"].items() if v[0]},
            "echo": echo.get(t),
        }
    for t, e in echo.items():
        out["per_token"].setdefault(str(t), {
            "surface": tok.convert_ids_to_tokens(t), "n_occ": 0, "echo": e})
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("wrote", a.out, "tokens:", len(out["per_token"]),
          "occurrences:", n_occ, flush=True)


if __name__ == "__main__":
    main()
