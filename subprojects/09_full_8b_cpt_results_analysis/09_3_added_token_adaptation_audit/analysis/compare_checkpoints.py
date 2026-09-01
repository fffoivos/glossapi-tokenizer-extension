"""Cross-checkpoint comparison of the per-added-token behavioural audit.

Answers the question the experiment exists for: do the added tokens keep improving
to the 77B terminal, or do they peak with GreekMMLU at update 9536?
"""
import json, sys
import numpy as np

ORDER = [("step400-tokens2B", 400), ("step9536-tokens40B", 9536), ("main", 18284)]
POLY_START = 148480

def load(root):
    out = {}
    for name, upd in ORDER:
        try:
            out[name] = (upd, json.load(open(f"{root}/audit_{name}.json")))
        except FileNotFoundError:
            print(f"  (missing audit_{name}.json)")
    return out

def arrays(d, group, min_occ=4):
    rows = []
    for tid, v in d["per_token"].items():
        t = int(tid)
        if group == "modern" and t >= POLY_START: continue
        if group == "polytonic" and t < POLY_START: continue
        if v.get("n_occ", 0) < min_occ: continue
        rows.append(v)
    return rows

def main(root):
    data = load(root)
    if not data: return
    print(f"\n{'='*78}\nADDED-TOKEN BEHAVIOURAL AUDIT — trajectory\n{'='*78}")
    for group in ("modern", "polytonic"):
        print(f"\n## {group} tokens")
        print(f"{'update':>8} {'n>=4occ':>8} {'delta_logp p50':>15} {'frac tok<=0':>12} "
              f"{'hcos L11':>9} {'hcos L30':>9} {'echo top1':>10}")
        per_upd = {}
        for name, upd in ORDER:
            if name not in data: continue
            _, d = data[name]
            rows = arrays(d, group)
            if not rows: continue
            dl = np.array([r["mean_delta_logp"] for r in rows])
            h1 = np.array([r["mean_hidden_cos"] for r in rows if r.get("mean_hidden_cos") is not None])
            h2 = np.array([r["mean_hidden_cos_late"] for r in rows if r.get("mean_hidden_cos_late") is not None])
            ech = np.array([r["echo"]["rank"] for r in rows if r.get("echo")])
            print(f"{upd:>8} {len(rows):>8} {np.median(dl):>15.3f} {float((dl<=0).mean()):>11.2%} "
                  f"{np.median(h1) if h1.size else float('nan'):>9.3f} "
                  f"{np.median(h2) if h2.size else float('nan'):>9.3f} "
                  f"{float((ech==1).mean()) if ech.size else float('nan'):>9.1%}")
            per_upd[upd] = {r["surface"]: r for r in rows}
        # peak-vs-terminal per-token movement
        if 9536 in per_upd and 18284 in per_upd:
            common = set(per_upd[9536]) & set(per_upd[18284])
            d40 = np.array([per_upd[9536][s]["mean_delta_logp"] for s in common])
            d77 = np.array([per_upd[18284][s]["mean_delta_logp"] for s in common])
            diff = d77 - d40
            print(f"\n  peak(9536) -> terminal(18284), n={len(common)} shared tokens")
            print(f"    tokens IMPROVED  : {float((diff>0).mean()):.1%}")
            print(f"    tokens REGRESSED : {float((diff<0).mean()):.1%}")
            print(f"    mean delta change: {diff.mean():+.4f} nats  (p5 {np.quantile(diff,.05):+.3f} / "
                  f"p95 {np.quantile(diff,.95):+.3f})")
            worst = sorted(common, key=lambda s: (per_upd[18284][s]["mean_delta_logp"]
                                                  - per_upd[9536][s]["mean_delta_logp"]))[:8]
            print("    biggest regressions:")
            for s in worst:
                a, b = per_upd[9536][s]["mean_delta_logp"], per_upd[18284][s]["mean_delta_logp"]
                print(f"      {s!r:26s} {a:+.3f} -> {b:+.3f}  ({b-a:+.3f})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
