import json,random
rows=[json.loads(l) for l in open("raw_train.jsonl")]
CATS=["Generation","Open QA","Chat","Rewrite","Closed QA"]
by={c:[] for c in CATS}
for r in rows:
    if r.get("category") in by: by[r["category"]].append(r)
for c in CATS: by[c].sort(key=lambda r:r["prompt_id"])
rnd=random.Random(20260823)
sel=[]
for c in CATS:
    pick=rnd.sample(by[c],20)
    for r in pick:
        msgs=r.get("messages") or []
        user=[m for m in msgs if m.get("role")=="user"]
        asst=[m for m in msgs if m.get("role")=="assistant"]
        system=[m for m in msgs if m.get("role")=="system"]
        sel.append({"row_id":r["prompt_id"],"category":c,
            "n_turns":len(msgs),"has_system":bool(system),
            "system_en":system[0]["content"] if system else None,
            "prompt_en":r["prompt"],
            "reference_en":asst[0]["content"] if asst else None,
            "n_user_turns":len(user),"n_asst_turns":len(asst),
            "prompt_chars":len(r["prompt"]),
            "ref_chars":len(asst[0]["content"]) if asst else 0})
with open("sample_100.jsonl","w") as f:
    for s in sel: f.write(json.dumps(s,ensure_ascii=False)+"\n")
import collections
print("n =",len(sel))
print("multi-turn rows:",sum(1 for s in sel if s["n_turns"]>2))
print("with system prompt:",sum(1 for s in sel if s["has_system"]))
for c in CATS:
    g=[s for s in sel if s["category"]==c]
    print(f"  {c:11s} n={len(g):2d}  mean prompt {sum(x['prompt_chars'] for x in g)//len(g):5d} ch  mean ref {sum(x['ref_chars'] for x in g)//len(g):5d} ch  multiturn={sum(1 for x in g if x['n_turns']>2)}")
