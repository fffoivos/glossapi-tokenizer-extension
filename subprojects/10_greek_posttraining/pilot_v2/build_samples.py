import json, random, collections, os
D=os.path.dirname(os.path.abspath(__file__))
rows=[json.loads(l) for l in open(f'{D}/raw_train.jsonl')]
by=collections.defaultdict(list)
for r in rows: by[r['category']].append(r)
for c in by: by[c].sort(key=lambda r: r['prompt_id'])

def pack(r):
    msgs=[{"role":m["role"],"content":m["content"]} for m in (r.get("messages") or [])]
    sysm=[m for m in msgs if m["role"]=="system"]
    return {"row_id":r["prompt_id"],"category":r["category"],
            "system_en":sysm[0]["content"] if sysm else None,
            "prompt_en":r["prompt"],
            "messages_en":[m for m in msgs if m["role"]!="system"],
            "n_turns":len(msgs)}

# ---- Sample A: the 9 rows the owner reviewed, in artifact display order
A_IDS=['028bf602','07794700','6d2fe44a','749a5955','76816928','7bc6f393','7fe9c4e3','86786cab','89c788bc']
idx={r['prompt_id'][:8]:r for r in rows}
A=[pack(idx[i]) for i in A_IDS if i in idx]
assert len(A)==9, len(A)
with open(f'{D}/sample_A.jsonl','w') as f:
    for r in A: f.write(json.dumps(r,ensure_ascii=False)+"\n")

# ---- Sample B: proportional 100, new seed; Generation stratified by prompt length
QUOTA={'Generation':46,'Open QA':12,'Brainstorm':11,'Chat':8,'Rewrite':7,
       'Summarize':4,'Coding':4,'Classify':4,'Closed QA':3,'Extract':2}
assert sum(QUOTA.values())==101
QUOTA['Generation']=45  # 100 total
rnd=random.Random(20260824)
B=[]
for c,n in QUOTA.items():
    pool=by[c]
    if c=='Generation':
        pool=sorted(pool,key=lambda r: len(r['prompt']))
        band=len(pool)//3
        bands=[pool[:band],pool[band:2*band],pool[2*band:]]
        take=[n//3,n//3,n-2*(n//3)]
        pick=[]
        for bnd,k in zip(bands,take): pick+=rnd.sample(bnd,k)
    else:
        pick=rnd.sample(pool,n)
    B+=[pack(r) for r in pick]
assert len(B)==100, len(B)
with open(f'{D}/sample_B.jsonl','w') as f:
    for r in B: f.write(json.dumps(r,ensure_ascii=False)+"\n")

print('A:',len(A),'rows |',dict(collections.Counter(r['category'] for r in A)))
print('B:',len(B),'rows')
for c,n in sorted(collections.Counter(r['category'] for r in B).items(),key=lambda x:-x[1]):
    print(f'   {c:11s} {n:3d}')
print('overlap A/B:',len({r["row_id"] for r in A} & {r["row_id"] for r in B}))
