import json,urllib.request,urllib.parse,time
have={}
for l in open("raw_train.jsonl"):
    r=json.loads(l); have[r["prompt_id"]]=r
print("have",len(have))
BASE="https://datasets-server.huggingface.co/rows"; DS="HuggingFaceH4/no_robots"
added=0
for off in range(0,9500,100):
    url=f"{BASE}?dataset={urllib.parse.quote(DS)}&config=default&split=train&offset={off}&length=100"
    got=False
    for a in range(5):
        try:
            with urllib.request.urlopen(url,timeout=60) as r: d=json.load(r); got=True; break
        except Exception: time.sleep(2*(a+1))
    if not got: print("STILL FAIL",off); continue
    for row in d.get("rows",[]):
        rr=row["row"]
        if rr["prompt_id"] not in have: have[rr["prompt_id"]]=rr; added+=1
print("added",added,"total",len(have))
with open("raw_train.jsonl","w") as f:
    for k in sorted(have): f.write(json.dumps(have[k],ensure_ascii=False)+"\n")
