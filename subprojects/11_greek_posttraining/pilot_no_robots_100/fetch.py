import json,urllib.request,urllib.parse,time,collections,sys
BASE="https://datasets-server.huggingface.co/rows"
DS="HuggingFaceH4/no_robots"
out=[]
for off in range(0,9500,100):
    url=f"{BASE}?dataset={urllib.parse.quote(DS)}&config=default&split=train&offset={off}&length=100"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url,timeout=60) as r:
                d=json.load(r); break
        except Exception as e:
            if attempt==3: print("FAIL",off,e,file=sys.stderr); d={"rows":[]}
            else: time.sleep(2*(attempt+1))
    for row in d.get("rows",[]):
        out.append(row["row"])
    if off%2000==0: print("...",off,len(out),file=sys.stderr)
with open("raw_train.jsonl","w") as f:
    for r in out: f.write(json.dumps(r,ensure_ascii=False)+"\n")
c=collections.Counter(r.get("category") for r in out)
print("TOTAL",len(out)); print(json.dumps(c,ensure_ascii=False,indent=1))
