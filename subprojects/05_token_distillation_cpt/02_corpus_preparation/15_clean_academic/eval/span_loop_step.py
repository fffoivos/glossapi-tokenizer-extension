#!/usr/bin/env python3
"""One step of the modest span-annotation loop. Merges any new chunk result into all.json, computes
which batches are still unfinished, enforces spacing between chunk launches, and prints the next
action for the agent: DONE | WAIT <secs> | RUN <json batch-path list>.

Usage:  span_loop_step.py [--merge result.json]
"""
import argparse, glob, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ALLP = f"{HERE}/annotations_span/all.json"
STATEP = f"{HERE}/units/SPAN_loop_state.json"
SPACING = 180    # small gap only; gentleness now lives in SEQUENTIAL chunk execution
CHUNK = 10       # smaller chunks → checkpoint to all.json every ~50 min (throttle is slow)


def load_all():
    if os.path.exists(ALLP):
        return {a["unit_id"]: a for a in json.load(open(ALLP))["annotations"] if a.get("unit_id")}
    return {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--merge"); a = ap.parse_args()
    d = load_all()
    if a.merge and os.path.exists(a.merge):
        for x in json.load(open(a.merge)).get("annotations", []):
            if x.get("unit_id"):
                d[x["unit_id"]] = x
        os.makedirs(os.path.dirname(ALLP), exist_ok=True)
        json.dump({"annotations": list(d.values())}, open(ALLP, "w"), ensure_ascii=False)

    batchpaths = json.load(open(f"{HERE}/units/SPAN_batchpaths.json"))
    skipp = f"{HERE}/units/SPAN_skip.json"
    skip = set(json.load(open(skipp))) if os.path.exists(skipp) else set()   # batches that keep tripping AUP
    unfinished = [p for p in batchpaths if p not in skip and not all(u["unit_id"] in d for u in json.load(open(p)))]
    n_spans = sum(len(x.get("spans", [])) for x in d.values())
    state = json.load(open(STATEP)) if os.path.exists(STATEP) else {"last_launch": 0, "chunk": 0}
    now = time.time()

    sys.stderr.write(f"[loop] {len(d)} windows done · {n_spans} spans · {len(unfinished)}/{len(batchpaths)} batches left\n")
    if not unfinished:
        print("DONE"); return
    elapsed = now - state["last_launch"]
    if elapsed < SPACING:
        print(f"WAIT {int(SPACING - elapsed)}"); return
    chunk = unfinished[:CHUNK]
    state["last_launch"] = now; state["chunk"] = state.get("chunk", 0) + 1
    json.dump(state, open(STATEP, "w"))
    sys.stderr.write(f"[loop] launching chunk {state['chunk']}: {len(chunk)} batches\n")
    print("RUN"); print(json.dumps(chunk))


if __name__ == "__main__":
    main()
