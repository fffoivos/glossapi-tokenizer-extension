#!/usr/bin/env python3
"""Resumability filter: read a units dir's _args.json and emit ONLY the jobs whose output file does not yet
exist. Run this before every (re)launch and pass the result to the annotation workflow, so a stop+restart
never re-spends tokens on already-annotated docs. Local only — no Opus.
  Usage: pending_args.py units/STRUCT_RUN"""
import json, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "units/STRUCT_RUN"
if not os.path.isabs(d):
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), d)
args = json.load(open(f"{d}/_args.json"))
pending = [a for a in args if not os.path.exists(a["out"])]
json.dump(pending, open(f"{d}/_pending.json", "w"))
done = len(args) - len(pending)
print(f"{len(args)} total · {done} done · {len(pending)} PENDING → {d}/_pending.json")
if pending:
    print("\npass this to Workflow args:")
    print(json.dumps(pending))
else:
    print("\nall done — nothing to run.")
