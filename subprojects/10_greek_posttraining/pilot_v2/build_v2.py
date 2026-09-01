#!/usr/bin/env python3
"""Build the v2 comparison artifact: original | opus | sol, three columns, details on demand."""
import json, os, sys, collections, re

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from spec_data import SPEC, FAMILY
from runner import load_json

MODELS = ['opus', 'sol']


def gather(sample):
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(f'{D}/sample_{sample}.jsonl')}
    man = json.load(open(f'{D}/v2/batches/{sample}_manifest.json'))
    out = {m: {'s1': {}, 's2': {}} for m in MODELS}
    missing = collections.Counter()
    for m in MODELS:
        for b in man:
            for stage in ('1', '2'):
                p = f'{D}/v2/{sample}/{m}/stage{stage}_{b["tag"]}.json'
                if not os.path.exists(p):
                    missing[f'{m}/s{stage}'] += b['n']
                    continue
                try:
                    for r in load_json(p):
                        if r.get('row_id'):
                            out[m]['s' + stage][r['row_id']] = r
                except Exception:
                    missing[f'{m}/s{stage}:bad'] += b['n']
    rows = []
    for rid, s in src.items():
        en_u = [x['content'] for x in s['messages_en'] if x['role'] == 'user']
        en_a = [x['content'] for x in s['messages_en'] if x['role'] == 'assistant']
        rec = {'id': rid[:8], 'cat': s['category'], 'sample': sample,
               'sys_en': s.get('system_en'), 'p_en': s['prompt_en'],
               'u_en': en_u, 'r_en': en_a, 'm': {}}
        for m in MODELS:
            a = out[m]['s1'].get(rid) or {}
            b = out[m]['s2'].get(rid) or {}
            rec['m'][m] = {
                'has': bool(a or b),
                'sys': a.get('system_el'), 'p': a.get('prompt_el') or '',
                'u': a.get('user_turns_el') or [],
                'r': ([b.get('response_el')] if b.get('response_el') else []) + (b.get('assistant_turns_el') or []),
                'fam': a.get('family'), 'cls': a.get('translation_class'),
                'sh': a.get('source_handling'), 'oh': a.get('output_handling'),
                'sub': a.get('subtype') or '', 'haz': a.get('hazards') or [],
                'regt': a.get('target_register'), 'regu': b.get('register_used'),
                'dev': a.get('devices') or [], 'devn': a.get('device_notes') or '',
                'tr': a.get('transpositions') or [], 'de': a.get('derived_elements') or [],
                'gg': (a.get('guide_gaps') or []) + (b.get('guide_gaps') or []),
                'cn': a.get('constraint_notes') or '', 'cc': b.get('constraint_check') or '',
                'rp': a.get('reference_problems') or [], 'rc': b.get('reference_corrections') or [],
                'dn': b.get('defect_note') or '', 'cd': b.get('class_disagreement'),
                'devi': b.get('deviations') or [], 'sf': b.get('self_flags') or [],
                'tn': a.get('translator_notes') or '',
            }
        rows.append(rec)
    order = {c: i for i, c in enumerate(
        ['Generation', 'Open QA', 'Brainstorm', 'Chat', 'Rewrite', 'Summarize',
         'Coding', 'Classify', 'Closed QA', 'Extract'])}
    rows.sort(key=lambda r: (order.get(r['cat'], 99), r['id']))
    return rows, missing


allrows, miss = [], collections.Counter()
for s in ('A', 'B'):
    r, m = gather(s)
    allrows += r
    miss.update(m)

stats = {
    'n': len(allrows),
    'A': sum(1 for r in allrows if r['sample'] == 'A'),
    'B': sum(1 for r in allrows if r['sample'] == 'B'),
    'cat': collections.Counter(r['cat'] for r in allrows),
    'cov': {m: sum(1 for r in allrows if r['m'][m]['r']) for m in MODELS},
    'missing': dict(miss),
}
CATS = [c for c, _ in sorted(stats['cat'].items(), key=lambda x: -x[1])]

payload = {'rows': allrows, 'stats': {k: (dict(v) if isinstance(v, collections.Counter) else v)
                                      for k, v in stats.items()},
           'cats': CATS, 'models': MODELS, 'spec': SPEC, 'family': FAMILY}
prompts = {'s1': open(f'{D}/prompts/v2_stage1.md').read(),
           's2': open(f'{D}/prompts/v2_stage2.md').read()}

TPL = open(f'{D}/template_v2.html').read()
out = (TPL.replace('/*__DATA__*/', json.dumps(payload, ensure_ascii=False))
          .replace('/*__PROMPTS__*/', json.dumps(prompts, ensure_ascii=False)))
open(f'{D}/artifact_v2.html', 'w').write(out)

import re as _re, subprocess, tempfile, shutil
blocks = _re.findall(r'<script>(.*?)</script>', out, _re.S)
if shutil.which('node') and blocks:
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(blocks[-1]); tmp = f.name
    p = subprocess.run(['node', '--check', tmp], capture_output=True, text=True)
    if p.returncode:
        raise SystemExit('JS SYNTAX ERROR:\n' + p.stderr[:1200])
    print('js syntax: OK')
for tag in ('data', 'prompts'):
    m = _re.search(rf'<script id="{tag}" type="application/json">(.*?)</script>', out, _re.S)
    json.loads(m.group(1))
print('json: OK')
print(f'rows {stats["n"]} (A={stats["A"]} B={stats["B"]}) | coverage {stats["cov"]}')
if miss:
    print('MISSING:', dict(miss))
print('wrote artifact_v2.html', len(out), 'bytes')
