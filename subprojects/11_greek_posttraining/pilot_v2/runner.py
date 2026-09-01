#!/usr/bin/env python3
"""Build per-batch prompt files for both stages, and drive the Sol (codex) runs.

Opus runs are driven separately by a Workflow script that reads the same batch files.
Layout:  v2/<sample>/<model>/stage{1,2}_<cat>.json      outputs
         v2/batches/<sample>_<stage>_<cat>.txt          prompt files (model-agnostic)
"""
import json, os, sys, collections, subprocess, re

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
from spec_data import SPEC, FAMILY

BATCH = 10  # rows per model call


def family_of(cat):
    for fam, f in FAMILY.items():
        if cat in f['cats']:
            return fam
    raise KeyError(cat)


def strip_html(s):
    return re.sub(r'<[^>]+>', '', s).replace('&amp;', '&').replace('&nbsp;', ' ')


def family_block(cat):
    fam = family_of(cat)
    f = FAMILY[fam]
    return (f"## This row's family: {fam}\n\n{strip_html(f['desc'])}\n\n"
            f"Defect notes in this family: {strip_html(f['defect'])}.\n")


def category_block(cat):
    s = SPEC[cat]
    ess = '\n'.join(f'- {strip_html(e)}' for e in s['ess'])
    return (f"## This row's category: {cat}\n\n"
            f"**Shape.** {strip_html(s['shape'])}\n\n"
            f"**The thing that defines this category.** {strip_html(s['key'])}\n\n"
            f"**Essential — lose any of these and the row is destroyed:**\n{ess}\n\n"
            f"**Safe to adapt.** {strip_html(s['safe'])}\n\n"
            f"**Normally classified as.** {strip_html(s['cls'])}\n")


def build_batches(sample_path, sample_name):
    rows = [json.loads(l) for l in open(sample_path)]
    bycat = collections.defaultdict(list)
    for r in rows:
        bycat[r['category']].append(r)
    os.makedirs(f'{D}/v2/batches', exist_ok=True)
    manifest = []
    for cat, rr in bycat.items():
        for i in range(0, len(rr), BATCH):
            chunk = rr[i:i + BATCH]
            tag = f"{sample_name}_{cat.replace(' ', '')}_{i // BATCH}"
            manifest.append({'tag': tag, 'sample': sample_name, 'cat': cat,
                             'n': len(chunk), 'ids': [r['row_id'] for r in chunk]})
            for stage in (1, 2):
                tpl = open(f'{D}/prompts/v2_stage{stage}.md').read()
                body = (tpl.replace('{{FAMILY_BLOCK}}', family_block(cat))
                           .replace('{{CATEGORY_BLOCK}}', category_block(cat)))
                if stage == 1:
                    payload = chunk
                    inp = ("\n\n---\n\n# The rows\n\nEach row: `row_id`, `category`, `system_en` "
                           "(system prompt, may be null), `prompt_en`, `messages_en` (user and "
                           "assistant turns; the assistant turns are the English reference answer), "
                           "`n_turns`.\n\n```json\n" + json.dumps(payload, ensure_ascii=False, indent=1) + "\n```\n")
                else:
                    inp = ("\n\n---\n\n# The rows\n\nEach row carries `stage1` (what stage 1 decided, "
                           "including the Greek prompt) and `english` (the original row, whose "
                           "assistant turns are the reference answer).\n\n"
                           "{{STAGE1_JSON}}\n")
                open(f'{D}/v2/batches/{tag}_s{stage}.txt', 'w').write(body + inp)
    json.dump(manifest, open(f'{D}/v2/batches/{sample_name}_manifest.json', 'w'),
              ensure_ascii=False, indent=1)
    return manifest


def stage2_input(tag, model, sample_name):
    """Splice stage-1 output into the stage-2 prompt file for a given model."""
    man = {m['tag']: m for m in json.load(open(f'{D}/v2/batches/{sample_name}_manifest.json'))}[tag]
    s1 = {r['row_id']: r for r in load_json(f'{D}/v2/{sample_name}/{model}/stage1_{tag}.json')}
    src = {json.loads(l)['row_id']: json.loads(l)
           for l in open(f'{D}/sample_{sample_name}.jsonl')}
    rows = [{'row_id': i, 'stage1': s1.get(i), 'english': src[i]} for i in man['ids']]
    tpl = open(f'{D}/v2/batches/{tag}_s2.txt').read()
    return tpl.replace('{{STAGE1_JSON}}',
                       "```json\n" + json.dumps(rows, ensure_ascii=False, indent=1) + "\n```")


def load_json(path):
    """Tolerant JSON array load: strips fences and any preamble/postamble."""
    t = open(path).read().strip()
    t = re.sub(r'^```(?:json)?\s*', '', t)
    t = re.sub(r'\s*```$', '', t.strip())
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find('['), t.rfind(']')
        if a == -1 or b == -1:
            raise
        return json.loads(t[a:b + 1])


def run_sol(prompt_text, out_path, timeout=1800):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cmd = ['codex', 'exec', '-m', 'gpt-5.6-sol',
           '-c', 'model_reasoning_effort=medium',
           '--skip-git-repo-check', '--ephemeral',
           '-o', out_path, '-']
    p = subprocess.run(cmd, input=prompt_text, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stderr or '')[-600:]


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if what == 'build':
        for s in ('A', 'B'):
            m = build_batches(f'{D}/sample_{s}.jsonl', s)
            print(f'sample {s}: {len(m)} batches, {sum(x["n"] for x in m)} rows')
            for x in m:
                print(f'   {x["tag"]:26s} {x["cat"]:11s} n={x["n"]}')
