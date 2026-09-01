#!/usr/bin/env python3
"""Stage 3 - deterministic Greek quality gates. Stdlib only.

Every check is cheap, explainable, and runs before any human looks at a row.
Register detection is a documented HEURISTIC instrument, not ground truth.
"""
import json, re, unicodedata as ud, sys, os, collections

DIR = os.path.dirname(os.path.abspath(__file__))

GREEK = re.compile(r'[Ͱ-Ͽἀ-῿]')
LATIN = re.compile(r'[A-Za-z]')
POLYTONIC = re.compile(r'[ἀ-῿]')
FENCE = re.compile(r'```.*?```', re.S)
INLINE = re.compile(r'`[^`]*`')
LATEX = re.compile(r'\$\$?.*?\$\$?', re.S)
URL = re.compile(r'https?://\S+|www\.\S+')
TOKEN = re.compile(r'[^\W\d_]+', re.U)

IDENTITY = [
    'chatgpt', 'chat gpt', 'gpt-3', 'gpt-4', 'gpt-5', 'openai', 'open ai', 'anthropic',
    'claude', 'gemma', 'gemini', 'llama', 'qwen', 'deepseek', 'mistral',
    'γλωσσικό μοντέλο', 'μοντέλο τεχνητής νοημοσύνης', 'ως τεχνητή νοημοσύνη',
    'as an ai', 'language model',
]

INFORMAL = [r'\bεσύ\b', r'\bεσένα\b', r'\bσου\b', r'\bσε σένα\b', r'\bδικό σου\b', r'\bδική σου\b']
FORMAL = [r'\bεσείς\b', r'\bεσάς\b', r'\bσας\b', r'\bδικό σας\b', r'\bδική σας\b']


def strip_noise(t):
    t = FENCE.sub(' ', t or '')
    t = INLINE.sub(' ', t)
    t = LATEX.sub(' ', t)
    t = URL.sub(' ', t)
    return t


def greek_ratio(t, drop_proper=False):
    """Greek share of letters. drop_proper=True removes Capitalised Latin tokens first:
    preserved proper names (band names, game titles, organisations) are CORRECT behaviour,
    and counting them sank 11/100 rows below an 0.85 floor in v1."""
    t = strip_noise(t)
    if drop_proper:
        t = re.sub(r'\b[A-Z][A-Za-z.\-]*\b', ' ', t)
    g = len(GREEK.findall(t)); l = len(LATIN.findall(t))
    return (g / (g + l)) if (g + l) else 0.0


def mixed_script_tokens(t):
    """Latin/Greek homoglyph pollution shows up as a token containing BOTH scripts.
    Legitimate English terms are whole-Latin tokens, so they do not trip this."""
    bad = []
    for tok in TOKEN.findall(strip_noise(t)):
        if GREEK.search(tok) and LATIN.search(tok):
            bad.append(tok)
    return bad


def final_sigma_errors(t):
    """sigma must be ς word-finally and σ elsewhere.

    v2: a token ending in σ immediately followed by an apostrophe is ELISION
    (σ' αγαπώ = σε αγαπώ; Δώσ' μου = Δώσε μου), not an error. v1 flagged 7/7 rows on
    this alone - every one a false positive.

    v3: two further false-positive sources found on real data --
      - a single Greek letter followed by '.' is an ABBREVIATION part (κ.σ. = tablespoon),
        not a word ending in sigma;
      - a medial ς inside an abnormally long token is a CONCATENATION artifact
        (τραγουδίστριατραγουδοποιόςηθοποιός), which here faithfully mirrors the English
        source's own 'Singersongwriteractress'. That is a source defect, not a Greek error,
        so it becomes an advisory rather than a hard flag.
    Across v1-v3 every sigma hard-flag on this run was a false positive.
    """
    errs, advis = [], []
    s = strip_noise(t)
    for m in TOKEN.finditer(s):
        tok = m.group()
        if not GREEK.search(tok):
            continue
        nxt = s[m.end():m.end() + 1]
        if tok.endswith('σ') and nxt not in ("'", '’', 'ʼ') and not (len(tok) == 1 and nxt == '.'):
            errs.append(('sigma_not_final', tok))
        if 'ς' in tok[:-1]:
            (advis if len(tok) > 20 else errs).append(('final_sigma_medial', tok))
    return errs, advis


def accented_caps(t):
    """Correct Greek drops the tonos in all-caps runs."""
    out = []
    for tok in TOKEN.findall(strip_noise(t)):
        if len(tok) > 2 and GREEK.search(tok) and tok.upper() == tok:
            if any(ud.combining(c) for c in ud.normalize('NFD', tok)):
                out.append(tok)
    return out


def latin_words(t):
    return [w for w in TOKEN.findall(strip_noise(t)) if LATIN.search(w) and not GREEK.search(w)]


def register_of(t):
    """HEURISTIC address-axis detector. Counts 2nd-person markers only."""
    low = (t or '').lower()
    inf = sum(len(re.findall(p, low)) for p in INFORMAL)
    frm = sum(len(re.findall(p, low)) for p in FORMAL)
    if inf == 0 and frm == 0:
        return 'neutral', inf, frm
    if inf and frm:
        return 'mixed', inf, frm
    return ('ενικός' if inf else 'πληθυντικός'), inf, frm


def load(pat, n=5):
    out = {}
    for i in range(n):
        p = f'{DIR}/out/{pat}{i}.json'
        if not os.path.exists(p):
            print(f'  MISSING {p}', file=sys.stderr); continue
        try:
            for r in json.load(open(p)):
                out[r['row_id']] = r
        except Exception as e:
            print(f'  BAD JSON {p}: {e}', file=sys.stderr)
    return out


def main():
    src = {json.loads(l)['row_id']: json.loads(l) for l in open(f'{DIR}/sample_100.jsonl')}
    s1 = load('stage1_b'); s2 = load('stage2_b')
    print(f'source={len(src)} stage1={len(s1)} stage2={len(s2)}')

    recs = []
    for rid, s in src.items():
        a, b = s1.get(rid), s2.get(rid)
        r = {'row_id': rid, 'category': s['category'],
             'has_stage1': bool(a), 'has_stage2': bool(b), 'gates': {}, 'flags': [], 'advisories': []}
        if not (a and b):
            r['flags'].append('MISSING_STAGE'); recs.append(r); continue

        p_el = a.get('prompt_el') or ''
        turns = b.get('assistant_turns_el') or []
        resp = b.get('response_el') or ''
        full = resp + '\n' + '\n'.join(turns)

        r['translation_class'] = a.get('translation_class')
        r['hazards'] = a.get('hazards') or []
        r['target_register'] = a.get('target_register')
        r['register_used'] = b.get('register_used')

        g = r['gates']
        g['nfc_prompt'] = ud.normalize('NFC', p_el) == p_el
        g['nfc_response'] = ud.normalize('NFC', full) == full
        g['greek_ratio_prompt'] = round(greek_ratio(p_el), 3)
        g['greek_ratio_response'] = round(greek_ratio(full), 3)
        g['greek_ratio_response_nonproper'] = round(greek_ratio(full, drop_proper=True), 3)
        g['mixed_script'] = mixed_script_tokens(p_el) + mixed_script_tokens(full)
        fs_p, fa_p = final_sigma_errors(p_el); fs_r, fa_r = final_sigma_errors(full)
        g['final_sigma'] = fs_p + fs_r
        g['final_sigma_advisory'] = fa_p + fa_r
        g['accented_caps'] = accented_caps(p_el) + accented_caps(full)
        g['polytonic'] = bool(POLYTONIC.search(strip_noise(p_el + full)))
        g['latin_words'] = sorted(set(latin_words(full)))[:12]
        g['identity_leak'] = [k for k in IDENTITY if k in full.lower()]
        det, inf, frm = register_of(full)
        g['register_detected'] = det; g['register_markers'] = {'informal': inf, 'formal': frm}

        en_ref = ' '.join(m['content'] for m in s['messages_en'] if m['role'] == 'assistant')
        g['len_ratio_el_en'] = round(len(full) / len(en_ref), 2) if en_ref else None

        # ---- flags
        # ---- hard flags: a defect in the data
        if not g['nfc_prompt'] or not g['nfc_response']: r['flags'].append('NOT_NFC')
        if g['greek_ratio_response_nonproper'] < 0.85: r['flags'].append('LOW_GREEK_RATIO')
        if g['mixed_script']: r['flags'].append('MIXED_SCRIPT')
        if g['final_sigma']: r['flags'].append('FINAL_SIGMA')
        if g['accented_caps']: r['flags'].append('ACCENTED_CAPS')
        if g['polytonic']: r['flags'].append('POLYTONIC')
        if g['identity_leak']: r['flags'].append('IDENTITY_LEAK')
        tr = a.get('target_register')
        if tr in ('ενικός', 'πληθυντικός') and det in ('ενικός', 'πληθυντικός') and det != tr:
            r['flags'].append('REGISTER_MISMATCH_VS_TARGET')

        # ---- advisories: route to a human, do NOT count as defects.
        # The address-axis heuristic cannot separate formal-you from plural-you: in the
        # Three Little Pigs row the wolf says "το σπίτι σου" to one pig and "το σπίτι σας"
        # to two, which v1 scored as mixed register. Number != formality.
        if g['final_sigma_advisory']: r['advisories'].append('CONCATENATED_TOKEN')
        if det == 'mixed': r['advisories'].append('REGISTER_REVIEW')
        if b.get('register_used') and det not in ('neutral', 'mixed') and b['register_used'] != det:
            r['advisories'].append('REGISTER_SELFREPORT_DISAGREES')
        if g['len_ratio_el_en'] is not None and not (0.5 <= g['len_ratio_el_en'] <= 2.5):
            r['advisories'].append('LENGTH_OUTLIER')
        recs.append(r)

    with open(f'{DIR}/out/gated.json', 'w') as f:
        json.dump(recs, f, ensure_ascii=False, indent=1)

    clean = [r for r in recs if not r['flags']]
    adv = [r for r in recs if r['advisories']]
    print(f'\nclean rows (no hard flag): {len(clean)}/{len(recs)}')
    print(f'rows with advisories (human review): {len(adv)}/{len(recs)}')
    print(f'clean AND no advisory: {len([r for r in recs if not r["flags"] and not r["advisories"]])}/{len(recs)}')
    fc = collections.Counter(f for r in recs for f in r['flags'])
    ac = collections.Counter(f for r in recs for f in r['advisories'])
    print('hard flags:', json.dumps(fc, ensure_ascii=False, indent=1) if fc else 'none')
    print('advisories:', json.dumps(ac, ensure_ascii=False, indent=1) if ac else 'none')
    cc = collections.Counter(r.get('translation_class') for r in recs if r.get('translation_class'))
    print('classes:', json.dumps(cc, ensure_ascii=False, indent=1))
    rr = collections.Counter(r.get('target_register') for r in recs if r.get('target_register'))
    print('target registers:', json.dumps(rr, ensure_ascii=False, indent=1))
    hz = collections.Counter(h for r in recs for h in (r.get('hazards') or []))
    print('hazards:', json.dumps(hz, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
