import sys, json, re
import numpy as np, pandas as pd
import os
R = os.path.abspath(os.environ.get("QP_ROOT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.."))
SUB = sys.argv[1]
LO, HI = 0.6, 1.3
RX = re.compile(r"\b(?:height|tall)\b[^?]*?\bof (?:the |a )?(?:[a-z]+ )?(?:bicycle|bike)\b"
                r"(?!\s*(?:sign|rack|lane|path|wheel|tire|tyre|seat|saddle|rider|'s|stand|shed))", re.I)
UNIT = {'m': 1.0, 'meter': 1.0, 'meters': 1.0, 'cm': 100.0, 'centimeter': 100.0, 'centimeters': 100.0, 'mm': 1000.0,
        'millimeter': 1000.0, 'millimeters': 1000.0, 'ft': 3.28084, 'feet': 3.28084, 'foot': 3.28084, 'in': 39.3701,
        'inch': 39.3701, 'inches': 39.3701}


def unit_k(parse_unit, q):
    u = str(parse_unit or '').strip().lower()
    if u in UNIT:
        return UNIT[u]
    ql = q.lower()
    for pat, k in ((r'\b(?:cm|centimet)', 100.0), (r'\b(?:mm|millimet)', 1000.0), (r'\b(?:feet|foot|ft)\b', 3.28084), (r'\binch', 39.3701)):
        if re.search(pat, ql):
            return k
    return 1.0


LATE = re.compile(r'^(?:patch:|phys_bound|scene_prior|xrow_prior)')


def build(sp):
    D = pd.read_csv(f'{R}/submissions/{SUB}_detail.csv').set_index('Unnamed: 0')
    G = pd.read_csv(f'{R}/preds/{sp}_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv').set_index('Unnamed: 0').reindex(D.index)
    PU = {}
    for ln in open(f'{R}/geo/out/parse_{sp}_v2.jsonl'):
        x = json.loads(ln); PU[int(x['id'])] = (x.get('q') or {}).get('unit')
    P = {}
    for i in D.index:
        q = str(G.question[i])
        if not RX.search(q):
            continue
        k = unit_k(PU.get(int(i)), q); a = float(D.answer[i]); g = D.g[i]; c = D.cs[i]
        gin = pd.notna(g) and g > 0 and LO <= g / k <= HI
        cin = pd.notna(c) and c > 0 and LO <= c / k <= HI
        new = None
        if gin and not cin:
            new, why = float(g), 't148'
        elif cin and not gin:
            new, why = float(c), 't150'
        tag = 'unchanged' if new is None else None
        if new is not None and abs(new - a) <= 0.02 * abs(a):
            tag = 'unchanged (within 2%)'
        elif new is not None and abs(np.log(a / new)) <= np.log(1.15):
            tag = 't149'
        elif new is not None and LATE.match(str(D.answer_source[i])):
            tag = 'unchanged (later override)'
        if tag:
            continue
        P[str(int(i))] = {'value': float(new), 'why': f'rul058: bicycle height band [{LO},{HI}] m; {why} (was {a:.4g}, {D.answer_source[i]})'}
    json.dump(P, open(f'{R}/cache/tmp/ag_rul058_patch_{sp}.json', 'w'), indent=1)
    print('')


build('test')
