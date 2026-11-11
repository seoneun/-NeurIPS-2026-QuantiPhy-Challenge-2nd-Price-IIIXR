import sys, re, json, io, contextlib
import numpy as np, pandas as pd
sys.path.insert(0, 'analysis/v3/ag/rul389')
from smoothstep_fit import fit, kin_model, track

BASE, OUT = sys.argv[1], sys.argv[2]
t = pd.read_csv(BASE)
new = {}


def run(path, stop=None):
    src = open(path).read()
    if stop:
        src = src[:src.index(stop)]
    ns = {'__file__': path, '__name__': 'line2'}; argv = sys.argv; sys.argv = [path]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, 'exec'), ns)
    sys.argv = argv
    return ns


UNIT = {'m': 1.0, 'cm': 0.01, 'mm': 0.001, 'km': 1000.0}
WINGED = re.compile(r'\b(bird|eagle|plane|airplane|jet|butterfly|phoenix|dragon|seagull|owl|bat|aircraft|hawk|crow|pigeon|parrot|'
                    r'duck|goose|swan|moth|airliner|glider|wing|wings)s?\b')
SKIP_SRC = re.compile(r'^(patch|var_ans_plain|mo_guard)')


def target(o):
    p = (o.get('target') or {}).get('phrase') or (o.get('q') or {}).get('target')
    return re.sub(r'^(the|a|an)\s+', '', (p or '').lower().strip())


def size(o, dim):
    q = o.get('q') or {}
    return q.get('kind') == 'size' and q.get('dim') == dim


det = pd.read_csv('analysis/v3/ag/r9_main/levers/line2/in/pred_t089_detail.csv', index_col=0)
raw = pd.read_csv('preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
P = {}
for line in open('geo/out/parse_test_v2.jsonl'):
    o = json.loads(line); P[int(o['id'])] = o
byv = {}
for i, o in P.items():
    byv.setdefault(o['video_id'], []).append(i)
for ids in byv.values():
    for i in ids:
        if not size(P[i], 'width') or i not in det.index:
            continue
        tp = target(P[i])
        sibs = [j for j in ids if j != i and j in det.index and size(P[j], 'length') and target(P[j]) == tp]
        if not sibs:
            continue
        j = sibs[0]; d = det.loc[i]
        k = UNIT.get(P[j]['q'].get('unit') or 'm', 1.0) / UNIT.get(P[i]['q'].get('unit') or 'm', 1.0)
        lg = float(det.loc[j, 'g']) * k
        g, cs, ans, src = float(d.g), float(d.cs), float(d.answer), str(d.answer_source)
        mw = float(raw.loc[i, 'meas_value']) if i in raw.index else np.nan
        ml = float(raw.loc[j, 'meas_value']) * k if j in raw.index else np.nan
        rg = mw / ml if (np.isfinite(mw) and np.isfinite(ml) and ml > 0) else np.nan
        if (cs >= 0.8 * lg and d['cat'] in ('S2', 'D2') and not WINGED.search(tp) and not (SKIP_SRC.search(src) or 'patch' in src)
                and np.isfinite(g) and g > 0 and np.isfinite(lg) and lg > 0 and 0.15 <= g / lg <= 0.7
                and not (np.isfinite(rg) and abs(np.log(rg / (g / lg))) > np.log(1.15))
                and not abs(np.log(g / ans)) <= 0.01):
            new[i] = g

D5 = 'analysis/v3/ag/rul389'


def cart(tags):
    vs, acc, sizes = [], [], []; fixed = None
    for tag in tags:
        video = 'simulation_0' + tag[1:4]
        k, x, _ = track(tag, video, drop=(7, 8, 9) if tag == 's151ft' else ())
        e = pd.read_csv(f'{D5}/frozen/tracks_test_{tag}/{video}_extents.csv'); e = e[e.box_h.notna() & (e.border.fillna(0) == 0)]
        e = e[np.abs(e.box_h / np.median(e.box_h) - 1) < 0.12]; sizes += [float(np.median(e.box_h)), float(np.median(e.box_w))]
        x = np.sign(x[-1] - x[0]) * x
        b = fit(k, x, fix=fixed) if tag == 's151rt' else fit(k, x)
        if tag == 's151ft':
            fixed = (b[1], b[2])
        v, a = kin_model(b, 24); vs.append(v); acc.append(a)
    s = 0.369 / np.mean(sizes)
    return float(np.mean(vs) * 24 * s), float(np.mean(acc) * 576 * s)


for vid, tags in (('simulation_0151', ['s151ft', 's151rt']), ('simulation_0152', ['s152ft']), ('simulation_0153', ['s153ft', 's153rt'])):
    v1, a1 = cart(tags)
    for i in t.index[(t.video_id == vid) & t.question.str.contains('of the blue car at time 1s')]:
        new[i] = a1 if 'acceleration' in t.question[i] else v1

ns = run('analysis/v3/ag/rul370/reproduce.py', stop='tlv = []')
for i, n in ((1238, '0058'), (1251, '0059'), (1275, '0061'), (1288, '0062'), (1301, '0063')):
    new[i] = float(np.median([e for a, c, e in ns['vals'] if a == n]))

sp = run('analysis/v3/ag/rul378/reproduce.py')['sp']
new[592] = float(np.median(sp[:2])); new[631] = float(np.median(sp[2:]))

rows = [(i, v) for i, v in sorted(new.items()) if abs(np.log(v / float(t.parsed_value[i]))) >= 1e-9]
pd.DataFrame(rows, columns=['id0', 'value']).to_csv(OUT, index=False)
