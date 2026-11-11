import json, os, re
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
OUTD = os.path.dirname(os.path.abspath(__file__))
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
assert (A.id0.values == np.arange(len(A))).all()


def track(v, d, phrase=None, role=None):
    j = json.load(open(f'{R}/geo/out/{d}/{v}.json')); out = []
    for q in j['queries']:
        if (phrase and q['phrase'] != phrase) or (role and role not in (q.get('roles') or [])): continue
        o = j['objs'].get(str(q['obj'])) or []
        out.append([(k * j['stride'] / j['fps'], e['st']) for k, e in enumerate(o) if e and e.get('st')])
    return out, j['fps']


def fall_mpp(v, phrase, g):
    (P,), fps = track(v, 'tracks_test_v3ntmv', phrase=phrase)
    T = np.array([p[0] for p in P]); y = np.array([p[1]['c'][1] for p in P]); top = np.array([p[1]['box'][1] for p in P])
    yr = np.median(y[-10:]); imp = int(np.argmax(y >= yr - 2))
    m = (np.arange(len(T)) < imp) & (top > 1)
    c = np.polyfit(T[m], y[m], 2)
    return g / (2 * c[0]), dict(n=int(m.sum()), a_px=2 * c[0])


def static_box(v, phrase, d='tracks_test_v3ntmv'):
    tr, fps = track(v, d, phrase=phrase)
    P = max(tr, key=len); C = np.array([p[1]['c'] for p in P]); yr = np.median(C[-10:, 1])
    rest = np.abs(C[:, 1] - yr) < 3
    B = np.array([p[1]['box'] for p in P])[rest]; RR = np.array([p[1]['rr'][:2] for p in P])[rest]
    return np.median(B[:, 2] - B[:, 0]), np.median(B[:, 3] - B[:, 1]), np.median(RR.max(1)), float(np.median(C[rest, 0]))


RENDERS = ['simulation_0257', 'simulation_0258', 'simulation_0259', 'simulation_0260']
for v in RENDERS:
    assert A[A.video == v].question.str.contains('of the plant').any()
assert {A.video[i] for i in (908, 889, 886)} <= set(RENDERS)
est = {}
for v in RENDERS:
    txt = A[A.video == v].prior.iloc[0]
    m = re.fullmatch(r'acceleration of the (.+?) before [0-9.]+s = ([0-9.]+)m/s\^2', txt.strip())
    assert m
    obj, g = m.group(1), float(m.group(2))
    mpp, info = fall_mpp(v, obj, g)
    pw, phh, _, px = static_box(v, 'plant')
    js = json.load(open(f'{R}/geo/out/tracks_test_v2sc/{v}.json'))
    ph = [q['phrase'] for q in js['queries'] if q['phrase'].endswith('phone')]
    assert len(set(ph)) == 1
    _, _, rrL, fx = static_box(v, ph[0], d='tracks_test_v2sc')
    est[v] = dict(mpp=mpp, plant_w=pw * mpp, plant_h=phh * mpp, phone_w=rrL * mpp,
                  diag=f'{v[-4:]}: prior {obj} g {g}, fit n {info["n"]} a_px {info["a_px"]:.2f}, mpp {mpp:.6f}, '
                       f'plant box {pw}x{phh} px, {ph[0]} rr long {rrL} px')
med = {k: float(np.median([e[k] for e in est.values()])) for k in ('plant_w', 'plant_h', 'phone_w')}
res = {908: med['plant_w'], 889: med['plant_h'], 886: med['phone_w']}
out = {i: round(x, 5) for i, x in res.items()}

with open(os.path.join(OUTD, 'reproduce_out.csv'), 'w') as f:
    f.write('id0,value\n')
    for i in (908, 889, 886):
        f.write(f'{i},{float(out[i])!r}\n')
for e in est.values():
    print(e['diag'], '| w', round(e['plant_w'], 5), 'h', round(e['plant_h'], 5), 'phone', round(e['phone_w'], 5))
