import io, json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
G = pd.read_csv(f'{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
assert (G['Unnamed: 0'].values == np.arange(len(G))).all()
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
assert (A.id0.values == np.arange(len(A))).all()
out = {}


def csv_hop(x):
    return float(pd.read_csv(io.StringIO(pd.DataFrame({'v': [float(x)]}).to_csv(index=False))).v[0])

SCENE = [f'simulation_0{n}' for n in range(202, 210)]
s = A[A.video.isin(SCENE)].copy()
g = G.loc[s.id0.values]
assert (g.video_id.values == s.video.values).all()
s['geo'] = g.geo_value.values; s['mpp'] = g.meas_mpp.values; s['prior'] = g.ground_truth_prior.values
h = s[s.question.str.contains('height of the typewriter', case=False)]
d2 = h[h.mpp.notna()]
r1 = d2[d2.prior.str.contains('magnifying glass')].geo
r2 = d2[d2.prior.str.contains('typewriter')].geo
assert len(r1) == 2 and len(r2) == 2
route1, route2 = float(np.median(r1)), float(np.median(r2))
print('')
assert A.video[532] == 'simulation_0203' and 'height of the typewriter' in A.question[532]
out[532] = csv_hop(csv_hop(route2))


def box_track(v):
    j = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/{v}.json'))
    q = [q for q in j['queries'] if q.get('obj') is not None and q['phrase'] == 'tool box'][0]
    fr = j['objs'][str(q['obj'])]
    ok = [i for i, f in enumerate(fr) if isinstance(f, dict) and f.get('st')]
    t = np.array(j['fidx'])[ok] / j['fps']
    st = [fr[i]['st'] for i in ok]
    return j, t, np.array([s['c'] for s in st], float), np.array([sorted(s['rr'][:2], reverse=True) for s in st]), np.array([s['box'] for s in st])


G_ACC = 9.8
est = {}
for v in ('simulation_0183', 'simulation_0184'):
    j, t, c, rr, bx = box_track(v)
    rest = np.median(rr[t < 0.2], axis=0)
    m = t <= 0.29
    a_px = 2 * np.polyfit(t[m], c[m, 1], 2)[0]
    est[v] = dict(L_px=rest[0], H_px=rest[1], mpp_fall=G_ACC / a_px, box0=bx[0].tolist())
    print('')
assert np.abs(np.array(est['simulation_0183']['box0']) - np.array(est['simulation_0184']['box0'])).max() <= 8
rt = {k: float(np.mean([e[f'{k}_px'] * e['mpp_fall'] for e in est.values()])) for k in ('L', 'H')}
for i, dim in ((444, 'H'), (1432, 'H'), (1431, 'L')):
    assert A.video[i] in ('simulation_0183', 'simulation_0184') and (dim == 'H') == ('height' in A.question[i])
    out[i] = csv_hop(rt[dim])
print('tool box: L', repr(rt['L']), 'H', repr(rt['H']))

with open(os.path.join(HERE, 'reproduce_out.csv'), 'w') as fh:
    fh.write('id0,value\n')
    for i in sorted(out):
        fh.write(f'{i},{float(out[i])!r}\n')
print('wrote', os.path.join(HERE, 'reproduce_out.csv'), {i: out[i] for i in sorted(out)})
