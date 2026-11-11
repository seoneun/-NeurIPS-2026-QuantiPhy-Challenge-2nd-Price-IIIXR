import json, os, re
import numpy as np, pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
OUTD = os.path.dirname(os.path.abspath(__file__))
aud = pd.read_csv(os.path.join(R, 'analysis/v3/calib/audit_test.csv'))
aud['cur'] = pd.read_csv(os.path.join(R, 'submissions/pred_t007_id1.csv')).parsed_value.values
assert (aud.id0.values == np.arange(len(aud))).all()
A = aud.set_index('id0')
cur = A.cur.to_dict()


def row(video, qsub):
    s = aud[(aud.video == video) & aud.question.str.contains(qsub, regex=False)]
    assert len(s) == 1
    return int(s.id0.iloc[0])


def box(v, phrase, src, k=0):
    js = json.load(open(os.path.join(R, f'geo/out/{src}/{v}.json')))
    qs = [q for q in js['queries'] if q['phrase'] == phrase]
    o = qs[k]['obj']
    bs = [fr['st']['box'] for fr in js['objs'][str(o)] if isinstance(fr, dict) and fr.get('st') and fr['st'].get('box')
          and not fr['st'].get('border', 0)]
    b = np.median(np.array(bs, float), 0)
    return dict(w=b[2] - b[0], h=b[3] - b[1], n=len(bs), box=b)


def centre_at(v, phrase, src, t):
    js = json.load(open(os.path.join(R, f'geo/out/{src}/{v}.json')))
    o = [q for q in js['queries'] if q['phrase'] == phrase][0]['obj']
    i = js['fidx'].index(int(round(t * js['fps'])))
    return np.array(js['objs'][str(o)][i]['st']['c'], float)


res, diag = {}, {}

i3075 = row('internet_0044', 'height of the white cylinder')
c44 = box('internet_0044', 'white cylinder', 'tracks_test_v2sc')
res[3083] = c44['w'] / c44['h'] * cur[i3075]
diag[3083] = 't23'

i3150 = row('internet_0055', 'height of the walking person')
wk = box('internet_0055', 'walking person', 'tracks_test_v2sc')
s55 = cur[i3150] / wk['h']
bic = box('internet_0055', 'bicycle sign', 'tracks_test_v3ntmv')
res[3154] = bic['h'] * s55
res[3155] = bic['w'] * s55
diag[3154] = 't24'
diag[3155] = 't25'

i3116 = row('internet_0050', 'height of the person riding scooter')
rd50 = box('internet_0050', 'person riding scooter', 'tracks_test_v2sc')
s50 = cur[i3116] / rd50['h']
bb = box('internet_0050', 'basketball backboard', 'tracks_test_v2sc')
res[3121] = bb['w'] * s50
diag[3121] = 't26'

q = A.question[3256]
m = re.search(r'between ([0-9.]+)s and ([0-9.]+)s in (cm|m)\b', q); assert m
t0, t1, unit = float(m.group(1)), float(m.group(2)), m.group(3)
V = []
for j, s in aud[aud.video == A.video[3256]].iterrows():
    mv = re.search(r'velocity of the cup at ([0-9.]+)s in m/s', str(s.question))
    if mv and t0 <= float(mv.group(1)) <= t1:
        V.append((float(mv.group(1)), int(s.id0)))
V.sort()
assert [t for t, _ in V][0] == t0 and [t for t, _ in V][-1] == t1 and len(V) >= 2
kin = 0.0
for (ta, ia), (tb, ib) in zip(V[:-1], V[1:]):
    dt = round(tb - ta, 9)
    kin = kin + dt * (cur[ia] + cur[ib]) / 2
res[3256] = kin * {'cm': 100, 'm': 1}[unit]
cup = box(A.video[3256], 'cup', 'tracks_test_v3ntmv')
dpx = np.linalg.norm(centre_at(A.video[3256], 'cup', 'tracks_test_v3ntmv', t1) - centre_at(A.video[3256], 'cup', 'tracks_test_v3ntmv', t0))
i3250 = row(A.video[3256], 'height of the cup')
diag[3256] = ('t27')

order = [3083, 3154, 3155, 3121, 3256]
out = {i: round(float(res[i]), 4) for i in order}
with open(os.path.join(OUTD, 'reproduce_out.csv'), 'w') as f:
    f.write('id0,value\n')
    for i in order:
        f.write(f'{i},{float(out[i])!r}\n')
