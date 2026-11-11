import os, sys
import numpy as np, pandas as pd
R = '@QP_ROOT@'
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
A = R + '/analysis/v3/ag/rul378'
sys.path.insert(0, A)
from objs import load, table
M, V = 'tracks_test_v2sc', 'tracks_test_v3ntmv'


def track(v, d, phrase):
    j = load(v, d)
    for q in j.get('queries', []):
        if q.get('obj') is not None and q['phrase'].lower().strip() == phrase:
            t = table(j); return t[t.obj == int(q['obj'])].sort_values('t').reset_index(drop=True)
    raise KeyError((v, d, phrase))


def linefit(g, t0=-1e9, t1=1e9, uncut=True):
    h = g[(g.t >= t0) & (g.t <= t1) & ((g.border == 0) if uncut else True)]
    return float(np.hypot(np.polyfit(h.t, h.cx, 1)[0], np.polyfit(h.t, h.cy, 1)[0])), len(h)


def scale(v, d, phrase, speed):
    vpx, n = linefit(track(v, d, phrase))
    return speed / vpx, vpx, n


def chain_mpp(v):
    p = pd.read_csv(R + '/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', usecols=['video_id', 'meas_mpp'])
    return float(np.nanmedian(pd.to_numeric(p[p.video_id == v].meas_mpp, errors='coerce')))


free = pd.read_csv(T5 + '/free_rows.csv').set_index('id0')
out = {}


def put(ids, val):
    for i in ids:
        out[i] = float(val)


a = track('simulation_0219', V, 'small robot'); b = track('simulation_0220', V, 'small robot')
v20, n20 = linefit(b); mpp_line = 0.82 / v20; mpp_ch = chain_mpp('simulation_0220')
mpp_small = float(np.sqrt(mpp_line * mpp_ch))
put([644], 0.82)
hpx = float(np.median([a.bh.median(), b.bh.median()]))
put([645, 647], hpx * mpp_small)

sp = []
for v in ['simulation_0214', 'simulation_0217']:
    for d in (M, V):
        mpp, vpx, _ = scale(v, d, 'black cat', 0.94); vd, n = linefit(track(v, d, 'dog'), 1.25, 1.75)
        sp.append(vd * mpp)
put([581, 592, 631], np.mean(sp))

mpp, vpx, _ = scale('simulation_0216', V, 'black cat', 0.94); c = track('simulation_0216', V, 'black cat')
put([608, 1924], c.rw.median() * mpp)

put([641], free.loc[637, 'c26c'])

sides = []
for v, d in [('simulation_0231', V), ('simulation_0232', M)]:
    j = load(v, d); t = table(j)
    ch = t[(t.cx > 700) & (t.bh > 110) & (t.bh < 140)]
    oid = ch.obj.mode()[0]; ch = t[t.obj == oid]
    mpp, vpx, _ = scale(v, V, 'black cat', 0.714); sides.append(ch.rh.median() * mpp)
ct = track('simulation_0233', V, 'black cat'); vct, nct = linefit(ct)
ch33 = track('simulation_0233', V, 'chair'); top = ch33.bw.median() * 0.714 / vct
put([758, 1966], np.sqrt(np.mean(sides) * top))

for i in sorted(out):
    print(i, repr(out[i]))
