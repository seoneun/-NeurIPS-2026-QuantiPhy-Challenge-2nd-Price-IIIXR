import sys, os, json, numpy as np, pandas as pd
sys.path.insert(0, 'analysis/v3/ag/r9_main/levers/zdepth')
from zdlib import keys, load, centres, obj_for, at, P, T24, T25
IN = 'geo/out/tracks_test_v3ntmv'
D = pd.read_parquet('data/test/test_dataset.parquet')
SPEC = []
for i in (2710, 2717): SPEC.append((i, 'white cup', ('white cup', None), 'pink bottle', ('pink bottle', None), T24, 1, 100))
for i in (2723, 2730, 2737): SPEC.append((i, 'white ball', ('ball', 1.0), 'white ball', ('ball', 1.5), T24, 1, 100))
for i in (2720, 2727, 2734): SPEC.append((i, 'white ball', ('ball', 1.0), 'white ball', ('ball', 1.5), T24, 0.5, 100))
for i in (2508, 2512, 2516): SPEC.append((i, 'red ball', ('ball', 0.4), 'red ball', ('ball', 0.6), T25, 0.2, 100))
for i in (2774, 2777, 2780): SPEC.append((i, 'yoga ball', ('yoga ball', 1.0), 'yoga ball', ('soccer ball', 1.5), T25, 1, 1))
for i in (2775, 2778, 2781): SPEC.append((i, 'yoga ball', ('soccer ball', 1.5), 'person', ('person', None), T25, 1, 1))
def track(vid):
    f = f'{IN}/{vid}.json'
    return json.load(open(f)) if os.path.exists(f) else load(vid)[1]
te = []
for i, oa, ka, ob, kb, tm, div, u in SPEC:
    r = D.iloc[i]; K = keys(r.depth_info); d = track(r.video_id); W, H = d['W'], d['H']
    kv = lambda nm, t: [k['z'] for k in K if k['name'] == nm and (k['t'] == t if t is not None else k['t'] is None)][0]
    pts = []
    for ob_, (kn, kt) in [(oa, ka), (ob, kb)]:
        o = obj_for(d, ob_)[0]; fi, c, b = centres(d, o)
        uv = at(fi, c, kt, tm) if kt is not None else np.nanmedian(c, 0); pts.append((uv, kv(kn, kt)))
    te.append(dict(id0=i, value=float(np.linalg.norm(P(W, H, *pts[1][0], pts[1][1], 'Z') - P(W, H, *pts[0][0], pts[0][1], 'Z')) / div * u)))
pd.DataFrame(te).to_csv('analysis/v3/calib/r9/levers/zdepth_test.csv', index=False)
