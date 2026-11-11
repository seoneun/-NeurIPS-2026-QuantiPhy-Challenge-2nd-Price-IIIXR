import sys, json, ast, numpy as np, pandas as pd
sys.path.insert(0, 'analysis/v3/ag/r9_main/levers/kincons2')
from k2 import load, series
IN = 'geo/out/tracks_test_qv3ntmv_mq2off'
def lin(rows, lo, hi):
    S = series(rows); m = (S[:, 0] >= lo - 1e-9) & (S[:, 0] <= hi + 1e-9)
    vx = np.polyfit(S[m, 0], S[m, 1], 1)[0]; vy = np.polyfit(S[m, 0], S[m, 2], 1)[0]; return np.hypot(vx, vy)
def medh(rows, lo, hi): return float(np.median([r[5] for r in rows if lo - 1e-9 <= r[1] <= hi + 1e-9]))
def tire2d(v, t, Hs, wf):
    j = json.load(open(f'{IN}/{v}__all.json')); fps = j['fps']; R = []
    for f, e in zip(j['fidx'], j['objs']['1']):
        st = e.get('st') if isinstance(e, dict) else None
        if isinstance(st, str): st = ast.literal_eval(st)
        if st: b = st['box']; R.append((f / fps, st['c'][0], b[2] - b[0], b[3] - b[1], int(bool(st.get('border', 0)))))
    R = np.array(R); mw = np.median(R[:, 2]); R = R[(R[:, 4] == 0) & (R[:, 2] >= wf * mw)]; out = []
    for H in Hs:
        m = np.abs(R[:, 0] - t) <= H + 1e-9
        if m.sum() < 5 or R[m, 0].min() > t - H + 0.05: continue
        p = np.polyfit(R[m, 0] - t, R[m, 1], 2); s = 0.764 / np.median(R[m, 3])
        out.append(2 * p[0] * np.sign(p[1]) * s)
    return out
def person(v, ph, lo, hi, H, rad):
    _, _, R = load(v, ph, f'{IN}/{v}__all.json'); R2 = [r for r in R if not r[7]]
    vp = lin(R2, lo, hi); h = medh(R2, lo, hi); lat = vp * H / h
    return float(np.hypot(lat, rad))
OUT = [(2012, float(np.mean(tire2d('simulation_0151_segmented', 1.0, (0.125, 0.1667, 0.2083), 0.9))))]
for i, v, lo, hi, rad in [(1114, 'simulation_0026', 0.5, 1.5, (114.8356 - 114.8319) / 1), (1115, 'simulation_0027', 1.0, 2.0, (100.5996 - 100.5638) / 1),
                          (1692, 'simulation_0027_segmented', 0.0, 3.1, (100.5996 - 100.5638) / 1), (1224, 'simulation_0055', 1.0, 2.0, 0.0)]:
    OUT.append((i, person(v, 'person', lo, hi, 1.75, rad)))
T = pd.DataFrame(OUT, columns=['id0', 'value']); T['value'] = T.value.round(4)
T.to_csv('analysis/v3/calib/r9/levers/kincons2_test.csv', index=False)
