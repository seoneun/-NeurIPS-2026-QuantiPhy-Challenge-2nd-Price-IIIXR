import json, re, numpy as np, pandas as pd
R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv'); F = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
L = pd.read_csv(f'{T5}/labelled_rows.csv')
def tj(v, ts='v3ntmv'): return json.load(open(f'{R}/geo/out/tracks_test_{ts}/{v}.json'))
def objser(d, k):
    return np.array([(d['fidx'][i] / d['fps'], *o['st']['c'], *o['st']['rr'][:2]) for i, o in enumerate(d['objs'][str(k)]) if o and o.get('st')], float)
def dep(v, key, t):
    s = str(A[A.video == v].depth.dropna().iloc[0]); m = re.findall(rf't ?= ?{t}s,\s*distance_{key}_camera\s*=\s*([\d.]+)', s)
    return float(m[0] if m else re.findall(rf'distance_{key}_camera\s*=\s*([\d.]+)', s)[0])
def detmed(d, word, j):
    q = [q for q in d['queries'] if word in q['phrase']][0]; b = np.array([x for x in q['det_boxes'] if x], float)
    return np.median(b[:, j + 2] - b[:, j])
V = {}
d1, d2 = tj('simulation_0181'), tj('simulation_0182'); a1, a2 = objser(d1, 0), objser(d2, 0)
def afall(a):
    m = a[:, 0] <= 7 / 24 + 1e-9; return 2 * np.polyfit(a[m, 0], a[m, 2], 2)[0]
s1 = 9.8 / afall(a1); d0 = dep('simulation_0182', 'box', 0); ff = afall(a2) * d0 / 9.8; fdef = 50 / 36 * d2['W']; k = np.sqrt(ff / fdef)
w1, h1 = np.median(a1[:4, 3]), np.median(a1[:4, 4]); w2, h2 = np.median(a2[:4, 3]), np.median(a2[:4, 4])
hD2, hD3 = h1 * s1, h2 * d0 / ff
V[438] = w1 * s1 * k; V[439] = V[1427] = (hD2 + hD3) / 2 * k; V[1426] = np.sqrt(w1 * s1 * w2 * d0 / ff) * k
c2 = tj('simulation_0043'); kr = [q['obj'] for q in c2['queries'] if q['phrase'] == 'rock'][0]; ar = objser(c2, kr)
short = np.median(np.minimum(ar[:, 3], ar[:, 4]))
db = dep('simulation_0043', 'boat', 1); fC2 = detmed(c2, 'boat', 0) * db / 10.0
y1105 = float(L[(L.set == 'held') & (L.id0 == 1105)].y.iloc[0]); fC1 = detmed(tj('simulation_0024'), 'boat', 1) * dep('simulation_0024', 'boat', 1) / y1105
rockw = short * db / np.mean([fC2, fC1]); V[1170] = V[1697] = rockw
V[1748] = y1105
v53 = float(F.loc[2810, 'c26c'])
for i in (2929, 2937, 2945): V[i] = v53
if __name__ == '__main__':
    for i, v in sorted(V.items()): print(i, round(float(v), 4))
