import json, os
import numpy as np, pandas as pd
R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
TOOL5 = f'{R}/analysis/v3/ag/rul392/frozen' if R5FZ else f'{R}/cache/tmp/r5_tool'
FPS = 24.0
TD = 0.764


def objboxes(path, phrase):
    js = json.load(open(path))
    q = [q for q in js['queries'] if q['phrase'] == phrase][0]
    out = {}
    for i, f in enumerate(js['objs'][str(q['obj'])]):
        st = f.get('st') if isinstance(f, dict) else None
        if st and st.get('box'):
            out[js['fidx'][i]] = list(st['box']) + [1 if st.get('border') else 0]
    return out, js


def ext(tag, video):
    e = pd.read_csv(f'{TOOL5}/tracks_test_{tag}/{video}_extents.csv')
    return e[e.ok == 1]


def smooth(k, y, kq, mode):
    k = np.asarray(k, float); y = np.asarray(y, float)
    if mode[0] == 'poly':
        return np.polyval(np.polyfit(k, y, mode[1]), kq)
    out = []
    for q in np.atleast_1d(kq):
        m = np.abs(k - q) <= mode[1]
        out.append(np.polyval(np.polyfit(k[m], y[m], 2), q))
    return np.array(out)


def kin(kk, P, mode, ks=(12, 24)):
    res = {}
    for k in ks:
        q = np.array([k - 2, k - 1, k], float)
        S = np.stack([smooth(kk, P[:, j], q, mode) for j in range(P.shape[1])], 1)
        v1 = np.linalg.norm(S[2] - S[1]) * FPS; v0 = np.linalg.norm(S[1] - S[0]) * FPS
        res[k] = (v1, (v1 - v0) * FPS)
    return res


MODES = [('poly', 3), ('poly', 4), ('loc', 5)]
print('')
fr = pd.read_csv(f'{T5}/free_rows.csv')
for v in ['simulation_0150', 'simulation_0151']:
    print(f'  {v} depth_info: {fr[fr.video == v].depth.iloc[0]!r}')
t50, _ = objboxes(f'{R}/geo/out/tracks_test_v2sc/simulation_0150.json', 'tire')
t51, _ = objboxes(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0151.json', 'tire')
c50, _ = objboxes(f'{R}/geo/out/tracks_test_v2sc/simulation_0150.json', 'car')
c51, _ = objboxes(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0151.json', 'yellow car')
dt = [max(abs(np.array(t50[k][:4]) - np.array(t51[k][:4]))) for k in t50 if k in t51]
dc = [max(abs(np.array(c50[k][:4]) - np.array(c51[k][:4]))) for k in c50 if k in c51 and k != 21]
print('')

print('')
K = sorted(t50)
tire = np.array([[k, (t50[k][0] + t50[k][2]) / 2, t50[k][2] - t50[k][0], t50[k][3] - t50[k][1]] for k in K])
full = tire[(tire[:, 2] >= 78) & (tire[:, 3] >= 78)]
D50 = float(np.median(np.r_[full[:, 2], full[:, 3]])); mpp50 = TD / D50
print('')
front = np.array([[k, c50[k][0]] for k in sorted(c50) if 1 <= k <= 27])
rear = np.array([[k, c50[k][2]] for k in sorted(c50) if 22 <= k <= 40])
tx = dict(zip(full[:, 0], full[:, 1]))
def align(e):
    m = np.array([k in tx for k in e[:, 0]])
    A = np.c_[e[m, 1], np.ones(m.sum())]; b = np.array([tx[k] for k in e[m, 0]])
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = A @ p - b
    return p, r
pf, rf = align(front); pr, rr = align(rear)
print('')
comp = {}
for k, x in front: comp.setdefault(k, []).append(pf[0] * x + pf[1])
for k, x in rear: comp.setdefault(k, []).append(pr[0] * x + pr[1])
for k, x in full[:, :2]: comp.setdefault(k, []).append(x)
k50 = np.array(sorted(comp)); x50 = np.array([np.mean(comp[k]) for k in k50])
s50 = (x50[0] - x50) * mpp50
print(f'  composite frames {int(k50[0])}-{int(k50[-1])}; s(k): ' + ' '.join(f'{int(k)}:{s:.2f}' for k, s in zip(k50[::3], s50[::3])))
raw = np.diff(full[:, 1]) * -mpp50 * FPS
print('  raw tire backward 1-frame speeds: ' + ' '.join(f'{int(k)}:{v:.2f}' for k, v in zip(full[1:, 0], raw)))
res50 = {}
for src, (kk, ss) in {'tire': (full[:, 0], (full[0, 1] - full[:, 1]) * mpp50), 'composite': (k50, s50)}.items():
    for mode in MODES:
        r = kin(kk, ss[:, None], mode); res50[(src, mode)] = r
        print(f'  0150 {src:9s} {mode}: v(0.5) {r[12][0]:.2f} a(0.5) {r[12][1]:.2f} v(1) {r[24][0]:.2f} a(1) {r[24][1]:.2f}')

print('')
e = ext('r149t', 'simulation_0149')
e = e[e.border == 0]
W, H, cx, cy = 853, 480, 426.5, 240.0
k49 = e.frame.values.astype(float); xc = ((e.x0 + e.x1) / 2).values; yc = ((e.y0 + e.y1) / 2).values; h = e.box_h.values
fs = []
for i in range(4):
    ray = 14.244 * h[i] / TD
    fs.append(np.sqrt(ray ** 2 - (xc[i] - cx) ** 2 - (yc[i] - cy) ** 2))
f49 = float(np.median(fs))
print('')
sl, ic = np.polyfit(h, yc, 1)
print('')
def P49(f):
    ray = np.c_[xc - cx, yc - cy, np.full(len(xc), f)]
    nr = np.linalg.norm(ray, axis=1)
    return (TD * nr / h)[:, None] * ray / nr[:, None]
for f in [f49, 0.9 * f49, 1.1 * f49, 50 / 36 * W]:
    P = P49(f)
    rs = [kin(k49, P, m) for m in MODES]
    print(f'  f {f:7.1f}: ' + ' | '.join(f'{m[0]}{m[1]} v(.5) {r[12][0]:.2f} a(.5) {r[12][1]:.2f} v(1) {r[24][0]:.2f} a(1) {r[24][1]:.2f}' for m, r in zip(MODES, rs)))
P = P49(f49)
Ps = np.stack([smooth(k49, P[:, j], k49, ('poly', 4)) for j in range(3)], 1)
s49 = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Ps, axis=0), axis=1))]
d0 = np.linalg.norm(Ps[0]); print('')
u = Ps[-1] - Ps[0]; u /= np.linalg.norm(u)
dev = np.linalg.norm((Ps - Ps[0]) - np.outer((Ps - Ps[0]) @ u, u), axis=1)
print('')

print('')
s50f = lambda kq: np.interp(kq, k50, smooth(k50, s50, k50, ('poly', 4)))
for shift in [-2, -1, 0, 1, 2]:
    m = (k49 + shift >= k50[0]) & (k49 + shift <= k50[-1])
    A = np.c_[s50f(k49[m] + shift), np.ones(m.sum())]
    p, *_ = np.linalg.lstsq(A, s49[m], rcond=None); r = A @ p - s49[m]
    print('')
m = (k49 >= k50[0])
A = np.c_[s50f(k49[m])]; c = np.mean(s49[m] - s50f(k49[m]))
r = s49[m] - s50f(k49[m]) - c
print('')
for deg in (2,):
    pp = np.polyfit(k49, s49, deg); rc = np.polyval(pp, k49) - s49
    print('')

print('')
c52, _ = objboxes(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0152.json', 'yellow car')
t52, _ = objboxes(f'{R}/geo/out/tracks_test_v3ntmv/simulation_0152.json', 'tire')
fe = np.array([[k, c52[k][2]] for k in sorted(c52) if 15 <= k <= 28])
tr = np.array([[k, (t52[k][0] + t52[k][2]) / 2, t52[k][3] - t52[k][1]] for k in sorted(t52) if 28 <= k <= 36])
D52 = float(np.median(tr[:, 2])); mpp52 = TD / D52
for name, a in [('front edge 15-28', fe), ('rear tire 28-36', tr[:, :2])]:
    A = np.c_[s50f(a[:, 0]), np.ones(len(a))]
    p, *_ = np.linalg.lstsq(A, a[:, 1], rcond=None); r = A @ p - a[:, 1]
    q2 = np.polyfit(a[:, 0], a[:, 1], 2); rq = np.polyval(q2, a[:, 0]) - a[:, 1]
    print(f'  0152 {name}: x52 = A + B s50, B {p[0]:.1f} px/m, rms {r.std():.2f} px (motion {abs(a[-1,1]-a[0,1]):.0f} px); quadratic-alone rms {rq.std():.2f} px')
print(f'  0152 rear tire diameter {D52:.1f} px; tire px speed over frames 28-36: ' + ' '.join(f'{v:.1f}' for v in np.diff(tr[:, 1])))
vt = np.polyfit(tr[:, 0], tr[:, 1], 1)[0] * mpp52 * FPS
print('')
fe_v = np.diff(fe[:, 1]); print('  0152 front edge px/frame 16-28: ' + ' '.join(f'{v:.0f}' for v in fe_v))
r52 = (fe[-1, 1] - fe[-3, 1]) / 2 / np.polyfit(tr[:, 0], tr[:, 1], 1)[0]
q = np.polyfit(fe[:, 0], fe[:, 1], 2)
v24 = (np.polyval(q, 24) - np.polyval(q, 23)) / r52 * mpp52 * FPS
a24 = ((np.polyval(q, 24) - np.polyval(q, 23)) - (np.polyval(q, 23) - np.polyval(q, 22))) / r52 * mpp52 * FPS * FPS
print('')

print('')
for v, tag, hc, yh in [('simulation_0149', 'r149mk', 0.382 + sl * TD, ic), ('simulation_0150', 'r150mk', None, 240.0)]:
    m = ext(tag, v); m = m[(m.border == 0) & (m.box_h >= 40)]
    yt, yb = m.y0.median(), m.y1.median()
    if hc is None:
        yb_t = np.median([t50[k][3] for k in full[:, 0]]); hc = (yb_t - yh) * TD / D50
    Hm = (yb - yt) / (yb - yh) * hc
    print('')

print('')
def vprof(kk, P, ks, hw=4):
    out = {}
    for k in ks:
        q = np.array([k - 1, k], float)
        if k - 1 - hw < kk[0] - 1 or k + hw > kk[-1] + 1:
            continue
        S = np.stack([smooth(kk, P[:, j], q, ('loc', hw)) for j in range(P.shape[1])], 1)
        out[k] = np.linalg.norm(S[1] - S[0]) * FPS
    return out
v49 = vprof(k49, P49(f49), range(2, 36))
v50t = vprof(full[:, 0], ((full[0, 1] - full[:, 1]) * mpp50)[:, None], range(2, 36))
v50f = vprof(front[:, 0], ((front[0, 1] - front[:, 1]) * mpp50 * pf[0])[:, None], range(2, 36))
v52t = vprof(tr[:, 0], ((tr[:, 1] - tr[0, 1]) * mpp52)[:, None], range(2, 37), hw=3)
v52f = vprof(fe[:, 0], ((fe[:, 1] - fe[0, 1]) * mpp52 / r52)[:, None], range(2, 37))
print('')
for k in range(2, 37):
    g = lambda d: f'{d[k]:6.2f}' if k in d else '     -'
    print(f' {k:2d} | {g(v49)} | {g(v50t)} | {g(v50f)} | {g(v52f)} | {g(v52t)}')
v50 = {k: np.mean([d[k] for d in (v50t, v50f) if k in d]) for k in set(v50t) | set(v50f)}
for s in (-2, -1, 0, 1, 2, 3):
    ks = [k for k in v49 if k + s in v50]
    d = np.array([v49[k] - v50[k + s] for k in ks]); r = np.array([v49[k] / v50[k + s] for k in ks])
    print('')
