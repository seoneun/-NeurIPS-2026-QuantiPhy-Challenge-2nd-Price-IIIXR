import io, json, os, re
from decimal import Decimal
import cv2, numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(D, '..', '..', '..', '..'))
AUD = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
BEST = pd.read_csv(f'{R}/submissions/pred_t007_id1.csv')
bv = dict(zip(AUD.id0, BEST.parsed_value.values))
A = AUD.set_index('id0')


def prior_num(i, word):
    return re.search(word + r'[^=]*=\s*([0-9.]+)\s*m\b', str(A.loc[i, 'prior'])).group(1)


def load(v):
    for d in ('tracks_test_v2sc', 'tracks_test_v3ntmv', 'tracks_test'):
        p = f'{R}/geo/out/{d}/{v}.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if 'objs' in j:
                return d, j
    return None, None


def frame_table(j):
    rows = []
    for o, frs in j['objs'].items():
        for pos, fr in enumerate(frs):
            st = fr.get('st') if isinstance(fr, dict) else None
            if not st or not st.get('box'):
                continue
            x0, y0, x1, y1 = st['box']
            rr = st.get('rr') or [np.nan, np.nan, np.nan]
            rows.append(dict(obj=int(o), f=j['fidx'][pos], cx=st['c'][0], cy=st['c'][1], x0=x0, y0=y0, x1=x1, y1=y1,
                             bw=x1 - x0, bh=y1 - y0, rw=rr[0], rh=rr[1], ra=rr[2], area=st.get('area'),
                             border=st.get('border')))
    return pd.DataFrame(rows)


_T, FPS = {}, {}


def table(v):
    if v not in _T:
        d, j = load(v)
        buf = io.StringIO(); frame_table(j).to_csv(buf, index=False); buf.seek(0)
        _T[v] = pd.read_csv(buf); FPS[v] = j['fps']
    return _T[v]


def trk(v, o):
    d = table(v); return d[d.obj == o].sort_values('f').set_index('f')


out = {}
rocket_m = float(prior_num(340, 'rocket'))
v1 = []
for v in ('simulation_0145', 'simulation_0145_segmented'):
    g = trk(v, 1); full = g[(g.border == 0) & (g.rw > 95)]; mpp = rocket_m / full.rw.median()
    fps = FPS[v]; assert fps == 24
    k1 = int(round(1.0 * fps))
    s = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(full[['cx', 'cy']].values, axis=0).T))])
    p = np.polyfit(full.index.values, s, 2)
    fit = (np.polyval(p, k1) - np.polyval(p, k1 - 1)) * fps * mpp
    te = g[['x1', 'y1']]
    tail = np.mean([np.hypot(*(te.loc[k].values - te.loc[k - 1].values)) * fps * mpp for k in (k1 - 1, k1, k1 + 1)])
    v1.append((fit, tail))
r1 = np.mean([a[0] for a in v1]); r2 = np.mean([a[1] for a in v1]); vnew = np.mean([r1, r2])
print('')
out[340] = out[1850] = float(vnew)

sky_mm = float(Decimal(prior_num(499, 'skynova')) * 1000)
mac = json.load(open(f'{R}/patches/ag_rul312_patch_test.json'))['499']['why']
old_px = float(re.search(r'prior [0-9.]+ mm / ([0-9.]+) ?px', mac).group(1))
VAL = pd.read_csv(f'{R}/data/val/validation_dataset.csv')
lab = VAL[(VAL.video_id == 'simulation_0196') & (VAL.question.str.contains('plunge tube'))].ground_truth_posterior
assert len(lab) == 1
tube_mm = float(Decimal(str(lab.iloc[0]).strip()) * 1000)
cap = cv2.VideoCapture(R + '/data/test_480p/simulation_0197.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, 25)
im = cap.read()[1]; hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
T97 = table('simulation_0197')
white = ((hsv[..., 2] > 190) & (hsv[..., 1] < 80)).astype(np.uint8)
orange = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 25) & (hsv[..., 1] > 70)).astype(np.uint8)
n, lab_, st, cen = cv2.connectedComponentsWithStats(white)
comps = [tuple(int(z) for z in st[i]) for i in range(1, n) if 15 <= st[i][4] <= 2000 and st[i][1] >= 330 and
         orange[max(0, st[i][1] - 4):st[i][1] + st[i][3] + 4, max(0, st[i][0] - 4):st[i][0] + st[i][2] + 4].mean() >= 0.25]
comps.sort(key=lambda c: c[0])
col = [c for c in comps if c[0] < comps[0][0] + comps[0][2] and c[0] + c[2] > comps[0][0]]
cx0 = min(c[0] for c in col); cx1 = max(c[0] + c[2] for c in col); wcol = cx1 - cx0
b = T97[T97.obj == 11]; bx0, bx1 = b.x0.median(), b.x1.median(); bc, br = (bx0 + bx1) / 2, (bx1 - bx0) / 2
sphi = abs((cx0 + cx1) / 2 - bc) / br; cphi = np.sqrt(1 - sphi ** 2)
mppA = sky_mm / (wcol / cphi)
g = T97[T97.obj == 9].sort_values('f'); P = g[['cx', 'cy']].values
u = (P[-1] - P[0]) / np.linalg.norm(P[-1] - P[0]); nrm = np.array([-u[1], u[0]])
liquid = ((hsv[..., 0] >= 5) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 50)).astype(np.uint8)
ws = []
for s_ in np.linspace(0.1, 0.9, 9):
    c0 = P[0] + s_ * (P[-1] - P[0]); ts = np.arange(-40, 40.5, 0.5)
    pts = (c0[None, :] + ts[:, None] * nrm[None, :]).round().astype(int)
    vv = liquid[pts[:, 1].clip(0, 479), pts[:, 0].clip(0, 479)]; i0 = len(ts) // 2
    lo = hi = i0
    while lo > 0 and vv[lo - 1]: lo -= 1
    while hi < len(ts) - 1 and vv[hi + 1]: hi += 1
    ws.append((hi - lo + 1) * 0.5)
wB = float(np.median(ws)); mppB = tube_mm / wB
mpp = (mppA + mppB) / 2; k = mpp / (sky_mm / old_px)
print('')
for i in (499, 500, 501, 502, 503):
    out[i] = float(bv[i] * k)

puck_m = float(prior_num(284, 'puck'))
puck = trk('simulation_0126', 1); mpp = puck_m / puck.rw.median()
cap = cv2.VideoCapture(R + '/data/test_480p/simulation_0126.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
hsv = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2HSV)
m = (((hsv[..., 0] < 10) | (hsv[..., 0] > 170)) & (hsv[..., 1] > 90) & (hsv[..., 2] > 60)).astype(np.uint8)
m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
n, lab_, st, cen = cv2.connectedComponentsWithStats(m)
bl = sorted([(cen[k_][0], cen[k_][1], st[k_][2], st[k_][3], st[k_][4]) for k_ in range(1, n) if st[k_][4] > 800], key=lambda z: z[1])
top, right = bl[0], bl[-1]
dmm = np.hypot(top[0] - right[0], top[1] - right[1]) * mpp
m4 = trk('simulation_0126', 4); m2 = trk('simulation_0126', 2)
dtrk = np.hypot(m4.cx.iloc[-1] - m2.cx.iloc[-1], m4.cy.iloc[-1] - m2.cy.iloc[-1]) * mpp
print('')
out[284] = float(dmm)

car_m = float(prior_num(95, 'yellow car'))
same = {i: A[(A.video == 'simulation_0082') & (A.question.str.strip() == A.loc[i, 'question'].strip())].index[0] for i in (94, 95)}
a = trk('simulation_0083', 0); b = trk('simulation_0083', 1)
L = car_m * b.rw.median() / a.rw.median()
v82 = bv[same[95]]
out[95] = float((L + round(v82, 3)) / 2)
out[94] = float(bv[same[94]])
print('')



def collected(x):
    return float(pd.read_csv(io.StringIO(f'v\n{float(x)!r}\n')).v.iloc[0])


with open(os.path.join(D, 'reproduce_out.csv'), 'w') as f:
    f.write('id0,value\n')
    for i in (340, 1850, 499, 500, 501, 502, 503, 284, 95, 94):
        f.write(f'{i},{collected(out[i])!r}\n')
for i in (340, 1850, 499, 500, 501, 502, 503, 284, 95, 94):
    print(i, 'computed', repr(out[i]), 'written', repr(collected(out[i])))
