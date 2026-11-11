import io, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from trk import load, fitpar, R
FROZEN = os.path.join(HERE, 'frozen', 'owl_wheel_boxes.json')
W, H_IMG = 853, 480
TYRE_V = ['simulation_0218', 'simulation_0219']


def big_robot(v):
    (fps, d), o = load(v)
    b = o['big robot'].copy()
    b['x0'] = b.bx - b.bw / 2; b['x1'] = b.bx + b.bw / 2; b['y1'] = b.by + b.bh / 2
    return fps, b


def rerun_owl():
    import cv2, torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from torchvision.ops import nms
    mp = os.path.join(R, 'models/owlv2-base-patch16-ensemble')
    op = Owlv2Processor.from_pretrained(mp)
    om = Owlv2ForObjectDetection.from_pretrained(mp).to('cuda').eval()
    res = {}
    for v in TYRE_V:
        fps, b = big_robot(v)
        full = b[(b.x0 > 3) & (b.x1 < W - 3)]
        cap = cv2.VideoCapture(os.path.join(R, 'data/test_480p', v + '.mp4'))
        res[v] = []
        for _, r in full.iloc[::2].iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.f)); ok, im = cap.read()
            if not ok:
                continue
            y0 = int(r.y1 - 0.25 * r.bh) - 4; y1 = int(min(480, r.y1 + 4)); x0 = int(max(0, r.x0 - 4)); x1 = int(min(W, r.x1 + 4))
            crop = cv2.resize(im[y0:y1, x0:x1], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = op(text=[["a wheel", "a tire"]], images=rgb, return_tensors='pt').to('cuda')
            with torch.no_grad():
                out = om(**inp)
            side = max(crop.shape[:2])
            pr = op.image_processor.post_process_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[side, side]], device='cuda'))[0]
            if len(pr['boxes']) == 0:
                res[v].append(dict(f=int(r.f), boxes=[], scores=[], labels=[])); continue
            keep = nms(pr['boxes'].float(), pr['scores'].float(), 0.3)
            res[v].append(dict(f=int(r.f), boxes=[[float(x) for x in bx] for bx in pr['boxes'][keep].cpu().numpy()],
                               scores=[float(s) for s in pr['scores'][keep].cpu().numpy()], labels=[int(l) for l in pr['labels'][keep].cpu().numpy()]))
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    json.dump(res, open(FROZEN, 'w'))
    print('wrote', FROZEN)


if '--rerun' in sys.argv:
    rerun_owl()
    sys.exit(0)

import cv2
A = pd.read_csv(os.path.join(R, 'analysis/v3/calib/audit_test.csv'))
cur = pd.read_csv(os.path.join(R, 'submissions/pred_t007_id1.csv')).parsed_value.values
assert (A.id0.values == np.arange(len(A))).all() and len(cur) == len(A)
num = lambda txt, pat: float(re.search(pat, txt).group(1))
g4 = lambda i: float(f'{A.g[i]:.4g}')
out, const = {}, {}


def csv_hop(x):
    return float(pd.read_csv(io.StringIO(pd.DataFrame({'v': [float(x)]}).to_csv(index=False))).v[0])


def edge_fit(df, axis='x', t0=0, t1=99):
    c = 'bx' if axis == 'x' else 'by'; s = 'bw' if axis == 'x' else 'bh'; lim = W if axis == 'x' else H_IMG
    lo = df[c] - df[s] / 2; hi = df[c] + df[s] / 2
    fwd = (df[c].iloc[-1] - df[c].iloc[0]) > 0
    e = hi if fwd else lo
    ok = (e > 3) & (e < lim - 3) & (df.t >= t0) & (df.t <= t1)
    t = df.t[ok].values; y = e[ok].values
    p = np.polyfit(t, y, 2)
    rms = float(np.sqrt(np.mean((np.polyval(p, t) - y) ** 2)))
    return dict(a=abs(2 * p[0]), v0=p[1] * (1 if fwd else -1), n=int(ok.sum()), rms=rms, p=p, fwd=fwd)


for i in (639, 1941, 643):
    assert A.prior[i] == A.prior[639] and A.question[i] == A.question[639]
V_BIG = num(A.prior[639], r'big robot\s*=\s*([0-9.]+)\s*m/s')
Hh, Oo, vcen = {}, {}, {}
for v in TYRE_V:
    fps, b = big_robot(v)
    t = b.f.values / fps
    m = (b.x0 > 3) & (b.x1 < W - 3)
    vpx = np.polyfit(t[m.values], b['bx'][m].values, 1)[0]
    vcen[v] = vpx
    s = V_BIG / vpx
    full = b[(b.x0 > 3) & (b.x1 < W - 3)]
    cap = cv2.VideoCapture(os.path.join(R, 'data/test_480p', v + '.mp4'))
    diam = []
    for _, r in full.iterrows():
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.f)); ok, im = cap.read()
        if not ok:
            continue
        sy = im.shape[0] / 480.0
        y1 = int(r.y1 * sy); y0 = int((r.y1 - 0.22 * r.bh) * sy); x0 = int(max(0, r.x0)); x1 = int(min(W, r.x1))
        gg = cv2.medianBlur(cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY), 3)
        cs = cv2.HoughCircles(gg, cv2.HOUGH_GRADIENT, dp=1, minDist=25, param1=120, param2=22, minRadius=10, maxRadius=32)
        if cs is None:
            continue
        cs = cs[0]
        cs = cs[(cs[:, 1] > 0.25 * gg.shape[0]) & (cs[:, 1] < 0.85 * gg.shape[0])]
        diam += list(2 * cs[:, 2])
    q = np.percentile(np.array(diam), [10, 25, 50, 75, 90])
    Hh[v] = float(q[2] * s)
    print('')
VPX_OWL = round(float(vcen['simulation_0218']), 1)
assert all(abs(VPX_OWL / vcen[v] - 1) <= 0.02 for v in TYRE_V)
S = V_BIG / VPX_OWL
FZ = json.load(open(FROZEN))
for v in TYRE_V:
    ds = []
    for fr in FZ[v]:
        if not fr['boxes']:
            continue
        for bx in np.array(fr['boxes'], dtype=np.float32) / 3.0:
            w, h = bx[2] - bx[0], bx[3] - bx[1]
            if 15 <= h <= 50 and 0.75 <= w / h <= 1.33:
                ds.append((w + h) / 2)
    ds = np.array(ds)
    q = np.percentile(ds, [25, 50, 75])
    Oo[v] = float(q[1] * S)
    print('')
gm = {v: float(np.sqrt(Hh[v] * Oo[v])) for v in TYRE_V}
tyre = float(np.mean(list(gm.values())))
for i in (639, 1941, 643):
    out[i] = csv_hop(tyre)

(fps, _), o = load('simulation_0168')
v_b = num(A.prior[1394], r'bike\s*=\s*([0-9.]+)\s*m/s')
(t0s, d0), (t1s, d1) = [(float(a), float(b)) for a, b in re.findall(r't=([0-9.]+)s, distance_bike_camera = ([0-9.]+)m', A.depth[1394])]
assert t0s == 0.0
D_WHEEL, D_TIRE = 0.745, 0.782
m_wheel = float(np.mean([cur[385], cur[388]])); m_tire = float(cur[392])
for i in (385, 388):
    assert "diameter of the bike's wheel" in A.question[i] and A.video[i] in ('simulation_0165', 'simulation_0166')
assert "police car's tire" in A.question[392] and A.video[392] == 'simulation_0166'
assert abs(m_wheel / D_WHEEL - 1) <= 0.02 and abs(m_tire / D_TIRE - 1) <= 0.02
s0 = (d0 ** 2 - d1 ** 2 + (v_b * t1s) ** 2) / (2 * v_b * t1s); bb = np.sqrt(d0 ** 2 - s0 ** 2)
db = lambda x: np.sqrt(bb ** 2 + (s0 - v_b * x) ** 2)
wh, ti = o['wheel'], o['tire']
CX, CY = W / 2, H_IMG // 2


def van_speed(dw, dt):
    f = float((wh.bh[wh.t <= 1.5] * db(wh.t[wh.t <= 1.5]) / dw).median())
    tt = ti[ti.bh >= 20]
    dv = f * dt / tt.bh.values
    ray = np.stack([(tt.bx.values - CX) / f, (tt.by.values - CY) / f, np.ones(len(tt))], 1)
    P3 = ray / np.linalg.norm(ray, axis=1, keepdims=True) * dv[:, None]
    return f, float(np.hypot(np.polyfit(tt.t, P3[:, 0], 1)[0], np.polyfit(tt.t, P3[:, 2], 1)[0]))


f, v3d = van_speed(D_WHEEL, D_TIRE)
g_len = g4(1392); ref_len = float(np.mean([cur[391], cur[396]]))
assert 'length of the police car' in A.question[1392] and all('length of the police car' in A.question[i] for i in (391, 396))
r2 = cur[1394] * ref_len / g_len
new1394 = float(np.mean([v3d, r2]))
_, v3d_m = van_speed(m_wheel, m_tire)
new1394_m = float(np.mean([v3d_m, r2]))
print('')
const[1394] = (dict(D_WHEEL=(D_WHEEL, m_wheel), D_TIRE=(D_TIRE, m_tire)), new1394_m)
out[1394] = csv_hop(new1394_m if os.environ.get('R4_COMPUTED') == '1' else new1394)

assert A.prior[369] == A.prior[1548]
A_TURTLE = num(A.prior[369], r'turtle\s*=\s*([0-9.]+)\s*m/s')
(fps, _), o = load('simulation_0158')
st = A_TURTLE / edge_fit(o['turtle'], 'x')['a']
sh = o['shark']; ins = sh[(sh.bx - sh.bw / 2 > 3) & (sh.bx + sh.bw / 2 < W - 3)]
L158 = float(ins.bw.median() * st); g159 = g4(1548)
out[1548] = csv_hop(float(np.mean([L158, g159])))
fs = edge_fit(sh, 'x', 0, 1.6)
tq = num(A.question[369], r'at time ([0-9.]+)\s*s')
out[369] = csv_hop((fs['v0'] + fs['a'] * tq) * st)
print('')

A_PHX = num(A.prior[957], r'=\s*([0-9.]+)\s*m/s')
(fps, d), o = load('simulation_0271'); df = o['phoenix']
fe = edge_fit(df, 'x')
p0, v0, ac, rms = fitpar(df.t, -(df.cx - df.cx.iloc[0]))
sc = float(np.sqrt((A_PHX / fe['a']) * (A_PHX / ac)))
out[957] = csv_hop(float(df.bh.rolling(3, center=True).median().max() * sc))
print('')

with open(os.path.join(HERE, 'reproduce_out.csv'), 'w') as fh:
    fh.write('id0,value\n')
    for i in sorted(out):
        fh.write(f'{i},{float(out[i])!r}\n')
print('wrote', os.path.join(HERE, 'reproduce_out.csv'), out, 'constants:', const)
