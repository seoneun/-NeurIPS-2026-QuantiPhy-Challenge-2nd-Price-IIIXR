import json, os, re
import numpy as np, pandas as pd, cv2

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
OUTD = os.path.dirname(os.path.abspath(__file__))
A = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
assert (A.id0.values == np.arange(len(A))).all()


def prior_speed(v, pattern):
    txt = A[A.video == v].prior.iloc[0].strip()
    m = re.fullmatch(pattern, txt)
    assert m
    return m


def tj(v, d='tracks_test_v2sc'):
    for dd in (d, 'tracks_test_v3ntmv', 'tracks_test'):
        p = f'{R}/geo/out/{dd}/{v}.json'
        if os.path.exists(p):
            return json.load(open(p))


def read_all(v):
    cap = cv2.VideoCapture(f'{R}/data/test_480p/{v}.mp4'); fr = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        fr.append(im)
    return fr, cap.get(cv2.CAP_PROP_FPS)


res, diag = {}, {}

v = A.video[208]
lift = float(prior_speed(v, r'the lifting speed of the charging post = ([0-9.]+)m/s').group(1))
js = tj(v); o = {q['phrase']: q['obj'] for q in js['queries']}['gray truncated cone']
st = [f['st'] for f in js['objs'][str(o)]]
cy = np.array([s['c'][1] for s in st]); d3 = np.abs(cy[3:] - cy[:-3]) / 3 * js['fps']
v_lift = float(np.median(d3[d3 > 0.5 * d3.max()])); mpp = lift / v_lift
bw = float(np.median([s['box'][2] - s['box'][0] for s in st]))
res[208] = round(bw * mpp, 4)
diag[208] = 't17'

for rid in (1019, 1022):
    v = A.video[rid]
    spd = float(prior_speed(v, r'speed of the column below the drone =\s*([0-9.]+)m/s').group(1))
    js = tj(v); fr, _ = read_all(v)
    o = [q['obj'] for q in js['queries'] if 'prior' in (q.get('roles') or [])]; assert len(o) == 1
    seq = [f['st'] for f in js['objs'][str(o[0])]]
    t = np.array(js['fidx']) / js['fps']; p = np.polyfit(t, [s['c'][1] for s in seq], 1); mpp = spd / abs(p[0])
    ws = []
    for i in range(6, 67, 4):
        x0, y0, x1, y1 = seq[i]['box']
        g = cv2.cvtColor(fr[i][max(0, y0 - 4):y1 + 4, max(0, x0 - 6):x1 + 6], cv2.COLOR_BGR2GRAY)
        for thr in (90, 120):
            runs = []
            for row in g < thr:
                best = cur = 0
                for q in row:
                    cur = cur + 1 if q else 0; best = max(best, cur)
                runs.append(best)
            runs = np.array(runs); prow = int(np.argmax(runs))
            col = np.where((runs >= 12) & (runs < 0.7 * runs.max()))[0]; col = col[col > prow]
            if len(col) >= 5:
                ws.append(float(np.median(runs[col[len(col) // 4: 3 * len(col) // 4 + 1]])))
    w = float(np.median(ws))
    res[rid] = round(w * mpp, 4)
    diag[rid] = f'raw {w * mpp!r}: column width {w} px (n {len(ws)}) x ({spd} m/s / {abs(p[0]):.3f} px/s)'


def blobs(im, bg, thr=45, amin=400):
    d = np.abs(im.astype(np.int16) - bg.astype(np.int16)).sum(2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    out = []
    for i in range(1, n):
        if st[i, 4] >= amin:
            out.append(dict(c=cen[i], area=st[i, 4], box=st[i, :4], gray=float(g[lab == i].mean())))
    return sorted(out, key=lambda b: -b['area'])


def person_tracks(v, pobj, pspd):
    fr, fps = read_all(v)
    bg = np.median(np.stack(fr[::3]), axis=0).astype(np.uint8)
    H, W = fr[0].shape[:2]
    tr = {0: [], 1: []}; last = None
    for i, im in enumerate(fr):
        bb = blobs(im, bg)[:2]
        if len(bb) < 2:
            continue
        if last is None:
            a, b = bb
        else:
            d00 = np.linalg.norm(bb[0]['c'] - last[0]) + np.linalg.norm(bb[1]['c'] - last[1])
            d01 = np.linalg.norm(bb[0]['c'] - last[1]) + np.linalg.norm(bb[1]['c'] - last[0])
            a, b = (bb[0], bb[1]) if d00 <= d01 else (bb[1], bb[0])
        tr[0].append((i, a)); tr[1].append((i, b)); last = (a['c'], b['c'])
    res_ = {}
    for k in (0, 1):
        ar = np.median([b['area'] for _, b in tr[k]])
        ok = [(i, b) for i, b in tr[k] if 0.7 * ar <= b['area'] <= 1.4 * ar and b['box'][0] > 2 and
              b['box'][0] + b['box'][2] < W - 2 and b['box'][1] > 2 and b['box'][1] + b['box'][3] < H - 2]
        t = np.array([i / fps for i, _ in ok]); cx = np.array([b['c'][0] for _, b in ok]); cy = np.array([b['c'][1] for _, b in ok])
        px = np.polyfit(t, cx, 1); py = np.polyfit(t, cy, 1)
        res_[k] = dict(px=px, py=py, sp=np.hypot(px[0], py[0]), gray=np.median([b['gray'] for _, b in tr[k]]), n=len(ok))
    kb = min(res_, key=lambda k: res_[k]['gray']); kw = 1 - kb
    kp = kb if pobj == 'black' else kw
    mpp = pspd / res_[kp]['sp']
    return res_, kb, kw, mpp


PT = {}
for rid in (67, 71, 73, 74):
    v = A.video[rid]
    m = prior_speed(v, r'speed of the person in the (black|white) shirt = ([0-9.]+)m/s')
    pobj, pspd = m.group(1), float(m.group(2))
    mt = re.search(r'black shirt and the person in the white shirt at ([0-9.]+)s in meters', A.question[rid]); assert mt
    t = float(mt.group(1))
    if v not in PT:
        PT[v] = person_tracks(v, pobj, pspd)
    rs, kb, kw, mpp = PT[v]
    pb = np.array([np.polyval(rs[kb]['px'], t), np.polyval(rs[kb]['py'], t)])
    pw = np.array([np.polyval(rs[kw]['px'], t), np.polyval(rs[kw]['py'], t)])
    dist = np.linalg.norm(pb - pw) * mpp
    res[rid] = round(float(f'{dist:.3f}'), 4)
    diag[rid] = (f'raw {float(dist)!r}: {v} t {t} s, prior {pobj} {pspd} m/s, mpp {mpp:.6f}, d_px {np.linalg.norm(pb - pw):.2f}, '
                 f'n black {rs[kb]["n"]} white {rs[kw]["n"]}')

order = [208, 1019, 1022, 67, 71, 73, 74]
with open(os.path.join(OUTD, 'reproduce_out.csv'), 'w') as f:
    f.write('id0,value\n')
    for i in order:
        f.write(f'{i},{float(res[i])!r}\n')
