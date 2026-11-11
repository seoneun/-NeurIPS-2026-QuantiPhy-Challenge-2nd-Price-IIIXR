import json, cv2, numpy as np, pandas as pd
D = 'geo/out/tracks_test_v3ntmv'


def run(v):
    d = json.load(open(f'{D}/{v}.json')); ph = [q['phrase'] for q in d['queries']]; fps = d['fps']
    c = cv2.VideoCapture(f'data/test/{v}.mp4'); S = c.get(3) / d['W']
    k = [i for i, p in enumerate(ph) if 'box' in p][0]; o = d['objs'][str(k)]
    js = [i for i, x in enumerate(o) if x and x.get('st')]
    a = np.array([(d['fidx'][i], *o[i]['st']['box']) for i in js], float); a[:, 1:] *= S
    bh = a[:, 4] - a[:, 2]
    f0 = d['fidx'][js[0]]; b = a[0, 1:5]
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2; w, h = (b[2] - b[0]) * 0.35, (b[3] - b[1]) * 0.35
    nf = int(round(0.6 * fps)) + 2; F = []
    while len(F) < nf:
        ok, im = c.read()
        if not ok: break
        F.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    x0, y0, x1, y1 = int(cx - w), int(cy - h), int(cx + w), int(cy + h); T = F[f0][y0:y1, x0:x1].astype(np.float32); px, py = x0, y0; tr = []
    for i in range(f0, len(F)):
        X0 = max(0, int(px) - 40); Y0 = max(0, int(py) - 60); Sb = F[i][Y0:Y0 + (y1 - y0) + 200, X0:X0 + (x1 - x0) + 80].astype(np.float32)
        if Sb.shape[0] < T.shape[0] or Sb.shape[1] < T.shape[1]: break
        m = cv2.matchTemplate(Sb, T, cv2.TM_CCOEFF_NORMED); _, mv, _, (x, y) = cv2.minMaxLoc(m)
        px, py = X0 + x, Y0 + y; tr.append((i, px, py, mv))
    tr = np.array(tr, float); t = tr[:, 0] / fps; yy = tr[:, 2] - tr[0, 2]
    msk = t <= 0.48 + 1e-9; cf = np.polyfit(t[msk], yy[msk], 2)
    mm = 9.8 / (2 * cf[0]) * 1000
    return mm, float(np.median(bh))


rows = []
for v, i in [('simulation_0191', 466), ('simulation_0191_segmented', 1913), ('simulation_0192', 472)]:
    mm, bh = run(v)
    rows.append(dict(id0=i, value=bh * mm / 1000))
pd.DataFrame(rows).to_csv('analysis/v3/calib/r9/levers/sizecons_test.csv', index=False)
