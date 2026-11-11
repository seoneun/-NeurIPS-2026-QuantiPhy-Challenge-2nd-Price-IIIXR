import sys, json, cv2, numpy as np, pandas as pd
D = 'analysis/v3/ag/r9_main/levers/sbfix/in'
b = pd.read_csv(sys.argv[1]); b['id0'] = b.id - 1; b = b.set_index('id0')


def frames(v, n):
    c = cv2.VideoCapture(f'data/test/{v}.mp4'); F = []
    while len(F) < n:
        ok, im = c.read()
        if not ok: break
        F.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32))
    return F


def track(F, box):
    x0, y0, x1, y1 = box; T = F[0][y0:y1, x0:x1]; out = []; px, py = x0, y0
    sub = lambda a, b_, c: 0 if (a - 2 * b_ + c) == 0 else 0.5 * (a - c) / (a - 2 * b_ + c)
    for i in range(len(F)):
        X0 = max(0, int(px) - 60); Y0 = max(0, int(py) - 250); S = F[i][Y0:Y0 + (y1 - y0) + 500, X0:X0 + (x1 - x0) + 120]
        m = cv2.matchTemplate(S, T, cv2.TM_CCOEFF_NORMED); _, mv, _, ml = cv2.minMaxLoc(m); x, y = ml
        dx = sub(m[y, x - 1], m[y, x], m[y, x + 1]) if 0 < x < m.shape[1] - 1 else 0
        dy = sub(m[y - 1, x], m[y, x], m[y + 1, x]) if 0 < y < m.shape[0] - 1 else 0
        px, py = X0 + x + dx, Y0 + y + dy; out.append((i, px, py, mv))
    return np.array(out)


def fall_scale(v, box, fps, tmax, nfr):
    a = track(frames(v, nfr + 1), box); t = a[:, 0] / fps; m = t <= tmax + 1e-9
    c = np.polyfit(t[m], a[m, 2], 2)
    return 9.8 / (2 * c[0])


rows = []
def put(i, v):
    rows.append((int(i), float(v)))

s81 = fall_scale('simulation_0181', (400, 262, 660, 420), 24.0, 7 / 24, 8)
s82 = fall_scale('simulation_0182', (400, 262, 590, 410), 24.0, 7 / 24, 8)
put(438, round(np.sqrt(301 * 328) * s81, 3))
put(439, round(np.sqrt(199 * 220) * s81, 3))
put(1427, round(np.sqrt(186 * 203) * s82, 3))

for i, v in ((474, 'simulation_0192'), (1921, 'simulation_0192_segmented')):
    d = json.load(open(f'geo/out/tracks_test_v3ntmv/{v}.json')); S = 1920 / d['W']
    bx = np.array([o['st']['box'] for o in d['objs']['0'][:5]], float) * S; bw = float(np.median(bx[:, 2] - bx[:, 0]))
    bb = bx[0]; cx, cy, w, h = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2, (bb[2] - bb[0]) * .35, (bb[3] - bb[1]) * .35
    s = fall_scale(v, (int(cx - w), int(cy - h), int(cx + w), int(cy + h)), 120.0, 0.48, 60)
    put(i, round(bw * s, 3))

for i, v, k in ((5, 'simulation_0012', '1'), (1648, 'simulation_0012_segmented', '0')):
    d = json.load(open(f'geo/out/tracks_test_v3ntmvSIL/{v}.json')); S = 1920 / d['W']
    a = np.array([x['st']['box'] for x in d['objs'][k] if x and x.get('st')], float) * S
    step = float(np.mean(np.diff(a[:, 0]))); bw = a[:, 2] - a[:, 0]
    body = float(np.median(bw[bw <= 150])); sc = 6.0 / (step * 24.0)
    put(i, round(body * sc, 3))

for i in (2437, 2447, 2457, 2410, 2419, 2428, 2238, 2249, 2260):
    put(i, 1.5)

for i, v in pd.read_csv('output/work/lev108/sbfix_copied_values.csv').itertuples(index=False):
    put(i, v)

o = pd.DataFrame(rows, columns=['id0', 'value']).sort_values('id0')
o = o[~np.isclose(o.value.values, b.parsed_value[o.id0].values, rtol=1e-6)]
o.to_csv('analysis/v3/calib/r9/levers/sbfix_test.csv', index=False)
