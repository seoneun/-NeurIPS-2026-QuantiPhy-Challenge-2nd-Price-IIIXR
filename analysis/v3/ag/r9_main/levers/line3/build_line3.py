import sys, re, json, io, contextlib
import numpy as np, pandas as pd
from scipy.optimize import least_squares
sys.path.insert(0, 'analysis/v3/ag/r9_main/levers/line3')
import deskcorner as WD

BASE, OUT = sys.argv[1], sys.argv[2]
AG = 'analysis/v3/ag'; IN = 'analysis/v3/ag/r9_main/levers/line3/in'
t = pd.read_csv(BASE)
G = pd.read_csv('submissions/pred_t002_detail.csv', index_col=0).g
new = {}


def run(path, stop=None):
    src = open(path).read()
    if stop:
        src = src[:src.index(stop)]
    ns = {'__file__': path, '__name__': 'line3'}; argv = sys.argv; sys.argv = [path]
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, path, 'exec'), ns)
    sys.argv = argv
    return ns


Q = pd.read_parquet('data/test/test_dataset.parquet')
per = WD.pixels(f'{IN}/ag_w1_deskcorner_masks.npz')
for v in WD.VIDS:
    di = str(Q[Q.video_id == v].depth_info.iloc[0]); n, f = WD.NEAR.search(di), WD.FAR.search(di)
    if not (per[v]['ok_BR'] and n and f):
        continue
    px = {k: per[v][k] for k in ('BL', 'BR', 'FR', 'FL')}; dn, df = float(n.group(1)), float(f.group(1))
    s = WD.solve_case(px, WD.fit_focal(px, dn, df) * WD.W, dn, df)
    if not (s['front'] >= df - dn and s['length'] >= s['width'] and abs(s['FLpx'][1] - px['FL'][1]) <= 10):
        continue
    for i in t.index[(t.video_id == v) & t.question.str.contains(r'What is the (?:length|width) of the table in meters')]:
        new[i] = float(s['length' if 'length' in t.question[i] else 'width'])

ns = run(f'{AG}/rul386/virus_values.py')
new[214] = round(float(ns['routeB']), 4)
for i, tt in ((215, 1.0), (216, 2.0)):
    dpx = np.hypot(np.polyval(ns['pLx'], tt) - np.polyval(ns['pRx'], tt), np.polyval(ns['pLy'], tt) - np.polyval(ns['pRy'], tt))
    new[i] = round(float(dpx * ns['mR']), 4)

ns = run(f'{AG}/rul395/reproduce.py', stop="c2 = tj('simulation_0043')")
s1, d0, ff, k, w1, h1, w2, h2 = (ns[x] for x in ('s1', 'd0', 'ff', 'k', 'w1', 'h1', 'w2', 'h2'))
new.update({438: w1 * s1, 439: h1 * s1, 1426: w2 * d0 / ff * k, 1427: h2 * d0 / ff * k})

for i, fn in ((2929, 'geo/out/tracks_test_v3ntmv/captured_0062.json'), (2937, f'geo/out/tracks_test_v3ntmv/captured_0062s.json'),
              (2945, f'geo/out/tracks_test_v3ntmv/captured_0062x.json')):
    c = json.load(open(fn)); qb = {q['phrase']: np.median(np.array([x for x in q['det_boxes'] if x], float), 0) for q in c['queries']}
    kb = float(re.search(r'distance_trashbin_camera = ([\d.]+)', t.depth_info[i]).group(1))
    kg = float(re.search(r'distance_blue_shopping_bag_camera = ([\d.]+)', t.depth_info[i]).group(1))
    if qb['trash bin'][3] < qb['blue shopping bag'][3] and kb < kg:
        new[i] = float(G[i]) * kg / kb

for i in (1169, 1751, 1489):
    new[i] = float(G[i])

new[758] = float(run(f'{AG}/rul378/reproduce.py')['top'])

js = json.load(open(f'geo/out/tracks_test_v3ntmv/simulation_0180.json'))
q = [q for q in js['queries'] if q['phrase'] == 'trolley'][0]
fr = [(int(k), o['st']['box']) for k, o in zip(js['fidx'], js['objs'][str(q['obj'])]) if isinstance(o, dict) and o.get('st') and o['st'].get('box')]
F = np.array([r[0] for r in fr]); B = np.array([r[1] for r in fr], float)
W, H = js['W'], js['H']; tt = F / js['fps']; cx = (B[:, 0] + B[:, 2]) / 2; cyy = (B[:, 1] + B[:, 3]) / 2
D0, A = 12.589, 1.507


def model(p, tt):
    f, phi = p
    al = np.arctan((cx[0] - W / 2) / f); be = np.arctan((cyy[0] - H / 2) / f)
    dh = D0 * np.cos(be); yv = D0 * np.sin(be)
    X = dh * np.sin(al) + A * tt ** 2 / 2 * np.sin(phi); Z = dh * np.cos(al) + A * tt ** 2 / 2 * np.cos(phi)
    return W / 2 + f * X / Z, np.sqrt(X ** 2 + Z ** 2 + yv ** 2)


best = None
for f0 in (600, 900, 1200, 1600, 2200):
    for ph0 in np.linspace(-np.pi, np.pi, 9):
        r = least_squares(lambda p: model(p, tt)[0] - cx, [f0, ph0], bounds=([200, -2 * np.pi], [6000, 2 * np.pi]))
        if best is None or r.cost < best.cost:
            best = r
new[1418] = float(np.median(B[:3, 3] - B[:3, 1]) * D0 / best.x[0])

ns = run(f'{AG}/rul386/reproduce.py', stop='r1 = float(np.median(own))')
for i, vid, trk in ((1628, 'simulation_0247', [('tracks_test_v2sc', 1), ('tracks_test_v3ntmv', 1)]),
                    (1985, 'simulation_0247_segmented', [('tracks_test_v2sc', 3), ('tracks_test_v3ntmv', 1)])):
    new[i] = float(np.median([1.277 * np.median(S[:, 4]) / ns['speed'](S) for S in (ns['series'](vid, o, d) for d, o in trk)]))

HP = 1.70
ns = run(f'{AG}/rul363/reproduce.py', stop='order = [3083')
for i, ip in ((3154, ns['i3150']), (3155, ns['i3150']), (3121, ns['i3116'])):
    new[i] = float(ns['res'][i]) * HP / float(ns['cur'][ip])
ns = run(f'{AG}/rul377/reproduce.py', stop="say(f'  3141")
new[3141] = float(ns['OUT'][3141][0]) * HP / float(ns['c26'][3134])

rows = [(i, v) for i, v in sorted(new.items()) if abs(np.log(v / float(t.parsed_value[i]))) >= 1e-9]
pd.DataFrame(rows, columns=['id0', 'value']).to_csv(OUT, index=False)
