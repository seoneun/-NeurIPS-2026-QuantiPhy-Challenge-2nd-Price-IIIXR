import csv, io, json, os, re
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '../../../..'))
t = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
b = pd.read_csv(f'{R}/submissions/pred_t007_id1.csv')
assert len(t) == len(b) and (b.video_id.values == t.video.values).all()
t['cur'] = b.parsed_value.values
cur = dict(zip(t.id0, t.cur))
FOC_LAB = json.load(open(f'{R}/analysis/v3/compliance/focal_spec_all.json'))['f_over_W']['lab']


def csv_roundtrip(x):
    return float(pd.read_csv(io.StringIO('v\n' + repr(float(x)) + '\n')).v.iloc[0])


def dump_tracks(v):
    src = next((d for d in ('tracks_test_v2sc', 'tracks_test_v3ntmv', 'tracks_test')
                if os.path.exists(os.path.join(R, f'geo/out/{d}/{v}.json'))), None)
    js = json.load(open(os.path.join(R, f'geo/out/{src}/{v}.json')))
    fps = js.get('fps', 24.0); fidx = js.get('fidx')
    phr = {str(i): q.get('phrase') for i, q in enumerate(js.get('queries', []))}
    lines = []
    for o, fr in js.get('objs', {}).items():
        for k, e in enumerate(fr):
            if not e or 'st' not in e or e['st'] is None:
                continue
            st = e['st']; bx = st.get('box'); c = st.get('c')
            f = fidx[k] if fidx and k < len(fidx) else k
            lines.append(f"{v}\t{src}\t{o}\t{phr.get(o)}\t{f}\t{f / fps:.4f}\t{c[0]:.1f}\t{c[1]:.1f}\t{bx[0]}\t{bx[1]}\t{bx[2]}\t{bx[3]}\t{st.get('area')}")
    cols = 'video src obj phrase frame t cx cy x0 y0 x1 y1 area'.split()
    return pd.read_csv(io.StringIO('\n'.join(lines) + '\n'), sep='\t', names=cols, dtype={'obj': str}), js


def row(video, qsub):
    m = t[(t.video == video) & t.question.str.contains(qsub, regex=False)]
    assert len(m) == 1
    return int(m.id0.iloc[0]), float(m.cur.iloc[0])


for i in (2175, 2180, 2185):
    assert t.loc[t.id0 == i, 'video'].iloc[0].startswith('captured_0009') and 'initial distance from the floor' in t.loc[t.id0 == i, 'question'].iloc[0]
dtext = str(t.loc[t.id0 == 2175, 'depth'].iloc[0])
dep, t_apex = {}, None
for m in re.finditer(r't=([\d.]+)s( \([^)]*\))?, distance_(\w+?)_camera ?= ?([\d.]+) ?m', dtext):
    dep[(m.group(3), float(m.group(1)))] = float(m.group(4))
    if m.group(2) and 'max rebounce' in m.group(2):
        t_apex = float(m.group(1))
d_ball0, d_floor, d_apex = dep[('ball', 0.0)], dep[('floor', 0.0)], dep[('ball', t_apex)]
i_diam, diam = row('captured_0009', 'diameter')
i_apx, apex_cm = row('captured_0009', 'maximum rebounce height')
r = diam / 2 / 100
APEX = round(apex_cm / 100, 4)

e, js = dump_tracks('captured_0009')
e = e[e.obj == '0'].set_index('frame')
W, H = js['W'], js['H']; F = FOC_LAB * W; CX, CY = W / 2, H / 2
fps = js.get('fps', 24.0)
K0, KA = 0, int(round(t_apex * fps))
pre = e.loc[(e.index > K0) & (e.index < KA), 'cy']
KI = int(pre.idxmax())
print('')


def bp(k, d):
    xn, yn = (e.loc[k, 'cx'] - CX) / F, (e.loc[k, 'cy'] - CY) / F; z = d / np.sqrt(1 + xn**2 + yn**2); return np.array([xn * z, yn * z, z])


size = lambda k: e.loc[k, 'x1'] - e.loc[k, 'x0']
Dc46 = d_apex + r; Dc37 = Dc46 * size(KA) / size(KI)
Pimp, Papex, P0 = bp(KI, Dc37), bp(KA, Dc46), bp(K0, d_ball0)
up = (Papex - Pimp) / np.linalg.norm(Papex - Pimp); alpha = np.arctan2(-up[2], -up[1])
n = np.array([0, np.cos(alpha), np.sin(alpha)]); hcam = n @ Pimp + r
rho = np.sqrt(d_floor**2 - hcam**2); e1 = np.array([1.0, 0, 0]); e2 = np.array([0, -np.sin(alpha), np.cos(alpha)])
lab = []
for phi in np.radians(np.arange(0, 360, 0.5)):
    Fp = hcam * n + rho * (np.cos(phi) * e1 + np.sin(phi) * e2)
    if Fp[2] <= 0: continue
    u, v = CX + F * Fp[0] / Fp[2], CY + F * Fp[1] / Fp[2]
    if 0 <= u < W and 0 <= v < H: lab.append(np.linalg.norm(P0 - Fp))
lab = np.array(lab)
route2 = float(np.median(lab)); drop = float((Papex - P0) @ n); route1 = APEX + drop
new9 = float(np.sqrt(route1 * route2)) * 100
assert len(lab) > 0 and abs(route1 / route2 - 1) < 0.15
unr = float(np.sqrt((apex_cm / 100 + drop) * route2)) * 100
print('')

assert t.loc[t.id0 == 1569, 'video'].iloc[0] == 'simulation_0205' and 'height of the typewriter' in t.loc[t.id0 == 1569, 'question'].iloc[0]
r07, r08 = row('simulation_0207', 'height of the typewriter'), row('simulation_0208', 'height of the typewriter')
route1_tw = float(np.median([r07[1], r08[1]]))

out = {i: round(new9, 4) for i in (2175, 2180, 2185)}
out[1569] = round(route1_tw, 4)
out = {i: csv_roundtrip(v) for i, v in out.items()}

with open(f'{HERE}/reproduce_out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id0', 'value'])
    for i, v in out.items():
        w.writerow([int(i), repr(float(v))])
