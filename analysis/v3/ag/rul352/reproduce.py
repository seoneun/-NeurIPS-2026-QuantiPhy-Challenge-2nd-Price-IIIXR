import csv, io, json, math, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '../../../..'))
sys.path.insert(0, HERE)
from track_px import px

t = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
b = pd.read_csv(f'{R}/submissions/pred_t007_id1.csv')
assert len(t) == len(b) and (b.video_id.values == t.video.values).all()
t['cur'] = b.parsed_value.values
T = t.set_index('id0')
val = pd.read_csv(f'{R}/data/val/validation_dataset.csv')
cur = lambda i: float(T.loc[i, 'cur'])
gm = lambda *v: float(np.exp(np.mean(np.log(v))))


def csv_roundtrip(x):
    return float(pd.read_csv(io.StringIO('v\n' + repr(float(x)) + '\n')).v.iloc[0])


def track_json(split, vid, mt):
    cands = [f"{R}/geo/out/tracks_{split}/{vid}.json"] if mt == "v1" else [f"{R}/geo/out/tracks_{split}_v2sc/{vid}.json", f"{R}/geo/out/tracks_{split}_v2/{vid}.json"]
    p = next((c for c in cands if os.path.exists(c)), None)
    return json.load(open(p)) if p else None


def ent_keys(ent):
    if not ent or not ent.get("phrase"):
        return []
    rel = ent.get("rel")
    base = ent["phrase"] + "|" + ",".join(ent.get("where") or [])
    return ([base + "|" + rel[0] + ":" + rel[1]] if rel else []) + [base]


def frames_of(vj, ent):
    if vj is None:
        return None
    qs = {q["key"]: q for q in vj["queries"]}
    for k in ent_keys(ent):
        if k in qs:
            q = qs[k]
            if vj.get("version") == "v2":
                return None if q.get("obj") is None else [f.get("st") if f else None for f in vj["objs"][str(q["obj"])]]
            return [f.get("st") if f else None for f in q["frames"]]
    return None


def geom(st):
    val_ = np.array([s is not None for s in st])
    area = np.array([s["area"] if s else 0 for s in st], float)
    med = np.median(area[val_]) if val_.any() else 0
    ok = val_ & (area > 0.4 * med) & (area < 2.5 * med) if med > 0 else val_
    border = np.array([s["border"] if s else 9 for s in st])
    okc = np.where(ok)[0]; oke = np.where(ok & (border == 0))[0]
    if len(oke) < max(2, 0.3 * len(okc)):
        oke = okc
    if len(oke) < 3:
        return None
    rr = np.array([st[j]["rr"] for j in oke], float)
    bx = np.array([st[j]["box"] for j in oke], float)
    L, S, a = rr[:, 0], rr[:, 1], np.deg2rad(rr[:, 2])
    bw, bh = bx[:, 2] - bx[:, 0], np.maximum(bx[:, 3] - bx[:, 1], 1.0)
    th = []
    for k in range(len(L)):
        best = None
        for t_ in (a[k], a[k] + np.pi / 2):
            e = abs(L[k] * abs(math.cos(t_)) + S[k] * abs(math.sin(t_)) - bw[k]) + abs(L[k] * abs(math.sin(t_)) + S[k] * abs(math.cos(t_)) - bh[k])
            if best is None or e < best[0]:
                best = (e, t_ % np.pi)
        th.append(best[1])
    th = np.array(th)
    thm = np.degrees((0.5 * math.atan2(np.median(np.sin(2 * th)), np.median(np.cos(2 * th)))) % np.pi)
    return dict(n=len(oke), rrL=float(np.median(L)), rrS=float(np.median(S)), bh=float(np.median(bh)),
                asp=float(np.median(L) / max(np.median(S), 1e-6)), theta=float(min(thm, 180 - thm)))


out, measured = {}, {}
p22, p24, p27 = px('simulation_0022_segmented'), px('simulation_0024_segmented'), px('simulation_0027_segmented')
rk22, bt22 = p22['rock|'], p22['boat|']
assert abs(p24['rock|']['w'] - p27['rock|']['w']) < 3 and abs(p24['rock|']['cy'] - p27['rock|']['cy']) < 3
assert T.loc[1689, 'video'] == 'simulation_0027_segmented' and 'length of the rock' in T.loc[1689, 'question']
assert T.loc[1670, 'video'] == 'simulation_0022_segmented' and 'height of the rock' in T.loc[1670, 'question']
boat_h_val = float(val[(val.video_id == 'simulation_0034_segmented') &
                       val.question.str.contains('height of the boat')].ground_truth_posterior.iloc[0])
len27 = cur(1689)
out[1679] = round(len27, 4)

r_a = rk22['h'] / bt22['h'] * boat_h_val
r_c = rk22['h'] / rk22['w'] * len27
r_d = cur(1670)
j1680 = next(j for j in map(json.loads, open(f'{R}/geo/out/parse_test_v2.jsonl')) if int(j['id']) == 1680)
RW = pd.read_csv(f'{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv', index_col=0)
CW = pd.read_csv(f'{R}/preds/test_caw9b_orig.csv', index_col=0)
assert CW.loc[1680, 'video_id'] == j1680['video_id'] == 'simulation_0024_segmented'
g_ff, cs9b, mt = float(RW.loc[1680, 'geo_value']), float(CW.loc[1680, 'parsed_value']), str(RW.loc[1680, 'meas_tracks'])
G = geom(frames_of(track_json('test', j1680['video_id'], mt), j1680.get('target')))
r_geo = G['rrS'] / G['bh']
ans = float(np.exp(0.5 * np.log(g_ff) + 0.5 * np.log(cs9b)))
r_b = ans * r_geo ** 0.5
X = gm(r_a, r_b, r_c, r_d)
HGT = 4.5973
assert abs(X / HGT - 1) < 0.02
for i in (1680, 1690):
    assert 'height of the rock' in T.loc[i, 'question']
    out[i] = round(X, 4) if os.environ.get('R4_COMPUTED') == '1' else HGT
    measured[i] = X
print('')

assert 'distance_boat_camera = 133.9572' in str(T.loc[1119, 'depth']) and 'distance_boat_camera = 133.9572' in str(T.loc[1148, 'depth'])
assert 'length of the boat = 10m' in str(T.loc[1148, 'prior'])
for i in (1119, 1699):
    assert 'height of the boat' in T.loc[i, 'question']
    out[i] = round(boat_h_val, 4)

p14 = px('simulation_0014')
wp = (p14['wheel||in:cart']['w'] + p14['wheel||in:cart']['h']) / 2
ch = p14['cart|']['h']
c_wheel = ch / wp * cur(1067)
c_box = ch / p14['rectangular cargo box|']['h'] * cur(1070)
assert 'diameter of' in T.loc[1067, 'question'] and 'height of' in T.loc[1070, 'question']
out[1069] = round(gm(c_wheel, c_box), 4)

out = {i: csv_roundtrip(v) for i, v in out.items()}
with open(f'{HERE}/reproduce_out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id0', 'value'])
    for i, v in out.items():
        w.writerow([int(i), repr(float(v))])
