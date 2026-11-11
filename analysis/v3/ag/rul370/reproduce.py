import csv, json, os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, '../../../..'))
sys.path.insert(0, HERE)
_argv, sys.argv = sys.argv, sys.argv[:1]
from road_metrology import depth, track, boxes
from road_poles import scan, frame
sys.argv = _argv

t = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')
b = pd.read_csv(f'{R}/submissions/pred_t007_id1.csv')
assert len(t) == len(b)
t['cur'] = b.parsed_value.values
cur = dict(zip(t.id0, t.cur))
val = pd.read_csv(f'{R}/data/val/validation_dataset.csv')


def vlab(video, qsub):
    m = val[(val.video_id == video) & val.question.str.contains(qsub, regex=False)]
    assert len(m) == 1
    return float(m.ground_truth_posterior.iloc[0])


H_PED = vlab('simulation_0060', 'height of the walking pedestrian')
H_YCAR = vlab('simulation_0060', 'height of the turning yellow car')
W_PURPLE = vlab('simulation_0060', 'width of the purple car')
W_GREEN = vlab('simulation_0060', 'width of the green car')
FR, TD = 14, 0.58


def fped_fcar(js, d, W, H):
    out = []
    for nm, key, Hm in (('pedestrian', 'human', H_PED), ('turning yellow car', 'yellow_car', H_YCAR)):
        bb = boxes(js, nm)[FR]; Z = d[(key, TD)]
        r2 = ((bb[0] + bb[2]) / 2 - W / 2) ** 2 + ((bb[1] + bb[3]) / 2 - H / 2) ** 2
        out.append(np.sqrt((bb[3] - bb[1]) ** 2 * Z ** 2 / Hm ** 2 - r2))
    return out


vals = []; gp = {}
for n in ('0058', '0059', '0061', '0062', '0063'):
    v = f'simulation_{n}'; js = track(v); d = depth(v); W, H = js['W'], js['H']
    ped = boxes(js, 'walking pedestrian'); sl = np.nanmedian(boxes(js, 'streetlight'), 0); ycar = boxes(js, 'turning yellow car')
    ok = ~np.isnan(ped).any(1) & (ped[:, 1] > 2) & (ped[:, 3] < H - 2)
    hp = ped[ok, 3] - ped[ok, 1]; yb = ped[ok, 3]
    (yh, k), *_ = np.linalg.lstsq(np.c_[np.ones_like(hp), hp], yb, rcond=None)
    fp, fc = fped_fcar(js, d, W, H)
    gp[n] = {'yh': yh, 'k': k, 'pv': abs(fp / fc - 1) < 0.3}
    if abs(fp / fc - 1) < 0.3:
        vals.append((n, 'P', H_PED * k * (sl[3] - sl[1]) / (sl[3] - yh)))
    cars = []
    for nm, Wm in (('purple car', W_PURPLE), ('green car', W_GREEN)):
        bx = boxes(js, nm)
        if bx is None: continue
        bb = np.nanmedian(bx, 0)
        if np.allclose(bb, np.nanmedian(ycar, 0)): continue
        cars.append(((bb[2] - bb[0]) / Wm, bb[3]))
    if len(cars) == 2:
        (p1, y1), (p2, y2) = cars; hc = (y1 - y2) / (p1 - p2); yhc = y1 - p1 * hc
        gp[n].update(hc=hc, yhc=yhc)
        vals.append((n, 'C', hc * (sl[3] - sl[1]) / (sl[3] - yhc)))
    elif len(cars) == 1:
        (p1, y1), = cars
        vals.append((n, 'C1', (sl[3] - sl[1]) / p1))
SL = float(np.median([x[2] for x in vals]))
print('streetlight routes:', [(a, c, round(e, 3)) for a, c, e in vals], '-> scene value', SL)

tlv = []
for n in ('0059', '0061'):
    v = f'simulation_{n}'; js = track(v); tl = np.nanmedian(boxes(js, 'traffic light'), 0); g = gp[n]
    im = frame(v, FR); xc = int(round((tl[0] + tl[2]) / 2))
    base = scan(im, xc, int(tl[3]) + 1, +1); top = scan(im, xc, int(tl[1]) - 1, -1, thr=40, gap=2, half=3, off=(9, 14))
    for conv, yt in (('sign', top), ('head', tl[1])):
        tlv.append((n, 'P', conv, H_PED * g['k'] * (base - yt) / (base - g['yh'])))
        if 'hc' in g: tlv.append((n, 'C', conv, g['hc'] * (base - yt) / (base - g['yhc'])))
TL = float(np.median([x[3] for x in tlv]))
print('traffic-light routes:', [(a, c, cv, round(e, 3)) for a, c, cv, e in tlv], '-> scene value', TL)

q1180 = t.loc[t.id0 == 1180, 'question'].iloc[0]
assert t.loc[t.id0 == 1180, 'video'].iloc[0] == 'simulation_0045'
sib = [int(t[(t.video == v) & (t.question == q1180)].id0.iloc[0]) for v in ('simulation_0046', 'simulation_0050')]
s_sib = [cur[i] for i in sib]; cs = float(t[t.id0 == 1180].cs.iloc[0])
STEP = float(np.median(s_sib + [cs]))
print('first step: siblings', dict(zip(sib, s_sib)), 'CaW', cs, '->', STEP)

out = {}
for i in (1238, 1251, 1275, 1288, 1301):
    assert 'height of the streetlight' in t.loc[t.id0 == i, 'question'].iloc[0]
    out[i] = round(SL, 3)
out[1300] = round(TL, 3)
out[1180] = round(STEP, 3)

with open(f'{HERE}/reproduce_out.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id0', 'value'])
    for i, v in out.items():
        w.writerow([int(i), repr(float(v))])
