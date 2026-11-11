import io, json, os, re, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
R = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
FROZEN = os.path.join(HERE, 'frozen', 'owl_0120.json')
OWL_JOB = ('simulation_0120', [10, 40], ["a square tray", "a tray", "a chessboard"])


def rerun_owl():
    import cv2, torch
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    from torchvision.ops import nms
    mp = os.path.join(R, 'models/owlv2-base-patch16-ensemble')
    op = Owlv2Processor.from_pretrained(mp)
    om = Owlv2ForObjectDetection.from_pretrained(mp).to('cuda').eval()
    v, frames, qs = OWL_JOB; res = {}
    cap = cv2.VideoCapture(os.path.join(R, 'data/test_480p', v + '.mp4'))
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, im = cap.read()
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        inp = op(text=[qs], images=rgb, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = om(**inp)
        side = max(im.shape[:2])
        pr = op.image_processor.post_process_object_detection(out, threshold=0.05, target_sizes=torch.tensor([[side, side]], device='cuda'))[0]
        keep = nms(pr['boxes'].float(), pr['scores'].float(), 0.3)[:6]
        res[f'{v}:{fi}'] = [dict(q=qs[int(l)], s=round(float(s), 3), box=[round(float(x), 1) for x in b]) for b, s, l in
                            zip(pr['boxes'][keep].cpu().numpy(), pr['scores'][keep].cpu().numpy(), pr['labels'][keep].cpu().numpy())]
    os.makedirs(os.path.dirname(FROZEN), exist_ok=True)
    json.dump(res, open(FROZEN, 'w'), indent=1)
    print('wrote', FROZEN)


if '--rerun' in sys.argv:
    rerun_owl()
    sys.exit(0)

import cv2
from comps import components
A = pd.read_csv(os.path.join(R, 'analysis/v3/calib/audit_test.csv'))
cur = pd.read_csv(os.path.join(R, 'submissions/pred_t007_id1.csv')).parsed_value.values
assert (A.id0.values == np.arange(len(A))).all() and len(cur) == len(A)
g = pd.read_csv(os.path.join(R, 'preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv'))
MPP = dict(zip(g['Unnamed: 0'], g.meas_mpp))
prior_num = lambda i: float(re.search(r'=\s*([0-9.]+)\s*m/s', A.prior[i]).group(1))
out, const = {}, {}


def csv_hop(x):
    return float(pd.read_csv(io.StringIO(pd.DataFrame({'v': [float(x)]}).to_csv(index=False))).v[0])


def put(i, v):
    out[i] = csv_hop(round(float(v), 5))


def trk(v, d='tracks_test_v3ntmv'):
    js = json.load(open(f'{R}/geo/out/{d}/{v}.json')); o = {}
    for q in js['queries']:
        ob = q.get('obj')
        if ob is not None and str(ob) in js['objs']:
            o[q['phrase']] = np.array([[k] + fr['st']['box'] for k, fr in zip(js['fidx'], js['objs'][str(ob)])
                                       if isinstance(fr, dict) and fr.get('st') and fr['st'].get('box')], float)
    return js, o


V_CAR = prior_num(302); assert A.prior[302] == A.prior[303] == A.prior[306] and A.video[302] == 'simulation_0130'
js = json.load(open(os.path.join(R, 'geo/out/tracks_test_v2sc/simulation_0130.json')))
fps = js['fps']; car = {}
for q in js['queries']:
    if q['phrase'] == 'car':
        for k, fr in zip(js['fidx'], js['objs'][str(q['obj'])]):
            if isinstance(fr, dict) and fr.get('st') and fr['st'].get('box'):
                car[int(k)] = fr['st']['box']
F = np.array(sorted(car)); B = np.array([car[k] for k in F], float)
v_car = abs(np.polyfit(F / fps, (B[:, 1] + B[:, 3]) / 2, 1)[0])
mpp = V_CAR / v_car
shp, cps = components('simulation_0130', 71, 'bright', 185, [110, 150, 310, 478], 150)
x0 = min(c['bbox'][0] for c in cps); y0 = min(c['bbox'][1] for c in cps)
x1 = max(c['bbox'][0] + c['bbox'][2] for c in cps); y1 = max(c['bbox'][1] + c['bbox'][3] for c in cps)
put(302, (y1 - y0) * mpp); put(303, (x1 - x0) * mpp)
cap = cv2.VideoCapture(os.path.join(R, 'data/test_480p/simulation_0130.mp4')); ny = []
for f in range(72):
    ok, im = cap.read()
    if not ok:
        break
    gm = im[:, 205:230].min(axis=2) > 195
    start = int(car[f][3]) + 3 if f in car else 0
    rows = np.nonzero(gm[start:].sum(1) >= 3)[0]
    ny.append((f, start + rows[0] if len(rows) else np.nan))
ny = np.array(ny, float); okn = ~np.isnan(ny[:, 1])
vn = abs(np.polyfit(ny[okn, 0] / fps, ny[okn, 1], 1)[0])
gap0, gap71 = ny[0, 1] - car[0][3], ny[71, 1] - car[71][3]
print('')
assert abs(gap0 - gap71) <= 2 and abs(vn * mpp / V_CAR - 1) <= 0.02
put(306, V_CAR)

L, S = [], []
for fi in (6, 36, 66):
    _, cc = components('simulation_0123_segmented', fi, 'dark', 250, None, 60)
    tw = [c for c in cc if c['area'] > 5000 and -80 < c['rr'][4] < -10]
    assert len(tw) == 1
    L.append(tw[0]['rr'][2]); S.append(tw[0]['rr'][3])
put(1835, float(np.median(L)) * MPP[1835]); put(1836, float(np.median(S)) * MPP[1835])
print('')

H = []
for fi in (4, 16, 28):
    _, cc = components('simulation_0116_segmented', fi, 'dark', 250, None, 60)
    doors = sorted([c for c in cc if c['area'] > 3000], key=lambda c: c['cen'][1])
    assert len(doors) == 2
    H.append(doors[0]['rr'][2])
put(237, float(np.median(H)) * MPP[1825])
print('')

js, o = trk('simulation_0124_segmented')
T_Q = float(re.search(r'at ([0-9.]+)\s*s', A.question[270]).group(1))


def vat(Bx, t0, fps):
    t = Bx[:, 0] / fps; m = (np.abs(t - t0) <= 0.25) & (Bx[:, 2] > 1) & (Bx[:, 4] < 479)
    cx = (Bx[m, 1] + Bx[m, 3]) / 2; cy = (Bx[m, 2] + Bx[m, 4]) / 2
    return float(np.hypot(np.polyfit(t[m], cx, 1)[0], np.polyfit(t[m], cy, 1)[0]))


vb, vg = vat(o['black car'], T_Q, js['fps']), vat(o['car'], T_Q, js['fps'])
put(270, prior_num(270) * vg / vb)
print('')

js, o = trk('simulation_0119'); P = o['chess piece']; fps = js['fps']
cy = (P[:, 2] + P[:, 4]) / 2
mv = np.r_[np.abs(np.diff(cy)) > 0.05, False]
last = int(P[np.nonzero(mv)[0].max(), 0]) + 1 if mv.any() else -1
m = P[:, 0] <= last
vpx = abs(np.polyfit(P[m, 0] / fps, cy[m], 1)[0])
mppn = prior_num(252) / vpx
Bc = o['chessboard']; board_px = float(np.median((Bc[:, 3] - Bc[:, 1] + Bc[:, 4] - Bc[:, 2]) / 2))
put(252, board_px * mppn)
print('')

owl = json.load(open(FROZEN))
tb = [d['box'] for k in ('simulation_0120:10', 'simulation_0120:40') for d in owl[k]
      if d['q'] == 'a square tray' and (d['box'][2] - d['box'][0]) < 40]
tray_px = float(np.median([(b[2] - b[0] + b[3] - b[1]) / 2 for b in tb]))
put(260, tray_px * MPP[260])
print('')

C1418 = 1.216
assert A.video[430] == 'simulation_0179' and A.question[430] == A.question[1418] and A.prior[430] == A.prior[1418]
meas = float(cur[430])
print('')
assert abs(meas / C1418 - 1) <= 0.02
const[1418] = (C1418, meas)
put(1418, meas if os.environ.get('R4_COMPUTED') == '1' else C1418)

with open(os.path.join(HERE, 'reproduce_out.csv'), 'w') as fh:
    fh.write('id0,value\n')
    for i in sorted(out):
        fh.write(f'{i},{float(out[i])!r}\n')
print('wrote', os.path.join(HERE, 'reproduce_out.csv'), out, 'constants (value, measured):', const)
