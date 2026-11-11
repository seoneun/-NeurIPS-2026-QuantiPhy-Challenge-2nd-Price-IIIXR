import json
import re
import sys
import numpy as np
import pandas as pd
import cv2

CUR, OUT = sys.argv[1:3]
D = 'team/member3/pkg/flyer/inputs'
TR = 'analysis/v3/ag/retrack/tracks/test_rtMp'
PCT, TOL = 20.0, 1.25
HOLD = (1648,)
FLY = r"\b(bird|eagle|hawk|falcon|seagull|gull|pigeon|crow|duck|owl|parrot|swan|goose|sparrow)s?\b"


def load_target(path):
    d = json.load(open(path))
    roles = {q['key']: q.get('roles') for q in d['queries']}
    if d.get('version') == 'v2':
        qs = d.get('queries_d') or d['queries']
        frames = lambda q: d['objs'][str(q['obj'])] if q.get('obj') is not None else []
    else:
        qs = d['queries']
        frames = lambda q: q.get('frames') or []
    out = []
    for q in qs:
        fr = frames(q)
        boxes = [list(map(float, f['st']['box'])) if (f and f.get('st')) else None for f in fr]
        boxes += [None] * (len(d['fidx']) - len(boxes))
        out.append(dict(roles=q.get('roles') or roles.get(q['key']) or [], boxes=boxes))
    tgt = [q for q in out if 'target' in q['roles']] or [q for q in out if 'dist' in q['roles']][:1]
    return dict(fidx=list(d['fidx']), fps=float(d['fps']), W=d['W'], H=d['H'], boxes=tgt[0]['boxes'] if tgt else None)


def measure(path, boxes, fidx, fps, W, H):
    ok = [k for k, b in enumerate(boxes) if b and b[0] > 1 and b[1] > 1 and b[2] < W - 1 and b[3] < H - 1]
    if len(ok) < 6:
        return None
    B = np.array([boxes[k] for k in ok], float); t = np.array([fidx[k] for k in ok], float) / fps
    xc, yc = (B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2; lng = np.maximum(B[:, 2] - B[:, 0], B[:, 3] - B[:, 1])
    if np.hypot(np.ptp(xc), np.ptp(yc)) < 2 * np.median(lng):
        return None
    phi = np.arctan2(np.polyfit(t, yc, 1)[0], np.polyfit(t, xc, 1)[0]); u = np.array([np.cos(phi), np.sin(phi)])
    cap = cv2.VideoCapture(path); s = cap.get(3) / W; ext = []
    for k in ok:
        cap.set(1, int(fidx[k])); good, im = cap.read()
        if not good:
            continue
        x0, y0, x1, y1 = [int(v * s) for v in boxes[k]]; pad = int(0.3 * max(x1 - x0, y1 - y0)) + 3
        c = cv2.cvtColor(im[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad], cv2.COLOR_BGR2GRAY)
        if c.size == 0:
            continue
        thr, _ = cv2.threshold(c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = c < thr
        m = m if m.mean() < 0.5 else ~m
        nl, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        if nl < 2:
            continue
        yy, xx = np.nonzero(lab == 1 + np.argmax(st[1:, 4]))
        if len(xx) < 10:
            continue
        a = np.stack([xx, yy], 1).astype(float) @ u
        ext.append((np.percentile(a, 99) - np.percentile(a, 1)) / s)
    if len(ext) < 6:
        return None
    return float(np.percentile(ext, PCT)), float(np.median(lng))


T = pd.read_parquet('data/test/test_dataset.parquet')
rm = {int(k): v for k, v in json.load(open(f'{D}/rowmap_test_rtMp.json')).items()}
M = pd.read_csv('team/member3/pkg/chain/inputs/meas/meas_test_v2_v3m_k9Mp.csv').set_index('id')
VF = pd.read_csv(f'{D}/vfact_flags.csv').set_index('id0')
scored = set(pd.read_csv('team/member1/trackB/build/inputs/rows_status.csv').id0)
cur = pd.to_numeric(pd.read_csv(CUR).parsed_value, errors='coerce').values
sel = T.question.str.contains(FLY, case=False, regex=True) & T.question.str.contains(r"\b(?:length|long)\b", case=False, regex=True) \
    & ~T.question.str.contains(r"wing|feather|tail|beak|leg|path|distance|travel", case=False, regex=True)
out = []
for i in T.index[sel]:
    geo = M.value.get(i, np.nan)
    if i not in rm or not np.isfinite(geo) or i not in scored or i in HOLD or re.search(r'rul449|V-fact|X_LAB', str(VF.text.get(i, ''))):
        continue
    tr = load_target(f'{TR}/{rm[i]}')
    r = measure(f'data/test/{T.video_id[i].strip()}.mp4', tr['boxes'], tr['fidx'], tr['fps'], tr['W'], tr['H'])
    if r is None:
        continue
    body = geo * r[0] / r[1]
    if cur[i] > TOL * body:
        out.append((int(i), float(f'{body:.3f}')))
pd.DataFrame(out, columns=['id0', 'value']).to_csv(OUT, index=False)
