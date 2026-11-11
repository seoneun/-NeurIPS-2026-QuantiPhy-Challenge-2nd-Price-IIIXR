import json, os, re, sys
import numpy as np, pandas as pd, cv2
sys.path.insert(0, os.path.dirname(__file__))
from road_metrology import depth, track, boxes
R = '@QP_ROOT@'

def frame(v, fi):
    cap = cv2.VideoCapture(f'{R}/data/test_480p/{v}.mp4'); cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    return cap.read()[1].astype(float)

def scan(im, xc, y_from, step, thr=28, gap=4, half=2, off=(6, 11)):
    H, W = im.shape[:2]; y = y_from; last = y_from; miss = 0
    while 0 <= y < H:
        c = im[y, max(0, xc - half):xc + half + 1]
        bg = np.r_[im[y, max(0, xc - off[1]):max(0, xc - off[0])], im[y, min(W - 1, xc + off[0]):min(W, xc + off[1])]]
        if len(bg) == 0: break
        d = np.linalg.norm(c[:, None, :] - np.median(bg, 0)[None, None, :], axis=2).max()
        if d > thr: last = y; miss = 0
        else:
            miss += 1
            if miss > gap: break
        y += step
    return last

out = []
for n in sys.argv[1:]:
    v = f'simulation_{n}'; js = track(v); Hh = js['H']
    ped = boxes(js, r'walking pedestrian|^pedestrian'); sl = boxes(js, r'streetlight'); tl = boxes(js, r'traffic light')
    pc = boxes(js, r'purple car'); gc = boxes(js, r'green car')
    r = {'v': n}
    ok = ~np.isnan(ped).any(1) & (ped[:, 1] > 2) & (ped[:, 3] < Hh - 2)
    hp = ped[ok, 3] - ped[ok, 1]; yb = ped[ok, 3]
    (yh, k), *_ = np.linalg.lstsq(np.c_[np.ones_like(hp), hp], yb, rcond=None)
    r.update(yhP=round(yh, 1), hcP=round(1.87 * k, 2))
    cars = []
    for b, Wm in ((pc, 1.86), (gc, 2.03)):
        if b is None: continue
        bb = np.nanmedian(b, 0)
        if not any(np.allclose(bb, c[2]) for c in cars): cars.append((Wm, bb[3], bb))
    if len(cars) == 2:
        (W1, y1, b1), (W2, y2, b2) = cars
        p1, p2 = (b1[2] - b1[0]) / W1, (b2[2] - b2[0]) / W2
        hcC = (y1 - y2) / (p1 - p2); yhC = y1 - p1 * hcC
        r.update(yhC=round(yhC, 1), hcC=round(hcC, 2))
    elif len(cars) == 1:
        W1, y1, b1 = cars[0]; r.update(car1=f'{W1}m w{b1[2]-b1[0]:.0f} yb{y1:.0f}')
    objs = {}
    if sl is not None:
        b = np.nanmedian(sl, 0); objs['SL'] = (b[1], b[3])
    if tl is not None:
        b = np.nanmedian(tl, 0); im = frame(v, 14); xc = int(round((b[0] + b[2]) / 2))
        base = scan(im, xc, int(b[3]) + 1, +1); top = scan(im, xc, int(b[1]) - 1, -1, thr=40, gap=2, half=3, off=(9, 14))
        r['TLscan'] = f'head{b[1]:.0f}-{b[3]:.0f} sign_top{top} base{base}'
        objs['TL_sign'] = (top, base); objs['TL_head'] = (b[1], base)
    for nm, (yt, ybS) in objs.items():
        r[f'{nm}_P'] = round(1.87 * k * (ybS - yt) / (ybS - yh), 2)
        if 'hcC' in r: r[f'{nm}_C'] = round(r['hcC'] * (ybS - yt) / (ybS - r['yhC']), 2)
        elif len(cars) == 1:
            W1, y1, b1 = cars[0]
            r[f'{nm}_C1'] = round(W1 * (ybS - yt) / (b1[2] - b1[0]) * (y1 - yh) / (ybS - yh), 2)
            r[f'{nm}_C1raw'] = round(W1 * (ybS - yt) / (b1[2] - b1[0]), 2)
    out.append(r)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 30)
print(pd.DataFrame(out).to_string(index=False))
