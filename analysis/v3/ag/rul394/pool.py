import json, os, sys
import cv2, numpy as np, pandas as pd
R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
det = pd.read_csv((R + '/analysis/v3/ag/rul394/frozen' if R5FZ else R + '/cache/tmp/rul394') + '/det_0225_markings_owlv2.csv')
b = det[(det.frame == 0) & (det.idx == 1)].iloc[0]; y0, y1 = int(round(b.y0)), int(round(b.y1))
def cat_speed(v):
    for d in ['tracks_test_v2sc', 'tracks_test_v2', 'tracks_test_v3ntmv']:
        if os.path.exists(f'{R}/geo/out/{d}/{v}.json'): break
    js = json.load(open(f'{R}/geo/out/{d}/{v}.json')); fps = float(js['fps']); fidx = js['fidx']
    q = [q for q in js['queries'] if q['key'].startswith('black cat') or q.get('phrase', '') == 'black cat'] if 'key' in js['queries'][0] else None
    if not q:
        q = [q for q in js['queries'] if 'black cat' in json.dumps(q)[:200]]
    o = js['objs'][str(q[0]['obj'])]
    it = o.items() if isinstance(o, dict) else enumerate(o)
    t, x = [], []
    for k, e in it:
        if e and 'st' in e:
            f = int(k) if isinstance(o, dict) else fidx[k]; t.append(f / fps); x.append(e['st']['c'][0])
    p = np.polyfit(t, x, 1); return abs(p[0]), len(t)
def spacing(v):
    cap = cv2.VideoCapture(f'{R}/data/test_480p/{v}.mp4'); prof = []
    for fi in [0, 12, 24, 37, 49]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, im = cap.read()
        if not ok: continue
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float); prof.append(g[y0 + 3:y1 - 3, :].mean(0))
    p = np.median(prof, 0); p = p - np.convolve(p, np.ones(31) / 31, 'same'); p = p[40:-40]
    ac = np.correlate(p - p.mean(), p - p.mean(), 'full')[len(p) - 1:]; ac /= ac[0]
    lags = np.arange(len(ac)); m = (lags >= 20) & (lags <= 300)
    pk = [int(l) for l in lags[m] if ac[l] > ac[l - 1] and ac[l] >= ac[l + 1]]
    top = sorted(pk, key=lambda l: -ac[l])[:3]
    pitch = float(np.median(np.diff(sorted(top))))
    vpx, n = cat_speed(v); s = 1.041 / vpx
    return pitch * s, pitch, s, [(l, round(float(ac[l]), 3)) for l in top]


if __name__ == '__main__':
    for v in ['simulation_0225', 'simulation_0227']:
        sp, pitch, s, top = spacing(v)
        print('')
