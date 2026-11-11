"""diagnostic: 0077 two-person track; torso colour per obj and speed in box-heights per frame (fps-free ratio)."""
import json, sys, numpy as np
sys.path.insert(0, 'geo'); import track_common as tc
for v, tr in (('0077', 'v2sc'), ('0076', 'v2repro'), ('0076', 'v2sc')):
    J = json.load(open(f'geo/out/tracks_test_{tr}/simulation_{v}.json')); fi = J['fidx']
    fr = tc.load_video(f'data/test_480p/simulation_{v}.mp4')
    print(v, tr, 'frames', len(fr), fr[0].shape, 'fidx', fi[:3], fi[-1])
    for k, L in J['objs'].items():
        hs, cs, cols = [], [], []
        for j, s in enumerate(L):
            if not s or not s.get('st'): continue
            x1, y1, x2, y2 = s['st']['box']; hs.append(y2 - y1); cs.append((fi[j], s['st']['c']))
            if j % 12 == 6 and fi[j] < len(fr):
                im = fr[fi[j]]; sc = im.shape[0] / 480.0
                yy1, yy2 = int((y1 + 0.25 * (y2 - y1))), int((y1 + 0.45 * (y2 - y1))); xx1, xx2 = int(x1 + 0.3 * (x2 - x1)), int(x1 + 0.7 * (x2 - x1))
                cols.append(im[yy1:yy2, xx1:xx2].reshape(-1, 3).mean(0))
        f0, c0 = cs[0]; f1, c1 = cs[-1]
        pxs = np.hypot(c1[0] - c0[0], c1[1] - c0[1]) / (f1 - f0)
        print('  obj', k, 'median h %.1f' % np.median(hs), 'px/frame %.3f' % pxs, 'h/frame %.4f' % (pxs / np.median(hs)), 'torso rgb', np.round(np.mean(cols, 0)).tolist() if cols else None)
