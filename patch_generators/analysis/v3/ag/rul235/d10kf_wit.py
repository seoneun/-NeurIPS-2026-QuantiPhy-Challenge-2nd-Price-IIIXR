"""diagnostic witness: pixel speed ratio of the two tracked persons in alternate track runs (fps-free)."""
import json, glob, numpy as np
for v in ('0076', '0077', '0078'):
    for f in sorted(glob.glob(f'geo/out/tracks_test_v2*/simulation_{v}.json')):
        J = json.load(open(f)); fi = np.array(J['fidx'], float); O = J['objs']
        if len(O) < 2: continue
        sp = {}; xs = {}
        for k, L in O.items():
            c = np.array([s['st']['c'] if s and s.get('st') else [np.nan, np.nan] for s in L], float)
            ok = np.isfinite(c[:, 0]); 
            if ok.sum() < 5: continue
            # robust: median frame-to-frame displacement per frame over valid consecutive samples
            idx = np.where(ok)[0]; d = np.hypot(*(np.diff(c[idx], axis=0).T)) / np.diff(fi[idx])
            sp[k] = np.median(d); xs[k] = (c[idx[0]].round(0).tolist(), c[idx[-1]].round(0).tolist(), int(ok.sum()))
        ks = sorted(sp)
        if len(ks) == 2:
            print(v, f.split('/')[2], {k: round(sp[k], 3) for k in ks}, 'ratio %.3f' % (sp[ks[0]] / sp[ks[1]]), xs, 'query obj', J['queries'][0]['obj'])
