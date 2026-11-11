import os
import sys, json, copy
sys.path.insert(0, os.path.join(os.environ.get('QP_ROOT', '.'), 'analysis/v3/ag/rul089'))
import vc
import numpy as np, pandas as pd
MV = vc.MV; M1 = MV.M1; R = vc.R
split = 'test'
raw = pd.read_csv(f'{R}/preds/{split}_{vc.RAW}_raw.csv').set_index('Unnamed: 0')
det = pd.read_csv(f'{R}/submissions/sub_xt_Nzz24_detail.csv').set_index('Unnamed: 0')
vids = sys.argv[1].split(',')
LOG = []
orig = M1.track_depth_fn
def wrap(depth, keys, trk):
    f = orig(depth, keys, trk)
    if trk is not None:
        s = np.sqrt(np.array([trk.st[j]['area'] for j in trk.okc], float))
        LOG.append(dict(keys=keys, q=trk.q.get('key'), t=np.round(trk.t, 3).tolist(), s=np.round(s, 1).tolist(),
                        c=[list(np.round(trk.st[j]['c'], 0)) for j in trk.okc], D=None if f is None else np.round(f(trk.t), 3).tolist()))
    return f
M1.track_depth_fn = wrap
fmed = vc.setup(split, set(vids), tracks='v2sc')
byv, vjs = MV.load_split_tracks(split, 'e')
for v in vids:
    rs = byv[v]
    vj, src = vc.load_vj(split, v, 'v2sc')
    print('==', v, src, 'fidx', vj['fidx'][:5], len(vj['fidx']), 'fps', raw[raw.video_id == v].fps.iloc[0])
    LOG.clear()
    out, V = vc.measure(rs, vj, fmed)
    for r in rs:
        i = r['id']
        print(i, r['question'][:70], '| raw', raw.geo_value.get(i), raw.meas_mode.get(i), raw.meas_tracks.get(i), raw.merge_src.get(i),
              '| rem', round(out[i]['value'], 4), out[i]['mode'], '| det', det.answer.get(i), det.answer_source.get(i), det.cs.get(i),
              '| tk', r.get('target_dkeys'), r['q'].get('kind'), r['q'].get('t'), r['q'].get('t1'), r['q'].get('t2'))
    print('depth', rs[0]['depth'])
    seen = set()
    for L in LOG:
        k = (str(L['keys']), L['q'])
        if k in seen: continue
        seen.add(k)
        print('LOG', L['keys'], L['q'])
        for t, s, c, d in zip(L['t'], L['s'], L['c'], L['D'] or [None] * len(L['t'])):
            print(f'   {t:.3f} s={s} c={c} D={d}')
