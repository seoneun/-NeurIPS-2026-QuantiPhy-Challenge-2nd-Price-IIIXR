import os, sys, json, copy
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
FF = json.load(open(f'{R}/geo/out/file_fps_test.json'))
QO = {'size': 0, 'dist2': 0, 'other': 0, 'dist': 0, 'disp': 0, 'path': 0, 'maxheight': 0, 'speed': 1, 'avgspeed': 1,
      'hvel': 1, 'vvel': 1, 'accel': 2}
PO = {'size': 0, 'speed': 1, 'accel': 2, 'gravity': 2}


def qnorm(s):
    return ' '.join(str(s).lower().split())


def detect(summary):
    s = pd.read_csv(summary)
    s = s[(s.split == 'test') & s.video.str.startswith('internet') & ~s.video.str.endswith('_segmented')]
    s = s[(s.pred_drop >= 0.10) & (s.series.str.startswith('trk') | (s.series == 'f99')) & (s.med_x >= 2)]
    ok = (s.frac_dbl >= s.pred_drop / 2) & ((s.gap_mode - np.round(1 / s.pred_drop)).abs() <= 1) & (s.gap_mode_n >= 8)
    return sorted(set(s[ok].video))




def compute(summary):
    vids = detect(summary)
    sys.path.insert(0, f'{R}/analysis/v3/ag/holder')
    import remeasure as RM
    MV = RM.MV
    Fr = pd.read_csv(f'{T5}/free_rows.csv')
    Pq = {}
    for l in open(f'{R}/geo/out/parse_test_v2.jsonl'):
        r = json.loads(l); Pq[r['id']] = r
    fmed = RM.setup('test', vids=set(vids))
    byv, vjs = MV.load_split_tracks('test', 'e')
    out = []
    for v in vids:
        o1, _ = RM.measure(byv[v], vjs[v], fmed)
        vj2 = copy.deepcopy(vjs[v]); vj2['fps'] = float(FF[v]['file'])
        o2, _ = RM.measure(byv[v], vj2, fmed)
        r_ = FF[v]['file'] / FF[v]['csv']
        f = Fr[Fr.video == v]
        for _, x in f.iterrows():
            q = Pq[x.id0]; ts = [q['q'].get(k) for k in ('t', 't1', 't2')]; timed = any(t for t in ts if t is not None)
            e = QO.get(x.kind, np.nan) - PO.get(x.prior_kind, np.nan)
            if not timed and (not np.isfinite(e) or e == 0):
                continue
            a, b = o1.get(x.id0, {}).get('value', np.nan), o2.get(x.id0, {}).get('value', np.nan)
            if e == 0:
                continue
            if not timed:
                new = x.c26c * r_ ** e
            elif np.isfinite(a) and np.isfinite(b) and a > 0:
                new = x.c26c * b / a
            elif np.isfinite(b):
                sib = [y for y in f.itertuples() if y.id0 != x.id0 and qnorm(y.question).split(' at ')[0] == qnorm(x.question).split(' at ')[0]
                       and np.isfinite(o1.get(y.id0, {}).get('value', np.nan))]
                if not sib:
                    continue
                y = sib[0]; sc = y.c26c / o1[y.id0]['value']
                new = b * sc
            else:
                continue
            if not np.isfinite(new) or abs(new / x.c26c - 1) < 0.005:
                continue
            out.append(dict(id0=int(x.id0), new=new))
    return pd.DataFrame(out)


