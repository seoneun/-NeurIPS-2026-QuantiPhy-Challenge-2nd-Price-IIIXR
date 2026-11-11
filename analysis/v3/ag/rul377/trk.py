import json, numpy as np, pandas as pd
R = '@QP_ROOT@'

def load(video, tset='v2sc', split='test'):
    p = f'{R}/geo/out/tracks_{split}_{tset}/{video}.json' if tset else f'{R}/geo/out/tracks_{split}/{video}.json'
    return json.load(open(p))

def frames(video, key, tset='v2sc', split='test', j=None):
    j = j or load(video, tset, split)
    q = next(q for q in j['queries'] if q['key'] == key)
    fr = j['objs'][str(q['obj'])] if q.get('obj') is not None else q.get('frames')
    rows = []
    for k, e in enumerate(fr):
        st = e.get('st') if e else None
        f = int(j['fidx'][k]); d = dict(k=k, frame=f, t=f / float(j['fps']))
        if st:
            b = st['box']; d.update(x0=b[0], y0=b[1], x1=b[2], y1=b[3], cx=st['c'][0], cy=st['c'][1], area=st['area'],
                                    rr_long=st['rr'][0], rr_short=st['rr'][1], border=int(bool(st['border'])))
        rows.append(d)
    return pd.DataFrame(rows), j
