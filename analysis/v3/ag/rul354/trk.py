import json, os
import numpy as np
import pandas as pd

R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))


def load(v, dirs=('tracks_test_v2sc', 'tracks_test_v3ntmv')):
    for d in dirs:
        p = os.path.join(R, 'geo/out', d, v + '.json')
        if os.path.exists(p):
            break
    else:
        return None, {}
    js = json.load(open(p))
    fps = float(js['fps'])
    fidx = np.array(js['fidx'], float)
    out = {}
    for q in js['queries']:
        o = q.get('obj')
        if o is None or str(o) not in js.get('objs', {}):
            continue
        rows = []
        for f, fr in zip(fidx, js['objs'][str(o)]):
            st = fr.get('st') if isinstance(fr, dict) else None
            if not st or not st.get('box'):
                continue
            x0, y0, x1, y1 = st['box']
            rr = st.get('rr') or [np.nan, np.nan, np.nan]
            rows.append(dict(f=int(f), t=f / fps, cx=st['c'][0], cy=st['c'][1], bx=(x0 + x1) / 2, by=(y0 + y1) / 2,
                             bw=x1 - x0, bh=y1 - y0, rw=max(rr[0], rr[1]), rh=min(rr[0], rr[1]),
                             area=st.get('area', np.nan), border=st.get('border', 0)))
        key = q['phrase']
        while key in out:
            key += "'"
        out[key] = pd.DataFrame(rows)
    return (fps, d), out


def fitpar(t, s, t0=-1e9, t1=1e9):
    t = np.asarray(t, float); s = np.asarray(s, float)
    m = (t >= t0) & (t <= t1) & np.isfinite(s)
    if m.sum() < 4:
        return np.nan, np.nan, np.nan, np.nan
    c = np.polyfit(t[m], s[m], 2)
    rms = float(np.sqrt(np.mean((np.polyval(c, t[m]) - s[m]) ** 2)))
    return c[2], c[1], 2 * c[0], rms


def axis(df, x='cx', y='cy'):
    P = df[[x, y]].values.astype(float)
    P0 = P - P.mean(0)
    u = np.linalg.svd(P0, full_matrices=False)[2][0]
    d = P[-1] - P[0]
    if np.dot(u, d) < 0:
        u = -u
    return u, (P - P[0]) @ u
