import json, os
import numpy as np, pandas as pd
R = '@QP_ROOT@'


def load(v, prefer=('tracks_test_v2sc', 'tracks_test_v3ntmv', 'tracks_test')):
    for d in prefer:
        p = f'{R}/geo/out/{d}/{v}.json'
        if os.path.exists(p):
            j = json.load(open(p))
            if 'objs' in j:
                return d, j
    return None, None


def phrase_objs(j):
    out = {}
    for q in j.get('queries', []):
        if q.get('obj') is not None:
            out.setdefault(q['phrase'].lower(), q.get('obj'))
    return out


def table(j):
    rows = []
    fps = float(j['fps'])
    for o, frs in j['objs'].items():
        for pos, fr in enumerate(frs):
            st = fr.get('st') if isinstance(fr, dict) else None
            if not st or not st.get('box'):
                continue
            x0, y0, x1, y1 = st['box']
            rr = st.get('rr') or [np.nan, np.nan, np.nan]
            f = j['fidx'][pos]
            b = int(x0 <= 2 or y0 <= 2 or x1 >= j['W'] - 3 or y1 >= j['H'] - 3)
            rows.append(dict(obj=int(o), f=f, t=f / fps, cx=st['c'][0], cy=st['c'][1], x0=x0, y0=y0, x1=x1, y1=y1,
                             bw=x1 - x0, bh=y1 - y0, rw=rr[0], rh=rr[1], area=st.get('area', np.nan), border=b))
    return pd.DataFrame(rows)


def local_acc(t, x, y, tc, half):
    m = np.abs(t - tc) <= half + 1e-9
    if m.sum() < 4:
        return np.nan, np.nan, int(m.sum())
    px = np.polyfit(t[m] - tc, x[m], 2); py = np.polyfit(t[m] - tc, y[m], 2)
    a = np.hypot(2 * px[0], 2 * py[0]); v = np.hypot(px[1], py[1])
    return a, v, int(m.sum())
