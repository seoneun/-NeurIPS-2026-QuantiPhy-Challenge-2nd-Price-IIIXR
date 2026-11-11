import json, os, sys
import numpy as np, pandas as pd
R = '@QP_ROOT@'


def load(v, d):
    p = f'{R}/geo/out/{d}/{v}.json'
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    return j if 'objs' in j else None


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
                             bw=x1 - x0, bh=y1 - y0, rw=rr[0], rh=rr[1], border=b))
    return pd.DataFrame(rows)


def phrases(j):
    out = {}
    for q in j.get('queries', []):
        if q.get('obj') is not None:
            out.setdefault(int(q['obj']), set()).add(q['phrase'].lower()[:28])
    return out


def mpp_of(v):
    p = f'{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv'
    if not hasattr(mpp_of, 'df'):
        mpp_of.df = pd.read_csv(p)
    d = mpp_of.df
    s = pd.to_numeric(d[d.video_id == v]['meas_mpp'], errors='coerce')
    return float(np.nanmedian(s)) if len(s) else np.nan


def summary(v):
    print('')
    for d, tag in (('tracks_test_v2sc', 'M'), ('tracks_test_v3ntmv', 'V')):
        j = load(v, d)
        if j is None:
            print(''); continue
        t = table(j); ph = phrases(j); fps = float(j['fps'])
        print('')
        for o, g in t.groupby('obj'):
            g = g.sort_values('f'); nb = g[g.border == 0]
            sp = np.nan
            if len(nb) > 3:
                df = np.diff(nb.f.values); ok = df > 0
                dd = np.hypot(np.diff(nb.cx.values), np.diff(nb.cy.values))[ok] / df[ok] * fps
                sp = float(np.median(dd))
            print('')


if __name__ == '__main__':
    for v in sys.argv[1:]:
        summary(v)
