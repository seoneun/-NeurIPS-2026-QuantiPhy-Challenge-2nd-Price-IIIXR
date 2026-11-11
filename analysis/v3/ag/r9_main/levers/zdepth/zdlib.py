import re, json, difflib, os, numpy as np
ROOT = '.'
ORDER = ['v2sc', 'v2', 'v3ntmv', '']
T24 = lambda k: k / 24.0
T25 = lambda k: np.floor(k * 25 / 24 + 0.5) / 25.0
def keys(di):
    out = []
    for line in str(di).replace(';', '\n').split('\n'):
        m = re.search(r'distance_([a-z_]+?)_camera\s*=\s*([\d.]+)\s*(cm|m)?', line.lower())
        if not m: continue
        tm = re.search(r't\s*=\s*([\d.]+)\s*s', line.lower())
        out.append(dict(name=m.group(1).replace('_', ' '), t=float(tm.group(1)) if tm else None,
                        z=float(m.group(2)) / (100 if m.group(3) == 'cm' else 1)))
    return out
def load(vid, order=ORDER):
    for td in order:
        f = f'{ROOT}/geo/out/tracks_test' + ('_' + td if td else '') + f'/{vid}.json'
        if os.path.exists(f):
            d = json.load(open(f))
            if 'objs' in d and 'queries' in d: return td, d
    return None, None
def centres(d, oid):
    o = d['objs'][str(oid)]
    st = [(json.loads(x['st'].replace("'", '"')) if isinstance(x['st'], str) else x['st']) if x and x.get('st') else None for x in o]
    c = np.array([s['c'] if s else [np.nan, np.nan] for s in st], float)
    b = np.array([s['box'] if s else [np.nan] * 4 for s in st], float)
    fi = np.array(d.get('fidx') or list(range(len(o))), float)[:len(o)]
    return fi, c, b
def sim(a, b): return difflib.SequenceMatcher(None, a, b).ratio()
def obj_for(d, name):
    qs = [q for q in d['queries'] if q.get('obj') is not None]
    if not qs: return None, None, 0
    q = max(qs, key=lambda q: sim(q['phrase'], name)); return q['obj'], q['phrase'], sim(q['phrase'], name)
def at(fi, c, t, tmap):
    tt = np.array([tmap(k) for k in fi]); m = np.isfinite(c[:, 0])
    if not m.any(): return np.array([np.nan, np.nan])
    i = np.argsort(np.abs(tt - t) + np.where(m, 0, 99))[:2]
    if np.abs(tt[i] - t).min() > 0.1: return np.array([np.nan, np.nan])
    w = 1 / (np.abs(tt[i] - t) + 1e-3); return (c[i] * w[:, None]).sum(0) / w.sum()
def P(W, H, u, v, z, conv):
    fp = 0.596 * W; x, y = (u - W / 2) / fp, (v - H / 2) / fp
    if conv == 'range': z = z / np.sqrt(1 + x * x + y * y)
    return z * np.array([x, y, 1.0])
