import json, os, re, sys
import numpy as np, pandas as pd
R = '@QP_ROOT@'
t = pd.read_csv(f'{R}/analysis/v3/calib/audit_test.csv')

def depth(v):
    q = ' '.join(t[t.video == v].depth.dropna().astype(str).unique())
    out = {}
    for m in re.finditer(r't=([\d.]+)s, distance_(\w+?)_camera = ([\d.]+) ?m', q):
        out[(m.group(2), float(m.group(1)))] = float(m.group(3))
    return out

def track(v):
    tj = next((f'{R}/geo/out/{d}/{v}.json' for d in ('tracks_test_v2sc', 'tracks_test_v3ntmv', 'tracks_test')
               if os.path.exists(f'{R}/geo/out/{d}/{v}.json')))
    return json.load(open(tj))

def boxes(js, phrase_re):
    q = next((q for q in js['queries'] if re.search(phrase_re, q['phrase']) and q.get('obj') is not None), None)
    if q is None: return None
    arr = np.full((js['N'], 4), np.nan)
    for i, fi in enumerate(js['fidx']):
        fr = js['objs'][str(q['obj'])][i]
        st = fr.get('st') if isinstance(fr, dict) else None
        if st and st.get('box'): arr[fi] = st['box']
    return arr

rows = []
for n in sys.argv[1:]:
    v = f'simulation_{n}'; js = track(v); H, W = js['H'], js['W']; d = depth(v)
    ped = boxes(js, r'pedestrian'); sl = boxes(js, r'streetlight'); tl = boxes(js, r'traffic light')
    whl = boxes(js, r'wheel'); car = boxes(js, r'turning yellow car')
    res = {'v': n}
    def fz(b, Z, Hm):
        h = b[3] - b[1]; r2 = ((b[0] + b[2]) / 2 - W / 2) ** 2 + ((b[1] + b[3]) / 2 - H / 2) ** 2
        return np.sqrt(max((h * Z / Hm) ** 2 - r2, 1.0))
    for nm, b, key, Hm in (('fped', ped, 'human', 1.87), ('fwhl', whl, 'bike', 0.56), ('fcar', car, 'yellow_car', 1.65)):
        for tt, fi in ((0.58, 14), (1.58, 38)):
            Z = d.get((key, tt))
            res[f'{nm}{fi}'] = round(fz(b[fi], Z, Hm)) if b is not None and Z and not np.isnan(b[fi]).any() else np.nan
    if ped is not None:
        ok = ~np.isnan(ped).any(1) & (ped[:, 1] > 2) & (ped[:, 3] < H - 2)
        hp = ped[ok, 3] - ped[ok, 1]; yb = ped[ok, 3]
        A = np.c_[np.ones_like(hp), hp]; (yh, k), *_ = np.linalg.lstsq(A, yb, rcond=None)
        res.update(yh=round(yh, 1), k=round(k, 3), hc=round(1.87 * k, 2), pyb=f'{yb.min():.0f}-{yb.max():.0f}',
                   fitres=round(np.std(yb - A @ np.array([yh, k])), 1))
        for nm, b in (('SL', sl), ('TLhead', tl)):
            if b is None: continue
            bb = np.nanmedian(b, 0); hS = bb[3] - bb[1]
            res[f'{nm}box'] = ','.join(f'{x:.0f}' for x in bb)
            res[f'{nm}_m'] = round(1.87 * k * hS / (bb[3] - yh), 2)
    rows.append(res)
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).to_string(index=False))
