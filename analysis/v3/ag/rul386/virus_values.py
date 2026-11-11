import json
import numpy as np
R = '@QP_ROOT@'

def series(v, o):
    js = json.load(open(f'{R}/geo/out/tracks_test_v2sc/{v}.json')); out = []
    for i, f in enumerate(js['objs'].get(str(o), [])):
        st = f.get('st') if isinstance(f, dict) else None
        if st and st.get('box'):
            out.append(dict(f=js['fidx'][i], t=js['fidx'][i] / js['fps'], cx=st['c'][0], cy=st['c'][1],
                            box=st['box'], bd=int(bool(st.get('border', 0)))))
    return out, js['fps']

def fit(s, key):
    t = np.array([r['t'] for r in s]); y = np.array([r[key] for r in s])
    p = np.polyfit(t, y, 1); return p, float(np.sqrt(np.mean((y - np.polyval(p, t)) ** 2)))

L, fps = series('simulation_0106', 3); Rv, _ = series('simulation_0106', 1)
Lc = [r for r in L if not r['bd']]; Rc = [r for r in Rv if not r['bd']]
pLx, _ = fit(Lc, 'cx'); pLy, rL = fit(Lc, 'cy'); pRx, _ = fit(Rc, 'cx'); pRy, rR = fit(Rc, 'cy')
vL = np.hypot(pLx[0], pLy[0]); vR = np.hypot(pRx[0], pRy[0])
for name, s, k, idx in [('right virus top edge y0', Rv, 'y0', 1), ('left virus bottom edge y1', L, 'y1', 3)]:
    rows = [dict(t=r['t'], e=r['box'][idx]) for r in s if r['bd']]
    rows = [r for r in rows if 0 < r['e'] < 479]
    if len(rows) >= 4:
        t = np.array([r['t'] for r in rows]); e = np.array([r['e'] for r in rows]); p = np.polyfit(t, e, 1)
        print('')
mL = 1.7393 / vL; mR = 1.5667 / vR
print('')
routeA = 1.7393; routeB = 1.5667 * vL / vR
print('')
cx0, cy0 = 427.0, 240.0
for t, rid, cur in [(1.0, 215, 5.2794), (2.0, 216, 7.0430)]:
    xl, yl = np.polyval(pLx, t), np.polyval(pLy, t); xr, yr = np.polyval(pRx, t), np.polyval(pRy, t)
    dpx = np.hypot(xl - xr, yl - yr)
    per = np.hypot((xl - cx0) * mL - (xr - cx0) * mR, (yl - cy0) * mL - (yr - cy0) * mR)
    print('')
    print(f'VALUE {rid} {per:.6f}')
print(f'VALUE 214 {routeA:.6f}')
