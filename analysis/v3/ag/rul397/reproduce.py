import json, re
import numpy as np
import pandas as pd

R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
C = f'{R}/analysis/v3/ag/rul397/frozen' if R5FZ else f'{R}/cache/tmp/rul397'
F = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
det = lambda v, p: pd.read_csv(f'{C}/det/test_{v}_{p}_owlv2.csv')
meas = lambda v: pd.read_csv(f'{C}/orig/measure_test_{v}_orig.csv').set_index('id0')
out, info = {}, []


def boxes(d):
    d = d.copy(); d['cx'] = (d.x0 + d.x1) / 2; d['cy'] = (d.y0 + d.y1) / 2
    d['w'] = d.x1 - d.x0; d['h'] = d.y1 - d.y0; d['L'] = d[['w', 'h']].max(axis=1)
    return d


c = boxes(det('internet_0072', 'credit-card')); c = c[c.idx == 0].set_index('frame')
fps = 30.0
cm_px = 8.56 / np.median([c.L[f] for f in (14, 15, 16)])
step = lambda a, b: float(np.hypot(c.cx[b] - c.cx[a], c.cy[b] - c.cy[a]))
cd = step(14, 16) / 2 * fps * cm_px; fd = step(15, 16) * fps * cm_px; bd = step(14, 15) * fps * cm_px
r1 = float(meas('internet_0072').loc[3271, 'prodM'])
out[3271] = r1
info.append(f'3271 route1 chain geo {r1:.2f}; route2 OWLv2 boxes, ruler {8.56 / cm_px:.1f} px: CD {cd:.2f} FD {fd:.2f} BD {bd:.2f} cm/s')
a = abs(step(19, 20) - step(12, 13)) * fps / (7 / fps) * cm_px
info.append(f'3273 check: |a| from detections {a:.0f} cm/s^2 (cur {F.c26c[3273]:.1f})')

p = boxes(det('internet_0033', 'plate-of-tomatoes')); p = p[(p.idx == 0) & (p.w > 30) & (p.w < 60)].set_index('frame')
fk = boxes(det('internet_0033', 'fork')); fk = fk[fk.idx == 0]
m_px = 0.203 / float(np.median(fk.L))
fps = 25.0


def pos(f):
    q = p.loc[[g for g in (f - 1, f, f + 1) if g in p.index]]
    return np.array([q.cx.median(), q.cy.median()])


st = json.load(open(f'{R}/patches/ag_rul425_patch_test.json'))
for rid, t1, t2 in ((3021, 0.0, 0.52), (3022, 1.0, 1.6)):
    f1, f2 = int(round(t1 * fps)), int(round(t2 * fps))
    r2 = float(np.linalg.norm(pos(f2) - pos(f1))) * m_px / (t2 - t1)
    r1 = float(re.search(r'2D geo ([0-9.]+)', st[str(rid)]['why']).group(1))
    out[rid] = float(np.sqrt(r1 * r2))
    info.append(f'{rid} route1 rul425 geo {r1:.4f}; route2 OWLv2 plate f{f1}-f{f2}, fork ruler {0.203 / m_px:.1f} px: {r2:.4f} m/s; '
                f'new = geometric mean {out[rid]:.4f}')

q = open(f'{C}/orig/q_0003.txt').read()
wpz = float(re.search(r"key 'pizza\|'.*?med box (\d+)x(\d+)", q).group(1))
cm_px = float(meas('internet_0003').loc[2021, 'prodM']) / wpz
pp = boxes(det('internet_0003', 'pepperoni')); pp = pp[pp.score >= 0.3]
dpx = float(np.median((pp.w + pp.h) / 2))
for rid in (2024, 2955):
    out[rid] = dpx * cm_px
info.append('t21')

tw = json.load(open(f'{R}/patches/ag_twin_patch_test.json'))
g = float(re.search(r'= ([0-9.]+) \(was', tw['3030']['why']).group(1))
out[3030] = float(F.c26c[3028]) * 0.83
info.append('t22')

for k in sorted(out):
    print(f'{k}, {out[k]:.6g}')
print('\n'.join(info))
