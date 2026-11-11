import sys
import numpy as np
from scipy.optimize import least_squares
R = '@QP_ROOT@'
sys.path.insert(0, R + '/analysis/v3/ag/rul399')
from trk import load, table, phrase_objs

V3, V2 = ('tracks_test_v3ntmv',), ('tracks_test_v2sc',)


def obj_frames(v, phrase, pref=V3, uncut=True):
    src, j = load(v, pref); T = table(j); ob = phrase_objs(j)[phrase]
    d = T[T.obj == ob].sort_values('f')
    return (d[d.border == 0] if uncut else d), float(j['fps'])


def med(d, col):
    return float(np.median(d[col]))


out = {}
d, fps = obj_frames('simulation_0253', 'box', V3, uncut=False)
mpp = 0.611 / 42.4
u = np.array([d.cx.values[-1] - d.cx.values[0], d.cy.values[-1] - d.cy.values[0]]); u /= np.linalg.norm(u)
dd = d[d.border == 0]; t = dd.t.values
s = ((dd.cx.values - d.cx.values[0]) * u[0] + (dd.cy.values - d.cy.values[0]) * u[1]) * mpp
pc = np.polyfit(t, s, 3); acub = np.polyder(pc, 2)
def model(q, tt):
    s0, v0, a0, c = q; c = max(c, 1e-6)
    return s0 + a0 / c * tt - (a0 / c - v0) * (1 - np.exp(-c * tt)) / c
q = least_squares(lambda q: model(q, t) - s, [0, 1, 5, 0.1]).x
vdam = lambda tc: q[2] / q[3] - (q[2] / q[3] - q[1]) * np.exp(-q[3] * tc)
adam = lambda tc: q[2] - q[3] * vdam(tc)
for rid, tc in (('859', 1.0), ('860', 1.5), ('861', 1.75)):
    out[rid] = 0.5 * (np.polyval(acub, tc) + adam(tc))
vac, _ = obj_frames('simulation_0222', 'vacuum cleaner', V3)
mpp22 = 0.438 / med(vac, 'rw')
bench22, _ = obj_frames('simulation_0222', 'wooden bench', V3)
shade22, _ = obj_frames('simulation_0222', 'lampshade', V3)
stand22_v2, _ = obj_frames('simulation_0222', 'tv stand', V2, uncut=False)
stand16, _ = obj_frames('simulation_0216', 'tv stand', V3, uncut=False)
bench16, _ = obj_frames('simulation_0216', 'wooden bench', V3, uncut=False)
shade16, _ = obj_frames('simulation_0216', 'lampshade', V3, uncut=False)
px_a = med(bench22, 'bw') * med(stand16, 'bw') / med(bench16, 'bw')
over = med(shade22, 'bw') * (med(shade16, 'x1') - med(stand16, 'x1')) / med(shade16, 'bw')
px_b = med(stand22_v2, 'bw') - over
out['666'] = 0.5 * (px_a + px_b) * mpp22
out['665'] = med(bench22, 'bh') * mpp22
spd = []; hts = []
for v in ('simulation_0231', 'simulation_0232'):
    c, fps = obj_frames(v, 'black cat', V3, uncut=False)
    spd.append(np.hypot(c.cx.values[-1] - c.cx.values[0], c.cy.values[-1] - c.cy.values[0]) / (c.t.values[-1] - c.t.values[0]))
    hts.append(med(c, 'bh'))
mpp_cat = 0.714 / np.mean(spd)
cat30, _ = obj_frames('simulation_0230', 'black cat', V2)
out['732'] = med(cat30, 'rw') * mpp_cat
out['733'] = float(np.mean(hts)) * mpp_cat
ring, _ = obj_frames('simulation_0113', 'swim ring', V3); pool, _ = obj_frames('simulation_0113', 'pool', V3)
box_r = np.median((pool.bw + pool.bh) / 2) / np.median((ring.bw + ring.bh) / 2)
rr_r = np.median((pool.rw + pool.rh) / 2) / np.median((ring.rw + ring.rh) / 2)
out['231'] = 0.5 * (box_r + rr_r) * 1.0
ball, _ = obj_frames('simulation_0160', 'ball', V3); tab, _ = obj_frames('simulation_0160', 'billiard table', V3)
ball_px = float(np.median((ball.bw + ball.bh + ball.rw + ball.rh) / 4))
out['372'] = med(tab, 'bh') * 0.0757 / ball_px
out['854'] = 0.5 * (np.polyval(np.polyder(pc), 0.5) + vdam(0.5))
tires = []
for v in ('simulation_0145', 'simulation_0145_segmented'):
    rk, _ = obj_frames(v, 'rocket', V3); rk = rk[rk.rw > 90]
    ti, _ = obj_frames(v, 'tire', V3)
    tires.append(float(np.median((ti.bw + ti.bh) / 2)) * 2.9 / med(rk, 'rw'))
out['338'] = out['1848'] = float(np.mean(tires))

for k in sorted(out, key=int):
    print(k, round(float(out[k]), 6))

