import sys, numpy as np, pandas as pd
base, W, out = sys.argv[1:4]
D = 'analysis/v3/ag/r9_main/levers/revisit/in'
b = pd.read_csv(base); b['id0'] = b.id - 1; b = b.set_index('id0'); cur = b.parsed_value; qst = b.question

F = 0.5959 * 1280; CX, CY = 640, 360
P = lambda x, y, Z: Z * np.array([(x - CX) / F, (y - CY) / F, 1.0])
rows = []
ids = {'captured_0008': (2160, 2162, 2163), 'captured_0008s': (2165, 2167, 2168), 'captured_0008x': (2170, 2172, 2173)}
for v, (i_init, i_reb, i_vel) in ids.items():
    t = pd.read_csv(f'{W}/trk_{v}.csv').set_index('n')
    con = int(t.loc[25:35, 'bot'].idxmax()); apex = int(t.loc[con + 1:con + 10, 'cy'].idxmin())
    pf = P(t.cx[con], t.bot[con], 1.725)
    k24 = int(round(1.0 * 24))
    h0 = np.linalg.norm(P(t.cx[k24], t.cy[k24], 1.401) - pf) * 100
    ha = np.linalg.norm(P(t.cx[apex], t.cy[apex], 1.5225) - pf) * 100
    rows += [(i_init, round(h0, 1)), (i_reb, round(ha, 1))]
    w = (t.right - t.left).astype(float); Rb = 0.116
    K = np.mean([(1.401 + Rb) * w[k24], (1.5225 + Rb) * w[apex]])
    n = np.arange(int(t.index[(t.cy.diff() > 2)].min()), con)
    Zc = K / w[n].values; Pc = np.array([Zc * (t.cx[n].values - CX) / F, Zc * (t.cy[n].values - CY) / F, Zc]).T
    tt = n / 24.0; fits = [np.polyfit(tt, Pc[:, j], 2) for j in range(3)]
    vel = lambda T: np.array([np.polyval(np.polyder(p), T) for p in fits])
    ts = np.linspace(0.6, 1.25, 1301); sp = np.array([np.linalg.norm(vel(x)) for x in ts]); t_r = ts[sp.argmin()]
    rows.append((i_vel, round(9.8 * (1.2 - t_r), 3)))
it1 = pd.DataFrame(rows, columns=['id0', 'value'])

FLAP = r'bird|eagle|phoenix|dragon|butterfly|bat|owl|parrot|crow|gull|duck|hawk|falcon'
wg = pd.read_csv(f'{D}/wing_rows.csv', float_precision='round_trip'); wg = wg[wg.id0.map(qst).str.contains(FLAP, case=False)]
it2 = pd.DataFrame({'id0': wg.id0, 'value': wg.id0.map(cur) * np.sqrt(wg.g_w5 / wg.g)})

c = pd.read_csv(f'{D}/c8_rows.csv', float_precision='round_trip')
c = c[~c.id0.map(qst).str.contains(r'speed|velocity|acceleration|how fast', case=False)]
c['a'] = c.base * np.where(np.isfinite(c.w), c.f ** c.w, 1.0)
k = c[np.isclose(c.id0.map(cur), c.base, rtol=1e-6) & (np.abs(c.a / c.base - 1) > 1e-9)]
it3 = pd.DataFrame({'id0': k.id0, 'value': k.a})

T = pd.read_csv(f'{D}/r02_rows.csv', float_precision='round_trip').set_index('id0'); TSTAR = 0.6
GM = ['dis_size_gm', 'rul263_dimgm', 'd3lab_gm', 'rul263_unstable_gm']
g, cs = T.g.astype(float), T.cs.astype(float); lo, hi = np.minimum(g, cs), np.maximum(g, cs)
sc = T.answer_source.isin(GM) & (hi / lo >= 1.5) & (T.answer > lo * 1.0001) & (T.answer < hi / 1.0001) & np.isclose(T.index.map(cur), T.answer, rtol=1e-6)
it4 = pd.DataFrame({'id0': T.index[sc], 'value': lo[sc] ** (1 - TSTAR) * hi[sc] ** TSTAR})

kc = pd.read_csv(f'{D}/kinoff_rows.csv', float_precision='round_trip'); kc['cur'] = kc.id0.map(cur)
kc = kc[((1 - kc.cur / kc.a_t) * kc.tk * kc.fps).abs() < 0.02].copy()
kc['value'] = kc.a_t * (1 - 0.2 / (kc.tk * kc.fps))
kc = kc[(kc.value / kc.cur - 1).abs() > 1e-6]
it6 = kc[['id0', 'value']]

for i, d in ((1, it1), (2, it2), (3, it3), (4, it4), (6, it6)): d.to_csv(f'{W}/item{i}.csv', index=False)
it = {i: pd.read_csv(f'{W}/item{i}.csv') for i in (1, 2, 3, 4, 6)}
it[3] = it[3][~it[3].id0.isin(it[4].id0)]
allr = pd.concat([it[i] for i in (1, 2, 3, 4, 6)]); assert allr.id0.is_unique
allr.to_csv(out, index=False)
