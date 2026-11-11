import sys, os, json, numpy as np, pandas as pd
A = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, A); import trk
R = trk.R; Q = '--quiet' in sys.argv
c26 = pd.read_csv(f'{R}/submissions/pred_t014_id1.csv').parsed_value.values
det = lambda f: pd.read_csv(f'{A}/det/{f}').assign(w=lambda d: d.x1 - d.x0, h=lambda d: d.y1 - d.y0)
def say(*a):
    if not Q: print(*a)
def inside(d, W, H): return d[(d.x0 > 1) & (d.y0 > 1) & (d.x1 < W - 1) & (d.y1 < H - 1)]
OUT = {}

v, _ = trk.frames('internet_0067', 'acorn|', 'v3ntmv'); v = v.set_index('frame')
m, _ = trk.frames('internet_0067', 'acorn|', ''); m = m.set_index('frame')
g = det('test_internet_0067_acorn_gdino.csv'); g = g[g.idx == 1].set_index('frame'); g['cx'] = (g.x0 + g.x1) / 2
fps = 30.0
h1 = float(v.loc[3:6].eval('y1 - y0').median()); h2 = float(g.h.median())
r1 = {3239: (v.cx[6] - v.cx[5]) * fps * 0.08 / h1, 3240: (v.x0[7] - v.x0[6]) * fps * 0.08 / h1}
r2 = {3239: (g.cx[6] - g.cx[5]) * fps * 0.08 / h2, 3240: (m.x0[7] - m.x0[6]) * fps * 0.08 / h2}
say(f'0067: acorn px height V median f3-6 {h1:.1f}, G-DINO idx1 median f3-6 {h2:.1f}; M (cap) box h f3-6 {m.loc[3:6].eval("y1 - y0").median():.1f}')
for i in (3239, 3240):
    OUT[i] = (r1[i], r2[i]); say(f'  {i}: route1 {r1[i]:.4f} route2 {r2[i]:.4f} m/s (c26c {c26[i]:.4f})')

mpp48 = 0.005904
w = det('test_internet_0048_wall-pad_owlv2.csv'); single = w[(w.w > 80) & (w.w < 115) & (w.h > 250)]
OUT[3102] = (96.3 * mpp48 * 100, float(single.w.median()) * mpp48 * 100)
say(f'0048: OWLv2 single-pad boxes n={len(single)} widths {sorted(single.w.round(1).tolist())} heights median {single.h.median():.1f}')
say(f'  3102: route1 {OUT[3102][0]:.3f} route2 {OUT[3102][1]:.3f} cm (c26c {c26[3102]:.3f})')

def phys(video, rid):
    ts = 'v2sc' if os.path.exists(f'{R}/geo/out/tracks_test_v2sc/{video}.json') else ''
    d, j = trk.frames(video, 'person riding scooter|', ts); d = inside(d.dropna(subset=['x0']), j['W'], j['H'])
    hp = float((d.y1 - d.y0).median()); return c26[rid] / hp, hp, len(d)
s53, hp53, n53 = phys('internet_0053', 3134)
b, j = trk.frames('internet_0053', 'bench|', 'v2sc'); b = inside(b.dropna(subset=['x0']), j['W'], j['H'])
ob = det('test_internet_0053_bench_owlv2.csv'); ob = ob[ob.idx == 0]
OUT[3141] = (float((b.x1 - b.x0).median()) * s53, float(ob.w.median()) * s53)
say(f'0053: rider box h {hp53:.1f} px (n {n53}) -> {s53:.6f} m/px; bench track w {(b.x1 - b.x0).median():.1f} px (n {len(b)}), OWLv2 w {ob.w.median():.1f} h {ob.h.median():.1f}')
say(f'  3141: route1 {OUT[3141][0]:.4f} route2 {OUT[3141][1]:.4f} m (c26c {c26[3141]:.4f}); bench height check OWLv2 h x scale {ob.h.median() * s53:.3f} vs c26c 3142 {c26[3142]:.4f}')
s54, hp54, n54 = phys('internet_0054', 3144)
cv, j = trk.frames('internet_0054', 'canopy|', 'v3ntmv'); cv = cv.dropna(subset=['x0'])
oc = det('test_internet_0054_canopy_owlv2.csv'); oc = oc[(oc.w > 300) & (oc.h < 80)]
OUT[3147] = (float((cv.x1 - cv.x0).median()) * s54, float(oc.w.median()) * s54)
say(f'0054: rider box h {hp54:.1f} px (n {n54}) -> {s54:.6f} m/px; canopy V track w {(cv.x1 - cv.x0).median():.1f} px (n {len(cv)}), OWLv2 strip w {oc.w.median():.1f} (n {len(oc)})')
say(f'  3147: route1 {OUT[3147][0]:.4f} route2 {OUT[3147][1]:.4f} m (c26c {c26[3147]:.4f})')

def ffall(tset):
    d, j = trk.frames('internet_0029', 'ball|', tset); d = d.dropna(subset=['cy']); f = float(j['fps'])
    p = np.polyfit(d.t, d.cy, 2); a = 2 * p[0]; y = lambda t: np.polyval(p, t)
    return (y(0.40) - y(0.30)) * 9.8 / a, a, f
d1, a1, f1 = ffall('v2sc'); d2, a2, f2 = ffall('v3ntmv')
r1_3001 = 121.6 * 9.8 / 3977
OUT[3001] = (r1_3001, d2)
say(f'0029: fps {f1}; a_px M {a1:.0f} V {a2:.0f} px/s^2; 3001: route1 (rul047) {r1_3001:.4f}, M refit {d1:.4f}, route2 (V fit) {d2:.4f} m (c26c {c26[3001]:.4f})')

for i in sorted(OUT):
    print(i, round(float(np.sqrt(OUT[i][0] * OUT[i][1])), 6))
