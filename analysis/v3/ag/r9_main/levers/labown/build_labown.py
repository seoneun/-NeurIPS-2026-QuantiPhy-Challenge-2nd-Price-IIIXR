import sys, re, numpy as np, pandas as pd
F = 0.5959 * 1280; CX, CY = 640, 360
def P(x, y, Z): return Z * np.array([(x - CX) / F, (y - CY) / F, 1.0])
q = pd.read_parquet('data/test/test_dataset.parquet').reset_index(drop=True)
def key(di, pat): return float(re.search(pat + r'\s*=\s*([0-9.]+)m', di).group(1))
out = []
for v, (i0, ia) in {'captured_0009s': (2180, 2182), 'captured_0009x': (2185, 2187)}.items():
    di = q.loc[i0].depth_info
    zb0 = key(di, r't=0\.0s, distance_ball_camera'); zf = key(di, r'distance_floor_camera')
    za = key(di, r't=1\.92s \(max rebounce\), distance_ball_camera')
    t = pd.read_csv(f'output/work/labown/trk_{v}.csv').set_index('n')
    con = int(t.loc[30:42, 'bot'].idxmax()); apex = int(t.loc[con + 1:, 'cy'].idxmin())
    pf = P(t.cx[con], t.bot[con], zf); p0 = P(t.cx[0], t.cy[0], zb0); pa = P(t.cx[apex], t.cy[apex], za)
    out += [(i0, round(np.linalg.norm(p0 - pf) * 100, 1)), (ia, round(np.linalg.norm(pa - pf) * 100, 1))]
pd.DataFrame(out, columns=['id0', 'value']).to_csv(sys.argv[1], index=False)
