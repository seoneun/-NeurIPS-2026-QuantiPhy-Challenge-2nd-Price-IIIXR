import sys, numpy as np, pandas as pd
F = 0.5959 * 1280; CX, CY = 640, 360
def P(x, y, Z): return Z * np.array([(x - CX) / F, (y - CY) / F, 1.0])
t = pd.read_csv('output/work/labtime/trk_captured_0009.csv').set_index('n')
con = int(t.loc[30:42, 'bot'].idxmax()); apex = int(t.loc[con + 1:, 'cy'].idxmin())
pf = P(t.cx[con], t.bot[con], 2.0285)
p0 = P(t.cx[0], t.cy[0], 1.3210)
pa = P(t.cx[apex], t.cy[apex], 1.5795)
h0, ha = np.linalg.norm(p0 - pf) * 100, np.linalg.norm(pa - pf) * 100
out = [(i, round(h0, 1)) for i in (2175, 2180, 2185)] + [(i, round(ha, 1)) for i in (2177, 2182, 2187)]
pd.DataFrame(out, columns=['id0', 'value']).to_csv(sys.argv[1], index=False)
