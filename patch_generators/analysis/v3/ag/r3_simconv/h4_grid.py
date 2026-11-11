"""H4: grid of label shifts s (frames): y = a (tk - s/fps) on the 36 drop rows (s_d) and on the 48 other kinoffset rows (s_o);
prints the (s_d, s_o) pairs whose predicted probe3 delta fits both portal readings (D2 row-MRA in (-4.44,-2.22), D3 in
(-1.76, 0.0)). Also checks best17v == best18v on scored rows and lists validation free-fall / from-rest speed rows."""
import re, sys
import numpy as np, pandas as pd
R = '/data/02_Work/QuantiPhy-Challenge-2026'
sys.path.insert(0, f'{R}/analysis/v3/calib'); import fusion_calib as FC
M = pd.read_csv(f'{R}/analysis/v3/calib/r3/inputs/scored_mask_3010.csv'); sc = M.scored.values.astype(bool)
b17 = pd.read_csv(f'{R}/submissions/sub_xt_best17v_id1.csv').parsed_value.values
b18 = pd.read_csv(f'{R}/submissions/sub_xt_best18v_id1.csv').parsed_value.values
print('best17v vs best18v differing scored rows:', int((~np.isclose(b17, b18, rtol=1e-9) & sc).sum()))
K = pd.read_csv(f'{R}/analysis/v3/ag/r3_simconv/h4_kinoffset_rows.csv')
S = np.round(np.arange(0, 1.01, 0.1), 2); ok = []
def dl(y):
    g = FC.row_mra(K.p3.values, y) - FC.row_mra(K.b18.values, y)
    return g[K['cat'].values == 'D2'].sum(), g[K['cat'].values == 'D3'].sum()
for sd in S:
    for so in S:
        s = np.where(K['drop'].values, sd, so); y = K.a.values * (K.tk.values - s / K.fps.values)
        d2, d3 = dl(y)
        if -4.44 < d2 < -2.22 and -1.76 < d3 < 0.0: ok.append((sd, so, round(d2, 1), round(d3, 1)))
print('consistent (s_drop, s_other, D2, D3):', ok)
for sd in S:
    y = K.a.values * (K.tk.values - np.where(K['drop'].values, sd, 0) / K.fps.values); print('s_d=%.1f s_o=0: D2 %.1f D3 %.1f' % ((sd,) + dl(y)), end=' | ')
print()
V = pd.read_csv(f'{R}/analysis/v3/calib/audit_val.csv')
m = V.question.str.contains(r'speed|velocity|acceleration', case=False) & (V.prior.str.contains(r'acceleration|9\.8|9\.7|grav', case=False, na=False) | V.question.str.contains('fall|drop', case=False))
for _, r in V[m & (V.src_type == 'simulation')].iterrows():
    print(r.id0, r.video, r['cat'], '|', r.question[:90], '| prior:', str(r.prior)[:70], '| y=', r.y)
