import sys, numpy as np, pandas as pd
R = '@QP_ROOT@'; C9 = f'{R}/analysis/v3/calib/r9'
W, H45 = sys.argv[1], sys.argv[2]; TAG, TAG2 = sys.argv[3], sys.argv[4]
rd = lambda p: pd.read_csv(p).parsed_value.astype(float).values
eq = lambda a, b: np.isclose(a, b, rtol=1e-9)
n, h = rd(f'{W}/pred_t028_id1.csv'), rd(H45)
AFIX = [2834, 2846]
L = pd.read_csv(f'{C9}/levers/h45m_test.csv')
k = ~L.id0.isin(AFIX)
assert eq(L.value[k].values, n[L.id0[k].values]).all()
I = L.copy()
I.to_csv(f'{W}/imp_{TAG}.csv', index=False, float_format='%.17g')
m = h.copy(); m[I.id0.values] = I.value.values
B = pd.concat([pd.read_csv(f'{C9}/b89/{f}_test.csv').assign(src=f) for f in ('b89t2', 'b89m2', 'b89x', 'b89y2')])
assert B.id0.is_unique and len(B) == 89
WB = pd.read_csv(f'{R}/cache/tmp/r9/work/b89y2/wb_rows.csv'); WB = WB[(WB.split == 'test') & (WB.trig == True)]
print('t2', WB.id.tolist())
Bb = B[~B.id0.isin(WB.id)]
hold = eq(m[Bb.id0.values], n[Bb.id0.values])
S = Bb[~hold].assign(r9h45=m[Bb.id0[~hold].values], r9n=n[Bb.id0[~hold].values], pred_t025=h[Bb.id0[~hold].values])
S[['id0', 'src', 'value', 'r9n', 'r9h45', 'pred_t025', 'status']].to_csv(f'{W}/{TAG2}_skipped.csv', index=False, float_format='%.17g')
print('t3', len(Bb), 't4', int(hold.sum()), '; skipped', sorted(S.id0.tolist()))
A = Bb[hold]
O = pd.concat([pd.DataFrame(dict(id0=A.id0, value=A.value, rule=A.src + ':' + A.status)),
               pd.DataFrame(dict(id0=WB.id, value=WB.new, rule='b89y2:white-ball trigger (' + WB.branch.astype(str) + ')'))]).sort_values('id0')
O.to_csv(f'{W}/imp_{TAG2}.csv', index=False, float_format='%.17g')
print('imp_r9h45b rows', len(O), O.rule.str.split(':').str[0].value_counts().to_dict(),
      't5', int((~eq(O.value.values, m[O.id0.values])).sum()))
