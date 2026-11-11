import sys, pandas as pd
base, out = sys.argv[1:3]
W = 'output/work/v15r03/meas'
b = pd.read_csv(base); b['id0'] = b.id - 1; b = b.set_index('id0')
def ld(tr, fx): return pd.read_csv(f'{W}/meas_test_v2_v15_{tr}_{fx}.csv').set_index('id')
A, B, A3, B3 = ld('v2sc', 'abcdef'), ld('v2sc', 'abdef'), ld('v3ntmv', 'abcdef'), ld('v3ntmv', 'abdef')
lev = []
for i in pd.read_csv('analysis/v3/ag/r9_main/levers/v15r03/in/rows.csv').id0:
    F = float(b.loc[i, 'parsed_value'])
    c = [(abs(F / B.loc[i, 'value'] - 1), A.loc[i, 'value'] / B.loc[i, 'value'], 'v2sc')]
    if pd.notna(B3.loc[i, 'value']) and pd.notna(A3.loc[i, 'value']) and B3.loc[i, 'value'] > 0:
        c.append((abs(F / B3.loc[i, 'value'] - 1), A3.loc[i, 'value'] / B3.loc[i, 'value'], 'v3ntmv'))
    c.sort(); lev.append((i, F * c[0][1]))
pd.DataFrame(lev, columns=['id0', 'value']).to_csv(out, index=False)
