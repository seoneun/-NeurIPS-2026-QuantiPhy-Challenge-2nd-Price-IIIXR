import sys, re
import os
import pandas as pd
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')); base, out = sys.argv[1:3]
RS = f'{R}/team/member1/trackB/rounds'
B = pd.read_csv(base); B = pd.Series(B.parsed_value.astype(float).values, index=B.id - 1)
a = pd.read_csv(f'{RS}/imp_S2/cand_optional.csv')[['id0', 'value']].assign(rule='r9_survey1_imp_S2_rigid_geo')
f = pd.read_csv(f'{RS}/pc_inet/cand_fwd5.csv')
f['ratio'] = f.rule.str.extract(r'ratio=([\d.]+)')[0].astype(float)
B2 = B.copy(); B2[a.id0.values] = a.value.values
f = pd.DataFrame({'id0': f.id0, 'value': f.id0.map(B2) * f.ratio, 'rule': 'r9_survey1_pc_inet_fwd_ratio'})
pd.concat([a[~a.id0.isin(f.id0)], f]).to_csv(f'{out}/imp_r9e.csv', index=False)
print('rows in both rules:', int(a.id0.isin(f.id0).sum()))
print('')
