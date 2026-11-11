import sys, numpy as np, pandas as pd
import os
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
h_csv, src_csv, out = sys.argv[1:4]
FALLBACK = {'base_z1s2c', 'base_h1chain', '01_h2.csv', '02_aud_tag.csv', '03b_aud_geo_z.csv', 'opt_A_copyfix.csv',
            'opt_B_h1chain_to_z1s2c.csv', '77_rul404.csv', '80_rul427.csv'}
Hf = pd.read_csv(h_csv); h = Hf.parsed_value.astype(float).values.copy()
p17 = pd.read_csv(f'{R}/submissions/pred_t003_id1.csv').parsed_value.astype(float).values
r8c = pd.read_csv(f'{R}/analysis/v3/calib/r8/cand/pred_t157_id1.csv').parsed_value.astype(float).values
A = pd.read_csv(f'{R}/team/member2/trackB/audit/rows_audit_1024.csv').set_index('id0')
fl = pd.read_csv(f'{R}/analysis/v3/calib/r9/flagged_pred_t157.csv').set_index('id0')
bad = set(fl.index[fl.our_bucket.isin(['4a_fact_other_item', '4b_fact_runtime_or_dev', '5_ourA_unexplained', '6_no_recipe'])])
src = pd.read_csv(src_csv).set_index('id0')['src'].astype(str).str.split('|').str[0].str.split(':').str[0]
eq = lambda a, b: np.isclose(a, b, rtol=1e-9)
ids = [i for i in A.index if A.at[i, 'team_status'] != 'X_LAB_VAL' and not eq(h[i], p17[i]) and eq(r8c[i], p17[i])
       and i not in bad and src.get(i, '') in FALLBACK]
rows = pd.DataFrame({'id0': ids, 'cat': A.loc[ids, 'cat'].values, 'h': h[ids], 'new': r8c[ids], 'h_src': src.reindex(ids).values})
h[ids] = r8c[ids]
o = Hf.copy(); o['parsed_value'] = h; o.to_csv(out, index=False); rows.to_csv(out.replace('.csv', '_rows.csv'), index=False)
print('')
