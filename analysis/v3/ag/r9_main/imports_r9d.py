import sys
import os
import pandas as pd
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')); out = sys.argv[1]
FB = ('base_z1s2c', 'base_h1chain', '01_h2', '02_aud_tag', '03b_aud_geo_z', 'opt_A', 'opt_B', '77_rul404', '80_rul427')
q = pd.read_csv(f'{R}/analysis/v3/calib/r9/q1/q1c_rows.csv')
src = pd.read_csv(os.environ.get('QP_HSRC') or f'{R}/team/member1/trackB/submission/pred_t017_id1_src.csv').set_index('id0').src.str.split('|').str[0]
p17 = pd.read_csv(f'{R}/submissions/pred_t003_id1.csv'); p17 = pd.Series(p17.parsed_value.values, index=p17.id - 1)
q = q[(q.decision == 'putback_ok') & q.id0.map(lambda i: str(src.get(i, '')).startswith(FB))]
pd.DataFrame({'id0': q.id0, 'value': q.id0.map(p17), 'rule': 'r9_q1c_putback_ok'}).to_csv(f'{out}/imp_q1c.csv', index=False)
print(f'imp_q1c {len(q)} rows', q.cat.value_counts().to_dict())
