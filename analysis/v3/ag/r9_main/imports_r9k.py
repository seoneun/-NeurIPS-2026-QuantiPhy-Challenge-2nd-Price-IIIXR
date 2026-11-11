import os, sys
import pandas as pd
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
out, c15f = sys.argv[1:3]
def vals(p): return pd.read_csv(p).parsed_value.values
c15, p17 = pd.read_csv(c15f).value.values, vals(f'{R}/submissions/pred_t003_id1.csv')
rows = [(i, c15[i], 'r9k_pred_t018_for_zteam_value') for i in [1943, 1944, 1948, 1898, 2410, 2419, 2428, 2949, 3011]]
rows += [(i, p17[i], 'r9k_q1e_allowed_on_researched_layer') for i in [1121, 1346]]
pd.DataFrame(rows, columns=['id0', 'value', 'rule']).to_csv(f'{out}/imp_r9k.csv', index=False)
print('')
