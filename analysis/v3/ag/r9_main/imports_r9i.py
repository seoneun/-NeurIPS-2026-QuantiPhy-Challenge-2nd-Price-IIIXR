import os, sys
import pandas as pd
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
out = sys.argv[1]
p17 = pd.read_csv(f'{R}/submissions/pred_t003_id1.csv').parsed_value.values
q = pd.read_csv(f'{R}/analysis/v3/calib/r9/q1/q1c_rows.csv'); q = q[(q.decision == 'putback_ok') & ~q.h34_fallback.astype(bool)]
pd.DataFrame({'id0': q.id0, 'value': p17[q.id0.values], 'rule': 'r9i_q1c_putback_researched_layer'}).to_csv(f'{out}/imp_r9i.csv', index=False)
print('')
