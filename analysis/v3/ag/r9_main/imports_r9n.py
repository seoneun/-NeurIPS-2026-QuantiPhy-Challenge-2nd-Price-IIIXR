import os, sys
import pandas as pd
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
q = pd.read_csv(f'{R}/analysis/v3/calib/r9/q1/q7b_rows.csv'); q = q[q.equal == 'within5pct']
pd.DataFrame({'id0': q.id0, 'value': q.rederived_value, 'rule': 'r9n_q7b_rederived_own_inputs'}).to_csv(f'{sys.argv[1]}/imp_r9n.csv', index=False)
print('')
