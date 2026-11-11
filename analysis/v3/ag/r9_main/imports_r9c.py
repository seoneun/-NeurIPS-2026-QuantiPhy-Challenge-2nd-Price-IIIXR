import sys, re
import os
import pandas as pd
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')); out = sys.argv[1]; Q = f'{R}/analysis/v3/calib/r9'
a = pd.read_csv(f'{Q}/q1/q1a_rows.csv'); a = a[a.decision == 'A'][['id0', 'replacement_value']].assign(rule='r9_q1a_A_replaced')
b = pd.read_csv(f'{Q}/q1/q1b_rows.csv'); b = b[b.keep_putback == 'no'][['id0', 'replacement_value']].assign(rule='r9_q1b_A_replaced')
pd.concat([a, b]).rename(columns={'replacement_value': 'value'}).to_csv(f'{out}/imp_q1.csv', index=False)
norm = lambda s: re.sub(r'\s+', ' ', str(s).strip().lower())
V = pd.read_csv(f'{R}/data/val/validation_dataset.csv'); V = V[V.video_id == 'simulation_0015']
vq = {norm(q): float(y) for q, y in zip(V.question, V.ground_truth_posterior)}
T = pd.read_csv(f'{R}/submissions/pred_t003_id1.csv'); T['id0'] = T.id - 1
H = pd.read_csv(f'{R}/team/member1/trackB/build/inputs/h1_rows.csv').set_index('id0')
t = T[T.video_id.isin(['simulation_0016', 'simulation_0017', 'simulation_0017_segmented']) & T.question.map(norm).isin(vq)]
t = t[t.id0.map(H.status) == 'X_LAB_VAL']
pd.DataFrame({'id0': t.id0, 'value': t.question.map(norm).map(vq), 'rule': 'r9_atrium_Vfact_val0015'}).to_csv(f'{out}/imp_atrium.csv', index=False)
s = pd.read_csv(f'{Q}/levers/survey2_test.csv'); s = s[s.rule.str.startswith(('member3_bundle_rej2_jm', 'member3_v35_D3labmotion'))]
s.to_csv(f'{out}/imp_survey2.csv', index=False)
print('')
