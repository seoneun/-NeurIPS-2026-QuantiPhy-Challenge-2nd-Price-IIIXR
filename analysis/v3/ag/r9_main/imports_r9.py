import sys, re
import os
import pandas as pd
R = os.environ.get('QP_ROOT') or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
SW = f'{R}/team/member2/trackB'
put, srcf, out, c15f = sys.argv[1:5]
FB = ('base_z1s2c', 'base_h1chain', '01_h2', '02_aud_tag', '03b_aud_geo_z', 'opt_A', 'opt_B', '77_rul404', '80_rul427')
def load(p):
    d = pd.read_csv(p); return pd.Series(d.parsed_value.astype(float).values, index=d.id - 1)
T = pd.read_csv(f'{R}/submissions/pred_t003_id1.csv'); T['id0'] = T.id - 1
p17, P, c15 = load(f'{R}/submissions/pred_t003_id1.csv'), load(put), pd.read_csv(c15f).set_index('id0').value.astype(float)
eq = lambda a, b: abs(a - b) <= 1e-9 * max(1.0, abs(b))
tl = pd.read_csv(f'{SW}/audit/teamline_candidates.csv')
A = pd.read_csv(f'{SW}/audit/rows_audit_1024.csv').set_index('id0')
tl = tl[tl.c15_eq_pre & tl.id0.map(lambda i: not eq(P[i], p17[i])) & (tl.id0.map(A.audit_verdict) == 'X:X_LAB')]
pd.DataFrame({'id0': tl.id0, 'value': tl.id0.map(p17), 'rule': 'r9_q3_member2_allowed_block_Vfact'}).to_csv(f'{out}/imp_blocks2.csv', index=False)
lab = set(pd.read_csv(f'{R}/team/member1/trackB/build/inputs/h1_rows.csv').query("status=='X_LAB_VAL'").id0)
src = pd.read_csv(srcf).set_index('id0').src.str.split('|').str[0]
def qty(q):
    q = q.lower()
    if 'acceleration' in q: return 'acc'
    if 'speed' in q or 'velocity' in q or 'how fast' in q: return 'speed'
    if 'distance' in q or 'how far' in q or 'displacement' in q: return 'dist'
    return 'size'
T['cat'] = T.inference_type.str[0] + T.video_type.str[1]
S = T[(T.video_source == 'simulation') & (T.question.map(qty) == 'size') & ~T.id0.isin(lab) & T.cat.isin(['S2', 'D2'])]
S = S[S.id0.map(lambda i: str(src.get(i, '')).startswith(FB) and not eq(P[i], p17[i]))]
pd.DataFrame({'id0': S.id0, 'value': S.id0.map(c15), 'rule': 'r9_q2_pred_t018_simsize'}).to_csv(f'{out}/imp_pred_t018_simsize.csv', index=False)
print('')
