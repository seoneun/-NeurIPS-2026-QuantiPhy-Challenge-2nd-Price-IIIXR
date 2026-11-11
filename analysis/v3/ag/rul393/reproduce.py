import json, os, pandas as pd
R = '@QP_ROOT@'; HERE = os.path.dirname(os.path.abspath(__file__))
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
CT5 = f'{HERE}/frozen' if R5FZ else f'{R}/cache/tmp'
fr = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
H_new = float(fr.loc[1680, 'c26c'])
W_new = json.load(open(f'{CT5}/ag_rul395_patch_test.json'))['1697']['value']
P = {1690: H_new, 1696: H_new, 1681: W_new, 1691: W_new}
for i, v in sorted(P.items()):
    print(i, v)
