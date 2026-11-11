import os
import numpy as np, pandas as pd
R = '@QP_ROOT@'
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
F = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')


def c26(i): return float(F.loc[i, 'c26c'])


def proposals():
    return [dict(id0=1325, new=round(float(np.median([c26(i) for i in [1250, 1274, 1287, 1300, 1313]])), 4)),
            dict(id0=1460, new=round(c26(1450), 4)),
            dict(id0=1227, new=round(float(np.median([c26(i) for i in [1097, 1132, 1211, 1220]])), 4))]


if __name__ == '__main__':
    for p in proposals():
        print(p['id0'], p['new'])
