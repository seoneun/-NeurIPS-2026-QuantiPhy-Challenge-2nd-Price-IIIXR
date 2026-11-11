import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pool
R = '@QP_ROOT@'
sub = pd.read_csv(R + '/submissions/pred_t014_id1.csv'); c26 = dict(zip(sub.index, sub.parsed_value))
r2 = c26[701]
rows = []
for i0, v in [(1952, 'simulation_0225'), (1956, 'simulation_0227')]:
    sp, pitch, s, top = pool.spacing(v)
    rows.append((i0, float(np.sqrt(sp * r2))))
rows.append((1395, float(c26[399])))
out = {}
for i0, new in rows:
    print(f'{i0}, {new:.6g}')
    out[str(i0)] = {'value': new}
