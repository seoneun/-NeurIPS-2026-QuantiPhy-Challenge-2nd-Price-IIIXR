import contextlib, io, os, runpy
import numpy as np, pandas as pd
R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
TOOL5 = f'{R}/analysis/v3/ag/rul392/frozen' if R5FZ else f'{R}/cache/tmp/r5_tool'
HERE = os.path.dirname(os.path.abspath(__file__))
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_path(os.path.join(HERE, 'garage_r5.py'))
MODES = g['MODES']




r2_1541 = float(np.median([g['res50'][('tire', m)][24][0] for m in MODES]))
v1541 = round(r2_1541, 2)
def strip_h(tag, video, hc, yh):
    e = g['ext'](tag, video); e = e[(e.border == 0) & (e.box_h >= 40)]
    yt, yb = e.y0.median(), e.y1.median()
    return (yb - yt) / (yb - yh) * hc
hc49 = 0.382 + g['sl'] * g['TD']; yh49 = g['ic']
yb50 = np.median([g['t50'][k][3] for k in g['full'][:, 0]]); hc50 = (yb50 - 240.0) * g['TD'] / g['D50']
r1_1519 = strip_h('r149mk', 'simulation_0149', hc49, yh49)
r2_1519 = strip_h('r150mk', 'simulation_0150', hc50, 240.0)
v1519 = round((r1_1519 + r2_1519) / 2, 2)
P = g['P49'](g['f49'])
r1_1523 = float(np.median([g['kin'](g['k49'], P, m)[24][0] for m in MODES]))
ms = pd.read_csv(f'{TOOL5}/measure_test_simulation_0149_r149t.csv')
r1b_1523 = float(ms.loc[ms.id0 == 1523, 'chain_new'].iloc[0])
v1523 = round((r1_1523 + r1b_1523) / 2, 2)

vals = {1541: v1541, 1519: v1519, 1523: v1523}
for k, v in vals.items():
    print(f'{k},{v}')

