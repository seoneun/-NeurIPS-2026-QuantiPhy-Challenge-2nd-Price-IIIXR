import os, sys, json
import numpy as np, pandas as pd
R = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'
W = f'{R}/analysis/v3/ag/rul400'

FR = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
T = pd.read_pickle(f'{R}/analysis/v3/ag/rul346/tab.pkl'); FT = T[T.split == 'test'].set_index('id0').F
sw_path = f'{R}/analysis/v3/ag/rul400/frozen/swap35b.csv' if R5FZ else f'{R}/cache/tmp/rul400/swap35b.csv'
if not os.path.exists(sw_path):
    os.system(f'{sys.executable} {W}/swap35b.py > /dev/null 2>&1')
SW = pd.read_csv(sw_path); GS = SW[SW.reading == 'S'].set_index('id0').value


def tape_value():
    j = json.load(open(f'{R}/geo/out/tracks_test_v3ntmv/captured_0040.json'))
    o = [str(q['obj']) for q in j['queries'] if q['phrase'] == 'tape'][0]
    px = np.median([max(e['st']['box'][2] - e['st']['box'][0], max(e['st']['rr'][:2])) for e in j['objs'][o] if e and e.get('st')])
    raw = pd.read_csv(f'{R}/preds/test_geo_v4ntrj5sQ2srngsr_d_vw_raw.csv')
    f = float(raw.meas_f.iloc[2745]) * j['W']
    return px * 0.973 / f * 100


ROWS = [([2745, 2753, 2761], tape_value())]
ids = [2614, 2623, 2632]
ROWS.append((ids, float(np.median([FR.g[i] * FT[i] for i in ids]))))
for ids in ([2639, 2648, 2657], [2646, 2655, 2664], [2641, 2650, 2659], [2645, 2654, 2663]):
    ROWS.append((ids, float(np.median([GS[i] * FT[i] for i in ids]))))

out = {}
for ids, val in ROWS:
    for i in ids:
        new = float(val)
        out[str(i)] = dict(value=new)
        print(f'{i},{new:.6g}')
