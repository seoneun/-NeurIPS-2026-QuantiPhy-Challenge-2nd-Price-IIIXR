import json, os
import numpy as np, pandas as pd
R = '@QP_ROOT@'
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'


def series(v, o, d='tracks_test_v2sc'):
    js = json.load(open(f'{R}/geo/out/{d}/{v}.json')); out = []
    for i, f in enumerate(js['objs'].get(str(o), [])):
        st = f.get('st') if isinstance(f, dict) else None
        if st and st.get('box') and not st.get('border'):
            x0, y0, x1, y1 = st['box']
            out.append((js['fidx'][i], js['fidx'][i] / js['fps'], st['c'][0], st['c'][1], x1 - x0))
    return np.array(out)


def speed(S):
    return float(np.hypot(np.polyfit(S[:, 1], S[:, 2], 1)[0], np.polyfit(S[:, 1], S[:, 3], 1)[0]))


F = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
vA = 1.7393
own = []
for v, d, o in [('simulation_0247', 'tracks_test_v2sc', 1), ('simulation_0247', 'tracks_test_v3ntmv', 1),
                ('simulation_0247_segmented', 'tracks_test_v2sc', 3), ('simulation_0247_segmented', 'tracks_test_v3ntmv', 1)]:
    S = series(v, o, d); own.append(1.277 * np.median(S[:, 4]) / speed(S))
r1 = float(np.median(own)); oth = [float(F.c26c[i]) for i in (817, 1977, 820, 826, 1980, 830)]; r2 = float(np.median(oth))
vD = float(np.sqrt(r1 * r2))
g = pd.read_csv(f'{R}/analysis/v3/ag/rul372/rows.csv').set_index('id0')
vG = float(g.new[431])
for i, v in ((214, vA), (1628, vD), (1985, vD), (431, vG)):
    print(i, round(v, 6))
