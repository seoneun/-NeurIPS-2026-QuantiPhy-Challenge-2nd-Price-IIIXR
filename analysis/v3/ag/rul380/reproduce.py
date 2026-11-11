import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kin import trk, slope
R = '@QP_ROOT@'
import os
R5FZ = os.environ.get('R5_FROZEN', '1') != '0'
T5 = f'{R}/analysis/v3/ag/r5_common/frozen' if R5FZ else f'{R}/cache/tmp/r5'




def dog_speed(v, ts, t):
    Tc, Cc, _ = trk(v, 'black cat', ts, need_nb=True)
    mpp = 0.94 / np.linalg.norm(slope(Tc, Cc, -1, 99)[0])
    Td, Cd, _ = trk(v, 'dog', ts)
    return float(np.linalg.norm(slope(Td, Cd, t - 0.25, t + 0.25)[0]) * mpp)


def box_ff(v, phrase, t, ts='v3ntmv', w=0.125):
    T, C, _ = trk(v, phrase, ts)
    f = T <= 0.30 + 1e-9
    P = C[f] - C[f].mean(0); u = np.linalg.svd(P)[2][0]
    aff = abs(2 * np.polyfit(T[f], P @ u, 2)[0])
    return float(np.linalg.norm(slope(T, C, t - w, t + w)[0]) * 9.8 / aff)


def gmean(x):
    return float(np.exp(np.mean(np.log(x))))


def main():
    fr = pd.read_csv(f'{T5}/free_rows.csv').set_index('id0')
    out = []
    for id0, t in ((1616, 1.0), (1617, 1.5)):
        r1 = [dog_speed(v, ts, t) for v in ('simulation_0214', 'simulation_0217') for ts in ('v2sc', 'v3ntmv')]
        r2 = [dog_speed('simulation_0213', ts, t) for ts in ('v2sc', 'v3ntmv')]
        out.append((id0, gmean([np.mean(r1), np.mean(r2)])))
    g1428 = float(fr.loc[1428, 'g'])
    for id0, t, d2id in ((1429, 1.0, 441), (1430, 1.5, 442)):
        a = box_ff('simulation_0184', 'tool box', t); b = box_ff('simulation_0183', 'tool box', t)
        c = float(fr.loc[d2id, 'c26c']); d = float(fr.loc[id0, 'g']) * 9.8 / g1428
        out.append((id0, gmean([a, b, c, d])))
    for id0, new in out:
        print(id0, round(new, 4))


if __name__ == '__main__':
    main()
