import sys, numpy as np, pandas as pd
sys.path.insert(0, 'analysis/v3/ag/r9_main/levers/kincons3'); sys.path.insert(0, 'analysis/v3/ag/r9_main/levers/kincons2')
from k2 import load, series
from prof3 import kfit
IN = 'geo/out/tracks_test_qv3ntmv_mq2off'
BAND = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4)
def trk(v, ph):
    j, q, R = load(v, ph, f'{IN}/{v}__all.json'); return R, series(R)
def acc(S, t, sc):
    return [kfit(S, t, H)[1] * sc for H in BAND if kfit(S, t, H)]
OUT = []
R, S = trk('simulation_0221', 'vacuum cleaner'); sc = 0.438 / float(np.median([r[4] for r in R]))
OUT.append((663, float(np.median(acc(S, 2.0, sc)))))
R, S = trk('internet_0072', 'credit card'); sc = 8.56 / float(np.median([r[8][0] for r in R]))
OUT.append((3273, float(np.median(acc(S, 0.5, sc)))))
T = pd.DataFrame(OUT, columns=['id0', 'value']); T['value'] = T.value.round(4)
T.to_csv('analysis/v3/calib/r9/levers/kincons3_test.csv', index=False)
